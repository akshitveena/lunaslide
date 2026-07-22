"""Stage 1 model building blocks.

The boulder detector intentionally requires a lunar-domain YOLO checkpoint.
Generic COCO YOLO weights have no lunar-boulder class and must not be used to
generate operational hazard evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import box_iou

from .contracts import BoulderDetection


class ModelUnavailableError(RuntimeError):
    """Raised when an optional model runtime or domain checkpoint is absent."""


class YOLOv8BoulderDetector:
    """Thin, explicit adapter around a *lunar fine-tuned* YOLOv8 checkpoint."""

    def __init__(self, checkpoint: str | Path, confidence_threshold: float = 0.25) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ModelUnavailableError(
                f"Lunar YOLOv8 checkpoint not found: {checkpoint}. "
                "Train/fetch a checkpoint whose label set contains 'boulder'."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError(
                "ultralytics is required for YOLOv8 inference. Install requirements.txt first."
            ) from exc
        self.model = YOLO(str(checkpoint))
        self.confidence_threshold = confidence_threshold

    def predict(self, image: np.ndarray) -> list[BoulderDetection]:
        """Return boulder candidates in the input image's pixel coordinates."""
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        result = results[0]
        names = result.names
        detections: list[BoulderDetection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id]).lower()
            if class_name != "boulder":
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                BoulderDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                )
            )
        return detections


def train_yolov8_boulders(dataset_yaml: str | Path, output_project: str | Path, epochs: int = 100) -> None:
    """Train a lunar-boulder YOLOv8 model from an Ultralytics dataset YAML file."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelUnavailableError("ultralytics is required for YOLOv8 training.") from exc
    # COCO weights initialise generic visual features; labels are replaced by
    # the dataset's explicit single 'boulder' class during training.
    YOLO("yolov8n.pt").train(data=str(dataset_yaml), epochs=epochs, project=str(output_project), name="boulder")


def build_mask_rcnn_boulder_refiner(pretrained_backbone: bool = True) -> nn.Module:
    """Create the two-class Mask R-CNN used to refine boulder instances.

    Class 0 is background and class 1 is boulder.  Train this model on
    instance masks before using it for inference.
    """
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return maskrcnn_resnet50_fpn(weights=None, weights_backbone=weights_backbone, num_classes=2)


class MaskRCNNBoulderRefiner:
    """Refine YOLO candidates with a Mask R-CNN trained on boulder masks."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", confidence_threshold: float = 0.4) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ModelUnavailableError(f"Mask R-CNN checkpoint not found: {checkpoint}")
        self.device, self.confidence_threshold = torch.device(device), confidence_threshold
        self.model = build_mask_rcnn_boulder_refiner(pretrained_backbone=False).to(self.device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True)["model"])

    def refine(self, image: np.ndarray, candidates: list[BoulderDetection]) -> list[tuple[BoulderDetection, np.ndarray]]:
        tensor = torch.from_numpy(image).float().div(255).repeat(3, 1, 1).to(self.device)
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        if not candidates or len(prediction["boxes"]) == 0:
            return []
        candidate_boxes = torch.tensor([item.bbox_xyxy for item in candidates], device=self.device)
        overlaps = box_iou(prediction["boxes"], candidate_boxes).amax(dim=1)
        accepted = (prediction["scores"] >= self.confidence_threshold) & (overlaps >= 0.1)
        refined = []
        for box, score, mask in zip(prediction["boxes"][accepted], prediction["scores"][accepted], prediction["masks"][accepted]):
            x1, y1, x2, y2 = (float(value) for value in box.tolist())
            refined.append((BoulderDetection((x1, y1, x2, y2), float(score)), mask[0].cpu().numpy() >= 0.5))
        return refined


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class ResNet50UNet(nn.Module):
    """Binary historical-debris segmenter with an optional pretrained encoder.

    Output is logits, not probabilities.  Use ``torch.sigmoid`` only at
    inference/evaluation time; train with ``BCEWithLogitsLoss`` or a combined
    Dice/BCE objective.
    """

    def __init__(self, pretrained_encoder: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained_encoder else None
        encoder = resnet50(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.decode3 = _DecoderBlock(2048, 1024, 512)
        self.decode2 = _DecoderBlock(512, 512, 256)
        self.decode1 = _DecoderBlock(256, 256, 128)
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        encoded = self.layer4(skip3)
        x = self.decode3(encoded, skip3)
        x = self.decode2(x, skip2)
        x = self.decode1(x, skip1)
        x = self.head(x)
        return nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
