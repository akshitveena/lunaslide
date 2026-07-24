"""End-to-end Stage 1 inference and visual-evidence export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import GeoReference, VisualEvidence
from .geospatial import merge_georeference
from .preprocessing import enhance_lunar_image


def _texture_roughness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_32F).std())


def _load_curve_checkpoint(path: str | Path, device):
    import torch
    from .enhancement_model import CurveEnhancer
    model = CurveEnhancer().to(device).eval()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model


def _load_segmenter_checkpoint(path: str | Path, device):
    import torch
    from .models import ResNet50UNet
    model = ResNet50UNet(pretrained_encoder=False).to(device).eval()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model


def run_stage1(
    image_path: str | Path,
    output_dir: str | Path,
    georef: GeoReference,
    *,
    enhancer_checkpoint: str | Path | None = None,
    yolo_checkpoint: str | Path | None = None,
    maskrcnn_checkpoint: str | Path | None = None,
    debris_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> VisualEvidence:
    """Create a complete, auditable Stage 1 evidence package.

    Model-dependent outputs are included only when their trained checkpoint is
    provided. This prevents a partial installation from being misrepresented as
    an all-clear hazard assessment.

    Any CRS, affine transform, or pixel size the source raster carries is
    folded into ``georef`` so Stage 3 can align this evidence with Stage 2's
    hazard grid.  Values the caller supplied explicitly are preserved.
    """
    image_path, output_dir = Path(image_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    georef = merge_georeference(georef, image_path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    enhanced, report = enhance_lunar_image(raw)
    if enhancer_checkpoint:
        import torch
        runtime = torch.device(device)
        model = _load_curve_checkpoint(enhancer_checkpoint, runtime)
        source = torch.from_numpy(enhanced).float().div(255).unsqueeze(0).unsqueeze(0).to(runtime)
        with torch.inference_mode():
            enhanced = (model(source)[0][0, 0].cpu().numpy() * 255).round().astype(np.uint8)
    enhanced_path = output_dir / "enhanced.png"
    cv2.imwrite(str(enhanced_path), enhanced)

    boulders = []
    if yolo_checkpoint:
        from .models import YOLOv8BoulderDetector
        boulders = YOLOv8BoulderDetector(yolo_checkpoint).predict(enhanced)
    if maskrcnn_checkpoint:
        from .models import MaskRCNNBoulderRefiner
        if not boulders:
            raise ValueError("Mask R-CNN refinement requires YOLO boulder candidates.")
        refined = MaskRCNNBoulderRefiner(maskrcnn_checkpoint, device=device).refine(enhanced, boulders)
        boulders = []
        for index, (detection, mask) in enumerate(refined):
            mask_path = output_dir / f"boulder_instance_{index}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            boulders.append(type(detection)(detection.bbox_xyxy, detection.confidence, instance_mask_path=str(mask_path)))

    debris_path: str | None = None
    if debris_checkpoint:
        import torch
        runtime = torch.device(device)
        model = _load_segmenter_checkpoint(debris_checkpoint, runtime)
        tensor = torch.from_numpy(enhanced).float().div(255).repeat(3, 1, 1).unsqueeze(0).to(runtime)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        debris = (probabilities >= 0.5).astype(np.uint8) * 255
        mask_file = output_dir / "historical_debris_mask.png"
        cv2.imwrite(str(mask_file), debris)
        debris_path = str(mask_file)

    evidence = VisualEvidence(
        georef=georef,
        boulders=boulders,
        historical_debris_mask_path=debris_path,
        shadow_fraction=report.shadow_fraction,
        texture_roughness=_texture_roughness(enhanced),
        preprocessing_report=report.to_dict(),
        model_versions={
            "enhancer": str(enhancer_checkpoint) if enhancer_checkpoint else "classical-gamma-clahe",
            "boulder_detector": str(yolo_checkpoint) if yolo_checkpoint else "not-run",
            "boulder_refiner": str(maskrcnn_checkpoint) if maskrcnn_checkpoint else "not-run",
            "debris_segmenter": str(debris_checkpoint) if debris_checkpoint else "not-run",
        },
    )
    (output_dir / "visual_evidence.json").write_text(json.dumps(evidence.to_dict(), indent=2) + "\n", encoding="utf-8")
    return evidence
