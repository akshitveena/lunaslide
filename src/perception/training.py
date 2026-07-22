"""Training loops for Stage 1 models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .enhancement_model import CurveEnhancer, SelfGuidedEnhancementLoss
from .models import ResNet50UNet, build_mask_rcnn_boulder_refiner


def _device(value: str) -> torch.device:
    return torch.device(value if value != "auto" else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    numerator = 2 * (probabilities * targets).sum(dim=(1, 2, 3)) + 1
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def train_enhancer(loader: DataLoader, epochs: int, output: str | Path, device: str = "auto") -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime = _device(device)
    model, criterion = CurveEnhancer().to(runtime), SelfGuidedEnhancementLoss().to(runtime)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        model.train()
        for image in loader:
            image = image.to(runtime)
            enhanced, curves = model(image)
            loss = criterion(image, enhanced, curves)["total"]
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save({"model": model.state_dict(), "kind": "curve_enhancer"}, target)
    return target


def train_segmenter(loader: DataLoader, epochs: int, output: str | Path, device: str = "auto") -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime = _device(device)
    model = ResNet50UNet(pretrained_encoder=True).to(runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for image, mask in loader:
            image, mask = image.to(runtime), mask.to(runtime)
            logits = model(image)
            loss = bce(logits, mask) + _dice_loss(logits, mask)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save({"model": model.state_dict(), "kind": "resnet50_unet_historical_debris"}, target)
    return target


def train_mask_rcnn(loader: DataLoader, epochs: int, output: str | Path, device: str = "auto") -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime = _device(device)
    model = build_mask_rcnn_boulder_refiner(pretrained_backbone=True).to(runtime)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            images = [image.to(runtime) for image in images]
            targets = [{key: value.to(runtime) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save({"model": model.state_dict(), "kind": "mask_rcnn_boulder"}, target)
    return target
