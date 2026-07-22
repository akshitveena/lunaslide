"""Datasets used to train the Stage 1 enhancer and debris segmenter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import enhance_lunar_image

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in _EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def read_gray(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized, _ = enhance_lunar_image(image)
    return torch.from_numpy(normalized).float().div(255.0).unsqueeze(0)


class LowLightImageDataset(Dataset[torch.Tensor]):
    def __init__(self, image_dir: str | Path) -> None:
        self.paths = image_paths(image_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Enhancer input is intentionally normalized but not CLAHE-enhanced.
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read {self.paths[index]}")
        return torch.from_numpy(image).float().div(255.0).unsqueeze(0)


class DebrisSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs images/masks with identical relative paths and file names."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path) -> None:
        self.image_root, self.mask_root = Path(image_dir), Path(mask_dir)
        self.paths = image_paths(self.image_root)
        missing = [path for path in self.paths if not (self.mask_root / path.relative_to(self.image_root)).exists()]
        if missing:
            raise ValueError(f"Missing masks for {len(missing)} images; first: {missing[0].name}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        image = read_gray(path).repeat(3, 1, 1)
        mask_path = self.mask_root / path.relative_to(self.image_root)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        mask = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)
        return image, mask


class BoulderInstanceDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Mask R-CNN data using ``.npz`` instance masks beside each image.

    For ``images/train/a.png``, provide ``masks/train/a.npz`` containing a
    boolean/uint8 array named ``masks`` with shape ``[instances, height, width]``.
    """

    def __init__(self, image_dir: str | Path, mask_dir: str | Path) -> None:
        self.image_root, self.mask_root = Path(image_dir), Path(mask_dir)
        self.paths = image_paths(self.image_root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.paths[index]
        image = read_gray(image_path).repeat(3, 1, 1)
        mask_path = (self.mask_root / image_path.relative_to(self.image_root)).with_suffix(".npz")
        if not mask_path.exists():
            raise ValueError(f"Missing instance masks: {mask_path}")
        masks = np.load(mask_path)["masks"].astype(bool)
        if masks.ndim != 3 or masks.shape[0] == 0:
            raise ValueError(f"{mask_path} must contain one or more [N,H,W] masks")
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError(f"Empty boulder mask in {mask_path}")
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks.astype(np.uint8)),
            "image_id": torch.tensor([index]),
        }
        return image, target
