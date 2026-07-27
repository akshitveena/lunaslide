"""Run the trained two-class detector on a lunar image and count features.

Tiles the image at the detector's training scale (256 px) and applies the same
classical enhancement it was trained on — running a detector on a different
preprocessing than it saw in training is the train/serve skew we fixed for the
enhancer, and it applies just as much here.

The returned reliability (validated recall) travels with the counts so Stage 3
knows how far to trust them: a low-recall detector's *detections* are useful
hazard evidence, but its *absence* of detections cannot clear a site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .lroc import iter_tiles
from .prepare import enhance_image


@dataclass(frozen=True)
class DetectionSummary:
    boulders: int
    craters: int
    tiles: int
    checkpoint: str
    recall: float | None = None   # validated recall, if known

    def counts(self) -> dict[str, int]:
        return {"boulder": self.boulders, "crater": self.craters}


def detect_features(
    image_path: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    size_px: int = 256,
    conf: float = 0.25,
    recall: float | None = None,
) -> DetectionSummary:
    """Count boulders and craters in an image with the trained YOLO."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read {image_path}")

    boulders = craters = tiles = 0
    for tile, _, _ in iter_tiles(gray, size_px, size_px):
        enhanced, _ = enhance_image(tile)  # same classical path the detector trained on
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        result = model(rgb, conf=conf, device=device, verbose=False)[0]
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            boulders += int((classes == 0).sum())
            craters += int((classes == 1).sum())
        tiles += 1
    return DetectionSummary(boulders, craters, tiles, str(checkpoint), recall)
