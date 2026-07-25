"""Propose and filter boulder candidates for SAM-assisted labelling.

Boulders in LROC NAC imagery read as small, high-contrast features: a sunlit cap
beside a hard-edged shadow.  This module finds those candidates cheaply so SAM
only has to refine a handful of prompts per patch, and filters SAM's masks down
to boulder-like shapes.

The output is *draft* labels for human review, never final training data
unreviewed — a detector trained on unverified auto-labels inherits their
mistakes.  Pure geometry lives here (testable offline); SAM and I/O live in
``scripts/label_boulders.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    """An axis-aligned pixel box, ``x1 <= x2`` and ``y1 <= y2``."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_yolo(self, image_w: int, image_h: int, cls: int = 0) -> str:
        """YOLO line: ``cls cx cy w h`` normalised to ``[0, 1]``."""
        cx = (self.x1 + self.x2) / 2 / image_w
        cy = (self.y1 + self.y2) / 2 / image_h
        return f"{cls} {cx:.6f} {cy:.6f} {self.width / image_w:.6f} {self.height / image_h:.6f}"


def propose_candidates(
    gray: np.ndarray,
    *,
    max_boulder_px: int = 24,
    min_contrast: float = 18.0,
    max_candidates: int = 40,
    max_ramp: float = 2.5,
) -> list[tuple[int, int]]:
    """Return ``(x, y)`` points likely to sit on a boulder.

    A white top-hat isolates bright features smaller than ``max_boulder_px``
    (boulder sunlit caps); candidates are the brightest such blobs that also
    have strong local contrast, since a boulder is bright *against* nearby
    shadow.  Returns points, not boxes — SAM turns each into a precise mask.

    Candidates sitting on a large-scale illumination ramp are rejected: a
    top-hat still fires on the bright side of a terminator or crater-shadow
    edge, but a boulder is a local peak on roughly flat ground, not a point on a
    steep bright-to-dark gradient.  ``max_ramp`` is the largest background
    gradient (DN per pixel, after heavy blur) a candidate may sit on.
    """
    if gray.ndim != 2:
        raise ValueError("propose_candidates expects a single-channel image.")
    image = gray.astype(np.uint8)
    ksize = 2 * max_boulder_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # Large-scale illumination field: blur away boulder-scale detail, then its
    # gradient magnitude marks terminator and crater-shadow boundaries.
    background = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=float(max_boulder_px))
    gy, gx = np.gradient(background)
    ramp = np.hypot(gx, gy)

    # Keep only strong bright residuals, then take one point per connected blob.
    threshold = max(min_contrast, float(np.percentile(tophat, 99)))
    mask = (tophat >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates: list[tuple[int, int, float]] = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 2 or area > max_boulder_px * max_boulder_px:
            continue
        cx, cy = centroids[label]
        ix, iy = int(round(cx)), int(round(cy))
        if ramp[iy, ix] > max_ramp:  # on a shadow/terminator boundary, not a boulder
            continue
        candidates.append((ix, iy, float(tophat[iy, ix])))

    candidates.sort(key=lambda c: c[2], reverse=True)
    return [(x, y) for x, y, _ in candidates[:max_candidates]]


def box_from_mask(mask: np.ndarray) -> Box | None:
    """Tight bounding box of a boolean mask, or ``None`` if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def is_boulder_like(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    *,
    min_area: int = 4,
    max_area_fraction: float = 0.08,
    max_aspect: float = 3.0,
    min_fill: float = 0.45,
    min_brightness_margin: float = 6.0,
) -> bool:
    """Whether a SAM mask is a boulder: small, compact, and a bright cap.

    Geometry rejects terrain-scale segments (too large), cracks/ridges (too
    elongated), and ragged masks (low box fill).  When ``image`` is given, a
    photometric check also requires the mask to be brighter than its immediate
    surroundings by ``min_brightness_margin`` DN — a boulder's sunlit cap is
    locally bright, so this rejects SAM masks that latched onto a crater shadow
    or the dark side of an edge.
    """
    area = int(mask.sum())
    if area < min_area or area > max_area_fraction * mask.size:
        return False
    box = box_from_mask(mask)
    if box is None or box.width == 0 or box.height == 0:
        return False
    aspect = max(box.width, box.height) / min(box.width, box.height)
    if aspect > max_aspect:
        return False
    if area / (box.width * box.height) < min_fill:
        return False
    if image is not None:
        ring = _ring(mask)
        if not ring.any():
            return False
        values = image.astype(np.float32)
        if float(values[mask].mean()) <= float(values[ring].mean()) + min_brightness_margin:
            return False
    return True


def _ring(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Boolean ring of pixels just outside ``mask``."""
    solid = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    dilated = cv2.dilate(solid, kernel) > 0
    return dilated & ~mask.astype(bool)


def deduplicate(boxes: list[Box], iou_threshold: float = 0.5) -> list[Box]:
    """Drop near-duplicate boxes (different prompts hitting one boulder)."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b.width * b.height, reverse=True):
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union
