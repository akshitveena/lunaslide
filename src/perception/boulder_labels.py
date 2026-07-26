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

    ramp = _illumination_ramp(image, max_boulder_px)

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
    if not passes_geometry(mask, min_area=min_area, max_area_fraction=max_area_fraction,
                           max_aspect=max_aspect, min_fill=min_fill):
        return False
    if image is not None:
        ring = _ring(mask)
        if not ring.any():
            return False
        values = image.astype(np.float32)
        if float(values[mask].mean()) <= float(values[ring].mean()) + min_brightness_margin:
            return False
    return True


def passes_geometry(
    mask: np.ndarray,
    *,
    min_area: int = 4,
    max_area_fraction: float = 0.08,
    max_aspect: float = 3.0,
    min_fill: float = 0.45,
) -> bool:
    """Compact, small, roughly-round shape gate shared by boulders and craters.

    Photometry (boulder cap vs crater shadow) is handled separately by
    :func:`classify_relief`; this only rejects terrain-scale segments, ridges,
    and ragged masks so both feature types pass the same size/shape sieve.
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
    return area / (box.width * box.height) >= min_fill


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


# --- Boulder vs crater by shadow geometry -----------------------------------
#
# A boulder is positive relief: sunlit cap on the sun side, shadow cast beyond
# it on the anti-sun side.  A crater is negative relief: its sun-facing interior
# wall is in shadow while the far (anti-sun) wall is lit.  So across a feature,
# the sun-side is BRIGHTER than the anti-sun-side for a boulder and DARKER for a
# crater.  That single asymmetry, measured against the sun direction, separates
# them -- and turns a boulder-only detector into a two-class one.

CLASS_NAMES = ("boulder", "crater")


def estimate_sun_vector(image: np.ndarray) -> tuple[float, float]:
    """Unit vector pointing toward the sun in image coordinates (x right, y down).

    Cast shadows are the darkest pixels and sit on the anti-sun side of whatever
    cast them, so from a shadow the nearest lit ground lies sunward.  Averaging
    the direction to the nearest bright region over all shadow pixels recovers
    the sun azimuth without needing image metadata.
    """
    gray = image.astype(np.float32)
    bright = gray >= np.percentile(gray, 75)
    shadow = gray <= np.percentile(gray, 20)
    if not bright.any() or not shadow.any():
        return (1.0, 0.0)
    # Distance to the nearest bright pixel; its negative gradient points sunward.
    distance = cv2.distanceTransform((~bright).astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(distance)
    sx, sy = -float(gx[shadow].mean()), -float(gy[shadow].mean())
    norm = float(np.hypot(sx, sy))
    return (1.0, 0.0) if norm < 1e-6 else (sx / norm, sy / norm)


def classify_relief(
    mask: np.ndarray,
    image: np.ndarray,
    sun_vector: tuple[float, float],
    *,
    margin: float = 4.0,
    pad: int = 4,
) -> str | None:
    """Classify a feature as ``"boulder"``, ``"crater"``, or ``None`` (ambiguous).

    Splits a window around the feature into a sun-facing half and an anti-sun
    half and compares their mean brightness.  Brighter toward the sun is a
    boulder (lit cap, shadow beyond); brighter away from the sun is a crater
    (near wall shadowed, far wall lit).  Independent of which part SAM
    segmented, because it reads the whole local neighbourhood's asymmetry.
    """
    box = box_from_mask(mask)
    if box is None:
        return None
    h, w = image.shape
    x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
    x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    window = image[y1:y2, x1:x2].astype(np.float32)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    yy, xx = np.mgrid[y1:y2, x1:x2]
    # Signed distance along the sun direction from the feature centre.
    projection = (xx - cx) * sun_vector[0] + (yy - cy) * sun_vector[1]
    sun_side, anti_side = projection > 0, projection < 0
    if not sun_side.any() or not anti_side.any():
        return None
    difference = float(window[sun_side].mean()) - float(window[anti_side].mean())
    if difference > margin:
        return "boulder"
    if difference < -margin:
        return "crater"
    return None


def _illumination_ramp(image: np.ndarray, sigma_px: int) -> np.ndarray:
    """Large-scale illumination gradient magnitude (marks shadow boundaries)."""
    background = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=float(sigma_px))
    gy, gx = np.gradient(background)
    return np.hypot(gx, gy)


def propose_dark_candidates(
    gray: np.ndarray,
    *,
    max_feature_px: int = 24,
    min_contrast: float = 18.0,
    max_candidates: int = 40,
    max_ramp: float = 2.5,
) -> list[tuple[int, int]]:
    """Propose points on small dark features (crater shadows/interiors).

    The black-hat mirror of :func:`propose_candidates`, and it applies the same
    illumination-ramp rejection: a black-hat also fires on the dark side of a
    terminator, so candidates on a steep large-scale gradient are dropped.
    Relief classification decides afterwards whether each is truly a crater.
    """
    if gray.ndim != 2:
        raise ValueError("propose_dark_candidates expects a single-channel image.")
    image = gray.astype(np.uint8)
    ksize = 2 * max_feature_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    ramp = _illumination_ramp(image, max_feature_px)
    threshold = max(min_contrast, float(np.percentile(blackhat, 99)))
    mask = (blackhat >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[int, int, float]] = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 2 or area > max_feature_px * max_feature_px:
            continue
        cx, cy = centroids[label]
        ix, iy = int(round(cx)), int(round(cy))
        if ramp[iy, ix] > max_ramp:
            continue
        candidates.append((ix, iy, float(blackhat[iy, ix])))
    candidates.sort(key=lambda c: c[2], reverse=True)
    return [(x, y) for x, y, _ in candidates[:max_candidates]]


def deduplicate_features(
    features: list[tuple[Box, int]], iou_threshold: float = 0.4
) -> list[tuple[Box, int]]:
    """Drop overlapping boxes across all classes, keeping the largest.

    A single feature can be proposed by both the bright and dark passes and come
    back as a boulder AND a crater; one object cannot be both, so cross-class
    overlaps are collapsed here rather than only within a class.
    """
    kept: list[tuple[Box, int]] = []
    for box, cls in sorted(features, key=lambda bc: bc[0].width * bc[0].height, reverse=True):
        if all(_iou(box, k) < iou_threshold for k, _ in kept):
            kept.append((box, cls))
    return kept
