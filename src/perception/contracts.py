"""Portable output contracts for Stage 1 visual evidence.

Stage 1 reports what is visible in an orbital image.  It deliberately does
not classify future slope failure; that remains the responsibility of Stage 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GeoReference:
    """Spatial identity shared by image-derived and DEM-derived products.

    ``transform`` follows GDAL affine order: (x_origin, x_pixel, x_rotation,
    y_origin, y_rotation, y_pixel).  It is optional only for non-georeferenced
    training crops; production inference must supply it.
    """

    image_id: str
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    acquisition_utc: str | None = None
    source: str | None = None
    ground_sample_distance_m: float | None = None


@dataclass(frozen=True)
class BoulderDetection:
    """A detected present-day boulder in pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "boulder"
    instance_mask_path: str | None = None


@dataclass
class VisualEvidence:
    """Stage 1 evidence passed to Stage 3 after spatial alignment.

    ``historical_debris_mask_path`` represents evidence of prior mass wasting,
    not a forecast of a future event.
    """

    georef: GeoReference
    boulders: list[BoulderDetection] = field(default_factory=list)
    historical_debris_mask_path: str | None = None
    shadow_fraction: float | None = None
    texture_roughness: float | None = None
    preprocessing_report: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    evidence_kind: Literal["visual"] = "visual"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation without writing a file."""
        return asdict(self)
