"""Immutable Stage 1 site definitions copied from Stage 2 coordinates.

This module deliberately does not import or change ``main.py``.  It records
the fixed sites already used by the completed Stage 2 workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage2Site:
    name: str
    latitude: float
    longitude: float
    stage2_kind: str
    image_url: str | None = None
    image_kind: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# The Apollo 15 image is the official LROC browse orthophoto for the 0.5 m/px
# product.  It is a compact visual demonstration asset.  For training and
# quantitative analysis, use the full GeoTIFF URL from the LROC product page.
STAGE2_FIXED_SITES: dict[str, Stage2Site] = {
    "apollo15": Stage2Site(
        name="Apollo 15 Landing Site (Safe Zone)",
        latitude=26.13,
        longitude=3.63,
        stage2_kind="safe",
        image_url=(
            "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
            "LROLRC_2001/EXTRAS/BROWSE/NAC_DTM/APOLLO15/"
            "NAC_DTM_APOLLO15_M111578606_50CM.BROWSE.PNG"
        ),
        image_kind="LROC NAC orthophoto browse PNG (0.5 m/px product)",
    ),
    "shackleton": Stage2Site(
        name="Shackleton Crater (South Pole Hazards)",
        latitude=-89.5,
        longitude=0.0,
        stage2_kind="extreme",
    ),
}


def get_stage2_site(key: str) -> Stage2Site:
    try:
        return STAGE2_FIXED_SITES[key]
    except KeyError as exc:
        choices = ", ".join(STAGE2_FIXED_SITES)
        raise ValueError(f"Unknown Stage 2 site {key!r}; choose one of: {choices}") from exc
