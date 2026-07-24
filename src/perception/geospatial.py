"""Read spatial metadata off a georeferenced Stage 1 input.

Stage 3 has to overlay Stage 1's visual evidence on Stage 2's hazard grid.  That
is only possible if the pixel-to-ground mapping travels with the evidence, so
this module lifts it out of the source raster and into ``GeoReference``.
Without it every Stage 1 export carried ``transform: null`` and the two stages
could not be aligned at all.

Reading is best-effort by design: PNG previews carry no georeferencing, and a
missing transform must degrade to ``None`` rather than to a fabricated one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from .contracts import GeoReference

# IAU mean radius of the Moon, for converting angular pixel sizes to metres.
MOON_RADIUS_M = 1_737_400.0


@dataclass(frozen=True)
class RasterGeometry:
    """Spatial facts recovered from an image file."""

    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    ground_sample_distance_m: float | None = None

    def is_empty(self) -> bool:
        return self.crs is None and self.transform is None


def _ground_sample_distance(crs, transform) -> float | None:
    """Pixel size in metres, or ``None`` when the units cannot be established.

    The north-south pixel size is used because it is latitude-independent in
    the equirectangular projections the lunar products are published in, where
    east-west spacing shrinks by ``cos(latitude)``.
    """
    north_south = abs(transform.e)
    if north_south == 0:
        return None
    try:
        if crs.is_geographic:
            return math.radians(north_south) * MOON_RADIUS_M
        units = (crs.linear_units or "").lower()
    except Exception:
        return None
    if units in {"metre", "meter", "m"}:
        return float(north_south)
    return None


def read_raster_geometry(path: str | Path) -> RasterGeometry:
    """Recover CRS, affine transform, and pixel size from an image.

    Returns an empty :class:`RasterGeometry` for plain images, for files with
    no spatial reference, and when rasterio is not installed — Stage 1 must
    stay runnable on a classical-only install.
    """
    try:
        import rasterio
    except ImportError:
        return RasterGeometry()

    try:
        with rasterio.open(str(path)) as src:
            transform = src.transform
            crs = src.crs
            if transform is None or transform.is_identity:
                return RasterGeometry(crs=str(crs) if crs else None)
            return RasterGeometry(
                crs=str(crs) if crs else None,
                # GDAL ordering: (x_origin, x_pixel, x_rotation,
                #                 y_origin, y_rotation, y_pixel).
                transform=tuple(float(value) for value in transform.to_gdal()),
                ground_sample_distance_m=(
                    _ground_sample_distance(crs, transform) if crs else None
                ),
            )
    except Exception:
        return RasterGeometry()


def merge_georeference(georef: GeoReference, path: str | Path) -> GeoReference:
    """Fill gaps in a caller-supplied ``GeoReference`` from the source raster.

    Values the caller set explicitly always win: an operator correcting known-
    bad product metadata must not be silently overruled by the file.
    """
    geometry = read_raster_geometry(path)
    if geometry.is_empty():
        return georef
    return replace(
        georef,
        crs=georef.crs if georef.crs is not None else geometry.crs,
        transform=georef.transform if georef.transform is not None else geometry.transform,
        ground_sample_distance_m=(
            georef.ground_sample_distance_m
            if georef.ground_sample_distance_m is not None
            else geometry.ground_sample_distance_m
        ),
    )
