"""Fetch imagery and elevation for the *same* ground footprint.

Stage 3 combines a boulder count (from an image) with slope hazard (from a DEM).
That is only meaningful if both describe the same patch of ground — otherwise a
count from one place is divided by the area of another.  At the lunar south pole
both a 1 m/px LROC NAC mosaic and a 5 m/px LOLA DEM exist as windowable Cloud
Optimized GeoTIFFs in the *same* polar-stereographic projection, so a single
lat/lon + size yields a co-registered image/DEM pair.

The two rasters have different resolutions, so the same ground span is a
different number of pixels in each (1000 m -> 1000 px image, 200 px DEM); both
cover the identical footprint, which is the point.  Boulder density from the
image and hazard from the DEM then refer to one area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.physics.dem_loader import _ASC, _project_to_raster

NAC_SPOLE_URL = _ASC + "Lunar_safed/LMAP/SouthPolar/LRO_LROC_NAC_SPOLE-90_Mosaic_1m_controlled_cog.tif"
LOLA_SPOLE_URL = _ASC + "Lunar_safed/LMAP/SouthPolar/ldem_87s_5mpp_cog.tif"


@dataclass(frozen=True)
class CoRegisteredPatch:
    """An image and DEM covering the same square of lunar surface."""

    image: np.ndarray            # uint8 NAC, size_m / image_res px
    elevation: np.ndarray        # float32 LOLA, size_m / dem_res px
    latitude: float
    longitude: float
    span_m: float
    image_res_m: float
    dem_res_m: float
    image_nodata_fraction: float

    @property
    def area_km2(self) -> float:
        return (self.span_m / 1000.0) ** 2

    @property
    def dem_grid_spacing(self) -> tuple[float, float]:
        # Polar stereographic near the pole is very close to isotropic.
        return (self.dem_res_m, self.dem_res_m)


def _window_at(src, lon: float, lat: float, span_m: float) -> np.ndarray:
    from rasterio.windows import from_bounds

    x, y = _project_to_raster(src, lon, lat)
    half = span_m / 2.0
    window = from_bounds(x - half, y - half, x + half, y + half, transform=src.transform)
    return src.read(1, window=window, boundless=True, masked=True)


def fetch_coregistered(
    latitude: float,
    longitude: float,
    span_m: float = 1000.0,
    *,
    timeout_s: int = 60,
    verbose: bool = False,
) -> CoRegisteredPatch | None:
    """Fetch a co-registered NAC image + LOLA DEM for one south-polar coordinate.

    Returns ``None`` (never a guess) if either raster is unavailable there or the
    image window is mostly nodata.  Only valid near the south pole, where both
    products exist in the same projection.
    """
    if latitude > -80.0:
        if verbose:
            print("Co-registration is south-polar only (both COGs cover <= -80 deg).")
        return None
    try:
        import rasterio
        from rasterio.env import Env
    except ImportError:
        return None

    try:
        with Env(GDAL_HTTP_TIMEOUT=timeout_s, GDAL_HTTP_MAX_RETRY=2, VSI_CACHE=True):
            with rasterio.open(NAC_SPOLE_URL) as img_src:
                image_res = abs(img_src.transform.a)
                raw_img = _window_at(img_src, longitude, latitude, span_m)
            with rasterio.open(LOLA_SPOLE_URL) as dem_src:
                dem_res = abs(dem_src.transform.a)
                raw_dem = _window_at(dem_src, longitude, latitude, span_m)
                dem_scale = float(dem_src.scales[0]) if dem_src.scales else 1.0
    except Exception as error:
        if verbose:
            print(f"Co-registration fetch failed: {error}")
        return None

    image = np.ma.getdata(raw_img).astype(np.uint8)
    image_invalid = np.ma.getmaskarray(raw_img)
    nodata_fraction = float(image_invalid.mean()) if image_invalid.size else 1.0
    if image.size == 0 or nodata_fraction > 0.5:
        if verbose:
            print(f"Image window unusable ({nodata_fraction:.0%} nodata).")
        return None

    elevation = np.ma.getdata(raw_dem).astype(np.float64) * dem_scale
    dem_invalid = np.ma.getmaskarray(raw_dem) | ~np.isfinite(elevation)
    if dem_invalid.any() and not dem_invalid.all():
        elevation[dem_invalid] = float(np.median(elevation[~dem_invalid]))

    return CoRegisteredPatch(
        image=image,
        elevation=elevation.astype(np.float32),
        latitude=latitude,
        longitude=longitude,
        span_m=span_m,
        image_res_m=image_res,
        dem_res_m=dem_res,
        image_nodata_fraction=nodata_fraction,
    )
