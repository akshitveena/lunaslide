"""Diagnostic probe for the USGS LOLA mosaic.

Reports the mosaic's coordinate reference, bounds, and pixel size, then checks
that the Stage 2 sites resolve to sensible pixel windows.  Use it to establish
whether the mosaic is published in degrees or metres, and with which longitude
convention, before trusting any streamed patch.

This lives in ``scripts/`` rather than ``tests/`` on purpose: it needs network
access, so it is a diagnostic, not a test.  Previously it sat in ``tests/`` and
issued its HTTP request at import time, which made ``pytest`` hang on
collection.

    python3 -m scripts.probe_lola_mosaic
"""

from __future__ import annotations

import sys

from src.physics.dem_loader import (
    LOLA_118M_URL,
    bounds_are_degrees,
    fetch_patch,
    lonlat_to_raster_xy,
)

SITES = (("Apollo 15", 26.13, 3.63), ("Shackleton", -89.5, 0.0))


def main() -> int:
    try:
        import rasterio
    except ImportError:
        print("rasterio is not installed. Run: pip install -r requirements.txt")
        return 1

    print(f"Opening {LOLA_118M_URL}\n")
    try:
        with rasterio.open(LOLA_118M_URL) as src:
            bounds = tuple(src.bounds)
            in_degrees = bounds_are_degrees(*bounds)
            print(f"  CRS          {src.crs}")
            print(f"  shape        {src.shape}")
            print(f"  bounds       {bounds}")
            print(f"  pixel size   {abs(src.transform.a)} x {abs(src.transform.e)}")
            print(f"  nodata       {src.nodata}")
            print(f"  units        {'degrees' if in_degrees else 'metres'} (inferred from bounds)")
            print(f"  longitude    {'0..360' if bounds[0] >= 0 else '-180..180'}\n")

            for name, latitude, longitude in SITES:
                x, y = lonlat_to_raster_xy(longitude, latitude, bounds)
                inside = bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
                print(f"  {name:<12} lon={longitude:>7.2f} lat={latitude:>6.2f} "
                      f"-> ({x:.2f}, {y:.2f}) {'inside' if inside else 'OUTSIDE'}")
    except Exception as error:
        print(f"Could not open the mosaic: {error}")
        return 1

    print("\nStreaming a 64 px patch at each site:")
    for name, latitude, longitude in SITES:
        patch = fetch_patch(latitude, longitude, size_px=64, cache_dir=None, verbose=False)
        if patch is None:
            print(f"  {name:<12} FAILED")
            continue
        elevation = patch.elevation
        print(f"  {name:<12} {elevation.shape} spacing={patch.grid_spacing_m:.1f} m/px "
              f"range=[{elevation.min():.1f}, {elevation.max():.1f}] m "
              f"nodata={patch.nodata_fraction:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
