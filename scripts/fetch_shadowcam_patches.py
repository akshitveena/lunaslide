"""Fetch genuinely dark ShadowCam patches from permanently shadowed regions.

ShadowCam is a LROC-heritage camera built to image PSRs -- it is ~800x more
sensitive than NAC, so inside a shadowed crater it records faint-but-real signal
where NAC would see black.  That makes its PSR imagery the correct domain to
test a low-light enhancer, unlike the well-lit NAC strips the CurveEnhancer was
first (unfairly) judged on.

Patches are kept only if they are *dark with signal*: low mean brightness but
non-trivial local variation and little nodata.  A flat near-zero patch (e.g. the
deepest Shackleton interior) is the no-signal case no enhancer can fix, and is
excluded.

    python3 -m scripts.fetch_shadowcam_patches --max-patches 400

Writes grayscale patches + a manifest to data/stage1/shadowcam_patches/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.perception.lroc import iter_tiles

SHADOWCAM_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/"
    "Lunar_safed/LMAP/SouthPolar/ShadowCam_SPOLE-90_Mosaic_1m_cog.tif"
)
OUTPUT = Path("data/stage1/shadowcam_patches")

# South-polar PSRs with known faint interior signal in ShadowCam.
PSRS = [
    ("faustini", -87.1, 77.0),
    ("shoemaker", -88.1, 44.9),
    ("haworth", -87.5, -5.0),
    ("cabeus", -85.3, -35.0),
    ("nobile", -85.2, 53.5),
]


def is_dark_with_signal(tile: np.ndarray, lo: float, hi: float, min_std: float, max_fill: float) -> bool:
    fill = float(np.mean(tile == 0))
    mean = float(tile.mean())
    return fill <= max_fill and lo <= mean <= hi and float(tile.std()) >= min_std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patches", type=int, default=400)
    parser.add_argument("--per-psr", type=int, default=100)
    parser.add_argument("--size-px", type=int, default=256)
    parser.add_argument("--window-px", type=int, default=4000, help="area sampled per PSR")
    parser.add_argument("--mean-lo", type=float, default=6.0)
    parser.add_argument("--mean-hi", type=float, default=75.0, help="upper bound keeps it genuinely dark")
    parser.add_argument("--min-std", type=float, default=8.0, help="require faint recoverable signal")
    parser.add_argument("--max-fill", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    try:
        import rasterio
        from rasterio.env import Env
        from rasterio.windows import Window

        from src.physics.dem_loader import _project_to_raster
    except ImportError:
        print("rasterio is required; run: pip install -r requirements.txt")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    kept = 0
    print(f"Streaming dark PSR patches from ShadowCam (1 m/px)\n")

    with Env(GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=3, VSI_CACHE=True):
        with rasterio.open(SHADOWCAM_URL) as src:
            for name, lat, lon in PSRS:
                if kept >= args.max_patches:
                    break
                x, y = _project_to_raster(src, lon, lat)
                row, col = src.index(x, y)
                half = args.window_px // 2
                window = Window(col - half, row - half, args.window_px, args.window_px)
                area = src.read(1, window=window, boundless=True, fill_value=0)
                kept_here = 0
                for tile, r, c in iter_tiles(area, args.size_px, args.size_px):
                    if kept >= args.max_patches or kept_here >= args.per_psr:
                        break
                    if not is_dark_with_signal(tile, args.mean_lo, args.mean_hi,
                                               args.min_std, args.max_fill):
                        continue
                    fname = f"{name}_r{r}_c{c}.png"
                    cv2.imwrite(str(args.output / fname), tile)
                    manifest.append({"patch": fname, "psr": name, "lat": lat, "lon": lon,
                                     "mean": float(tile.mean()), "std": float(tile.std())})
                    kept += 1
                    kept_here += 1
                print(f"  {name:11} kept {kept_here:3d}  (total {kept})", flush=True)

    if not manifest:
        print("\nNo dark-with-signal patches found; nothing written.")
        return 1

    means = np.array([m["mean"] for m in manifest])
    (args.output / "manifest.json").write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "ShadowCam_SPOLE-90_Mosaic_1m",
        "note": "Genuinely dark PSR imagery with faint signal -- the enhancer's intended domain.",
        "count": len(manifest), "patches": manifest,
    }, indent=2) + "\n")
    print(f"\nWrote {kept} dark PSR patches -> {args.output}")
    print(f"  brightness: mean {means.mean():.1f}/255 ({means.mean()/255*100:.0f}%), "
          f"range [{means.min():.0f}, {means.max():.0f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
