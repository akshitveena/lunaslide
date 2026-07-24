"""Build a hazard dataset from real lunar topography.

Sampling is **band-oriented**, because the equirectangular lunar products are
strip-organised rather than tiled: each block is one full-width scanline, so a
128-row window costs 128 full-width strips (~47 MB for the 59 m merge) however
few columns are wanted.  Requesting patches independently transferred ~47 MB
each and produced truncated range reads under concurrency, losing a third of
the sample.  Reading the whole band costs the same 47 MB and yields ~1,400
disjoint patches — roughly 30x less data for the same sample count.

The trade-off is stated plainly: patches from one band share a latitude, so
sampling is stratified by band rather than independent.  In exchange every band
spans all longitudes, so latitude cannot confound the nearside/farside split.

Three further design choices, each fixing a defect in the previous dataset:

1. **One product, one resolution.**  Hazard estimates are resolution-dependent
   (Apollo 15 sheds 0.23% at 118 m/px and 1.07% at 59 m/px), so mixing products
   by latitude would make latitude a proxy for resolution.
2. **No synthetic terrain, ever.**  The previous builder fell back to procedural
   craters chosen by the label being predicted.  Patches that cannot be read are
   dropped and counted *by reason*, never substituted.
3. **Uncensored targets.**  Features are terrain statistics computable without
   simulating; targets come from the automaton at a fixed budget.

    python3 -m scripts.build_hazard_dataset --bands 24 --per-band 20

Writes ``data/stage2/hazard_dataset.csv``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.physics.dem_loader import PRODUCTS_BY_KEY, fetch_latitude_band
from src.physics.response import relaxation_response, terrain_features

OUTPUT = Path("data/stage2/hazard_dataset.csv")
PRODUCT_KEY = "lolakaguya_59m"
LATITUDE_LIMIT = 58.0  # inside the product's +/-60 coverage, with margin
CRIT = 0.577  # tan(30 deg)
MAX_NODATA_FRACTION = 0.02


def band_latitudes(count: int, seed: int) -> list[float]:
    """Latitudes sampled uniformly by area within the product's coverage.

    Uniform-in-latitude would over-weight high latitudes, where a degree covers
    less surface.  Sampling sin(latitude) uniformly corrects that.
    """
    rng = np.random.default_rng(seed)
    limit = np.sin(np.radians(LATITUDE_LIMIT))
    return sorted(np.degrees(np.arcsin(rng.uniform(-limit, limit, count))).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=int, default=24, help="latitude bands to stream")
    parser.add_argument("--per-band", type=int, default=20, help="patches cut from each band")
    parser.add_argument("--size-px", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    product = PRODUCTS_BY_KEY[PRODUCT_KEY]
    span_km = args.size_px * product.resolution_m / 1000.0
    target = args.bands * args.per_band
    print(f"Product : {product.name}")
    print(f"Sampling: {args.bands} latitude bands x {args.per_band} patches = {target} target")
    print(f"Patches : {args.size_px}px  (~{span_km:.1f} km across)")
    print(f"Budget  : {args.max_iter} CA iterations, crit={CRIT} (30 deg repose)")
    print(f"Coverage: +/-{LATITUDE_LIMIT:.0f} deg latitude, uniform by area\n")

    rng = np.random.default_rng(args.seed + 1)
    rows: list[dict] = []
    dropped: Counter[str] = Counter()

    for index, latitude in enumerate(band_latitudes(args.bands, args.seed), start=1):
        band = fetch_latitude_band(
            latitude, args.size_px, product=PRODUCT_KEY, verbose=True
        )
        if band is None:
            dropped["band unreadable"] += args.per_band
            print(f"  [{index}/{args.bands}] lat {latitude:+6.2f}  BAND FAILED", flush=True)
            continue

        width = band.elevation.shape[1]
        columns = rng.choice(
            max(1, width - args.size_px), size=min(args.per_band * 3, width - args.size_px),
            replace=False,
        )
        kept_here = 0
        for column in columns:
            if kept_here >= args.per_band:
                break
            cut = band.patch_at(int(column), args.size_px)
            if cut is None:
                dropped["window out of range"] += 1
                continue
            elevation, nodata_fraction = cut
            if nodata_fraction > MAX_NODATA_FRACTION:
                dropped["nodata"] += 1
                continue
            longitude = band.longitude_of(int(column), args.size_px)
            patch = elevation.astype(np.float64)
            row = {"latitude": latitude, "longitude": longitude, "band": index}
            row.update(terrain_features(patch, band.grid_spacing, CRIT))
            row.update(
                relaxation_response(
                    patch, band.grid_spacing, CRIT, max_iter=args.max_iter
                ).to_dict()
            )
            row["grid_spacing_y_m"] = band.grid_spacing_y_m
            row["grid_spacing_x_m"] = band.grid_spacing_x_m
            row["nodata_fraction"] = nodata_fraction
            # The farside is older and more heavily cratered than the nearside,
            # so this split is a real distribution shift.
            row["hemisphere"] = "nearside" if abs(longitude) <= 90.0 else "farside"
            rows.append(row)
            kept_here += 1
        print(f"  [{index}/{args.bands}] lat {latitude:+6.2f}  kept {kept_here:3d}  "
              f"total {len(rows)}", flush=True)

    if not rows:
        print("\nNo patches could be read; nothing written.")
        return 1

    frame = pd.DataFrame(rows).sort_values(["hemisphere", "longitude"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"\nWrote {len(frame)} rows -> {args.output}")
    print(f"  nearside {int((frame.hemisphere == 'nearside').sum())}  "
          f"farside {int((frame.hemisphere == 'farside').sum())}")
    print(f"  bands used: {frame.band.nunique()}/{args.bands}")
    print(f"  converged within budget: {frame.converged.mean() * 100:.1f}%")
    print(f"  toppled_fraction: min {frame.toppled_fraction.min():.5f}  "
          f"median {frame.toppled_fraction.median():.5f}  "
          f"max {frame.toppled_fraction.max():.5f}")
    if dropped:
        print("  dropped:")
        for reason, count in dropped.most_common():
            print(f"    {reason}: {count}")
    print(f"  built {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
