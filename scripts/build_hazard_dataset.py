"""Build a hazard dataset from real lunar topography.

Three design choices, each fixing a specific defect in the previous dataset:

1. **One product, one resolution.**  Hazard estimates are resolution-dependent
   (Apollo 15 sheds 0.23% at 118 m/px and 1.07% at 59 m/px).  Mixing products by
   latitude would make latitude a proxy for resolution and hand the model a
   confound.  Everything here is the 59 m LOLA+Kaguya merge, so sampling is
   restricted to its +/-60 deg coverage.

2. **No synthetic terrain, ever.**  The previous builder fell back to procedural
   craters chosen by the label being predicted.  A patch that cannot be streamed
   is dropped and counted, never substituted.

3. **Uncensored targets.**  Features are terrain statistics computable without
   simulating; targets come from the automaton at a fixed budget.  That split is
   what makes the model a surrogate rather than a tautology.

    python3 -m scripts.build_hazard_dataset --samples 400

Writes ``data/stage2/hazard_dataset.csv``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.physics.dem_loader import PRODUCTS_BY_KEY, fetch_patch
from src.physics.response import relaxation_response, terrain_features

OUTPUT = Path("data/stage2/hazard_dataset.csv")
PRODUCT_KEY = "lolakaguya_59m"
LATITUDE_LIMIT = 58.0  # inside the product's +/-60 coverage, with margin
CRIT = 0.577  # tan(30 deg)


def sample_locations(count: int, seed: int) -> list[tuple[float, float]]:
    """Uniform-by-area sampling over the sphere within the product's coverage.

    Sampling latitude uniformly would over-represent the poles, because a
    degree of latitude covers the same area everywhere but a degree of
    longitude does not.  Sampling ``sin(latitude)`` uniformly fixes that.
    """
    rng = np.random.default_rng(seed)
    limit = np.sin(np.radians(LATITUDE_LIMIT))
    latitudes = np.degrees(np.arcsin(rng.uniform(-limit, limit, count)))
    longitudes = rng.uniform(-180.0, 180.0, count)
    return list(zip(latitudes.tolist(), longitudes.tolist()))


def process(latitude: float, longitude: float, size_px: int, max_iter: int) -> dict | None:
    patch = fetch_patch(
        latitude, longitude, size_px=size_px, product=PRODUCT_KEY, verbose=False
    )
    if patch is None:
        return None
    elevation = patch.elevation.astype(np.float64)
    spacing = patch.grid_spacing
    row = {"latitude": latitude, "longitude": longitude}
    row.update(terrain_features(elevation, spacing, CRIT))
    response = relaxation_response(elevation, spacing, CRIT, max_iter=max_iter)
    row.update(response.to_dict())
    row["grid_spacing_y_m"] = patch.grid_spacing_y_m
    row["grid_spacing_x_m"] = patch.grid_spacing_x_m
    row["nodata_fraction"] = patch.nodata_fraction
    # Farside is more heavily cratered than nearside, so this is a genuine
    # distribution shift rather than a random split wearing a geographic label.
    row["hemisphere"] = "nearside" if abs(longitude) <= 90.0 else "farside"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--size-px", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    product = PRODUCTS_BY_KEY[PRODUCT_KEY]
    span_km = args.size_px * product.resolution_m / 1000.0
    print(f"Product : {product.name}")
    print(f"Patches : {args.samples} x {args.size_px}px  (~{span_km:.1f} km across)")
    print(f"Budget  : {args.max_iter} CA iterations, crit={CRIT} (30 deg repose)")
    print(f"Coverage: +/-{LATITUDE_LIMIT:.0f} deg latitude, uniform by area\n")

    locations = sample_locations(args.samples, args.seed)
    rows, failures = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, lat, lon, args.size_px, args.max_iter): (lat, lon)
            for lat, lon in locations
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:
                row, _ = None, print(f"  error: {error}")
            if row is None:
                failures += 1
            else:
                rows.append(row)
            if index % 25 == 0 or index == len(futures):
                print(f"  {index}/{len(futures)} processed, {len(rows)} kept, "
                      f"{failures} unavailable", flush=True)

    if not rows:
        print("\nNo patches could be streamed; nothing written.")
        return 1

    frame = pd.DataFrame(rows).sort_values(["hemisphere", "longitude"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    converged = frame["converged"].mean() * 100
    print(f"\nWrote {len(frame)} rows -> {args.output}")
    print(f"  nearside {int((frame.hemisphere == 'nearside').sum())}  "
          f"farside {int((frame.hemisphere == 'farside').sum())}")
    print(f"  converged within budget: {converged:.1f}%")
    print(f"  toppled_fraction: min {frame.toppled_fraction.min():.5f}  "
          f"median {frame.toppled_fraction.median():.5f}  "
          f"max {frame.toppled_fraction.max():.5f}")
    print(f"  built {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
