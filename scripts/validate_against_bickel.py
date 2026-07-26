"""Empirically validate the Stage 2 physics against observed lunar rockfalls.

The physics engine has only ever been checked for internal consistency (mass,
isotropy, convergence).  This script asks the real question: does the hazard it
predicts actually correspond to where rockfalls have been *observed*?

Ground truth is the Bickel et al. (2020) catalogue of 136,610 real lunar
rockfalls (Edmond, doi:10.17617/3.OG927P; fields NAC_ID, C_LON, C_LAT, R_DIAM).
A rockfall is direct evidence that mass wasting occurred at that spot.

Test design — a case/control comparison:

* **Cases:** N real rockfall coordinates, sampled from the catalogue.
* **Controls:** N random lunar coordinates in the same latitude band (matched
  for the cos-latitude area bias), which are overwhelmingly rockfall-free.
* For each, fetch the DEM patch and compute the physics hazard (slope failure
  fraction, max slope, vibration sensitivity).

If the physics is meaningful, cases should score significantly higher than
controls.  Reported with a Mann-Whitney U test (non-parametric; the hazard
distribution is heavily skewed) and the AUC of hazard as a rockfall classifier.

    python3 -m scripts.validate_against_bickel --samples 150

This is genuine external validation: the elevation, the physics, and the
rockfall labels are all independent real data.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CATALOGUE = Path("data/bickel/rockfalls_i80_06.csv")
OUTPUT_FIG = Path("figures/bickel_validation.png")


def hazard_at(lat: float, lon: float, size_px: int, max_iter: int) -> dict | None:
    """Physics hazard at a coordinate, or None if the DEM is unavailable there."""
    from src.physics.dem_loader import fetch_patch
    from src.physics.relaxation import compute_slope, simulate_mass_wasting

    patch = fetch_patch(lat, lon, size_px=size_px, verbose=False)
    if patch is None:
        return None
    H = patch.elevation.astype(np.float64)
    spacing = patch.grid_spacing
    crit_nom, crit_vib = np.tan(np.radians(30.0)), np.tan(np.radians(24.0))
    _, toppled, _ = simulate_mass_wasting(H, grid_spacing=spacing, crit=crit_nom, max_iter=max_iter)
    _, toppled_vib, _ = simulate_mass_wasting(H, grid_spacing=spacing, crit=crit_vib, max_iter=max_iter)
    slope_deg = np.degrees(np.arctan(compute_slope(H, spacing)))
    return {
        "toppled": float(toppled.mean()),
        "toppled_vib": float(toppled_vib.mean()),
        "max_slope": float(slope_deg.max()),
        "p95_slope": float(np.percentile(slope_deg, 95)),
    }


def sample_controls(n: int, lat_lo: float, lat_hi: float, seed: int) -> list[tuple[float, float]]:
    """Random coordinates, uniform by area, in the catalogue's latitude band."""
    rng = np.random.default_rng(seed)
    lo, hi = np.sin(np.radians(lat_lo)), np.sin(np.radians(lat_hi))
    lats = np.degrees(np.arcsin(rng.uniform(lo, hi, n)))
    lons = rng.uniform(-180, 180, n)
    return list(zip(lats.tolist(), lons.tolist()))


def collect(coords, size_px, max_iter, label) -> list[dict]:
    rows = []
    for i, (lat, lon) in enumerate(coords, 1):
        h = hazard_at(lat, lon, size_px, max_iter)
        if h is not None:
            rows.append(h)
        if i % 25 == 0:
            print(f"  {label}: {i}/{len(coords)} ({len(rows)} usable)", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--samples", type=int, default=150, help="rockfalls (and controls)")
    parser.add_argument("--size-px", type=int, default=96)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--min-diam", type=float, default=5.0, help="min rockfall diameter (m)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.catalogue.is_file() or args.catalogue.stat().st_size == 0:
        print(f"Catalogue not found at {args.catalogue}.\n"
              "Download it from Edmond (doi:10.17617/3.OG927P):\n"
              "  curl -sL -A 'Mozilla/5.0' "
              "https://edmond.mpg.de/api/access/datafile/102549 -o data/bickel/rockfalls_i80_06.csv")
        return 1

    import pandas as pd
    from scipy.stats import mannwhitneyu

    cat = pd.read_csv(args.catalogue)
    cat = cat[cat["R_DIAM"] >= args.min_diam]
    rng = np.random.default_rng(args.seed)
    picks = cat.sample(n=min(args.samples, len(cat)), random_state=args.seed)
    cases = list(zip(picks["C_LAT"].tolist(), picks["C_LON"].tolist()))
    lat_lo, lat_hi = float(cat["C_LAT"].min()), float(cat["C_LAT"].max())
    controls = sample_controls(len(cases), lat_lo, lat_hi, args.seed + 1)

    print(f"Catalogue: {len(cat)} rockfalls >= {args.min_diam} m; testing {len(cases)} "
          f"cases vs {len(controls)} controls.\n")
    case_rows = collect(cases, args.size_px, args.max_iter, "rockfall")
    control_rows = collect(controls, args.size_px, args.max_iter, "control ")

    if len(case_rows) < 20 or len(control_rows) < 20:
        print("\nToo few usable samples for a test.")
        return 1

    metrics = ["toppled", "toppled_vib", "max_slope", "p95_slope"]
    print(f"\n{'metric':<14}{'rockfall median':>18}{'control median':>18}{'p (MWU)':>12}{'AUC':>8}")
    print("-" * 70)
    results = {}
    for m in metrics:
        a = np.array([r[m] for r in case_rows])
        b = np.array([r[m] for r in control_rows])
        u, p = mannwhitneyu(a, b, alternative="greater")
        auc = u / (len(a) * len(b))  # U/(n*m) is the AUC of the ranking
        results[m] = {"case_median": float(np.median(a)), "control_median": float(np.median(b)),
                      "p_value": float(p), "auc": float(auc)}
        print(f"{m:<14}{np.median(a):>18.4f}{np.median(b):>18.4f}{p:>12.2e}{auc:>8.2f}")

    best = max(results, key=lambda m: results[m]["auc"])
    verdict = (
        f"Physics hazard is significantly higher at observed rockfalls: best signal "
        f"'{best}' (AUC {results[best]['auc']:.2f}, p {results[best]['p_value']:.1e})."
        if results[best]["p_value"] < 0.05 and results[best]["auc"] > 0.55 else
        "No significant association found at this resolution/sample."
    )
    print(f"\n{verdict}")
    _render(results, case_rows, control_rows, verdict, args)
    return 0


def _render(results, case_rows, control_rows, verdict, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["toppled", "toppled_vib", "max_slope", "p95_slope"]
    labels = {"toppled": "slope failure @30°", "toppled_vib": "failure @24° (load)",
              "max_slope": "max slope (deg)", "p95_slope": "p95 slope (deg)"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    fig.suptitle("Stage 2 physics vs observed rockfalls (Bickel et al. 2020)",
                 fontsize=14, fontweight="bold")
    for ax, m in zip(axes, metrics):
        a = [r[m] for r in case_rows]
        b = [r[m] for r in control_rows]
        ax.hist(b, bins=25, alpha=0.55, color="#999999", density=True, label="control")
        ax.hist(a, bins=25, alpha=0.55, color="#c1440e", density=True, label="rockfall")
        ax.set_title(f"{labels[m]}\nAUC {results[m]['auc']:.2f}, p {results[m]['p_value']:.1e}",
                     fontsize=10)
        ax.set_yticks([]); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.text(0.005, 0.01,
             f"{len(case_rows)} observed rockfalls vs {len(control_rows)} random controls, matched by "
             f"latitude band. AUC = P(hazard higher at a rockfall than a random site).\n{verdict}  "
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.20, top=0.86)
    fig.savefig(OUTPUT_FIG, dpi=140)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())
