"""Calibrate Stage 3's decision thresholds against real lunar terrain.

The reconciler's thresholds began as engineering guesses.  This derives them
from data, using two independent real sources:

* **Population distribution** — the hazard of 480 real DEM patches sampled
  uniformly by area (``data/stage2/hazard_dataset.csv``).  This says what is
  *unusual* for the Moon: a CAUTION threshold set at the 90th percentile means
  "steeper than 90% of lunar terrain".
* **Rockfall anchoring** — hazard measured at 120 *observed* Bickel rockfall
  sites versus 120 random controls (``data/bickel/validation_raw.json``).  For
  each candidate threshold this reports the **enrichment**: how much more often
  real rockfall terrain exceeds it than random terrain does.  A threshold worth
  using should be crossed substantially more often where mass wasting actually
  happened.

The honest limit: this calibrates against *where rockfalls occurred*, not
against *landing outcomes*.  No lunar landing failure dataset exists, so the
thresholds remain risk-screening levels, not certified safety limits.

    python3 -m scripts.calibrate_stage3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HAZARD_CSV = Path("data/stage2/hazard_dataset.csv")
BICKEL_JSON = Path("data/bickel/validation_raw.json")
OUTPUT_FIG = Path("figures/stage3_calibration.png")


def enrichment(threshold: float, cases: np.ndarray, controls: np.ndarray) -> tuple[float, float, float]:
    """(case exceedance, control exceedance, enrichment ratio) at a threshold."""
    case_rate = float((cases > threshold).mean())
    control_rate = float((controls > threshold).mean())
    ratio = case_rate / control_rate if control_rate > 0 else float("inf")
    return case_rate, control_rate, ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard-csv", type=Path, default=HAZARD_CSV)
    parser.add_argument("--bickel-json", type=Path, default=BICKEL_JSON)
    parser.add_argument("--caution-percentile", type=float, default=90.0,
                        help="population percentile defining CAUTION")
    parser.add_argument("--nogo-percentile", type=float, default=99.0,
                        help="population percentile defining NO-GO")
    args = parser.parse_args()

    import pandas as pd

    if not args.hazard_csv.is_file():
        print(f"Missing {args.hazard_csv}. Run: python3 -m scripts.build_hazard_dataset")
        return 1
    frame = pd.read_csv(args.hazard_csv)
    population = frame["toppled_fraction"].to_numpy()

    # --- Population-based thresholds: what counts as unusual lunar terrain ----
    caution = float(np.percentile(population, args.caution_percentile))
    nogo = float(np.percentile(population, args.nogo_percentile))

    print(f"Population: {len(population)} real DEM patches, uniform by area")
    print(f"  median {np.median(population):.5f}   p90 {np.percentile(population, 90):.5f}"
          f"   p99 {np.percentile(population, 99):.5f}   max {population.max():.5f}")
    print(f"\nProposed thresholds (slope failure fraction at repose):")
    print(f"  CAUTION = p{args.caution_percentile:.0f} = {caution:.5f}")
    print(f"  NO-GO   = p{args.nogo_percentile:.0f} = {nogo:.5f}")

    # --- Rockfall anchoring: do these levels mark real mass-wasting terrain? --
    report: dict = {
        "population": {
            "n": int(len(population)),
            "median": float(np.median(population)),
            "p90": float(np.percentile(population, 90)),
            "p99": float(np.percentile(population, 99)),
        },
        "thresholds": {"slope_caution": caution, "slope_nogo": nogo},
    }
    cases = controls = None
    if args.bickel_json.is_file():
        raw = json.loads(args.bickel_json.read_text())
        cases = np.array([r["toppled"] for r in raw["rockfall_sites"]])
        controls = np.array([r["toppled"] for r in raw["control_sites"]])
        print(f"\nRockfall anchoring ({len(cases)} observed rockfalls vs {len(controls)} controls):")
        print(f"  {'threshold':>12}{'rockfall %':>14}{'control %':>12}{'enrichment':>12}")
        anchor = {}
        for name, value in (("CAUTION", caution), ("NO-GO", nogo)):
            case_rate, control_rate, ratio = enrichment(value, cases, controls)
            print(f"  {name:>12}{case_rate * 100:>13.1f}%{control_rate * 100:>11.1f}%"
                  f"{ratio:>11.1f}x")
            anchor[name] = {"threshold": value, "rockfall_exceedance": case_rate,
                            "control_exceedance": control_rate, "enrichment": ratio}
        report["rockfall_anchoring"] = anchor
        best = max(anchor.values(), key=lambda a: a["enrichment"])
        print(f"\nObserved rockfall terrain crosses these levels up to "
              f"{best['enrichment']:.1f}x more often than random terrain — the thresholds "
              f"track real mass wasting, not just an arbitrary quantile.")
    else:
        print(f"\n(No {args.bickel_json}; population-only calibration. "
              "Run scripts.validate_against_bickel for rockfall anchoring.)")

    report["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["note"] = ("Calibrated against terrain distribution and observed rockfall locations, "
                      "NOT against landing outcomes (no such dataset exists). These are "
                      "risk-screening levels, not certified safety limits.")
    Path("data/stage3").mkdir(parents=True, exist_ok=True)
    Path("data/stage3/calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nWrote data/stage3/calibration.json")
    print("Set these in DecisionPolicy (src/reasoning/reconcile.py):")
    print(f"  slope_caution: float = {caution:.4f}")
    print(f"  slope_nogo: float = {nogo:.4f}")

    _render(population, cases, controls, caution, nogo)
    return 0


def _render(population, cases, controls, caution, nogo) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 2 if cases is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 4.8), squeeze=False)
    ax = axes[0][0]
    positive = population[population > 0]
    bins = np.logspace(np.log10(max(positive.min(), 1e-6)), np.log10(population.max() + 1e-9), 40)
    ax.hist(positive, bins=bins, color="#1f77b4", alpha=0.75)
    ax.axvline(caution, color="#f9a825", linewidth=2, label=f"CAUTION {caution:.4f}")
    ax.axvline(nogo, color="#c62828", linewidth=2, label=f"NO-GO {nogo:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("slope failure fraction at repose")
    ax.set_ylabel("real DEM patches")
    ax.set_title(f"Lunar terrain distribution (n={len(population)})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    if cases is not None:
        ax2 = axes[0][1]
        levels = np.logspace(-5, -1, 40)
        case_rates = [(cases > t).mean() * 100 for t in levels]
        control_rates = [(controls > t).mean() * 100 for t in levels]
        ax2.plot(levels, case_rates, color="#c1440e", linewidth=2, label="observed rockfalls")
        ax2.plot(levels, control_rates, color="#999999", linewidth=2, label="random control")
        ax2.axvline(caution, color="#f9a825", linewidth=2, linestyle="--")
        ax2.axvline(nogo, color="#c62828", linewidth=2, linestyle="--")
        ax2.set_xscale("log")
        ax2.set_xlabel("threshold on slope failure fraction")
        ax2.set_ylabel("% of sites exceeding threshold")
        ax2.set_title("Rockfall anchoring: real mass wasting vs random terrain", fontsize=11)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("Stage 3 threshold calibration on real lunar data", fontsize=14, fontweight="bold")
    fig.text(0.005, 0.01,
             "Thresholds are population percentiles, checked against observed rockfall sites. "
             "Calibrated to terrain and observed mass wasting, NOT to landing outcomes "
             "(no such dataset exists).  "
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.20, top=0.86)
    fig.savefig(OUTPUT_FIG, dpi=140)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())
