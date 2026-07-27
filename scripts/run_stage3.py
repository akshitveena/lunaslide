"""Stage 3 end-to-end: perception + physics -> landing verdict, per site.

For each target site this fetches the real DEM (Stage 2), summarises its slope
failure at quiescent and vibrational friction, pairs it with the site's Stage 1
visual evidence (real where an image exists, honest ``not-run`` where a detector
is untrained), and runs the reconciler to a GO / CAUTION / NO-GO verdict.  It is
the first point at which all three stages exchange data.

    python3 -m scripts.run_stage3

Writes figures/stage3_decisions.png and data/stage3/decisions.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.perception.contracts import GeoReference, VisualEvidence
from src.reasoning.hazard import summarise_hazard
from src.reasoning.reconcile import decide

OUTPUT_FIG = Path("figures/stage3_decisions.png")
OUTPUT_JSON = Path("data/stage3/decisions.json")
VERDICT_COLOUR = {"GO": "#2e7d32", "CAUTION": "#f9a825", "NO-GO": "#c62828"}


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    latitude: float
    longitude: float
    size_px: int = 400


SITES = (
    Site("apollo15", "Apollo 15", 26.13, 3.63),
    Site("shackleton", "Shackleton", -89.5, 0.0),
    Site("faustini", "Faustini (PSR)", -87.1, 77.0),
)


# The trained detector's validated recall (from scripts.train_boulder_detector).
DETECTOR_CKPT = Path("runs/detect/runs/boulder_crater/weights/best.pt")
DETECTOR_RECALL = 0.42  # boulder recall on the held-out split


def visual_for(site: Site) -> tuple[VisualEvidence, float | None]:
    """Real Stage 1 evidence for the site, honest about what did and didn't run.

    Apollo 15 has a registered LROC image: its shadow fraction is measured from
    real pixels, and if the trained detector exists it is actually *run* on the
    image, so real boulder/crater counts flow into the verdict (with the
    detector's recall, so Stage 3 knows not to clear on a weak model's silence).
    The polar sites have no visible-light product, so their detectors honestly
    report ``not-run``.
    """
    versions = {"enhancer": "classical-gamma-clahe",
                "boulder_detector": "not-run", "debris_segmenter": "not-run"}
    boulders: list = []
    shadow = None
    recall = None
    apollo_png = Path("data/stage1/apollo15/apollo15.png")
    if site.key == "apollo15" and apollo_png.exists():
        import cv2
        from src.perception.contracts import BoulderDetection
        from src.perception.prepare import to_unit_gray
        raw = cv2.imread(str(apollo_png), cv2.IMREAD_GRAYSCALE)
        shadow = float(np.mean(to_unit_gray(raw) < 0.12))
        if DETECTOR_CKPT.is_file():
            from src.perception.detect import detect_features
            summary = detect_features(apollo_png, DETECTOR_CKPT, recall=DETECTOR_RECALL)
            versions["boulder_detector"] = f"{DETECTOR_CKPT} (recall {DETECTOR_RECALL:.2f})"
            recall = DETECTOR_RECALL
            boulders = ([BoulderDetection((0, 0, 1, 1), 0.5, "boulder")] * summary.boulders
                        + [BoulderDetection((0, 0, 1, 1), 0.5, "crater")] * summary.craters)
            print(f"  detector ran: {summary.boulders} boulders, {summary.craters} craters "
                  f"across {summary.tiles} tiles", flush=True)
    ev = VisualEvidence(
        georef=GeoReference(image_id=site.key, source="LROC"),
        boulders=boulders, shadow_fraction=shadow, model_versions=versions)
    return ev, recall


def main() -> int:
    from src.physics.dem_loader import fetch_patch

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    decisions = []
    panels = []
    for site in SITES:
        print(f"{site.name} ...", flush=True)
        patch = fetch_patch(site.latitude, site.longitude, size_px=site.size_px, verbose=False)
        if patch is None:
            print("  DEM unavailable; skipped")
            continue
        hazard = summarise_hazard(patch.elevation, patch.grid_spacing, site.name)
        km = patch.elevation.shape[1] * patch.grid_spacing_x_m / 1000.0
        area_km2 = (patch.elevation.shape[0] * patch.grid_spacing_y_m / 1000.0) * km
        visual, recall = visual_for(site)
        decision = decide(visual, hazard, site_area_km2=area_km2, detector_recall=recall)
        decisions.append({"hazard": hazard.to_dict(), "decision": decision.to_dict()})
        panels.append((site, patch, hazard, decision))
        print(f"  {decision.verdict}  (hazard score {decision.hazard_score}, "
              f"nominal {hazard.toppled_fraction_nominal:.1%}, "
              f"under load {hazard.toppled_fraction_vibration:.1%})", flush=True)

    if not panels:
        print("No sites produced a verdict.")
        return 1

    OUTPUT_JSON.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Rule-based Stage 3 verdicts. Thresholds are engineering choices, "
                "not calibrated against landing outcomes.",
        "sites": decisions,
    }, indent=2) + "\n")

    _render(panels)
    print(f"\nWrote {OUTPUT_FIG} and {OUTPUT_JSON}")
    return 0


def _render(panels) -> None:
    from matplotlib.colors import LightSource
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1.0]})
    fig.suptitle("Stage 3 — landing verdict from perception + physics", fontsize=15,
                 fontweight="bold", y=0.98)
    for col, (site, patch, hazard, decision) in enumerate(panels):
        H = patch.elevation.astype(np.float64)
        dy, dx = patch.grid_spacing
        shade = LightSource(azdeg=315, altdeg=45).hillshade(H, vert_exag=1.0, dx=dx, dy=dy)
        ax = axes[0][col]
        ax.imshow(shade, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        colour = VERDICT_COLOUR[decision.verdict]
        ax.set_title(f"{site.name}", fontsize=12, fontweight="bold")
        ax.text(0.5, -0.08, decision.verdict, transform=ax.transAxes, ha="center", va="top",
                fontsize=15, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=colour, edgecolor="none"))

        ax2 = axes[1][col]
        ax2.axis("off")
        lines = [f"hazard score: {decision.hazard_score:.2f}",
                 f"slope failure (repose): {hazard.toppled_fraction_nominal:.1%}",
                 f"under descent load:     {hazard.toppled_fraction_vibration:.1%}",
                 f"vibration sensitivity:  {hazard.vibration_sensitivity:.1f}x",
                 f"max slope: {hazard.max_slope_deg:.0f} deg", ""]
        if decision.conflicts:
            lines.append("CONFLICTS:")
            lines += [f"  - {c}" for c in decision.conflicts]
        if decision.evidence_gaps:
            lines.append("EVIDENCE GAPS:")
            lines += [f"  - {g}" for g in decision.evidence_gaps]
        lines.append("REASONING:")
        lines += [f"  - {r}" for r in decision.rationale]
        ax2.text(0.0, 1.0, "\n".join(_wrap(lines)), transform=ax2.transAxes, va="top",
                 fontsize=7.6, family="monospace", color="#222")
    fig.text(0.005, 0.005,
             "Rule-based reconciler. Detectors report not-run (the trained YOLO is a one-frame "
             "proof of concept), so no site can be certified GO -- the honest epistemic rule.  "
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.09, top=0.92, hspace=0.18)
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=140)
    plt.close(fig)


def _wrap(lines: list[str], width: int = 52) -> list[str]:
    import textwrap
    out = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
        else:
            indent = "    " if line.startswith("  ") else ""
            out.extend(indent + w for w in textwrap.wrap(line, width))
    return out


if __name__ == "__main__":
    sys.exit(main())
