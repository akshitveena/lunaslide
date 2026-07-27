"""Stage 3 on a co-registered image/DEM patch — density that actually means something.

At the south pole a 1 m NAC image and a 5 m LOLA DEM cover the same ground, so
the boulder count (from the image) and the slope hazard (from the DEM) refer to
one footprint.  Boulder *density* is finally meaningful, which is the missing
piece for a real GO: elsewhere the image and DEM came from different places and
dividing a count by the wrong area was meaningless.

    python3 -m scripts.demo_coregistered_verdict --lat -88.0 --lon 180

Honest caveats, stated on the figure: the detector is a single-frame,
equatorially-trained proof of concept, so at the pole it is both weak and
out of domain; its recall keeps it from clearing a site regardless. This
demonstrates the *plumbing* for a real GO — same footprint, meaningful density —
not an operational clearance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.perception.contracts import BoulderDetection, GeoReference, VisualEvidence
from src.perception.prepare import enhance_image, to_unit_gray
from src.reasoning.coregister import fetch_coregistered
from src.reasoning.hazard import summarise_hazard
from src.reasoning.reconcile import decide

DETECTOR_CKPT = Path("runs/detect/runs/boulder_crater/weights/best.pt")
DETECTOR_RECALL = 0.42
OUTPUT_FIG = Path("figures/stage3_coregistered.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=-88.0)
    parser.add_argument("--lon", type=float, default=180.0)
    parser.add_argument("--span-m", type=float, default=1000.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Fetching co-registered image + DEM at {args.lat}, {args.lon} ...")
    patch = fetch_coregistered(args.lat, args.lon, span_m=args.span_m, verbose=True)
    if patch is None:
        print("No co-registered patch available there (needs <= -87 deg, lit, real DEM).")
        return 1
    print(f"  image {patch.image.shape} @ {patch.image_res_m:.0f} m/px, "
          f"DEM {patch.elevation.shape} @ {patch.dem_res_m:.0f} m/px, "
          f"shared footprint {patch.area_km2:.2f} km^2")

    # Physics on the DEM.
    hazard = summarise_hazard(patch.elevation, patch.dem_grid_spacing, f"{args.lat},{args.lon}")

    # Detector on the SAME-footprint image (tiled at training scale).
    boulders_n = craters_n = 0
    versions = {"enhancer": "classical-gamma-clahe",
                "boulder_detector": "not-run", "debris_segmenter": "not-run"}
    recall = None
    if DETECTOR_CKPT.is_file():
        from ultralytics import YOLO

        from src.perception.lroc import iter_tiles
        model = YOLO(str(DETECTOR_CKPT))
        import cv2
        for tile, _, _ in iter_tiles(patch.image, 256, 256):
            enhanced, _ = enhance_image(tile)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            result = model(rgb, conf=0.25, device=args.device, verbose=False)[0]
            if result.boxes is not None:
                cls = result.boxes.cls.cpu().numpy().astype(int)
                boulders_n += int((cls == 0).sum()); craters_n += int((cls == 1).sum())
        versions["boulder_detector"] = f"{DETECTOR_CKPT} (recall {DETECTOR_RECALL:.2f})"
        recall = DETECTOR_RECALL

    boulders = ([BoulderDetection((0, 0, 1, 1), 0.5, "boulder")] * boulders_n
                + [BoulderDetection((0, 0, 1, 1), 0.5, "crater")] * craters_n)
    visual = VisualEvidence(
        georef=GeoReference(image_id=f"{args.lat},{args.lon}", source="LROC NAC polar"),
        boulders=boulders, shadow_fraction=float(np.mean(to_unit_gray(patch.image) < 0.12)),
        model_versions=versions)

    decision = decide(visual, hazard, site_area_km2=patch.area_km2, detector_recall=recall)

    density_b = boulders_n / patch.area_km2
    print(f"\n  boulders {boulders_n} ({density_b:.0f}/km^2 over the SHARED footprint), "
          f"craters {craters_n}")
    print(f"  slope failure {hazard.toppled_fraction_nominal:.1%} @repose, "
          f"{hazard.toppled_fraction_vibration:.1%} under load")
    print(f"  VERDICT: {decision.verdict}  (hazard {decision.hazard_score})")
    for line in decision.rationale + decision.conflicts + decision.evidence_gaps:
        print(f"    - {line}")

    _render(patch, hazard, decision, boulders_n, craters_n, density_b, args)
    return 0


def _render(patch, hazard, decision, nb, nc, density_b, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    colour = {"GO": "#2e7d32", "CAUTION": "#f9a825", "NO-GO": "#c62828"}[decision.verdict]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    fig.suptitle(f"Stage 3 on co-registered image + DEM  ({args.lat}, {args.lon})",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(enhance_image(patch.image)[0], cmap="gray")
    axes[0].set_title(f"NAC image {patch.image_res_m:.0f} m/px\n"
                      f"detector: {nb} boulders, {nc} craters", fontsize=10)
    ls = LightSource(azdeg=315, altdeg=45)
    axes[1].imshow(ls.hillshade(patch.elevation.astype(float), vert_exag=1.0,
                                dx=patch.dem_res_m, dy=patch.dem_res_m), cmap="gray")
    axes[1].set_title(f"LOLA DEM {patch.dem_res_m:.0f} m/px\n"
                      f"failure {hazard.toppled_fraction_nominal:.1%} -> "
                      f"{hazard.toppled_fraction_vibration:.1%} under load", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].text(0.5, -0.09, decision.verdict, transform=axes[0].transAxes, ha="center",
                 va="top", fontsize=15, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=colour, edgecolor="none"))
    axes[0].text(1.02, -0.09, f"boulder density {density_b:.0f}/km^2 over the SAME {patch.area_km2:.2f} km^2",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=9)

    fig.text(0.005, 0.01,
             "Same footprint: count and hazard describe one patch, so density is meaningful. "
             "Detector is a single-frame equatorial proof of concept -- weak and out of domain at "
             "the pole -- so its recall correctly bars clearance. This shows the GO plumbing, not an "
             f"operational clearance.  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.13, top=0.88)
    fig.savefig(OUTPUT_FIG, dpi=140)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FIG}")


if __name__ == "__main__":
    sys.exit(main())
