"""SAM-assisted boulder and crater labelling for real LROC patches.

For each patch: propose candidates (bright top-hat blobs for boulder caps, dark
black-hat blobs for crater interiors), refine each with SAM, keep the
compact ones, and classify each as boulder or crater by its shadow geometry
relative to the sun (see boulder_labels.classify_relief).  Writes two-class
YOLO boxes plus a review overlay.  These are DRAFT labels: a human reviews and
corrects them before training -- a detector is only as good as its labels.

The enhancer is a switch.  Classical gamma+CLAHE is the default because it is
measurably sharper on well-lit NAC patches (detail 2388 vs 2121), and sharp
edges matter for small-feature detection; the learned CurveEnhancer is offered
for the low-light regime where it wins instead.  Whatever is chosen here is what
a detector must also be served at inference.

    python3 -m scripts.label_boulders --limit 12 --review-figure
    python3 -m scripts.label_boulders --enhancer learned   # low-light regime

Writes:
  data/stage1/boulders_yolo/images/*.png   copies of labelled patches
  data/stage1/boulders_yolo/labels/*.txt   YOLO boxes (class 0=boulder, 1=crater)
  data/stage1/boulders_yolo/dataset.yaml   class names for training
  data/stage1/boulders_yolo/review/*.png   per-patch overlays for human review
  figures/boulder_labelling_review.png      a contact sheet of a few overlays
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.perception.boulder_labels import (
    CLASS_NAMES,
    Box,
    box_from_mask,
    classify_relief,
    deduplicate_features,
    estimate_sun_vector,
    passes_geometry,
    propose_candidates,
    propose_dark_candidates,
)
from src.perception.prepare import enhance_image

OUTPUT = Path("data/stage1/boulders_yolo")
# BGR colours for the review overlay, indexed by class.
_COLOURS = {0: (0, 0, 255), 1: (255, 128, 0)}  # boulder red, crater blue


def label_patch(sam, enhanced: np.ndarray) -> list[tuple[Box, int]]:
    """Draft ``(box, class)`` features for one enhanced patch.

    class 0 = boulder (positive relief), class 1 = crater (negative relief).
    """
    sun = estimate_sun_vector(enhanced)
    points = propose_candidates(enhanced) + propose_dark_candidates(enhanced)
    if not points:
        return []
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    features: list[tuple[Box, int]] = []
    for x, y in points:
        result = sam(rgb, points=[[x, y]], labels=[1], verbose=False)
        masks = result[0].masks
        if masks is None or masks.data.shape[0] == 0:
            continue
        mask = masks.data[0].cpu().numpy() >= 0.5
        if not passes_geometry(mask):
            continue
        relief = classify_relief(mask, enhanced, sun)
        if relief is None:  # ambiguous asymmetry -> leave for the human
            continue
        box = box_from_mask(mask)
        if box is not None:
            features.append((box, CLASS_NAMES.index(relief)))
    # One object cannot be both a boulder and a crater: dedupe across classes.
    return deduplicate_features(features)


def draw_overlay(enhanced: np.ndarray, features: list[tuple[Box, int]]) -> np.ndarray:
    canvas = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for box, cls in features:
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), _COLOURS[cls], 1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=Path("data/stage1/lroc_patches"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--sam-model", default="sam_b.pt")
    parser.add_argument("--enhancer", choices=["classical", "learned"], default="classical",
                        help="classical gamma+CLAHE (default, sharper on NAC) or learned CurveEnhancer")
    parser.add_argument("--enhancer-checkpoint", type=Path, default=Path("checkpoints/enhancer.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--review-figure", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.patches.glob("*.png"))[: args.limit]
    if not paths:
        print(f"No patches at {args.patches}. Run: python3 -m scripts.build_lroc_patches")
        return 1

    try:
        from ultralytics import SAM
    except ImportError:
        print("ultralytics is required; run: pip install -r requirements.txt")
        return 1
    sam = SAM(args.sam_model)

    curve_model = None
    if args.enhancer == "learned":
        import torch

        from src.perception.enhancement_model import CurveEnhancer
        if not args.enhancer_checkpoint.is_file():
            print(f"No enhancer checkpoint at {args.enhancer_checkpoint}; train it first.")
            return 1
        state = torch.load(args.enhancer_checkpoint, map_location=args.device, weights_only=True)
        curve_model = CurveEnhancer().to(args.device).eval()
        curve_model.load_state_dict(state["model"])
    print(f"Enhancer: {args.enhancer}")

    for sub in ("images", "labels", "review"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: images\nval: images\n"
        f"names:\n  0: {CLASS_NAMES[0]}\n  1: {CLASS_NAMES[1]}\n")

    counts = {0: 0, 1: 0}
    labelled = 0
    overlays: list[np.ndarray] = []
    for i, path in enumerate(paths, 1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        enhanced, _ = enhance_image(gray, curve_model, device=args.device)
        features = label_patch(sam, enhanced)
        stem = path.stem
        shutil.copy(path, args.output / "images" / path.name)
        lines = [box.to_yolo(gray.shape[1], gray.shape[0], cls=cls) for box, cls in features]
        (args.output / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        overlay = draw_overlay(enhanced, features)
        cv2.imwrite(str(args.output / "review" / f"{stem}.png"), overlay)
        if len(overlays) < 8:
            overlays.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        n_b = sum(1 for _, c in features if c == 0)
        n_c = sum(1 for _, c in features if c == 1)
        counts[0] += n_b; counts[1] += n_c
        labelled += 1 if features else 0
        print(f"  [{i}/{len(paths)}] {stem}: {n_b} boulders, {n_c} craters", flush=True)

    print(f"\nDrafted {counts[0]} boulders + {counts[1]} craters across "
          f"{labelled}/{len(paths)} patches.")
    print(f"  labels -> {args.output/'labels'}    review overlays -> {args.output/'review'}")
    print("  REVIEW REQUIRED: inspect overlays, fix false positives/misses before training.")

    if args.review_figure and overlays:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = min(4, len(overlays))
        rows = (len(overlays) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)
        fig.suptitle("SAM-assisted labels (draft) — red = boulder, blue = crater",
                     fontsize=13, fontweight="bold")
        for ax in axes.ravel():
            ax.axis("off")
        for ax, img in zip(axes.ravel(), overlays):
            ax.imshow(img)
        fig.text(0.005, 0.01,
                 "Boulder (red) vs crater (blue) by shadow geometry relative to the sun. "
                 "Draft labels for human review, not final training data.  "
                 f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
                 fontsize=7.5, family="monospace", color="#333333")
        Path("figures").mkdir(exist_ok=True)
        fig.subplots_adjust(bottom=0.08, top=0.92)
        fig.savefig("figures/boulder_labelling_review.png", dpi=140)
        plt.close(fig)
        print("  review figure -> figures/boulder_labelling_review.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
