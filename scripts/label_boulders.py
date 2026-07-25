"""SAM-assisted boulder labelling for real LROC patches.

For each patch: propose boulder candidates (top-hat blobs), refine each with SAM
into a precise mask, keep the boulder-shaped ones, and write YOLO-format boxes
plus a review overlay.  These are DRAFT labels: a human reviews the overlays and
removes false positives / adds misses before the labels are used for training.
A detector is only as trustworthy as the labels behind it.

    python3 -m scripts.label_boulders --limit 12 --review-figure

Writes:
  data/stage1/boulders_yolo/images/*.png   copies of labelled patches
  data/stage1/boulders_yolo/labels/*.txt   YOLO boxes (class 0 = boulder)
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
    Box,
    box_from_mask,
    deduplicate,
    is_boulder_like,
    propose_candidates,
)
from src.perception.prepare import enhance_image

OUTPUT = Path("data/stage1/boulders_yolo")


def label_patch(sam, gray: np.ndarray) -> list[Box]:
    """Draft boulder boxes for one grayscale patch."""
    enhanced, _ = enhance_image(gray)  # classical: the stronger enhancer (measured)
    points = propose_candidates(enhanced)
    if not points:
        return []
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    boxes: list[Box] = []
    for x, y in points:
        result = sam(rgb, points=[[x, y]], labels=[1], verbose=False)
        masks = result[0].masks
        if masks is None or masks.data.shape[0] == 0:
            continue
        mask = masks.data[0].cpu().numpy() >= 0.5
        if is_boulder_like(mask, enhanced):  # photometric check needs the image
            box = box_from_mask(mask)
            if box is not None:
                boxes.append(box)
    return deduplicate(boxes)


def draw_overlay(gray: np.ndarray, boxes: list[Box]) -> np.ndarray:
    enhanced, _ = enhance_image(gray)
    canvas = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=Path("data/stage1/lroc_patches"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--sam-model", default="sam_b.pt")
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

    for sub in ("images", "labels", "review"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    total_boxes, labelled = 0, 0
    overlays: list[np.ndarray] = []
    for i, path in enumerate(paths, 1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        boxes = label_patch(sam, gray)
        stem = path.stem
        shutil.copy(path, args.output / "images" / path.name)
        lines = [b.to_yolo(gray.shape[1], gray.shape[0]) for b in boxes]
        (args.output / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        overlay = draw_overlay(gray, boxes)
        cv2.imwrite(str(args.output / "review" / f"{stem}.png"), overlay)
        if len(overlays) < 8:
            overlays.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        total_boxes += len(boxes)
        labelled += 1 if boxes else 0
        print(f"  [{i}/{len(paths)}] {stem}: {len(boxes)} boulders", flush=True)

    print(f"\nDrafted {total_boxes} boulder boxes across {labelled}/{len(paths)} patches.")
    print(f"  labels -> {args.output/'labels'}    review overlays -> {args.output/'review'}")
    print("  REVIEW REQUIRED: inspect overlays, fix false positives/misses before training.")

    if args.review_figure and overlays:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = min(4, len(overlays))
        rows = (len(overlays) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)
        fig.suptitle("SAM-assisted boulder labels (draft) — real LROC patches",
                     fontsize=13, fontweight="bold")
        for ax in axes.ravel():
            ax.axis("off")
        for ax, img in zip(axes.ravel(), overlays):
            ax.imshow(img)
        fig.text(0.005, 0.01,
                 "Red boxes: top-hat proposals refined by SAM, filtered to boulder-shaped masks. "
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
