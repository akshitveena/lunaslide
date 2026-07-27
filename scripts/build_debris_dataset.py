"""Build a debris/mass-wasting segmentation dataset from real Bickel labels.

Source: RMaM-2020 (Bickel et al.), the authors' own human-labelled rockfall set
on the Moon — real NAC tiles with real rockfall bounding boxes.  Boxes locate
mass-wasting features but do not give their shape, so SAM refines each box into a
pixel mask; the per-tile union is the binary debris label.

Real observed labels (Bickel) + SAM for geometry only.  The physics engine plays
no part, so there is no circularity — this is the honest path we could not find
earlier when no debris-mask dataset seemed to exist.

    python3 -m scripts.build_debris_dataset

Writes data/stage1/debris/{train,val}/{images,masks}/ for the segmenter.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np

from src.perception.prepare import enhance_image

ZIP = Path("data/rmam/RMaM-2020.zip")
OUT = Path("data/stage1/debris")
SPLITS = {"train": ("moon/train_images/", "moon/train_labels/train_labels_m.csv"),
          "val": ("moon/test_images/", "moon/test_labels/test_labels_m.csv")}


def read_boxes(zf: zipfile.ZipFile, csv_name: str) -> dict[str, list[tuple[int, int, int, int]]]:
    boxes: dict[str, list] = defaultdict(list)
    text = zf.read(csv_name).decode("utf-8", "replace")
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 6:
            continue
        name, x1, y1, x2, y2 = row[0], *row[1:5]
        try:
            box = (int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))
        except ValueError:
            continue  # empty-coord negative row: do not create a key
        boxes[name].append(box)
    return boxes


def _clean_boxes(boxes, w: int, h: int):
    """Clip to bounds and drop degenerate boxes that make SAM's encoder fail."""
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
        y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append((x1, y1, x2, y2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sam-model", default="sam_b.pt")
    parser.add_argument("--limit", type=int, default=0, help="cap tiles per split (0 = all)")
    args = parser.parse_args()

    if not args.zip.is_file():
        print(f"Missing {args.zip}. Download RMaM-2020 first.")
        return 1
    try:
        from ultralytics import SAM
    except ImportError:
        print("ultralytics required.")
        return 1
    sam = SAM(args.sam_model)
    zf = zipfile.ZipFile(args.zip)

    for split, (img_prefix, csv_name) in SPLITS.items():
        boxes_by_tile = read_boxes(zf, csv_name)
        positives = sorted(n for n in zf.namelist()
                           if n.startswith(img_prefix) and n.endswith(".tif")
                           and Path(n).name in boxes_by_tile)
        # Hard negatives (tiles with no rockfall) sharpen precision, but only in
        # training: a Dice on all-empty val masks would be trivially perfect and
        # inflate the score, so validation stays positives-only and honest.
        negatives = ([] if split == "val" else
                     sorted(n for n in zf.namelist()
                            if n.startswith(img_prefix) and n.endswith(".tif")
                            and Path(n).name not in boxes_by_tile))
        tiles = positives + negatives
        if args.limit:
            tiles = positives[: args.limit] + negatives[: args.limit]
        img_dir = args.out / split / "images"
        mask_dir = args.out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        for i, tile in enumerate(tiles, 1):
            name = Path(tile).name
            arr = cv2.imdecode(np.frombuffer(zf.read(tile), np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            enhanced, _ = enhance_image(arr)  # what the segmenter will see
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            h, w = arr.shape
            boxes = _clean_boxes(boxes_by_tile.get(name, []), w, h)
            mask = np.zeros(arr.shape, np.uint8)
            if not boxes:  # hard-negative tile: real image, empty debris mask
                stem = Path(name).stem
                cv2.imwrite(str(img_dir / f"{stem}.png"), arr)
                cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
                kept += 1
                continue
            sam_ok = False
            try:
                result = sam(rgb, bboxes=[list(b) for b in boxes], verbose=False)
                masks = result[0].masks
                if masks is not None and masks.data.shape[0] > 0:
                    for m in masks.data.cpu().numpy():
                        mask[m >= 0.5] = 255
                    sam_ok = True
            except Exception:
                sam_ok = False
            if not sam_ok:  # SAM failed/empty — fall back to the boxes themselves
                for x1, y1, x2, y2 in boxes:
                    mask[y1:y2, x1:x2] = 255
            stem = Path(name).stem
            cv2.imwrite(str(img_dir / f"{stem}.png"), arr)      # raw; read_gray enhances at train
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
            kept += 1
            if i % 50 == 0:
                print(f"  {split}: {i}/{len(tiles)}", flush=True)
        coverage = _mean_coverage(mask_dir)
        print(f"{split}: {kept} tiles, mean debris coverage {coverage:.1%}")

    print(f"\nWrote {args.out}/  — train/val images + masks")
    return 0


def _mean_coverage(mask_dir: Path) -> float:
    vals = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean() / 255.0
            for p in mask_dir.glob("*.png")]
    return float(np.mean(vals)) if vals else 0.0


if __name__ == "__main__":
    sys.exit(main())
