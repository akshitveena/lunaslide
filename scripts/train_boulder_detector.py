"""Train and honestly evaluate the two-class boulder/crater YOLO detector.

Trains YOLOv8 on the human-reviewed labels from data/stage1/boulders_yolo and
reports metrics on a held-out split.  These labels must be *reviewed* -- a
detector trained on the raw SAM drafts inherits their false positives.

    python3 -m scripts.train_boulder_detector --epochs 100

Writes the trained weights under runs/ and a detection figure to figures/.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data/stage1/boulders_yolo")
NAMES = {0: "boulder", 1: "crater"}


def build_split(data: Path, val_fraction: float, seed: int) -> tuple[Path, int, int]:
    """Write train/val image lists and a dataset.yaml; return (yaml, n_train, n_val).

    Split is by source NAC frame where possible, so patches from one frame do
    not appear in both train and val (that would leak near-duplicate terrain and
    inflate the metrics).  Falls back to a random split if every patch shares a
    frame.
    """
    images = sorted((data / "images").glob("*.png"))
    labelled = [p for p in images if (data / "labels" / f"{p.stem}.txt").exists()]
    if not labelled:
        raise SystemExit(f"No labelled patches in {data}.")

    # Frame id = filename up to the row/col suffix (e.g. nac_m1417928961lc).
    def frame(p: Path) -> str:
        return p.stem.split("_r")[0]

    frames = sorted({frame(p) for p in labelled})
    rng = random.Random(seed)
    if len(frames) >= 3:
        rng.shuffle(frames)
        n_val_frames = max(1, round(len(frames) * val_fraction))
        val_frames = set(frames[:n_val_frames])
        train = [p for p in labelled if frame(p) not in val_frames]
        val = [p for p in labelled if frame(p) in val_frames]
        split_kind = f"by frame ({len(frames)} frames)"
    else:
        shuffled = labelled[:]
        rng.shuffle(shuffled)
        cut = max(1, round(len(shuffled) * val_fraction))
        val, train = shuffled[:cut], shuffled[cut:]
        split_kind = "random (only 1-2 frames; same-frame leakage possible)"

    (data / "train.txt").write_text("\n".join(str(p.resolve()) for p in train) + "\n")
    (data / "val.txt").write_text("\n".join(str(p.resolve()) for p in val) + "\n")
    yaml = data / "dataset.yaml"
    yaml.write_text(
        f"path: {data.resolve()}\ntrain: train.txt\nval: val.txt\n"
        f"names:\n  0: boulder\n  1: crater\n")
    print(f"Split: {split_kind}  ->  train {len(train)}, val {len(val)}")
    return yaml, len(train), len(val)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base", default="yolov8n.pt")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is required; run: pip install -r requirements.txt")
        return 1

    yaml, n_train, n_val = build_split(args.data, args.val_fraction, args.seed)
    if n_val == 0:
        print("Empty validation split; add more patches.")
        return 1

    model = YOLO(args.base)  # COCO-pretrained backbone, heads replaced for 2 classes
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project="runs", name="boulder_crater", exist_ok=True,
        seed=args.seed, verbose=False,
    )
    metrics = model.val(data=str(yaml), device=args.device, verbose=False)

    print("\n--- Held-out evaluation ---")
    box = metrics.box
    print(f"  mAP50     {box.map50:.3f}")
    print(f"  mAP50-95  {box.map:.3f}")
    for i, name in NAMES.items():
        try:
            print(f"  {name:8} AP50 {box.ap50[i]:.3f}  (precision {box.p[i]:.3f}, recall {box.r[i]:.3f})")
        except (IndexError, TypeError):
            print(f"  {name:8} not present in val split")
    best = Path("runs/boulder_crater/weights/best.pt")
    print(f"\nWeights -> {best}")
    print(f"Trained {n_train} patches, evaluated {n_val}.  "
          f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    # Detection overlay on a few val images, as honest visual evidence.
    _detection_figure(model, args.data, args.device)
    return 0


def _detection_figure(model, data: Path, device: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.perception.prepare import enhance_image

    val = [Path(line) for line in (data / "val.txt").read_text().splitlines() if line.strip()]
    sample = val[:6]
    if not sample:
        return
    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Boulder/crater YOLO on held-out patches — red boulder, blue crater",
                 fontsize=13, fontweight="bold")
    for ax in axes.ravel():
        ax.axis("off")
    colours = {0: (1, 0, 0), 1: (0, 0.5, 1)}
    for ax, path in zip(axes.ravel(), sample):
        enhanced, _ = enhance_image(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        result = model(rgb, device=device, verbose=False)[0]
        ax.imshow(rgb)
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls.item())
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=colours.get(cls, (1, 1, 0)), linewidth=1.2))
    fig.text(0.005, 0.01, "Predictions on patches the detector never trained on.  "
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    Path("figures").mkdir(exist_ok=True)
    fig.subplots_adjust(bottom=0.08, top=0.90)
    fig.savefig("figures/boulder_detection_eval.png", dpi=140)
    plt.close(fig)
    print("Detection figure -> figures/boulder_detection_eval.png")


if __name__ == "__main__":
    sys.exit(main())
