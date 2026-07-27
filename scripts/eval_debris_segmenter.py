"""Evaluate a trained debris segmenter and (re)generate its evidence artifacts.

Loads the saved checkpoint, scores it on the held-out RMaM test tiles, and writes
metrics.json plus the evidence figure — so the reported numbers and picture always
come from the actual weights on disk, even if training was interrupted.

    python3 -m scripts.eval_debris_segmenter
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.train_debris_segmenter import (METRICS, ROOT, Resized, _render,
                                            evaluate)
from src.perception.data import DebrisSegmentationDataset
from src.perception.models import ResNet50UNet
from src.perception.training import _device

CKPT = Path("runs/segment/debris_segmenter.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        print(f"Missing checkpoint {args.ckpt}.")
        return 1

    val_ds = Resized(DebrisSegmentationDataset(ROOT / "val/images", ROOT / "val/masks"))
    train_n = len(list((ROOT / "train/images").glob("*.png")))
    runtime = _device(args.device)
    model = ResNet50UNet(pretrained_encoder=False).to(runtime)
    ckpt = torch.load(args.ckpt, map_location=runtime, weights_only=False)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate(model, DataLoader(val_ds, batch_size=8), runtime)
    metrics["epoch"] = ckpt.get("metrics", {}).get("epoch")
    print(f"held-out ({len(val_ds)} tiles): dice {metrics['dice']:.3f}  "
          f"iou {metrics['iou']:.3f}  P {metrics['pixel_precision']:.3f}  "
          f"R {metrics['pixel_recall']:.3f}")

    METRICS.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "RMaM-2020 (Bickel et al.) lunar rockfall labels; SAM-refined masks",
        "checkpoint": str(args.ckpt),
        "train_tiles": train_n, "val_tiles": len(val_ds),
        "best": metrics,
    }, indent=2) + "\n")
    _render(model, val_ds, runtime, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
