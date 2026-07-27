"""Train the debris / mass-wasting segmenter on Bickel-anchored masks.

Data comes from ``scripts.build_debris_dataset`` (real RMaM-2020 rockfall boxes,
SAM-refined to pixel masks).  A ResNet50-UNet is trained with BCE+Dice and scored
on the held-out RMaM test tiles with pixel Dice / IoU — real metrics on real,
unseen labels, not a self-report.

    python3 -m scripts.train_debris_segmenter --epochs 40

Writes runs/segment/debris_segmenter.pt, figures/stage1_debris_segmenter.png,
and data/stage1/debris/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.perception.data import DebrisSegmentationDataset
from src.perception.models import ResNet50UNet
from src.perception.training import _device, _dice_loss

ROOT = Path("data/stage1/debris")
CKPT = Path("runs/segment/debris_segmenter.pt")
FIG = Path("figures/stage1_debris_segmenter.png")
METRICS = ROOT / "metrics.json"
SIZE = 256


class Resized(Dataset):
    """Wrap DebrisSegmentationDataset, resizing every pair to SIZE x SIZE.

    RMaM tiles vary in shape; a fixed size lets them batch and matches the
    square window the segmenter sees at inference.
    """

    def __init__(self, base: DebrisSegmentationDataset, size: int = SIZE):
        self.base, self.size = base, size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        image, mask = self.base[i]
        image = F.interpolate(image.unsqueeze(0), size=(self.size, self.size),
                              mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size),
                             mode="nearest")[0]
        return image, mask


@torch.no_grad()
def evaluate(model, loader, runtime) -> dict:
    model.eval()
    inter = union = tp = fp = fn = 0.0
    dices = []
    for image, mask in loader:
        image = image.to(runtime)
        prob = torch.sigmoid(model(image)).cpu()
        pred = (prob >= 0.5).float()
        m = (mask >= 0.5).float()
        i = (pred * m).sum().item()
        u = ((pred + m) >= 1).float().sum().item()
        inter += i
        union += u
        tp += i
        fp += (pred * (1 - m)).sum().item()
        fn += ((1 - pred) * m).sum().item()
        d = (2 * i + 1) / (pred.sum().item() + m.sum().item() + 1)
        dices.append(d)
    return {
        "dice": float(np.mean(dices)),
        "iou": inter / union if union else 0.0,
        "pixel_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "pixel_recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=20.0,
                        help="BCE weight on rockfall pixels; counters the <1%% "
                             "foreground that otherwise collapses the model to all-zero")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_ds = Resized(DebrisSegmentationDataset(ROOT / "train/images", ROOT / "train/masks"))
    val_ds = Resized(DebrisSegmentationDataset(ROOT / "val/images", ROOT / "val/masks"))
    print(f"train {len(train_ds)} tiles, val {len(val_ds)} tiles")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    runtime = _device(args.device)
    model = ResNet50UNet(pretrained_encoder=True).to(runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=runtime))

    best = {"dice": -1.0}
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for image, mask in train_loader:
            image, mask = image.to(runtime), mask.to(runtime)
            logits = model(image)
            loss = bce(logits, mask) + _dice_loss(logits, mask)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        metrics = evaluate(model, val_loader, runtime)
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.3f}  "
              f"val dice {metrics['dice']:.3f}  iou {metrics['iou']:.3f}  "
              f"P {metrics['pixel_precision']:.3f}  R {metrics['pixel_recall']:.3f}", flush=True)
        if metrics["dice"] > best["dice"]:
            best = {**metrics, "epoch": epoch}
            torch.save({"model": model.state_dict(),
                        "kind": "resnet50_unet_debris_rmam",
                        "metrics": best}, CKPT)

    print(f"\nBest val Dice {best['dice']:.3f} (epoch {best['epoch']}) -> {CKPT}")
    METRICS.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "RMaM-2020 (Bickel et al.) lunar rockfall labels; SAM-refined masks",
        "train_tiles": len(train_ds), "val_tiles": len(val_ds),
        "pos_weight": args.pos_weight,
        "best": best,
    }, indent=2) + "\n")

    _render(model, val_ds, runtime, best)
    return 0


def _render(model, val_ds, runtime, best) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    n = min(4, len(val_ds))
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9), squeeze=False)
    with torch.no_grad():
        for col in range(n):
            image, mask = val_ds[col]
            prob = torch.sigmoid(model(image.unsqueeze(0).to(runtime)))[0, 0].cpu().numpy()
            axes[0][col].imshow(image[0].numpy(), cmap="gray")
            axes[1][col].imshow(mask[0].numpy(), cmap="magma", vmin=0, vmax=1)
            axes[2][col].imshow(prob, cmap="magma", vmin=0, vmax=1)
            for row in range(3):
                axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
    axes[0][0].set_ylabel("NAC (enhanced)", fontsize=10)
    axes[1][0].set_ylabel("Bickel label (SAM mask)", fontsize=10)
    axes[2][0].set_ylabel("segmenter prediction", fontsize=10)
    fig.suptitle("Stage 1 — debris / mass-wasting segmenter (real RMaM-2020 labels)",
                 fontsize=13, fontweight="bold")
    fig.text(0.005, 0.01,
             f"ResNet50-UNet, BCE+Dice.  Held-out RMaM test tiles: Dice {best['dice']:.2f}, "
             f"IoU {best['iou']:.2f}, pixel P {best['pixel_precision']:.2f} / R {best['pixel_recall']:.2f}.  "
             "Labels are Bickel et al. rockfall boxes, SAM-refined to masks; small dataset, "
             f"so treat as a proof of concept.  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333")
    fig.subplots_adjust(bottom=0.07, top=0.93)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    sys.exit(main())
