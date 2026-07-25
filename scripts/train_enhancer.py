"""Train the self-supervised CurveEnhancer on real LROC patches.

This is the one Stage 1 model that needs no labels: the CurveEnhancer learns
per-pixel enhancement curves from unpaired low-light imagery, guided only by
exposure, spatial-consistency, and smoothness objectives (Zero-DCE lineage).

It trains on the output of scripts.build_lroc_patches and, crucially, consumes
the *same* representation at train and inference time — prepare.to_unit_gray of
the raw image — so no train/serve skew (see the prepare module).

    python3 -m scripts.train_enhancer --epochs 40

Produces checkpoints/enhancer.pt and figures/enhancer_training.png (loss curves
plus before/after on held-out patches), so the run leaves auditable evidence
that the model learned something rather than just a weights blob.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PATCHES = Path("data/stage1/lroc_patches")
CHECKPOINT = Path("checkpoints/enhancer.pt")
FIGURE = Path("figures/enhancer_training.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=PATCHES)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    # Defaults chosen empirically on real LROC patches: 8 recursive curve steps
    # with strong curve smoothness (200) give clean enhancement without etching,
    # and a 0.5 exposure target lifts shadows without over-amplifying genuinely
    # low-signal (near-black) regions into their sensor-noise floor.
    parser.add_argument("--curve-steps", type=int, default=8)
    parser.add_argument("--target-exposure", type=float, default=0.5)
    parser.add_argument("--w-exposure", type=float, default=1.0)
    parser.add_argument("--w-spatial", type=float, default=1.0)
    parser.add_argument("--w-smoothness", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        print("PyTorch is required; run: pip install -r requirements.txt")
        return 1

    from src.perception.data import LowLightImageDataset
    from src.perception.enhancement_model import CurveEnhancer, SelfGuidedEnhancementLoss
    from src.perception.training import _device

    if not args.patches.exists():
        print(f"No patches at {args.patches}. Run: python3 -m scripts.build_lroc_patches")
        return 1

    dataset = LowLightImageDataset(args.patches)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    n_val = max(1, int(len(dataset) * args.val_fraction))
    val_index, train_index = permutation[:n_val], permutation[n_val:]
    train_set, val_set = Subset(dataset, train_index), Subset(dataset, val_index)

    device = _device(args.device)
    print(f"Device      {device}")
    print(f"Patches     {len(dataset)}  (train {len(train_set)}, val {len(val_set)})")
    print(f"Model       CurveEnhancer  "
          f"({sum(p.numel() for p in CurveEnhancer().parameters()):,} params)\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = CurveEnhancer(curve_steps=args.curve_steps).to(device)
    criterion = SelfGuidedEnhancementLoss(
        target_exposure=args.target_exposure,
        w_exposure=args.w_exposure,
        w_spatial=args.w_spatial,
        w_smoothness=args.w_smoothness,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def epoch_loss(loader, train: bool) -> dict[str, float]:
        model.train(train)
        totals: dict[str, float] = {}
        batches = 0
        for image in loader:
            image = image.to(device)
            with torch.set_grad_enabled(train):
                enhanced, curves = model(image)
                losses = criterion(image, enhanced, curves)
            if train:
                optimizer.zero_grad(); losses["total"].backward(); optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
        return {key: value / max(batches, 1) for key, value in totals.items()}

    history = {"train": [], "val": []}
    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_loss(train_loader, train=True)
        val_metrics = epoch_loss(val_loader, train=False)
        history["train"].append(train_metrics["total"])
        history["val"].append(val_metrics["total"])
        marker = ""
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({"model": model.state_dict(), "kind": "curve_enhancer",
                        "val_loss": best_val, "epoch": epoch}, args.output)
            marker = "  <- saved"
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"  epoch {epoch:3d}  train {train_metrics['total']:.4f}  "
                  f"val {val_metrics['total']:.4f}  "
                  f"(exp {val_metrics['exposure']:.3f} spa {val_metrics['spatial']:.3f} "
                  f"smo {val_metrics['smoothness']:.3f}){marker}", flush=True)

    improvement = (history['val'][0] - best_val) / max(history['val'][0], 1e-9) * 100
    print(f"\nBest val loss {best_val:.4f} (from {history['val'][0]:.4f}, "
          f"{improvement:.0f}% lower).  Checkpoint -> {args.output}")

    _render_evidence(args, model, val_set, history, device)
    return 0


def _render_evidence(args, model, val_set, history, device) -> None:
    import torch

    from src.perception.enhancement_model import CurveEnhancer

    # Reload the best checkpoint for the figure, not the last-epoch weights.
    best = CurveEnhancer(curve_steps=args.curve_steps).to(device).eval()
    best.load_state_dict(torch.load(args.output, map_location=device)["model"])

    examples = min(4, len(val_set))
    fig, axes = plt.subplots(2, examples + 1, figsize=(3.2 * (examples + 1), 6.4))
    fig.suptitle("CurveEnhancer on held-out real LROC patches", fontsize=14, fontweight="bold")

    axes[0, 0].plot(range(1, len(history["train"]) + 1), history["train"], label="train")
    axes[0, 0].plot(range(1, len(history["val"]) + 1), history["val"], label="val")
    axes[0, 0].set_title("Self-guided loss", fontsize=10)
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)
    axes[1, 0].axis("off")

    for column in range(examples):
        source = val_set[column].unsqueeze(0).to(device)
        with torch.inference_mode():
            enhanced = best(source)[0]
        before = source[0, 0].cpu().numpy()
        after = enhanced[0, 0].clamp(0, 1).cpu().numpy()
        axes[0, column + 1].imshow(before, cmap="gray", vmin=0, vmax=1)
        axes[0, column + 1].set_title(f"input (mean {before.mean():.2f})", fontsize=9)
        axes[1, column + 1].imshow(after, cmap="gray", vmin=0, vmax=1)
        axes[1, column + 1].set_title(f"enhanced (mean {after.mean():.2f})", fontsize=9)
        for row in (0, 1):
            axes[row, column + 1].set_xticks([]); axes[row, column + 1].set_yticks([])

    fig.text(0.005, 0.01,
             f"Self-supervised (no labels): under-exposure lift toward {args.target_exposure:g}, "
             f"Zero-DCE spatial-consistency, and curve-smoothness (w={args.w_smoothness:g}) on "
             f"unpaired real LROC NAC patches.  Rows: input vs enhanced on held-out patches.\n"
             "Enhancement is clean where signal exists; a near-black shadowed patch amplifies "
             "sensor quantisation, the documented limit of enhancing regions with little recorded signal.",
             fontsize=7.5, family="monospace", color="#333333")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.12, top=0.90)
    fig.savefig(args.figure, dpi=140)
    plt.close(fig)
    print(f"Evidence figure -> {args.figure}")


if __name__ == "__main__":
    sys.exit(main())
