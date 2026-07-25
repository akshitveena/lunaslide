"""Quantitatively compare the classical and learned enhancers on real patches.

There is no ground-truth "correctly enhanced" lunar image, so this uses
no-reference metrics chosen to reflect what enhancement is actually for:

* **detail** - variance of the Laplacian; higher means sharper structure.
* **entropy** - Shannon entropy of the histogram; higher means the tonal range
  is used more fully (information content).
* **shadow recovery** - mean local contrast inside the regions that were darkest
  in the raw image; higher means more detail pulled out of shadow, which is the
  headline purpose of the enhancer.
* **flat-region noise** - mean local contrast inside regions that were *flat* in
  the raw image; here any structure is amplified noise, so lower is better. This
  is what catches the learned enhancer's quantisation amplification.

No single number crowns a winner - that needs downstream detection accuracy,
which does not exist yet. This characterises the trade-off honestly instead.

    python3 -m scripts.compare_enhancers --enhancer-checkpoint checkpoints/enhancer.pt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.perception.prepare import enhance_image, to_unit_gray

OUTPUT = Path("figures")


def _local_std(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Per-pixel local standard deviation via box filters."""
    f = image.astype(np.float32)
    mean = cv2.blur(f, (kernel, kernel))
    mean_sq = cv2.blur(f * f, (kernel, kernel))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _entropy(u8: np.ndarray) -> float:
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def metrics(raw_u8: np.ndarray, out_u8: np.ndarray) -> dict[str, float]:
    raw_ls = _local_std(raw_u8)
    out_ls = _local_std(out_u8)
    dark = raw_u8 <= np.percentile(raw_u8, 25)          # darkest quarter of raw
    flat = raw_ls <= np.percentile(raw_ls, 15)          # flattest regions of raw
    return {
        "detail": float(cv2.Laplacian(out_u8, cv2.CV_64F).var()),
        "entropy": _entropy(out_u8),
        "shadow_recovery": float(out_ls[dark].mean()),   # higher better
        "flat_noise": float(out_ls[flat].mean()),        # lower better
        "mean": float(out_u8.mean() / 255.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=Path("data/stage1/lroc_patches"))
    parser.add_argument("--enhancer-checkpoint", type=Path, default=Path("checkpoints/enhancer.pt"))
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    paths = sorted(args.patches.glob("*.png"))[: args.limit]
    if not paths:
        print(f"No patches at {args.patches}. Run: python3 -m scripts.build_lroc_patches")
        return 1

    curve_model = None
    if args.enhancer_checkpoint.is_file():
        import torch

        from src.perception.enhancement_model import CurveEnhancer
        state = torch.load(args.enhancer_checkpoint, map_location=args.device, weights_only=True)
        curve_model = CurveEnhancer().to(args.device).eval()
        curve_model.load_state_dict(state["model"])
    else:
        print(f"No checkpoint at {args.enhancer_checkpoint}; cannot compare.")
        return 1

    print(f"Comparing on {len(paths)} real LROC patches ...")
    rows = {"classical": [], "learned": []}
    for i, path in enumerate(paths, 1):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        raw_u8 = (to_unit_gray(raw) * 255).astype(np.uint8)
        classical, _ = enhance_image(raw)
        learned, _ = enhance_image(raw, curve_model, device=args.device)
        rows["classical"].append(metrics(raw_u8, classical))
        rows["learned"].append(metrics(raw_u8, learned))
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    keys = ["detail", "entropy", "shadow_recovery", "flat_noise", "mean"]
    better = {"detail": "higher", "entropy": "higher", "shadow_recovery": "higher",
              "flat_noise": "lower", "mean": "~0.5"}
    agg = {m: {k: np.array([r[k] for r in rows[m]]) for k in keys} for m in rows}

    print(f"\n{'metric':<16}{'classical':>16}{'learned':>16}   better")
    print("-" * 66)
    for k in keys:
        c, l = agg["classical"][k], agg["learned"][k]
        print(f"{k:<16}{c.mean():>10.2f}±{c.std():<4.1f}{l.mean():>10.2f}±{l.std():<4.1f}   {better[k]}")

    # --- figure: per-metric distributions ---
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    fig.suptitle("Classical (gamma+CLAHE) vs learned (CurveEnhancer) on real LROC patches",
                 fontsize=14, fontweight="bold")
    for ax, k in zip(axes, ["detail", "entropy", "shadow_recovery", "flat_noise"]):
        ax.hist(agg["classical"][k], bins=30, alpha=0.55, label="classical", color="#1f77b4")
        ax.hist(agg["learned"][k], bins=30, alpha=0.55, label="learned", color="#c1440e")
        ax.axvline(agg["classical"][k].mean(), color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.axvline(agg["learned"][k].mean(), color="#c1440e", linestyle="--", linewidth=1.5)
        ax.set_title(f"{k}  ({better[k]} better)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_yticks([])

    verdict = (
        f"n={len(paths)} patches, no-reference metrics (no ground truth exists).  "
        f"Classical: sharper (detail {agg['classical']['detail'].mean():.0f} vs "
        f"{agg['learned']['detail'].mean():.0f}).  "
        f"Learned: {'cleaner' if agg['learned']['flat_noise'].mean() < agg['classical']['flat_noise'].mean() else 'noisier'} "
        f"in flat regions (noise {agg['learned']['flat_noise'].mean():.1f} vs "
        f"{agg['classical']['flat_noise'].mean():.1f}).  "
        f"A decisive winner needs downstream detection accuracy, not these metrics.  "
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    )
    fig.text(0.005, 0.01, verdict, fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "enhancer_comparison.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    print(f"\nWrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
