"""Stage 1 evidence figures: the trained enhancer on real lunar imagery.

For each target image this renders raw -> classical (gamma+CLAHE) -> learned
(CurveEnhancer) side by side, a luminance histogram, and the auditable evidence
contract (shadow fraction, texture roughness, and each model's version or its
honest ``not-run`` status).  A per-image detection panel is drawn only when
boulder or debris outputs exist, so the same figure grows into a full detection
view once those models are trained -- nothing here needs rewriting then.

    python3 -m scripts.generate_stage1_figures --enhancer-checkpoint checkpoints/enhancer.pt

Writes to figures/.  Runs with or without a trained checkpoint: without one, the
learned column is omitted and the classical path is shown alone, still honest.
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

from src.perception.contracts import GeoReference
from src.perception.pipeline import run_stage1
from src.perception.prepare import enhance_image, to_unit_gray

OUTPUT = Path("figures")


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _texture(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_32F).std())


def render_image(
    name: str,
    image_path: Path,
    output: Path,
    curve_model=None,
    device: str = "cpu",
    enhancer_checkpoint: Path | None = None,
) -> dict | None:
    if not image_path.exists():
        print(f"  {name}: missing {image_path}")
        return None
    raw = _load_gray(image_path)
    raw_unit = to_unit_gray(raw)

    classical, report = enhance_image(raw)  # gamma + CLAHE
    columns = ["Raw", "Classical (gamma+CLAHE)"]
    images = [raw_unit, classical.astype(np.float32) / 255.0]
    if curve_model is not None:
        learned, _ = enhance_image(raw, curve_model, device=device)
        columns.append("Learned (CurveEnhancer)")
        images.append(learned.astype(np.float32) / 255.0)

    n = len(images)
    fig = plt.figure(figsize=(4.6 * n, 8.4))
    grid = fig.add_gridspec(2, n, height_ratios=[2.4, 1.0], hspace=0.22, wspace=0.06)
    fig.suptitle(f"Stage 1 perception evidence — {name}", fontsize=15, fontweight="bold", y=0.98)

    for col, (title, img) in enumerate(zip(columns, images)):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nmean {img.mean():.2f}  |  texture "
                     f"{_texture((img * 255).astype(np.uint8)):.0f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax_hist = fig.add_subplot(grid[1, :2])
    for title, img in zip(columns, images):
        ax_hist.hist((img * 255).ravel(), bins=64, range=(0, 255), histtype="step",
                     linewidth=1.6, label=title)
    ax_hist.set_title("Luminance distribution", fontsize=10)
    ax_hist.set_xlabel("pixel value"); ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=8); ax_hist.grid(alpha=0.3)

    # Evidence contract, populated by the real pipeline. Uses the trained
    # enhancer when one is supplied, so the contract's metrics and version match
    # the learned column shown above rather than silently reporting classical.
    evidence = run_stage1(
        image_path, output / f"_stage1_{name.split()[0].lower()}",
        GeoReference(image_id=name, source="LROC"),
        enhancer_checkpoint=str(enhancer_checkpoint) if enhancer_checkpoint else None,
        device=device,
    )
    versions = evidence.model_versions
    ax_txt = fig.add_subplot(grid[1, 2] if n == 3 else grid[1, 2:])
    ax_txt.axis("off")
    lines = [
        "VisualEvidence contract",
        "-----------------------",
        f"shadow fraction    {evidence.shadow_fraction * 100:5.1f}%",
        f"texture roughness  {evidence.texture_roughness:5.1f}",
        f"gamma (classical)  {report.gamma:5.3f}",
        "",
        f"enhancer          {'trained' if versions['enhancer'] != 'classical-gamma-clahe' else 'classical'}",
        f"boulder_detector  {versions['boulder_detector']}",
        f"boulder_refiner   {versions['boulder_refiner']}",
        f"debris_segmenter  {versions['debris_segmenter']}",
    ]
    ax_txt.text(0.0, 0.98, "\n".join(lines), fontsize=9, family="monospace",
                va="top", transform=ax_txt.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f4f4", edgecolor="#999"))

    fig.text(0.005, 0.005,
             f"Real LROC imagery. 'not-run' means the model has no trained checkpoint and emits "
             f"no result -- never a false all-clear.  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(top=0.90, bottom=0.07)
    target = output / f"stage1_evidence_{name.split()[0].lower()}.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    print(f"  {name}: wrote {target}")
    return {"name": name, "shadow": evidence.shadow_fraction, "texture": evidence.texture_roughness}


def render_patch_gallery(
    patch_dir: Path, output: Path, curve_model, device: str, count: int = 6
) -> None:
    """Enhancer on a spread of real LROC patches, darkest-first."""
    if curve_model is None:
        return
    paths = sorted(patch_dir.glob("*.png"))
    if not paths:
        return
    ranked = sorted(paths, key=lambda p: _load_gray(p).mean())
    step = max(1, len(ranked) // count)
    chosen = ranked[::step][:count]

    fig, axes = plt.subplots(2, len(chosen), figsize=(2.6 * len(chosen), 5.6))
    fig.suptitle("CurveEnhancer across the brightness range of real LROC patches",
                 fontsize=13, fontweight="bold")
    for col, path in enumerate(chosen):
        raw = to_unit_gray(_load_gray(path))
        learned, _ = enhance_image(_load_gray(path), curve_model, device=device)
        after = learned.astype(np.float32) / 255.0
        axes[0, col].imshow(raw, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"raw {raw.mean():.2f}", fontsize=8)
        axes[1, col].imshow(after, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"enhanced {after.mean():.2f}", fontsize=8)
        for row in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("raw", fontsize=9)
    axes[1, 0].set_ylabel("enhanced", fontsize=9)
    fig.text(0.005, 0.01,
             "Darkest-to-brightest, left to right. Enhancement is clean where signal exists; the "
             "darkest near-black patch amplifies sensor quantisation -- the limit of enhancing "
             f"unrecorded signal.  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.16, top=0.88)
    target = output / "stage1_enhancer_gallery.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    print(f"  patch gallery: wrote {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhancer-checkpoint", type=Path, default=Path("checkpoints/enhancer.pt"))
    parser.add_argument("--apollo15", type=Path, default=Path("data/stage1/apollo15/apollo15.png"))
    parser.add_argument("--patches", type=Path, default=Path("data/stage1/lroc_patches"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    curve_model = None
    if args.enhancer_checkpoint.is_file():
        try:
            import torch

            from src.perception.enhancement_model import CurveEnhancer
            state = torch.load(args.enhancer_checkpoint, map_location=args.device, weights_only=True)
            curve_model = CurveEnhancer().to(args.device).eval()
            curve_model.load_state_dict(state["model"])
            print(f"Loaded trained enhancer: {args.enhancer_checkpoint}")
        except Exception as error:
            print(f"Could not load enhancer ({error}); showing classical path only.")
    else:
        print("No enhancer checkpoint; showing classical path only.")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.enhancer_checkpoint if curve_model is not None else None
    render_image("Apollo 15 Landing Site", args.apollo15, args.output, curve_model,
                 args.device, enhancer_checkpoint=checkpoint)
    render_patch_gallery(args.patches, args.output, curve_model, args.device)
    print(f"\nStage 1 figures -> {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
