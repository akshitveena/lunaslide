"""CLI for complete Stage 1 visual-evidence inference."""

from __future__ import annotations

import argparse

from .contracts import GeoReference
from .pipeline import run_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lunaslide Stage 1 on one lunar image.")
    parser.add_argument("image")
    parser.add_argument("output_dir")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--source")
    parser.add_argument("--gsd-m", type=float)
    parser.add_argument("--enhancer-checkpoint")
    parser.add_argument("--yolo-checkpoint")
    parser.add_argument("--maskrcnn-checkpoint")
    parser.add_argument("--debris-checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evidence = run_stage1(
        args.image, args.output_dir,
        GeoReference(image_id=args.image_id, source=args.source, ground_sample_distance_m=args.gsd_m),
        enhancer_checkpoint=args.enhancer_checkpoint, yolo_checkpoint=args.yolo_checkpoint,
        maskrcnn_checkpoint=args.maskrcnn_checkpoint,
        debris_checkpoint=args.debris_checkpoint, device=args.device,
    )
    print(f"Stage 1 evidence written for {evidence.georef.image_id}")


if __name__ == "__main__":
    main()
