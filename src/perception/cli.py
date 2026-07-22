"""Command-line entry point for the Stage 1 enhancement pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .preprocessing import enhance_lunar_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance one lunar grayscale image for Stage 1.")
    parser.add_argument("input", type=Path, help="Input image readable by OpenCV")
    parser.add_argument("output", type=Path, help="Output enhanced PNG/TIFF")
    parser.add_argument("--report", type=Path, help="Optional JSON preprocessing report")
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced, report = enhance_lunar_image(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), enhanced):
        raise SystemExit(f"Could not write image: {args.output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
