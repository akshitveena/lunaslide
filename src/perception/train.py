"""Run Stage 1 model training from labelled local data."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from .data import BoulderInstanceDataset, DebrisSegmentationDataset, LowLightImageDataset
from .training import train_enhancer, train_mask_rcnn, train_segmenter
from .models import train_yolov8_boulders


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 enhancer or debris segmenter.")
    sub = parser.add_subparsers(dest="model", required=True)
    enhancer = sub.add_parser("enhancer")
    enhancer.add_argument("--images", required=True)
    enhancer.add_argument("--output", required=True)
    segmenter = sub.add_parser("segmenter")
    segmenter.add_argument("--images", required=True)
    segmenter.add_argument("--masks", required=True)
    segmenter.add_argument("--output", required=True)
    yolo = sub.add_parser("yolo")
    yolo.add_argument("--dataset-yaml", required=True)
    yolo.add_argument("--output", required=True)
    maskrcnn = sub.add_parser("maskrcnn")
    maskrcnn.add_argument("--images", required=True)
    maskrcnn.add_argument("--masks", required=True)
    maskrcnn.add_argument("--output", required=True)
    for command in (enhancer, segmenter, yolo, maskrcnn):
        command.add_argument("--epochs", type=int, default=20)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.model == "enhancer":
        train_enhancer(DataLoader(LowLightImageDataset(args.images), batch_size=args.batch_size, shuffle=True), args.epochs, args.output, args.device)
    elif args.model == "segmenter":
        train_segmenter(DataLoader(DebrisSegmentationDataset(args.images, args.masks), batch_size=args.batch_size, shuffle=True), args.epochs, args.output, args.device)
    elif args.model == "yolo":
        train_yolov8_boulders(args.dataset_yaml, args.output, args.epochs)
    else:
        dataset = BoulderInstanceDataset(args.images, args.masks)
        train_mask_rcnn(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch))), args.epochs, args.output, args.device)


if __name__ == "__main__":
    main()
