"""Interactive review of draft boulder labels — the human-in-the-loop step.

Opens each patch with its SAM-drafted boxes and lets you correct them, then
writes clean YOLO labels.  This is the step that turns noisy auto-labels into
trustworthy training data; a detector trained on unreviewed labels inherits
their mistakes.

Boxes are two-class: red = boulder (class 0), blue = crater (class 1).

Controls (shown in the window title too):
  left-click a box        -> delete it (false positive)
  left-drag on empty      -> draw a new box (class = current add-class)
  b / c                   -> set add-class to boulder / crater
  t                       -> toggle the class of the box under the cursor
  n / right-arrow         -> save this patch and go to the next
  p / left-arrow          -> save this patch and go back
  u                       -> undo the last box you added
  r                       -> reset this patch to its original draft
  q / close window        -> save and quit

Progress is saved every time you change patches, so you can stop and resume.

    python3 -m scripts.review_boulders
    python3 -m scripts.review_boulders --data data/stage1/boulders_yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Interactive backend on purpose — this script opens a window.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


CLASS_COLOURS = {0: "red", 1: "deepskyblue"}  # 0 boulder, 1 crater


def load_boxes(label_path: Path, w: int, h: int) -> list[list[int]]:
    """Read YOLO ``cls cx cy w h`` into pixel ``[x1,y1,x2,y2,cls]``."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = (float(p) for p in parts)
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        boxes.append([x1, y1, x2, y2, int(cls)])
    return boxes


def save_boxes(label_path: Path, boxes: list[list[int]], w: int, h: int) -> None:
    lines = []
    for x1, y1, x2, y2, cls in boxes:
        cx = (x1 + x2) / 2 / w; cy = (y1 + y2) / 2 / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {abs(x2 - x1) / w:.6f} {abs(y2 - y1) / h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


class Reviewer:
    def __init__(self, images: list[Path], label_dir: Path):
        self.images = images
        self.label_dir = label_dir
        self.index = 0
        self.add_class = 0  # class assigned to newly drawn boxes
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.boxes: list[list[int]] = []
        self.original: list[list[int]] = []
        self.selector = RectangleSelector(
            self.ax, self._on_draw, useblit=True, button=[1],
            minspanx=3, minspany=3, spancoords="pixels", interactive=False,
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._load()

    def _load(self) -> None:
        path = self.images[self.index]
        self.gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.gray.shape
        self.label_path = self.label_dir / f"{path.stem}.txt"
        self.boxes = load_boxes(self.label_path, self.w, self.h)
        self.original = [b[:] for b in self.boxes]
        self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.gray, cmap="gray", vmin=0, vmax=255)
        for x1, y1, x2, y2, cls in self.boxes:
            self.ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=CLASS_COLOURS.get(cls, "red"), linewidth=1.3))
        n_b = sum(1 for b in self.boxes if b[4] == 0)
        n_c = sum(1 for b in self.boxes if b[4] == 1)
        adding = "boulder" if self.add_class == 0 else "crater"
        self.ax.set_title(
            f"[{self.index + 1}/{len(self.images)}] {self.images[self.index].name}   "
            f"red boulder={n_b}  blue crater={n_c}   adding: {adding}\n"
            "click=delete  drag=add  b/c=add-class  t=toggle  n/p=nav  u=undo  r=reset  q=quit",
            fontsize=9)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def _on_draw(self, epress, erelease) -> None:
        x1, y1 = int(epress.xdata), int(epress.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self.add_class])
        self._redraw()

    def _box_under(self, x: float, y: float) -> int | None:
        hits = [i for i, b in enumerate(self.boxes) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if not hits:
            return None
        return min(hits, key=lambda i: (self.boxes[i][2] - self.boxes[i][0])
                   * (self.boxes[i][3] - self.boxes[i][1]))

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        # Only treat as delete if the click lands inside an existing box; a
        # drag to add is handled by the RectangleSelector instead.
        hit = self._box_under(event.xdata, event.ydata)
        if hit is not None:
            del self.boxes[hit]
            self._redraw()

    def _save(self) -> None:
        save_boxes(self.label_path, self.boxes, self.w, self.h)

    def _on_key(self, event) -> None:
        if event.key in ("n", "right"):
            self._save()
            self.index = min(self.index + 1, len(self.images) - 1)
            self._load()
        elif event.key in ("p", "left"):
            self._save()
            self.index = max(self.index - 1, 0)
            self._load()
        elif event.key == "b":
            self.add_class = 0; self._redraw()
        elif event.key == "c":
            self.add_class = 1; self._redraw()
        elif event.key == "t" and event.inaxes == self.ax and event.xdata is not None:
            hit = self._box_under(event.xdata, event.ydata)
            if hit is not None:
                self.boxes[hit][4] ^= 1  # toggle boulder <-> crater
                self._redraw()
        elif event.key == "u" and self.boxes:
            self.boxes.pop(); self._redraw()
        elif event.key == "r":
            self.boxes = [b[:] for b in self.original]; self._redraw()
        elif event.key == "q":
            self._save(); plt.close(self.fig)

    def run(self) -> None:
        print(f"Reviewing {len(self.images)} patches. Close the window or press q to finish.")
        plt.show()
        self._save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/stage1/boulders_yolo"))
    args = parser.parse_args()

    images = sorted((args.data / "images").glob("*.png"))
    if not images:
        print(f"No patches at {args.data/'images'}. Run scripts.label_boulders first.")
        return 1
    if matplotlib.get_backend().lower() == "agg":
        print("matplotlib has no interactive backend here; run this in your local terminal.")
        return 1
    Reviewer(images, args.data / "labels").run()
    print("Review saved. Clean labels are in", args.data / "labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
