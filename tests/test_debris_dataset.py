"""Offline tests for the RMaM debris-dataset builder (no network, no SAM)."""

from __future__ import annotations

import io
import unittest

from scripts.build_debris_dataset import _clean_boxes, read_boxes


class _FakeZip:
    """Minimal stand-in exposing the one method read_boxes needs."""

    def __init__(self, text: str):
        self._text = text

    def read(self, _name: str) -> bytes:
        return self._text.encode("utf-8")


class TestReadBoxes(unittest.TestCase):
    def test_parses_valid_rows_and_skips_empty_negative_rows(self):
        csv = "1.tif,10,20,40,60,rockfall\n1.tif,5,5,15,15,rockfall\nneg0.tif,,,,,\n"
        boxes = read_boxes(_FakeZip(csv), "x")
        self.assertEqual(len(boxes["1.tif"]), 2)
        self.assertEqual(boxes["1.tif"][0], (10, 20, 40, 60))
        self.assertNotIn("neg0.tif", boxes)  # empty-coord negatives excluded


class TestCleanBoxes(unittest.TestCase):
    def test_clips_to_image_bounds(self):
        self.assertEqual(_clean_boxes([(-5, -5, 50, 50)], 40, 30), [(0, 0, 40, 30)])

    def test_drops_degenerate_boxes(self):
        # zero-width / zero-height boxes break SAM's prompt encoder
        self.assertEqual(_clean_boxes([(10, 10, 10, 20)], 100, 100), [])
        self.assertEqual(_clean_boxes([(10, 10, 20, 11)], 100, 100), [])

    def test_normalises_reversed_coordinates(self):
        self.assertEqual(_clean_boxes([(40, 60, 10, 20)], 100, 100), [(10, 20, 40, 60)])


if __name__ == "__main__":
    unittest.main()
