"""Offline tests for boulder-label geometry (no SAM, no network)."""

from __future__ import annotations

import unittest

import numpy as np

from src.perception.boulder_labels import (
    Box,
    box_from_mask,
    deduplicate,
    is_boulder_like,
    propose_candidates,
)


class TestBox(unittest.TestCase):
    def test_yolo_normalisation_centres_and_scales(self):
        line = Box(10, 20, 30, 60).to_yolo(100, 100)
        cls, cx, cy, w, h = line.split()
        self.assertEqual(cls, "0")
        self.assertAlmostEqual(float(cx), 0.20)   # (10+30)/2/100
        self.assertAlmostEqual(float(cy), 0.40)   # (20+60)/2/100
        self.assertAlmostEqual(float(w), 0.20)
        self.assertAlmostEqual(float(h), 0.40)


class TestBoxFromMask(unittest.TestCase):
    def test_tight_box(self):
        mask = np.zeros((32, 32), bool)
        mask[5:10, 8:12] = True
        box = box_from_mask(mask)
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (8, 5, 12, 10))

    def test_empty_mask_is_none(self):
        self.assertIsNone(box_from_mask(np.zeros((8, 8), bool)))


class TestIsBoulderLike(unittest.TestCase):
    def _disk(self, r: int, size: int = 64) -> np.ndarray:
        yy, xx = np.ogrid[:size, :size]
        return (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= r * r

    def test_compact_blob_accepted(self):
        self.assertTrue(is_boulder_like(self._disk(4)))

    def test_terrain_scale_segment_rejected(self):
        big = np.ones((64, 64), bool)  # whole patch
        self.assertFalse(is_boulder_like(big))

    def test_elongated_ridge_rejected(self):
        ridge = np.zeros((64, 64), bool)
        ridge[30:33, 5:60] = True  # long thin streak
        self.assertFalse(is_boulder_like(ridge))

    def test_tiny_speck_rejected(self):
        speck = np.zeros((64, 64), bool)
        speck[0, 0] = True
        self.assertFalse(is_boulder_like(speck))

    def test_bright_cap_accepted_with_photometry(self):
        # A compact blob that is brighter than its surroundings -> boulder.
        mask = self._disk(4)
        image = np.full((64, 64), 40, np.uint8)
        image[mask] = 200
        self.assertTrue(is_boulder_like(mask, image))

    def test_dark_blob_rejected_by_photometry(self):
        # Same geometry, but the mask is a shadow (darker than surroundings).
        mask = self._disk(4)
        image = np.full((64, 64), 150, np.uint8)
        image[mask] = 30  # crater-shadow blob
        self.assertFalse(is_boulder_like(mask, image))
        # Geometry alone (no image) would have accepted it:
        self.assertTrue(is_boulder_like(mask))


class TestDeduplicate(unittest.TestCase):
    def test_overlapping_boxes_collapse(self):
        a = Box(0, 0, 10, 10)
        b = Box(1, 1, 11, 11)   # ~0.68 IoU with a
        far = Box(50, 50, 60, 60)
        kept = deduplicate([a, b, far])
        self.assertEqual(len(kept), 2)

    def test_disjoint_boxes_kept(self):
        boxes = [Box(0, 0, 5, 5), Box(20, 20, 25, 25), Box(40, 40, 45, 45)]
        self.assertEqual(len(deduplicate(boxes)), 3)


class TestProposeCandidates(unittest.TestCase):
    def test_finds_a_bright_blob_on_dark_ground(self):
        gray = np.full((128, 128), 40, np.uint8)
        gray[60:66, 60:66] = 220  # bright boulder cap
        points = propose_candidates(gray)
        self.assertTrue(points)
        x, y = min(points, key=lambda p: (p[0] - 63) ** 2 + (p[1] - 63) ** 2)
        self.assertLess(abs(x - 63), 6)
        self.assertLess(abs(y - 63), 6)

    def test_flat_image_yields_no_candidates(self):
        self.assertEqual(propose_candidates(np.full((128, 128), 100, np.uint8)), [])

    def test_bright_blob_on_a_terminator_ramp_is_rejected(self):
        # A bright speck sitting on a steep bright-to-dark ramp (a shadow
        # boundary) must be dropped by the ramp filter but found without it.
        ramp = np.tile(np.linspace(230, 5, 128).astype(np.uint8), (128, 1))
        ramp[60:64, 60:64] = 255

        def near_blob(points):
            return any(abs(x - 62) < 6 and abs(y - 62) < 6 for x, y in points)

        self.assertTrue(near_blob(propose_candidates(ramp, max_ramp=1e9)))   # found without filter
        self.assertFalse(near_blob(propose_candidates(ramp, max_ramp=1.0)))  # rejected on the ramp

    def test_rejects_colour_input(self):
        with self.assertRaises(ValueError):
            propose_candidates(np.zeros((16, 16, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
