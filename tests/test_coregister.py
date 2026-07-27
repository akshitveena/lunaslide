"""Offline tests for co-registration logic (no network)."""

from __future__ import annotations

import unittest

import numpy as np

from src.reasoning.coregister import CoRegisteredPatch, fetch_coregistered


def _patch(span_m=1000.0):
    return CoRegisteredPatch(
        image=np.zeros((int(span_m), int(span_m)), np.uint8),
        elevation=np.zeros((int(span_m // 5), int(span_m // 5)), np.float32),
        latitude=-88.5, longitude=30.0, span_m=span_m,
        image_res_m=1.0, dem_res_m=5.0, image_nodata_fraction=0.0,
    )


class TestCoRegisteredPatch(unittest.TestCase):
    def test_shared_footprint_area(self):
        self.assertAlmostEqual(_patch(1000.0).area_km2, 1.0)
        self.assertAlmostEqual(_patch(2000.0).area_km2, 4.0)

    def test_image_and_dem_cover_the_same_ground_at_different_resolutions(self):
        p = _patch(1000.0)
        # 1000 m at 1 m/px vs 5 m/px -> 1000 px vs 200 px, same footprint.
        self.assertEqual(p.image.shape, (1000, 1000))
        self.assertEqual(p.elevation.shape, (200, 200))
        self.assertEqual(p.image.shape[0] * p.image_res_m, p.elevation.shape[0] * p.dem_res_m)

    def test_dem_grid_spacing_is_isotropic_at_pole(self):
        self.assertEqual(_patch().dem_grid_spacing, (5.0, 5.0))


class TestLatitudeGate(unittest.TestCase):
    def test_non_polar_returns_none_without_network(self):
        # Equatorial coordinate: rejected before any fetch is attempted.
        self.assertIsNone(fetch_coregistered(26.13, 3.63, verbose=False))


if __name__ == "__main__":
    unittest.main()
