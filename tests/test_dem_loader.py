"""Coordinate-handling tests for the LOLA patch loader.

These cover the pure geometry helpers and run without rasterio or a network,
which is deliberate: the coordinate bugs they guard against were invisible
precisely because every failure fell back to synthetic terrain.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.physics.dem_loader import (
    MOON_RADIUS_M,
    bounds_are_degrees,
    fetch_patch,
    lonlat_to_raster_xy,
    wrap_longitude,
)

# The global mosaic in each of the two conventions USGS publishes.
DEGREE_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
DEGREE_BOUNDS_0_360 = (0.0, -90.0, 360.0, 90.0)
METRE_BOUNDS = (
    -np.pi * MOON_RADIUS_M,
    -np.pi / 2 * MOON_RADIUS_M,
    np.pi * MOON_RADIUS_M,
    np.pi / 2 * MOON_RADIUS_M,
)


class TestBoundsDetection(unittest.TestCase):
    def test_degree_bounds_are_recognised(self):
        self.assertTrue(bounds_are_degrees(*DEGREE_BOUNDS))
        self.assertTrue(bounds_are_degrees(*DEGREE_BOUNDS_0_360))

    def test_metre_bounds_are_recognised(self):
        self.assertFalse(bounds_are_degrees(*METRE_BOUNDS))

    def test_polar_stereographic_metre_bounds_are_recognised(self):
        self.assertFalse(bounds_are_degrees(-304000.0, -304000.0, 304000.0, 304000.0))


class TestLongitudeWrapping(unittest.TestCase):
    def test_negative_longitude_folds_into_a_0_360_raster(self):
        self.assertAlmostEqual(wrap_longitude(-170.0, 0.0, 360.0), 190.0)

    def test_high_longitude_folds_into_a_signed_raster(self):
        self.assertAlmostEqual(wrap_longitude(190.0, -180.0, 180.0), -170.0)

    def test_longitude_already_in_range_is_unchanged(self):
        self.assertAlmostEqual(wrap_longitude(3.63, -180.0, 180.0), 3.63)
        self.assertAlmostEqual(wrap_longitude(3.63, 0.0, 360.0), 3.63)

    def test_partial_coverage_rasters_are_left_alone(self):
        # Wrapping is meaningless for a tile that does not span the globe.
        self.assertAlmostEqual(wrap_longitude(-170.0, 10.0, 20.0), -170.0)


class TestCoordinateProjection(unittest.TestCase):
    def test_degree_raster_receives_degrees(self):
        x, y = lonlat_to_raster_xy(3.63, 26.13, DEGREE_BOUNDS)
        self.assertAlmostEqual(x, 3.63)
        self.assertAlmostEqual(y, 26.13)

    def test_metre_raster_receives_metres(self):
        x, y = lonlat_to_raster_xy(3.63, 26.13, METRE_BOUNDS)
        self.assertAlmostEqual(x, np.radians(3.63) * MOON_RADIUS_M, places=3)
        self.assertAlmostEqual(y, np.radians(26.13) * MOON_RADIUS_M, places=3)

    def test_apollo15_and_shackleton_both_land_inside_the_mosaic(self):
        for name, lon, lat in (("apollo15", 3.63, 26.13), ("shackleton", 0.0, -89.5)):
            for bounds in (DEGREE_BOUNDS, DEGREE_BOUNDS_0_360, METRE_BOUNDS):
                with self.subTest(site=name, bounds=bounds[0]):
                    x, y = lonlat_to_raster_xy(lon, lat, bounds)
                    self.assertTrue(bounds[0] <= x <= bounds[2])
                    self.assertTrue(bounds[1] <= y <= bounds[3])

    def test_western_longitude_lands_inside_a_0_360_mosaic(self):
        # main.py samples longitudes from [-180, 180]; on a 0..360 mosaic every
        # negative one previously fell outside the raster and silently fell
        # back to synthetic terrain.
        x, _ = lonlat_to_raster_xy(-45.0, 0.0, DEGREE_BOUNDS_0_360)
        self.assertAlmostEqual(x, 315.0)
        self.assertTrue(DEGREE_BOUNDS_0_360[0] <= x <= DEGREE_BOUNDS_0_360[2])


class TestFetchValidation(unittest.TestCase):
    def test_rejects_impossible_requests(self):
        with self.assertRaises(ValueError):
            fetch_patch(0.0, 0.0, size_px=0)
        with self.assertRaises(ValueError):
            fetch_patch(120.0, 0.0)

    def test_unreachable_mosaic_returns_none_rather_than_guessing(self):
        patch = fetch_patch(
            26.13,
            3.63,
            size_px=8,
            url="https://invalid.invalid/no-such-mosaic.tif",
            timeout_s=1,
            cache_dir=None,
            verbose=False,
        )
        self.assertIsNone(patch)


if __name__ == "__main__":
    unittest.main()
