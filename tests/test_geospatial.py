"""Tests for Stage 1 georeferencing recovery and merge semantics."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.perception.contracts import GeoReference
from src.perception.geospatial import (
    MOON_RADIUS_M,
    RasterGeometry,
    _ground_sample_distance,
    merge_georeference,
    read_raster_geometry,
)


@dataclass
class StubTransform:
    e: float


@dataclass
class StubCRS:
    is_geographic: bool
    linear_units: str | None = None


class TestGroundSampleDistance(unittest.TestCase):
    def test_projected_metre_raster_reports_its_pixel_size(self):
        gsd = _ground_sample_distance(StubCRS(False, "metre"), StubTransform(-118.0))
        self.assertAlmostEqual(gsd, 118.0)

    def test_geographic_raster_is_converted_through_the_lunar_radius(self):
        # The LOLA mosaic's 118 m pixels are ~0.00389 degrees of latitude.
        degrees = math.degrees(118.0 / MOON_RADIUS_M)
        gsd = _ground_sample_distance(StubCRS(True), StubTransform(-degrees))
        self.assertAlmostEqual(gsd, 118.0, places=6)

    def test_unknown_linear_units_are_reported_as_unknown(self):
        self.assertIsNone(_ground_sample_distance(StubCRS(False, "us survey foot"), StubTransform(-1.0)))

    def test_degenerate_transform_is_reported_as_unknown(self):
        self.assertIsNone(_ground_sample_distance(StubCRS(False, "metre"), StubTransform(0.0)))


class TestReadRasterGeometry(unittest.TestCase):
    def test_plain_png_carries_no_georeferencing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((16, 16), dtype=np.uint8)))
            self.assertTrue(read_raster_geometry(path).is_empty())

    def test_missing_file_degrades_instead_of_raising(self):
        self.assertTrue(read_raster_geometry("/nonexistent/image.tif").is_empty())


class TestMergeGeoreference(unittest.TestCase):
    def test_caller_supplied_values_are_never_overwritten(self):
        original = GeoReference(
            image_id="APOLLO15",
            crs="IAU_2015:30100",
            transform=(1.0, 0.5, 0.0, 2.0, 0.0, -0.5),
            ground_sample_distance_m=0.5,
        )
        merged = merge_georeference(original, "/nonexistent/image.tif")
        self.assertEqual(merged, original)

    def test_ungeoreferenced_input_leaves_the_contract_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            cv2.imwrite(str(path), np.zeros((16, 16), dtype=np.uint8))
            original = GeoReference(image_id="preview")
            self.assertEqual(merge_georeference(original, path), original)

    def test_gaps_are_filled_from_the_raster(self):
        recovered = RasterGeometry(
            crs="IAU_2015:30110",
            transform=(100.0, 0.5, 0.0, 200.0, 0.0, -0.5),
            ground_sample_distance_m=0.5,
        )
        original = GeoReference(image_id="patch", source="LROC")
        merged = _merge_with(original, recovered)
        self.assertEqual(merged.crs, "IAU_2015:30110")
        self.assertEqual(merged.transform, (100.0, 0.5, 0.0, 200.0, 0.0, -0.5))
        self.assertEqual(merged.ground_sample_distance_m, 0.5)
        self.assertEqual(merged.image_id, "patch")
        self.assertEqual(merged.source, "LROC")

    def test_a_partially_specified_contract_keeps_only_its_own_values(self):
        recovered = RasterGeometry(
            crs="IAU_2015:30110",
            transform=(100.0, 0.5, 0.0, 200.0, 0.0, -0.5),
            ground_sample_distance_m=0.5,
        )
        # The operator overrides a known-bad product GSD but not the transform.
        original = GeoReference(image_id="patch", ground_sample_distance_m=0.4919)
        merged = _merge_with(original, recovered)
        self.assertEqual(merged.ground_sample_distance_m, 0.4919)
        self.assertEqual(merged.transform, (100.0, 0.5, 0.0, 200.0, 0.0, -0.5))


def _merge_with(georef: GeoReference, geometry: RasterGeometry) -> GeoReference:
    """Exercise merge_georeference against a synthesised raster geometry."""
    import src.perception.geospatial as module

    original = module.read_raster_geometry
    module.read_raster_geometry = lambda _path: geometry
    try:
        return module.merge_georeference(georef, "ignored")
    finally:
        module.read_raster_geometry = original


if __name__ == "__main__":
    unittest.main()
