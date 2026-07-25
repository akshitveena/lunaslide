"""Offline tests for LROC acquisition logic (no network, no rasterio)."""

from __future__ import annotations

import unittest

import numpy as np

from src.perception.lroc import (
    NacProduct,
    build_ode_params,
    is_informative,
    iter_tiles,
    parse_products,
)

# A minimal ODE response shaped like the real one, including the single-element
# collapse ODE applies (Product and Product_file as bare objects, not lists).
ODE_ONE = {
    "ODEResults": {
        "Status": "Success",
        "Products": {"Product": {
            "pdsid": "nac.m123rc",
            "Center_latitude": "26.1",
            "Center_longitude": "3.6",
            "Product_files": {"Product_file": [
                {"URL": "https://x/M123RC.IMG", "KBytes": "264467"},
                {"URL": "https://x/M123RC_pyr.tif", "KBytes": "22001"},
            ]},
        }},
    }
}
ODE_ERROR = {"ODEResults": {"Status": "ERROR", "Error": "Invalid IIPT"}}


class TestParseProducts(unittest.TestCase):
    def test_extracts_pyr_tif_and_metadata(self):
        products = parse_products(ODE_ONE)
        self.assertEqual(len(products), 1)
        p = products[0]
        self.assertEqual(p.product_id, "nac.m123rc")
        self.assertTrue(p.pyr_tif_url.endswith("_pyr.tif"))
        self.assertAlmostEqual(p.center_lat, 26.1)
        self.assertAlmostEqual(p.kbytes, 22001)

    def test_error_response_yields_nothing(self):
        self.assertEqual(parse_products(ODE_ERROR), [])

    def test_product_without_pyr_tif_is_skipped(self):
        response = {"ODEResults": {"Status": "Success", "Products": {"Product": {
            "pdsid": "nac.only_img",
            "Center_latitude": "0", "Center_longitude": "0",
            "Product_files": {"Product_file": {"URL": "https://x/A.IMG", "KBytes": "1"}},
        }}}}
        self.assertEqual(parse_products(response), [])


class TestDistance(unittest.TestCase):
    def test_wraps_across_the_antimeridian(self):
        product = NacProduct("p", 0.0, 179.0, "u", 1.0)
        # 179 deg to -179 deg is 2 deg apart, not 358.
        self.assertAlmostEqual(product.distance_deg(0.0, -179.0), 2.0, places=6)


class TestInformative(unittest.TestCase):
    def test_flat_black_margin_is_rejected(self):
        self.assertFalse(is_informative(np.zeros((64, 64), dtype=np.uint8)))

    def test_saturated_tile_is_rejected(self):
        self.assertFalse(is_informative(np.full((64, 64), 255, dtype=np.uint8)))

    def test_textured_tile_is_kept(self):
        rng = np.random.default_rng(0)
        tile = rng.integers(40, 210, (64, 64), dtype=np.uint8)
        self.assertTrue(is_informative(tile))

    def test_low_contrast_tile_is_rejected(self):
        tile = np.full((64, 64), 128, dtype=np.uint8)
        tile[0, 0] = 130  # negligible variation
        self.assertFalse(is_informative(tile))


class TestIterTiles(unittest.TestCase):
    def test_tiles_cover_the_grid_without_partial_windows(self):
        image = np.arange(300 * 260, dtype=np.uint8).reshape(300, 260)
        tiles = list(iter_tiles(image, size_px=128, stride=128))
        # floor(300/128)=2 rows, floor(260/128)=2 cols
        self.assertEqual(len(tiles), 4)
        for tile, _, _ in tiles:
            self.assertEqual(tile.shape, (128, 128))

    def test_overlapping_stride_yields_more_tiles(self):
        image = np.zeros((256, 256), dtype=np.uint8)
        self.assertEqual(len(list(iter_tiles(image, 128, 128))), 4)
        self.assertEqual(len(list(iter_tiles(image, 128, 64))), 9)

    def test_rejects_bad_geometry(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            list(iter_tiles(image, 0))
        with self.assertRaises(ValueError):
            list(iter_tiles(image, 4, stride=0))


class TestOdeParams(unittest.TestCase):
    def test_no_target_omits_the_spatial_box(self):
        params = build_ode_params(None, None, 2.0, 10)
        self.assertNotIn("minlatitude", params)
        self.assertEqual(params["pt"], "CDRNAC4")

    def test_target_adds_a_box_and_wraps_longitude(self):
        params = build_ode_params(26.0, -179.0, 2.0, 10)
        self.assertAlmostEqual(float(params["minlatitude"]), 24.0)
        # -181 deg wraps to 179 deg.
        self.assertAlmostEqual(float(params["westernlon"]), 179.0)


if __name__ == "__main__":
    unittest.main()
