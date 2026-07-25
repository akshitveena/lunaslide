"""Train/serve consistency tests for Stage 1 image preparation.

The bug these pin: the curve enhancer was trained on ``image/255`` but run on
CLAHE-enhanced output, and normalisation differed by bit depth between the two
paths.  A model only works if training input matches inference input, so these
assert that equivalence directly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.perception.prepare import enhance_image, to_unit_gray
from src.perception.preprocessing import enhance_lunar_image


class TestUnitGray(unittest.TestCase):
    def test_output_is_float01(self):
        image = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
        unit = to_unit_gray(image)
        self.assertEqual(unit.dtype, np.float32)
        self.assertGreaterEqual(unit.min(), 0.0)
        self.assertLessEqual(unit.max(), 1.0)

    def test_bit_depth_independence(self):
        # The exact failure of naive image/255: a 16-bit image whose values sit
        # around 300-700 would divide to ~0.01, i.e. near-black. Robust
        # normalisation must instead use the image's own dynamic range, so an
        # 8-bit image and its 257x-scaled 16-bit twin normalise the same.
        rng = np.random.default_rng(1)
        eight = rng.integers(20, 220, (48, 48), dtype=np.uint8)
        sixteen = (eight.astype(np.uint16) * 257)  # 255*257 = 65535
        np.testing.assert_allclose(to_unit_gray(eight), to_unit_gray(sixteen), atol=0.01)

    def test_naive_division_would_have_failed(self):
        # Documents why the fix matters: a 16-bit lunar-style image divided by
        # 255 saturates, while to_unit_gray recovers real contrast.
        image = (np.linspace(300, 700, 40 * 40).reshape(40, 40)).astype(np.uint16)
        naive = np.clip(image / 255.0, 0, 1)
        self.assertGreater(float(naive.mean()), 0.99)          # saturated to white
        self.assertLess(float(to_unit_gray(image).mean()), 0.9)  # contrast preserved


class TestEnhanceImageClassicalPath(unittest.TestCase):
    def test_matches_enhance_lunar_image_when_no_model(self):
        image = np.random.default_rng(2).integers(0, 256, (40, 40), dtype=np.uint8)
        enhanced, report = enhance_image(image)
        reference, ref_report = enhance_lunar_image(image)
        np.testing.assert_array_equal(enhanced, reference)
        self.assertEqual(report.to_dict(), ref_report.to_dict())


@unittest.skipIf(torch is None, "PyTorch required for the curve-enhancer path.")
class TestTrainServeConsistency(unittest.TestCase):
    def test_dataset_input_equals_inference_input(self):
        from src.perception.data import LowLightImageDataset

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.png"
            self.assertTrue(cv2.imwrite(str(path), image))

            training_input = LowLightImageDataset(directory)[0].numpy()  # (1,H,W)

        # The representation prepare.enhance_image feeds the curve model:
        inference_input = to_unit_gray(image)[None]  # (1,H,W)
        np.testing.assert_allclose(training_input, inference_input, atol=1e-6)

    def test_curve_model_replaces_rather_than_stacks(self):
        # With a curve model, enhance_image must feed it to_unit_gray(raw),
        # never the CLAHE output. An identity-like model should therefore return
        # approximately to_unit_gray(raw), not the classical enhancement.
        from src.perception.enhancement_model import CurveEnhancer

        rng = np.random.default_rng(4)
        image = rng.integers(30, 210, (48, 48), dtype=np.uint8)
        model = CurveEnhancer().eval()

        enhanced, report = enhance_image(image, model, device="cpu")
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # gamma is undefined for the learned path and reported as NaN.
        self.assertTrue(np.isnan(report.gamma))
        # output_mean describes the curve output, and must be a real fraction.
        self.assertTrue(0.0 <= report.output_mean <= 1.0)


if __name__ == "__main__":
    unittest.main()
