"""Behavioural tests for the corrected self-guided enhancement loss.

These pin the three calibration fixes so the loss cannot silently regress to the
version that produced grid and etch artifacts.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.perception.enhancement_model import CurveEnhancer, SelfGuidedEnhancementLoss


@unittest.skipIf(torch is None, "PyTorch is required for the enhancement loss tests.")
class TestExposureOnlyLifts(unittest.TestCase):
    def test_bright_image_is_not_penalised_for_being_bright(self):
        # A field already brighter than target must incur zero exposure loss --
        # the old symmetric |mean - target| dragged bright detail down to grey.
        loss = SelfGuidedEnhancementLoss(target_exposure=0.5)
        bright = torch.full((1, 1, 32, 32), 0.8)
        curves = torch.zeros((1, 8, 32, 32))
        self.assertAlmostEqual(float(loss(bright, bright, curves)["exposure"]), 0.0, places=6)

    def test_dark_image_incurs_exposure_loss(self):
        loss = SelfGuidedEnhancementLoss(target_exposure=0.5)
        dark = torch.full((1, 1, 32, 32), 0.1)
        curves = torch.zeros((1, 8, 32, 32))
        self.assertGreater(float(loss(dark, dark, curves)["exposure"]), 0.3)


@unittest.skipIf(torch is None, "PyTorch is required for the enhancement loss tests.")
class TestSpatialConsistency(unittest.TestCase):
    def test_uniform_scaling_preserves_structure(self):
        # A pure brightness offset keeps every region-to-neighbour difference,
        # so spatial-consistency loss must be ~0.
        rng = torch.Generator().manual_seed(0)
        source = torch.rand((1, 1, 32, 32), generator=rng)
        loss = SelfGuidedEnhancementLoss()
        shifted = (source + 0.1).clamp(0, 1)
        curves = torch.zeros((1, 8, 32, 32))
        self.assertLess(float(loss(source, shifted, curves)["spatial"]), 1e-3)

    def test_destroying_contrast_is_penalised(self):
        rng = torch.Generator().manual_seed(1)
        source = torch.rand((1, 1, 32, 32), generator=rng)
        flat = torch.full_like(source, 0.5)  # all structure removed
        loss = SelfGuidedEnhancementLoss()
        curves = torch.zeros((1, 8, 32, 32))
        self.assertGreater(float(loss(source, flat, curves)["spatial"]), 1e-3)


@unittest.skipIf(torch is None, "PyTorch is required for the enhancement loss tests.")
class TestSmoothnessAndWeights(unittest.TestCase):
    def test_jagged_curves_cost_more_than_smooth_ones(self):
        loss = SelfGuidedEnhancementLoss()
        image = torch.full((1, 1, 16, 16), 0.3)
        smooth = torch.zeros((1, 8, 16, 16))
        jagged = torch.zeros((1, 8, 16, 16))
        jagged[:, :, ::2, :] = 1.0  # alternating rows
        self.assertGreater(
            float(loss(image, image, jagged)["smoothness"]),
            float(loss(image, image, smooth)["smoothness"]),
        )

    def test_weights_scale_the_total(self):
        image = torch.full((1, 1, 32, 32), 0.2)
        curves = torch.zeros((1, 8, 32, 32))
        light = SelfGuidedEnhancementLoss(w_exposure=1.0)(image, image, curves)["total"]
        heavy = SelfGuidedEnhancementLoss(w_exposure=5.0)(image, image, curves)["total"]
        self.assertGreater(float(heavy), float(light))

    def test_configurable_curve_steps_round_trip(self):
        model = CurveEnhancer(curve_steps=4)
        out, curves = model(torch.rand((1, 1, 16, 16)))
        self.assertEqual(curves.shape[1], 4)
        self.assertEqual(out.shape, (1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
