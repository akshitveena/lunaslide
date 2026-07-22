import unittest

import numpy as np

from src.perception.preprocessing import EnhancementConfig, enhance_lunar_image
from src.perception.augmentations import augment_lunar_illumination


class TestLunarEnhancement(unittest.TestCase):
    def test_enhances_dark_image_without_mutating_input(self):
        image = np.full((64, 64), 300, dtype=np.uint16)
        image[16:48, 16:48] = 700
        original = image.copy()

        enhanced, report = enhance_lunar_image(image)

        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertEqual(enhanced.shape, image.shape)
        self.assertTrue(np.array_equal(image, original))
        self.assertLess(report.gamma, 1.0)
        self.assertGreater(float(enhanced.std()), 0.0)

    def test_rejects_colour_array_at_core_boundary(self):
        with self.assertRaises(ValueError):
            enhance_lunar_image(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_rejects_invalid_percentiles(self):
        with self.assertRaises(ValueError):
            enhance_lunar_image(
                np.zeros((8, 8), dtype=np.uint8),
                EnhancementConfig(lower_percentile=99, upper_percentile=1),
            )

    def test_training_augmentation_is_seeded_and_shape_preserving(self):
        image = np.full((48, 48), 120, dtype=np.uint8)
        first = augment_lunar_illumination(image, np.random.default_rng(8))
        second = augment_lunar_illumination(image, np.random.default_rng(8))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
