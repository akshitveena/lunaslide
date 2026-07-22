"""Training-time illumination augmentation for Stage 1 lunar imagery."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IlluminationAugmentationConfig:
    gamma_range: tuple[float, float] = (0.55, 1.55)
    gain_range: tuple[float, float] = (0.65, 1.25)
    shadow_probability: float = 0.5


def augment_lunar_illumination(
    image: np.ndarray,
    rng: np.random.Generator,
    config: IlluminationAugmentationConfig | None = None,
) -> np.ndarray:
    """Apply realistic brightness/gamma and soft-shadow variation to uint8 imagery.

    This is only for model training. Evaluation and operational inference use
    the deterministic preprocessing pipeline instead.
    """
    config = config or IlluminationAugmentationConfig()
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Illumination augmentation expects a uint8 grayscale image.")
    gamma = rng.uniform(*config.gamma_range)
    gain = rng.uniform(*config.gain_range)
    augmented = np.clip((image.astype(np.float32) / 255.0) ** gamma * gain, 0, 1)
    if rng.random() < config.shadow_probability:
        height, width = image.shape
        center = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        axes = (max(1, int(width * rng.uniform(0.25, 0.7))), max(1, int(height * rng.uniform(0.2, 0.6))))
        mask = np.ones_like(augmented, dtype=np.float32)
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 0.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, min(height, width) / 12))
        shadow_strength = rng.uniform(0.25, 0.7)
        augmented *= 1.0 - shadow_strength * (1.0 - mask)
    return np.rint(augmented * 255.0).astype(np.uint8)
