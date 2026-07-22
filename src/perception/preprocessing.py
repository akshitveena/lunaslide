"""Illumination-aware preprocessing for lunar orbital imagery.

This is an IllumiCurveNet-inspired *classical* pipeline: robust intensity
normalisation, image-adaptive gamma correction, and CLAHE.  It increases
contrast where photons are present; it cannot reconstruct terrain with no
recorded signal in permanently shadowed regions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    target_luminance: float = 0.55
    min_gamma: float = 0.45
    max_gamma: float = 1.45
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class EnhancementReport:
    input_p01: float
    input_p99: float
    input_mean: float
    gamma: float
    shadow_fraction: float
    output_mean: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _to_uint8(image: np.ndarray, config: EnhancementConfig) -> tuple[np.ndarray, float, float]:
    """Robustly map grayscale imagery of any numeric depth to uint8."""
    if image.ndim != 2:
        raise ValueError("Stage 1 preprocessing expects one-channel lunar imagery.")
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")

    low, high = np.percentile(finite, [config.lower_percentile, config.upper_percentile])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), float(low), float(high)
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), float(low), float(high)


def _adaptive_gamma(normalized: np.ndarray, config: EnhancementConfig) -> float:
    mean_luminance = float(np.mean(normalized))
    # gamma solves mean ** gamma = target.  Epsilon avoids log(0), and the
    # bounds prevent extreme amplification of sensor noise in dark images.
    gamma = np.log(config.target_luminance) / np.log(max(mean_luminance, 1e-6))
    return float(np.clip(gamma, config.min_gamma, config.max_gamma))


def enhance_lunar_image(
    image: np.ndarray, config: EnhancementConfig | None = None
) -> tuple[np.ndarray, EnhancementReport]:
    """Apply adaptive gamma correction followed by CLAHE.

    Returns an 8-bit enhanced grayscale image and an auditable report.  The
    original image is never modified.
    """
    config = config or EnhancementConfig()
    if not (0 < config.lower_percentile < config.upper_percentile < 100):
        raise ValueError("Percentiles must satisfy 0 < lower < upper < 100.")
    if not (0 < config.target_luminance < 1):
        raise ValueError("target_luminance must be between zero and one.")

    normalized_u8, low, high = _to_uint8(image, config)
    normalized = normalized_u8.astype(np.float32) / 255.0
    gamma = _adaptive_gamma(normalized, config)
    gamma_corrected = np.rint(np.power(normalized, gamma) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_grid_size,
    )
    enhanced = clahe.apply(gamma_corrected)
    report = EnhancementReport(
        input_p01=low,
        input_p99=high,
        input_mean=float(np.mean(normalized)),
        gamma=gamma,
        shadow_fraction=float(np.mean(normalized < 0.12)),
        output_mean=float(np.mean(enhanced) / 255.0),
    )
    return enhanced, report
