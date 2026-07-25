"""Single source of truth for turning a raw lunar image into model input.

Stage 1 had train/serve skew: the curve enhancer was *trained* on ``image/255``
(a naive normalisation that is wrong for any image that is not already 8-bit)
but *run* on the CLAHE-enhanced output of ``enhance_lunar_image`` — a different
representation entirely.  A model is only as good as the match between what it
sees in training and what it sees in production, and these did not match.

Everything now routes through here, so training and inference cannot drift:

* :func:`to_unit_gray` is the one normalisation — robust to bit depth, giving a
  float32 image in ``[0, 1]``.  It is the curve enhancer's input in both
  training and inference.
* :func:`enhance_image` is the one enhancement step.  The learned curve enhancer
  and classical gamma+CLAHE are *alternatives*, not a pipeline: the curve
  enhancer replaces the classical step rather than stacking on top of it, which
  is what its Zero-DCE-style self-supervised training assumes.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import EnhancementConfig, EnhancementReport, _to_uint8, enhance_lunar_image


def to_unit_gray(image: np.ndarray, config: EnhancementConfig | None = None) -> np.ndarray:
    """Robustly normalise grayscale imagery of any bit depth to float32 ``[0, 1]``.

    This is the curve enhancer's canonical input.  ``image/255`` only happens to
    work for 8-bit input; a 16-bit DEM-derived or LROC product divided by 255
    saturates to white.  Percentile normalisation handles every depth the same
    way, which is exactly why training and inference must both use it.
    """
    config = config or EnhancementConfig()
    normalized_u8, _, _ = _to_uint8(image, config)
    return (normalized_u8.astype(np.float32) / 255.0)


def _report_for(raw_gray: np.ndarray, enhanced: np.ndarray, config: EnhancementConfig) -> EnhancementReport:
    """Build an evidence report whose stats describe *this* enhanced output.

    Input statistics come from the raw image (they describe the scene, not the
    enhancer), while ``output_mean`` reflects whichever enhancer actually ran.
    """
    normalized_u8, low, high = _to_uint8(raw_gray, config)
    normalized = normalized_u8.astype(np.float32) / 255.0
    return EnhancementReport(
        input_p01=low,
        input_p99=high,
        input_mean=float(np.mean(normalized)),
        gamma=float("nan"),  # not defined for the learned enhancer
        shadow_fraction=float(np.mean(normalized < 0.12)),
        output_mean=float(np.mean(enhanced) / 255.0),
    )


def enhance_image(
    raw_gray: np.ndarray,
    curve_model=None,
    *,
    device: str = "cpu",
    config: EnhancementConfig | None = None,
) -> tuple[np.ndarray, EnhancementReport]:
    """Produce the enhanced 8-bit image and its report, from one raw grayscale.

    With no ``curve_model`` this is exactly classical gamma+CLAHE.  With a curve
    model it runs the learned enhancer on :func:`to_unit_gray` of the *raw*
    image — the same representation the model was trained on — instead of on the
    classical output.  The two are alternatives; they are never chained.
    """
    config = config or EnhancementConfig()
    if curve_model is None:
        return enhance_lunar_image(raw_gray, config)

    import torch

    unit = to_unit_gray(raw_gray, config)
    tensor = torch.from_numpy(unit).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = curve_model(tensor)[0]
    enhanced = np.rint(output[0, 0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return enhanced, _report_for(raw_gray, enhanced, config)
