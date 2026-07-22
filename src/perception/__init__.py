"""Stage 1 visual-perception components for Lunaslide.

Classical preprocessing intentionally remains importable without the optional
deep-learning stack, which makes diagnostics usable in lightweight setups.
"""

from .contracts import GeoReference, VisualEvidence
from .preprocessing import EnhancementConfig, EnhancementReport, enhance_lunar_image

__all__ = [
    "EnhancementConfig", "EnhancementReport", "GeoReference", "VisualEvidence",
    "ResNet50UNet", "YOLOv8BoulderDetector", "CurveEnhancer",
    "SelfGuidedEnhancementLoss", "enhance_lunar_image",
]


def __getattr__(name: str):
    """Lazily load torch-dependent models only when the caller requests one."""
    if name in {"ResNet50UNet", "YOLOv8BoulderDetector"}:
        from .models import ResNet50UNet, YOLOv8BoulderDetector
        return {"ResNet50UNet": ResNet50UNet, "YOLOv8BoulderDetector": YOLOv8BoulderDetector}[name]
    if name in {"CurveEnhancer", "SelfGuidedEnhancementLoss"}:
        from .enhancement_model import CurveEnhancer, SelfGuidedEnhancementLoss
        return {"CurveEnhancer": CurveEnhancer, "SelfGuidedEnhancementLoss": SelfGuidedEnhancementLoss}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
