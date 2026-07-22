import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.perception.contracts import GeoReference
    from src.perception.enhancement_model import CurveEnhancer, SelfGuidedEnhancementLoss
    from src.perception.pipeline import run_stage1


@unittest.skipIf(torch is None, "PyTorch is required for neural Stage 1 tests; install it in this environment.")
class TestStage1Pipeline(unittest.TestCase):
    def test_curve_model_and_self_guided_loss_are_finite(self):
        model = CurveEnhancer()
        source = torch.full((2, 1, 32, 32), 0.12)
        enhanced, curves = model(source)
        losses = SelfGuidedEnhancementLoss()(source, enhanced, curves)
        self.assertEqual(enhanced.shape, source.shape)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_classical_pipeline_exports_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.tile(np.linspace(10, 130, 64, dtype=np.uint8), (64, 1))
            source = root / "input.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            result = run_stage1(source, root / "result", GeoReference(image_id="smoke-test"))
            evidence = json.loads((root / "result" / "visual_evidence.json").read_text())
            self.assertEqual(result.georef.image_id, "smoke-test")
            self.assertEqual(evidence["model_versions"]["boulder_detector"], "not-run")
            self.assertTrue((root / "result" / "enhanced.png").exists())


if __name__ == "__main__":
    unittest.main()
