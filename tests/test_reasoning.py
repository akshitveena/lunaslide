"""Decision-logic tests for the Stage 3 reconciler.

These pin the behaviour that matters: the verdict follows the worst signal, the
descent-vibration conflict is caught, and — most importantly — missing evidence
can never be cleared GO.
"""

from __future__ import annotations

import unittest

from src.perception.contracts import BoulderDetection, GeoReference, VisualEvidence
from src.reasoning.contracts import Stage2Hazard
from src.reasoning.reconcile import DecisionPolicy, decide

COMPLETE_VERSIONS = {"boulder_detector": "runs/best.pt", "debris_segmenter": "unet.pt"}


def visual(boulders=0, craters=0, shadow=0.05, versions=None) -> VisualEvidence:
    dets = ([BoulderDetection((0, 0, 1, 1), 0.9, "boulder")] * boulders
            + [BoulderDetection((0, 0, 1, 1), 0.9, "crater")] * craters)
    return VisualEvidence(
        georef=GeoReference(image_id="t"),
        boulders=dets,
        shadow_fraction=shadow,
        model_versions=versions if versions is not None else dict(COMPLETE_VERSIONS),
    )


def hazard(nominal=0.0, vibration=0.0, max_slope=10.0, converged=True) -> Stage2Hazard:
    return Stage2Hazard(
        site="t", toppled_fraction_nominal=nominal, toppled_fraction_vibration=vibration,
        max_slope_deg=max_slope, grid_spacing_m=59.0, converged=converged,
    )


class TestClearSite(unittest.TestCase):
    def test_benign_site_with_complete_evidence_is_go(self):
        d = decide(visual(boulders=1, craters=1), hazard(nominal=0.0, vibration=0.005),
                   site_area_km2=1.0)
        self.assertEqual(d.verdict, "GO")
        self.assertFalse(d.evidence_gaps)


class TestPhysicalHazard(unittest.TestCase):
    def test_severe_slope_failure_is_nogo(self):
        d = decide(visual(), hazard(nominal=0.20, vibration=0.25), site_area_km2=1.0)
        self.assertEqual(d.verdict, "NO-GO")

    def test_moderate_slope_is_caution(self):
        d = decide(visual(), hazard(nominal=0.05, vibration=0.06), site_area_km2=1.0)
        self.assertEqual(d.verdict, "CAUTION")


class TestVibrationConflict(unittest.TestCase):
    def test_stable_but_vibration_sensitive_is_flagged_nogo(self):
        # 1% at rest, 20% under load, 20x sensitivity: the moonquake conflict.
        d = decide(visual(), hazard(nominal=0.01, vibration=0.20), site_area_km2=1.0)
        self.assertEqual(d.verdict, "NO-GO")
        self.assertTrue(any("descent-engine" in c for c in d.conflicts))

    def test_mild_vibration_sensitivity_is_caution(self):
        d = decide(visual(), hazard(nominal=0.005, vibration=0.03), site_area_km2=1.0)
        self.assertEqual(d.verdict, "CAUTION")
        self.assertTrue(d.conflicts)


class TestVisualHazard(unittest.TestCase):
    def test_dense_boulders_is_nogo(self):
        d = decide(visual(boulders=300), hazard(), site_area_km2=1.0)
        self.assertEqual(d.verdict, "NO-GO")

    def test_dense_craters_is_nogo(self):
        d = decide(visual(craters=200), hazard(), site_area_km2=1.0)
        self.assertEqual(d.verdict, "NO-GO")

    def test_density_scales_with_area(self):
        # 100 boulders over 4 km^2 = 25/km^2, below caution.
        d = decide(visual(boulders=100), hazard(), site_area_km2=4.0)
        self.assertEqual(d.verdict, "GO")


class TestEvidenceGaps(unittest.TestCase):
    def test_untrained_detector_can_never_be_go(self):
        # Physically pristine, but the boulder detector never ran.
        d = decide(visual(versions={"boulder_detector": "not-run"}),
                   hazard(nominal=0.0, vibration=0.0), site_area_km2=1.0)
        self.assertEqual(d.verdict, "CAUTION")
        self.assertTrue(any("boulder_detector" in g for g in d.evidence_gaps))

    def test_heavy_shadow_can_never_be_go(self):
        d = decide(visual(shadow=0.6), hazard(nominal=0.0, vibration=0.0), site_area_km2=1.0)
        self.assertEqual(d.verdict, "CAUTION")
        self.assertTrue(any("shadow" in g for g in d.evidence_gaps))

    def test_non_convergence_caps_at_caution(self):
        d = decide(visual(), hazard(nominal=0.0, vibration=0.0, converged=False),
                   site_area_km2=1.0)
        self.assertEqual(d.verdict, "CAUTION")

    def test_gap_never_masks_a_real_nogo(self):
        # A severe hazard with also-missing evidence is still NO-GO, not CAUTION.
        d = decide(visual(boulders=300, versions={"boulder_detector": "not-run"}),
                   hazard(nominal=0.2, vibration=0.25), site_area_km2=1.0)
        self.assertEqual(d.verdict, "NO-GO")


class TestCalibratedThresholds(unittest.TestCase):
    """Pin the data-derived thresholds so a silent revert to guesses fails.

    Values come from scripts/calibrate_stage3.py: p90/p99 of the hazard
    distribution over 480 real DEM patches, confirmed against observed Bickel
    rockfall sites (6.4x / 8.0x enrichment over random terrain).
    """

    def test_slope_thresholds_are_the_calibrated_values(self):
        policy = DecisionPolicy()
        self.assertAlmostEqual(policy.slope_caution, 0.0090, places=4)
        self.assertAlmostEqual(policy.slope_nogo, 0.0658, places=4)

    def test_thresholds_are_ordered_and_plausible(self):
        policy = DecisionPolicy()
        self.assertLess(policy.slope_caution, policy.slope_nogo)
        self.assertLess(policy.slope_nogo, policy.vibration_vibration_nogo)
        for value in (policy.slope_caution, policy.slope_nogo,
                      policy.vibration_vibration_nogo, policy.shadow_unverifiable):
            self.assertTrue(0.0 < value < 1.0)

    def test_terrain_at_the_population_median_is_cleared(self):
        # Median lunar terrain (~1.8e-4 failure) must not trip CAUTION on slope.
        visual = VisualEvidence(georef=GeoReference(image_id="t"), shadow_fraction=0.05,
                                model_versions=dict(COMPLETE_VERSIONS))
        median = decide(visual, hazard(nominal=0.00018, vibration=0.0007),
                        site_area_km2=1.0)
        self.assertEqual(median.verdict, "GO")


class TestDetectorReliabilityGate(unittest.TestCase):
    """A low-recall detector may flag hazard but may not clear a site."""

    def _clean_visual(self):
        return VisualEvidence(georef=GeoReference(image_id="t"), shadow_fraction=0.05,
                              model_versions=dict(COMPLETE_VERSIONS))

    def test_reliable_detector_can_clear(self):
        d = decide(self._clean_visual(), hazard(nominal=0.0, vibration=0.0),
                   site_area_km2=1.0, detector_recall=0.8)
        self.assertEqual(d.verdict, "GO")

    def test_low_recall_detector_cannot_clear(self):
        # Same clean scene, but the detector recalls too little to trust absence.
        d = decide(self._clean_visual(), hazard(nominal=0.0, vibration=0.0),
                   site_area_km2=1.0, detector_recall=0.42)
        self.assertEqual(d.verdict, "CAUTION")
        self.assertTrue(any("recall" in g for g in d.evidence_gaps))

    def test_low_recall_detections_still_flag_hazard(self):
        # A weak detector that DOES find many boulders is still believed for NO-GO.
        d = decide(visual(boulders=300), hazard(nominal=0.0, vibration=0.0),
                   site_area_km2=1.0, detector_recall=0.42)
        self.assertEqual(d.verdict, "NO-GO")

    def test_none_recall_is_backward_compatible(self):
        d = decide(self._clean_visual(), hazard(nominal=0.0, vibration=0.0),
                   site_area_km2=1.0)  # no recall passed
        self.assertEqual(d.verdict, "GO")


class TestScoreAndSerialisation(unittest.TestCase):
    def test_hazard_score_in_unit_range(self):
        d = decide(visual(boulders=500), hazard(nominal=0.5, vibration=0.6), site_area_km2=1.0)
        self.assertGreaterEqual(d.hazard_score, 0.0)
        self.assertLessEqual(d.hazard_score, 1.0)

    def test_decision_is_json_serialisable(self):
        import json
        d = decide(visual(), hazard(), site_area_km2=1.0)
        json.dumps(d.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
