"""Offline tests for the Stage 3 agent's safety guardrail and parsing.

The LLM call itself needs a running Ollama and is exercised by
scripts.run_stage3_agent; here we pin the parts that must be correct regardless
of what the model says: the enforcement floor and verdict parsing.
"""

from __future__ import annotations

import unittest

from src.perception.contracts import GeoReference, VisualEvidence
from src.reasoning.agent import _enforce, _parse_verdict
from src.reasoning.contracts import Stage2Hazard
from src.reasoning.reconcile import decide


def _rule(verdict_driver):
    """A LandingDecision produced by the real reconciler for a given scenario."""
    return verdict_driver


class TestEnforcementFloor(unittest.TestCase):
    """The agent may tighten the rule verdict, never loosen it."""

    def _decision(self, versions, nominal, vibration):
        visual = VisualEvidence(georef=GeoReference(image_id="t"), shadow_fraction=0.05,
                                model_versions=versions)
        hazard = Stage2Hazard("t", nominal, vibration, 20.0, 59.0, True)
        return decide(visual, hazard, site_area_km2=1.0)

    def test_agent_cannot_upgrade_a_capped_caution_to_go(self):
        # Rule engine caps at CAUTION (detector not-run); agent says GO -> enforced CAUTION.
        rule = self._decision({"boulder_detector": "not-run"}, 0.0, 0.0)
        self.assertEqual(rule.verdict, "CAUTION")
        self.assertEqual(_enforce("GO", rule), "CAUTION")

    def test_agent_cannot_soften_a_nogo(self):
        rule = self._decision({"boulder_detector": "x"}, 0.2, 0.25)
        self.assertEqual(rule.verdict, "NO-GO")
        self.assertEqual(_enforce("GO", rule), "NO-GO")
        self.assertEqual(_enforce("CAUTION", rule), "NO-GO")

    def test_agent_may_be_more_conservative(self):
        rule = self._decision({"boulder_detector": "x", "debris_segmenter": "x"}, 0.0, 0.0)
        self.assertEqual(rule.verdict, "GO")
        self.assertEqual(_enforce("NO-GO", rule), "NO-GO")
        self.assertEqual(_enforce("CAUTION", rule), "CAUTION")

    def test_unparseable_agent_verdict_defers_to_rules(self):
        rule = self._decision({"boulder_detector": "x", "debris_segmenter": "x"}, 0.0, 0.0)
        self.assertEqual(_enforce("banana", rule), rule.verdict)


class TestVerdictParsing(unittest.TestCase):
    def test_parses_final_verdict_line(self):
        self.assertEqual(_parse_verdict("reasoning...\nVERDICT: NO-GO"), "NO-GO")
        self.assertEqual(_parse_verdict("VERDICT: go"), "GO")
        self.assertEqual(_parse_verdict("blah\nVERDICT: Caution."), "CAUTION")

    def test_uses_the_last_verdict_line(self):
        self.assertEqual(_parse_verdict("VERDICT: GO\nactually\nVERDICT: NO-GO"), "NO-GO")

    def test_missing_verdict_is_conservative(self):
        self.assertEqual(_parse_verdict("no verdict here"), "CAUTION")


if __name__ == "__main__":
    unittest.main()
