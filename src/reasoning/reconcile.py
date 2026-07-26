"""Rule-based reconciliation of visual and physical hazard into a verdict.

Transparent by design: every signal, conflict, and evidence gap that moves the
verdict is recorded on the returned :class:`LandingDecision`, so the reasoning
can be audited rather than trusted.  The thresholds are engineering choices, not
values calibrated against real landing outcomes — no such ground truth is used
anywhere in this project — and are exposed on :class:`DecisionPolicy`.

The load-bearing rule is epistemic.  Absence of evidence is not evidence of
safety: an untrained detector or terrain hidden in shadow means the site cannot
be *certified*, so it is capped at CAUTION and can never be GO, however benign
the physics looks.  This mirrors Stage 1's ``not-run`` convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.perception.contracts import VisualEvidence

from .contracts import LandingDecision, Stage2Hazard, Verdict


@dataclass(frozen=True)
class DecisionPolicy:
    """Thresholds governing the verdict.  Engineering choices, not calibrated."""

    # Slope failure (fraction of cells shedding material at repose).
    slope_caution: float = 0.02
    slope_nogo: float = 0.10
    # Descent-engine vibration: a site stable now but that fails badly under a
    # lowered friction angle is a moonquake/landing-load conflict.
    vibration_ratio_caution: float = 3.0
    vibration_vibration_nogo: float = 0.15   # failure fraction under load
    # Feature density, per square kilometre.
    boulder_density_caution: float = 50.0
    boulder_density_nogo: float = 200.0
    crater_density_caution: float = 30.0
    crater_density_nogo: float = 120.0
    # Fraction of the scene in shadow above which it cannot be visually cleared.
    shadow_unverifiable: float = 0.35


def _rank(value: Verdict) -> int:
    return {"GO": 0, "CAUTION": 1, "NO-GO": 2}[value]


def _worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key=_rank)


def decide(
    visual: VisualEvidence,
    hazard: Stage2Hazard,
    *,
    site_area_km2: float,
    policy: DecisionPolicy | None = None,
) -> LandingDecision:
    """Weigh visual and physical hazard into a GO / CAUTION / NO-GO verdict."""
    policy = policy or DecisionPolicy()
    signals: dict[str, float] = {}
    conflicts: list[str] = []
    gaps: list[str] = []
    rationale: list[str] = []
    verdict: Verdict = "GO"
    score_terms: list[float] = []

    # --- Physical hazard: slope failure at the quiescent angle of repose ------
    slope = hazard.toppled_fraction_nominal
    signals["slope_failure_fraction"] = slope
    signals["max_slope_deg"] = hazard.max_slope_deg
    if slope >= policy.slope_nogo:
        verdict = _worst(verdict, "NO-GO")
        rationale.append(f"Severe slope failure: {slope:.1%} of terrain sheds material at repose.")
    elif slope >= policy.slope_caution:
        verdict = _worst(verdict, "CAUTION")
        rationale.append(f"Moderate slope failure: {slope:.1%} at repose.")
    score_terms.append(min(slope / policy.slope_nogo, 1.0))

    # --- Vibration sensitivity: the descent-load / moonquake conflict ---------
    vib = hazard.toppled_fraction_vibration
    signals["vibration_failure_fraction"] = vib
    signals["vibration_sensitivity"] = hazard.vibration_sensitivity
    if slope < policy.slope_nogo and hazard.vibration_sensitivity >= policy.vibration_ratio_caution \
            and vib >= policy.vibration_vibration_nogo:
        verdict = _worst(verdict, "NO-GO")
        conflicts.append(
            f"Stable at rest ({slope:.1%}) but {vib:.1%} fails under descent-engine "
            f"loading ({hazard.vibration_sensitivity:.1f}x): rim/slope collapse risk on landing.")
    elif hazard.vibration_sensitivity >= policy.vibration_ratio_caution and vib >= policy.slope_caution:
        verdict = _worst(verdict, "CAUTION")
        conflicts.append(
            f"Vibration-sensitive: failure grows {hazard.vibration_sensitivity:.1f}x under load "
            f"(to {vib:.1%}).")
    score_terms.append(min(vib / max(policy.vibration_vibration_nogo, 1e-6), 1.0))

    # --- Visual hazard: boulder and crater density ----------------------------
    area = max(site_area_km2, 1e-6)
    for name, count, cau, nogo in (
        ("boulder", sum(1 for b in visual.boulders if b.class_name == "boulder"),
         policy.boulder_density_caution, policy.boulder_density_nogo),
        ("crater", sum(1 for b in visual.boulders if b.class_name == "crater"),
         policy.crater_density_caution, policy.crater_density_nogo),
    ):
        density = count / area
        signals[f"{name}_density_per_km2"] = density
        if density >= nogo:
            verdict = _worst(verdict, "NO-GO")
            rationale.append(f"High {name} density: {density:.0f}/km^2.")
        elif density >= cau:
            verdict = _worst(verdict, "CAUTION")
            rationale.append(f"Elevated {name} density: {density:.0f}/km^2.")
        score_terms.append(min(density / nogo, 1.0))

    # --- Evidence gaps: absence of evidence is not evidence of safety ---------
    detector_versions = visual.model_versions or {}
    for model in ("boulder_detector", "debris_segmenter"):
        if detector_versions.get(model, "not-run") == "not-run":
            gaps.append(f"{model} did not run — site cannot be visually cleared.")
    shadow = visual.shadow_fraction
    if shadow is not None:
        signals["shadow_fraction"] = shadow
        if shadow >= policy.shadow_unverifiable:
            gaps.append(f"{shadow:.0%} of the scene is in shadow — hidden terrain not assessable.")

    if gaps:
        # Cannot certify GO without complete evidence; cap at CAUTION at best.
        verdict = _worst(verdict, "CAUTION")
        rationale.append("Verdict capped at CAUTION: evidence is incomplete (see gaps).")

    if not hazard.converged:
        gaps.append("Physics did not converge in budget — hazard may be underestimated.")
        verdict = _worst(verdict, "CAUTION")

    if verdict == "GO":
        rationale.append("Low slope failure, low vibration sensitivity, sparse boulders/craters, "
                         "and complete evidence — cleared for landing.")

    hazard_score = min(1.0, sum(score_terms) / len(score_terms)) if score_terms else 0.0
    return LandingDecision(
        site=hazard.site,
        verdict=verdict,
        hazard_score=round(hazard_score, 3),
        signals={k: round(v, 4) for k, v in signals.items()},
        conflicts=conflicts,
        evidence_gaps=gaps,
        rationale=rationale,
    )
