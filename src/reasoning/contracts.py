"""Contracts exchanged with the Stage 3 reconciler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Verdict = Literal["GO", "CAUTION", "NO-GO"]


@dataclass(frozen=True)
class Stage2Hazard:
    """Physics summary for a site, distilled from the mass-wasting simulation.

    ``toppled_fraction_nominal`` is failure at the quiescent angle of repose;
    ``toppled_fraction_vibration`` is failure once a descent engine lowers the
    effective friction angle.  Their ratio is the vibration sensitivity — a site
    stable at rest but sensitive under load is the case Stage 3 exists to catch.
    """

    site: str
    toppled_fraction_nominal: float
    toppled_fraction_vibration: float
    max_slope_deg: float
    grid_spacing_m: float
    converged: bool
    nominal_repose_deg: float = 30.0
    vibration_repose_deg: float = 24.0

    @property
    def vibration_sensitivity(self) -> float:
        base = max(self.toppled_fraction_nominal, 1e-6)
        return self.toppled_fraction_vibration / base

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vibration_sensitivity"] = self.vibration_sensitivity
        return data


@dataclass
class LandingDecision:
    """The Stage 3 verdict for a site, with every reason made explicit."""

    site: str
    verdict: Verdict
    hazard_score: float                       # 0 (safe) .. 1 (severe)
    signals: dict[str, float] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
