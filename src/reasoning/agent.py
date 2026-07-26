"""Local-LLM agentic layer over the Stage 3 reconciler.

A small local model (via Ollama, e.g. qwen2.5:14b) reads a site's evidence and
*decides for itself* which physics simulations to run — probing the failure
margin at friction angles it chooses — before writing a narrative landing
verdict.  This is the "agentic" layer: the model, not a fixed script, chooses
the experiments.

Two things keep it honest:

* **The rule engine stays authoritative.**  The deterministic reconciler
  (:func:`reconcile.decide`) is computed independently, and its epistemic cap is
  *enforced in code* — the agent may be more conservative than the rules but can
  never certify GO when evidence is missing or a real NO-GO stands.  The LLM
  narrates and explores; it does not get to overrule safety.
* **The tool runs the real simulator.**  ``simulate_at_friction`` executes the
  actual cellular automaton on the site's DEM, so the numbers the model reasons
  over are real, not hallucinated.

Everything is local: no API key, no network beyond localhost Ollama.  If Ollama
is unreachable the module degrades to the rule-based verdict with a note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from src.perception.contracts import VisualEvidence

from .contracts import LandingDecision, Stage2Hazard
from .hazard import summarise_hazard
from .reconcile import DecisionPolicy, decide

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:14b"

_SIMULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "simulate_at_friction",
        "description": (
            "Run the mass-wasting cellular automaton on this site's real elevation "
            "at a given effective angle of repose (friction angle) and return the "
            "fraction of terrain that fails. Lower angles model stronger "
            "descent-engine vibration. Call this to probe the site's margin."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "friction_angle_deg": {
                    "type": "number",
                    "description": "effective angle of repose, 15-35 degrees",
                }
            },
            "required": ["friction_angle_deg"],
        },
    },
}

_SYSTEM = """You are a lunar landing-safety analyst. Decide GO, CAUTION, or NO-GO for a site.

You are given visual evidence and a physics hazard summary. You have a tool,
simulate_at_friction, that runs the real slope-failure simulation at any friction
angle you choose (15-35 deg). A descent engine lowers the effective friction
angle, so probe lower angles (e.g. 21-24 deg) to test the landing-load margin.

Rules you must respect:
- A site that is stable at rest but fails badly under a lowered friction angle is
  a descent-load hazard -> at most CAUTION, often NO-GO.
- If the boulder or debris detector did not run, or much of the scene is in
  shadow, the site CANNOT be certified GO (evidence is incomplete) -> CAUTION.
- High slope failure or dense boulders/craters -> NO-GO.

Call the tool 1-3 times to gather evidence, then give your verdict. End with a
final line exactly: VERDICT: <GO|CAUTION|NO-GO>"""


@dataclass
class AgentResult:
    """The agent's narrative verdict alongside the authoritative rule verdict."""

    site: str
    agent_verdict: str
    enforced_verdict: str          # after the safety guardrail
    narrative: str
    tool_calls: list[dict] = field(default_factory=list)
    rule_decision: dict = field(default_factory=dict)
    used_llm: bool = True

    def to_dict(self) -> dict:
        return {
            "site": self.site,
            "agent_verdict": self.agent_verdict,
            "enforced_verdict": self.enforced_verdict,
            "narrative": self.narrative,
            "tool_calls": self.tool_calls,
            "rule_decision": self.rule_decision,
            "used_llm": self.used_llm,
        }


def _ollama_chat(model: str, messages: list[dict], tools: list[dict] | None, timeout: int) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    request = Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["message"]


def _enforce(agent_verdict: str, rule: LandingDecision) -> str:
    """Safety guardrail: the agent may tighten the rule verdict, never loosen it.

    The rule engine's verdict is the floor.  If the rules say CAUTION (e.g. an
    evidence gap) the agent cannot upgrade to GO; if the rules say NO-GO the agent
    cannot soften it.  The agent *can* be more conservative.
    """
    rank = {"GO": 0, "CAUTION": 1, "NO-GO": 2}
    agent = agent_verdict if agent_verdict in rank else rule.verdict
    return max(agent, rule.verdict, key=lambda v: rank[v])


def _parse_verdict(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "VERDICT:" in line.upper():
            token = line.split(":", 1)[1].strip().upper().strip(".")
            for v in ("NO-GO", "CAUTION", "GO"):
                if v in token:
                    return v
    return "CAUTION"  # unparseable -> conservative


def run_agent(
    visual: VisualEvidence,
    elevation: np.ndarray,
    grid_spacing: float | tuple[float, float],
    site: str,
    *,
    site_area_km2: float,
    hazard: Stage2Hazard | None = None,
    model: str = DEFAULT_MODEL,
    policy: DecisionPolicy | None = None,
    max_tool_calls: int = 3,
    timeout: int = 180,
) -> AgentResult:
    """Run the agentic verdict, with the rule engine as the enforced floor."""
    hazard = hazard or summarise_hazard(elevation, grid_spacing, site)
    rule = decide(visual, hazard, site_area_km2=site_area_km2, policy=policy)

    versions = visual.model_versions or {}
    evidence = {
        "site": site,
        "slope_failure_fraction_at_repose": round(hazard.toppled_fraction_nominal, 4),
        "max_slope_deg": round(hazard.max_slope_deg, 1),
        "boulders_detected": sum(1 for b in visual.boulders if b.class_name == "boulder"),
        "craters_detected": sum(1 for b in visual.boulders if b.class_name == "crater"),
        "shadow_fraction": visual.shadow_fraction,
        "boulder_detector_ran": versions.get("boulder_detector", "not-run") != "not-run",
        "debris_segmenter_ran": versions.get("debris_segmenter", "not-run") != "not-run",
        "site_area_km2": round(site_area_km2, 2),
    }
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "Site evidence:\n" + json.dumps(evidence, indent=2)},
    ]

    tool_calls: list[dict] = []
    try:
        for _ in range(max_tool_calls + 1):
            message = _ollama_chat(model, messages, [_SIMULATE_TOOL], timeout)
            calls = message.get("tool_calls") or []
            if not calls:
                messages.append(message)
                break
            messages.append(message)
            for call in calls[:max_tool_calls]:
                angle = float(call["function"]["arguments"].get("friction_angle_deg", 30))
                angle = max(15.0, min(35.0, angle))
                crit = float(np.tan(np.radians(angle)))
                from src.physics.relaxation import simulate_mass_wasting
                _, toppled, _ = simulate_mass_wasting(
                    np.asarray(elevation, float), grid_spacing=grid_spacing, crit=crit, max_iter=2000)
                fraction = float(toppled.mean())
                tool_calls.append({"friction_angle_deg": angle, "failed_fraction": round(fraction, 4)})
                messages.append({"role": "tool", "content": json.dumps(
                    {"friction_angle_deg": angle, "failed_fraction": round(fraction, 4)})})
        else:
            message = _ollama_chat(model, messages, None, timeout)  # force a final answer
        narrative = message.get("content", "")
        agent_verdict = _parse_verdict(narrative)
        used_llm = True
    except (URLError, TimeoutError, OSError, KeyError) as error:
        narrative = (f"[Ollama unavailable: {error}. Falling back to the rule-based verdict.]")
        agent_verdict = rule.verdict
        used_llm = False

    return AgentResult(
        site=site,
        agent_verdict=agent_verdict,
        enforced_verdict=_enforce(agent_verdict, rule),
        narrative=narrative.strip(),
        tool_calls=tool_calls,
        rule_decision=rule.to_dict(),
        used_llm=used_llm,
    )
