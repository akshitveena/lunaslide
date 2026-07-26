"""Stage 3 with the local-LLM agent: the model probes the physics, then decides.

For each site the agent (Ollama, qwen2.5:14b by default) reads the evidence,
calls the real slope-failure simulator at friction angles it chooses, and writes
a narrative GO / CAUTION / NO-GO verdict.  The deterministic reconciler is
computed alongside and enforced as the floor, so the agent can be more cautious
than the rules but never less.

    python3 -m scripts.run_stage3_agent
    python3 -m scripts.run_stage3_agent --model deepseek-r1:14b

Everything is local (localhost Ollama). Writes data/stage3/agent_decisions.json.
Falls back to the rule verdict per site if Ollama is unreachable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.perception.contracts import GeoReference, VisualEvidence
from src.reasoning.agent import DEFAULT_MODEL, run_agent

OUTPUT = Path("data/stage3/agent_decisions.json")
SITES = (("Apollo 15", 26.13, 3.63), ("Shackleton", -89.5, 0.0), ("Faustini (PSR)", -87.1, 77.0))


def _visual(site: str) -> VisualEvidence:
    versions = {"enhancer": "classical-gamma-clahe",
                "boulder_detector": "not-run", "debris_segmenter": "not-run"}
    shadow = None
    apollo = Path("data/stage1/apollo15/apollo15.png")
    if site.startswith("Apollo") and apollo.exists():
        import cv2
        from src.perception.prepare import to_unit_gray
        shadow = float(np.mean(to_unit_gray(cv2.imread(str(apollo), cv2.IMREAD_GRAYSCALE)) < 0.12))
    return VisualEvidence(georef=GeoReference(image_id=site, source="LROC"),
                          shadow_fraction=shadow, model_versions=versions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size-px", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    from src.physics.dem_loader import fetch_patch

    results = []
    for name, lat, lon in SITES:
        print(f"\n=== {name} ===", flush=True)
        patch = fetch_patch(lat, lon, size_px=args.size_px, verbose=False)
        if patch is None:
            print("  DEM unavailable; skipped")
            continue
        km = patch.elevation.shape[1] * patch.grid_spacing_x_m / 1000.0
        area = (patch.elevation.shape[0] * patch.grid_spacing_y_m / 1000.0) * km
        result = run_agent(_visual(name), patch.elevation, patch.grid_spacing, name,
                           site_area_km2=area, model=args.model, timeout=args.timeout)
        results.append(result.to_dict())
        probed = ", ".join(f"{c['friction_angle_deg']:.0f}deg->{c['failed_fraction']:.1%}"
                           for c in result.tool_calls) or "none"
        print(f"  agent probed: {probed}")
        print(f"  agent: {result.agent_verdict}  enforced: {result.enforced_verdict}  "
              f"rule: {result.rule_decision['verdict']}  (LLM: {result.used_llm})")

    if not results:
        print("No sites produced a verdict.")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "note": "Local-LLM agent chose the simulations; rule engine enforced as the floor. "
                "Thresholds uncalibrated. Fully local via Ollama.",
        "sites": results,
    }, indent=2) + "\n")
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
