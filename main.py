"""Lunaslide Stage 2 — build a hazard dataset from real lunar topography,
train the surrogate, and regenerate the figure set.

This is the entry point. It runs, in order:

    1. scripts.build_hazard_dataset  — stream real LOLA+Kaguya patches, run the
       cellular automaton on each, write data/stage2/hazard_dataset.csv
    2. scripts.train_hazard_model    — train the surrogate and evaluate it on a
       held-out hemisphere against explicit baselines
    3. scripts.generate_figures      — per-site panels and the reality check
    4. scripts.generate_ca_visuals   — CA evolution, vibration sensitivity, 3D

Each stage is also runnable on its own; see docs/THESIS.md for what each
produces and why.

    python3 main.py                  # everything
    python3 main.py --skip-figures   # dataset + model only
    python3 main.py --bands 12       # smaller dataset, faster

An earlier version of this file built its dataset from synthetic terrain chosen
by the hazard label it was trying to predict, because the elevation stream
silently returned empty windows on every request. That path is gone: patches
that cannot be streamed are dropped and counted, never substituted.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time


def run_module(module: str, argv: list[str]) -> bool:
    """Run a script module in-process, reporting rather than raising on failure."""
    print(f"\n{'=' * 78}\n  {module}\n{'=' * 78}")
    saved = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = [module, *argv]
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exit_code:
        if exit_code.code not in (0, None):
            print(f"\n{module} exited with code {exit_code.code}")
            return False
    except Exception as error:
        print(f"\n{module} failed: {type(error).__name__}: {error}")
        return False
    finally:
        sys.argv = saved
        print(f"\n[{module} finished in {time.perf_counter() - started:.1f}s]")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", type=int, default=24,
                        help="latitude bands to stream (each yields many patches)")
    parser.add_argument("--per-band", type=int, default=20,
                        help="patches cut from each band")
    parser.add_argument("--size-px", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=2000,
                        help="CA iteration budget; ~92%% of 128px patches converge by 2000")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="reuse the existing dataset CSV")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    print("LUNASLIDE — Stage 2: physics engine and hazard surrogate")
    print("Real LRO topography, streamed from USGS Cloud Optimized GeoTIFFs.")

    ok = True
    if not args.skip_dataset:
        ok &= run_module("scripts.build_hazard_dataset", [
            "--bands", str(args.bands),
            "--per-band", str(args.per_band),
            "--size-px", str(args.size_px),
            "--max-iter", str(args.max_iter),
        ])
    if ok:
        ok &= run_module("scripts.train_hazard_model", [])
    if ok and not args.skip_figures:
        run_module("scripts.generate_figures", [])
        run_module("scripts.generate_ca_visuals", [])

    print(f"\n{'=' * 78}")
    print("Outputs: data/stage2/hazard_dataset.csv, figures/")
    print("Write-up: docs/THESIS.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
