"""Distil a Stage 2 DEM patch into the hazard summary Stage 3 consumes."""

from __future__ import annotations

import numpy as np

from src.physics.relaxation import compute_slope, simulate_mass_wasting

from .contracts import Stage2Hazard


def summarise_hazard(
    elevation: np.ndarray,
    grid_spacing: float | tuple[float, float],
    site: str,
    *,
    nominal_repose_deg: float = 30.0,
    vibration_repose_deg: float = 24.0,
    max_iter: int = 2000,
) -> Stage2Hazard:
    """Run the mass-wasting simulation at quiescent and vibrational friction.

    Two runs of the same terrain: one at the dry angle of repose, one at the
    lower effective angle a descent engine induces.  The pair is exactly what
    Stage 3 needs to tell a resting-stable site from a landing-stable one.
    """
    elevation = np.asarray(elevation, dtype=np.float64)
    nominal_crit = float(np.tan(np.radians(nominal_repose_deg)))
    vibration_crit = float(np.tan(np.radians(vibration_repose_deg)))

    _, toppled_nominal, iters = simulate_mass_wasting(
        elevation, grid_spacing=grid_spacing, crit=nominal_crit, max_iter=max_iter)
    _, toppled_vibration, _ = simulate_mass_wasting(
        elevation, grid_spacing=grid_spacing, crit=vibration_crit, max_iter=max_iter)

    slope_deg = np.degrees(np.arctan(compute_slope(elevation, grid_spacing)))
    spacing_y = grid_spacing[0] if isinstance(grid_spacing, tuple) else grid_spacing
    return Stage2Hazard(
        site=site,
        toppled_fraction_nominal=float(toppled_nominal.mean()),
        toppled_fraction_vibration=float(toppled_vibration.mean()),
        max_slope_deg=float(slope_deg.max()),
        grid_spacing_m=float(spacing_y),
        converged=bool(iters < max_iter),
        nominal_repose_deg=nominal_repose_deg,
        vibration_repose_deg=vibration_repose_deg,
    )
