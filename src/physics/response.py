"""Uncensored descriptors of how terrain responds to a relaxation budget.

``iterations_to_relax`` — the number of steps until the automaton settles — is
a *stopping time*, and stopping times saturate.  Transport is diffusive, so a
patch of hazardous terrain does not converge within any affordable ceiling and
its iteration count pins to ``max_iter``, while flat terrain returns 0.  The
feature therefore degenerates into a near-binary indicator of the hazard class
rather than a measurement of the terrain, and inflates any score computed from
it.

Everything here is instead evaluated at a **fixed budget**.  "How much of the
pending failure did this terrain work through in N steps" is well defined
whether or not it finished, and does not saturate.

Terrain descriptors (:func:`terrain_features`) are separated from response
descriptors (:class:`RelaxationResponse`) on purpose: the former are cheap and
computable without simulating, the latter require the automaton.  That split is
what makes a surrogate model meaningful — it learns to predict the expensive
quantity from the cheap ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .relaxation import compute_slope, simulate_mass_wasting, total_excess


@dataclass(frozen=True)
class RelaxationResponse:
    """What a fixed relaxation budget did to a patch."""

    toppled_fraction: float
    mass_moved_m: float
    max_change_m: float
    initial_excess_m: float
    residual_excess_m: float
    excess_relieved_fraction: float
    residual_unstable_fraction: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict:
        return asdict(self)


def terrain_features(
    H: np.ndarray, grid_spacing: float | tuple[float, float], crit: float = 0.577
) -> dict[str, float]:
    """Cheap terrain statistics — no simulation required.

    These are the inputs a surrogate model is allowed to see.  Nothing here
    depends on the automaton having been run.
    """
    elevation = np.asarray(H, dtype=np.float64)
    slope = compute_slope(elevation, grid_spacing)
    excess = total_excess(elevation, grid_spacing, crit)
    return {
        "relief_m": float(elevation.max() - elevation.min()),
        "elevation_std_m": float(elevation.std()),
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
        "slope_p90": float(np.percentile(slope, 90)),
        "slope_p99": float(np.percentile(slope, 99)),
        "slope_max": float(slope.max()),
        # Cells already standing above the angle of repose, and by how much in
        # total. Both come from a single gather with no iteration, so they are
        # available long before the simulation would finish.
        "unstable_fraction": float((slope > crit).mean()),
        "excess_total_m": float(excess.sum()),
        "excess_max_m": float(excess.max()),
    }


def relaxation_response(
    H: np.ndarray,
    grid_spacing: float | tuple[float, float],
    crit: float = 0.577,
    *,
    max_iter: int = 500,
    connectivity: int = 4,
) -> RelaxationResponse:
    """Run the automaton for a fixed budget and describe what it did.

    ``excess_relieved_fraction`` is the intended replacement for
    ``iterations_to_relax``: the share of the initial above-repose material the
    automaton resolved within the budget.  It is bounded in ``[0, 1]``, defined
    whether or not the run converged, and does not pin to the ceiling.
    """
    initial = np.asarray(H, dtype=np.float64)
    before = total_excess(initial, grid_spacing, crit, connectivity=connectivity).sum()
    relaxed, toppled, iterations = simulate_mass_wasting(
        initial, grid_spacing=grid_spacing, crit=crit,
        max_iter=max_iter, connectivity=connectivity,
    )
    after_grid = total_excess(relaxed, grid_spacing, crit, connectivity=connectivity)
    after = float(after_grid.sum())
    change = relaxed - initial
    slope_after = compute_slope(relaxed, grid_spacing)
    return RelaxationResponse(
        toppled_fraction=float(toppled.mean()),
        # Deposits equal erosion under conservation of mass, so summing the
        # positive side counts relocated material exactly once.
        mass_moved_m=float(change[change > 0].sum()),
        max_change_m=float(np.abs(change).max()),
        initial_excess_m=float(before),
        residual_excess_m=after,
        excess_relieved_fraction=float(1.0 - after / before) if before > 0 else 1.0,
        residual_unstable_fraction=float((slope_after > crit).mean()),
        iterations=iterations,
        converged=iterations < max_iter,
    )
