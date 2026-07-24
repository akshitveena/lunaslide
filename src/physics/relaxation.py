"""Synchronous cellular-automaton model of gravity-driven lunar mass wasting.

The update rule is *synchronous*: every directional slope is measured against a
single frozen snapshot of the terrain, all fluxes are derived from that
snapshot, and the whole grid is updated in one vectorised step.  An earlier
revision swept the four neighbour directions sequentially, mutating the terrain
between directions, which made the result depend on the order the directions
happened to be listed in and biased transport along the first axis processed.
See ``docs/PHYSICS_BUGS.md``.

Two invariants are enforced, both covered by ``tests/test_physics_relaxation.py``:

* **Conservation of mass.** Material removed from a cell is deposited on its
  neighbour, and flux across the domain edge is masked to zero, so the total
  elevation sum is invariant to floating-point precision.
* **No inversion.** A cell never ends an iteration below a neighbour it just
  fed.  Without this the scheme is stable only for
  ``relax_factor <= 1 / connectivity`` and silently oscillates above it.
"""

from __future__ import annotations

import numpy as np

# Offsets to the neighbour a cell may shed material onto, as (row, column).
# Each set is closed under 90-degree rotation, which is what makes the update
# rule isotropic on the grid.
_VON_NEUMANN: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MOORE: tuple[tuple[int, int], ...] = _VON_NEUMANN + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _spacing_pair(grid_spacing: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise a spacing argument to ``(north_south, east_west)`` metres.

    A scalar means square cells.  Equirectangular lunar patches are not square:
    east-west spacing contracts by ``cos(latitude)``, so a patch at 70 degrees
    has 118 m rows and 40 m columns.
    """
    if isinstance(grid_spacing, (int, float)):
        pair = (float(grid_spacing), float(grid_spacing))
    else:
        values = tuple(float(value) for value in grid_spacing)
        if len(values) != 2:
            raise ValueError("grid_spacing must be a scalar or a (north_south, east_west) pair.")
        pair = values
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError("grid_spacing values must be positive.")
    return pair


def compute_slope(H: np.ndarray, grid_spacing: float | tuple[float, float]) -> np.ndarray:
    """Return the slope magnitude ``|grad H|`` of a DEM, in metres per metre.

    Uses central differences in the interior and one-sided differences at the
    edges (``numpy.gradient``), the standard estimator for gridded elevation.
    ``grid_spacing`` accepts a scalar for square cells or a
    ``(north_south, east_west)`` pair for projected patches that are not.
    """
    spacing_y, spacing_x = _spacing_pair(grid_spacing)
    d_dy, d_dx = np.gradient(np.asarray(H, dtype=np.float64), spacing_y, spacing_x)
    return np.hypot(d_dx, d_dy)


def _neighbour_exists(shape: tuple[int, ...], row_shift: int, column_shift: int) -> np.ndarray:
    """Mask of cells that actually have a neighbour at the given offset.

    ``numpy.roll`` is periodic, so without this mask the top edge would see the
    bottom edge as an adjacent cliff and dump material across the wrap.
    """
    valid = np.ones(shape, dtype=bool)
    if row_shift < 0:
        valid[0, :] = False
    elif row_shift > 0:
        valid[-1, :] = False
    if column_shift < 0:
        valid[:, 0] = False
    elif column_shift > 0:
        valid[:, -1] = False
    return valid


def _build_directions(
    shape: tuple[int, ...], spacing_y: float, spacing_x: float, connectivity: int
) -> list[tuple[int, int, float, np.ndarray]]:
    """Neighbour offsets with their true ground distance and validity mask."""
    offsets = _VON_NEUMANN if connectivity == 4 else _MOORE
    return [
        (
            row_shift,
            column_shift,
            float(np.hypot(row_shift * spacing_y, column_shift * spacing_x)),
            _neighbour_exists(shape, row_shift, column_shift),
        )
        for row_shift, column_shift in offsets
    ]


def _gather(
    terrain: np.ndarray, directions: list, crit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Downhill drop and excess-above-repose per direction, from one snapshot."""
    drops = np.empty((len(directions), *terrain.shape), dtype=np.float64)
    excesses = np.empty_like(drops)
    for index, (row_shift, column_shift, distance, exists) in enumerate(directions):
        neighbour = np.roll(terrain, (-row_shift, -column_shift), axis=(0, 1))
        drop = np.where(exists, terrain - neighbour, 0.0)
        drops[index] = drop
        excesses[index] = np.maximum(drop - crit * distance, 0.0)
    return drops, excesses


def total_excess(
    H: np.ndarray,
    grid_spacing: float | tuple[float, float],
    crit: float = 0.577,
    *,
    connectivity: int = 4,
) -> np.ndarray:
    """Material standing above the angle of repose, per cell, in metres.

    This is the quantity the automaton drives to zero.  Evaluated on a
    partially relaxed grid it measures how much failure is still pending — an
    uncensored alternative to counting iterations, which saturates at the
    iteration ceiling exactly on the hazardous class.
    """
    spacing_y, spacing_x = _spacing_pair(grid_spacing)
    terrain = np.asarray(H, dtype=np.float64)
    directions = _build_directions(terrain.shape, spacing_y, spacing_x, connectivity)
    return _gather(terrain, directions, crit)[1].sum(axis=0)


def simulate_mass_wasting(
    H: np.ndarray,
    grid_spacing: float | tuple[float, float] = 5.0,
    crit: float = 0.577,
    relax_factor: float = 0.2,
    max_iter: int = 500,
    *,
    connectivity: int = 4,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Relax slopes above the angle of repose until the terrain is stable.

    Args:
        H: Elevation grid in metres.
        grid_spacing: Ground distance between adjacent cell centres, in metres.
            A scalar for square cells, or a ``(north_south, east_west)`` pair
            for projected patches whose columns are closer together than their
            rows — which is every equirectangular patch away from the equator.
        crit: Critical slope (rise/run) above which material fails.  The default
            0.577 is ``tan(30 deg)``, a representative dry angle of repose for
            lunar regolith.  Lowering it models a reduced effective friction
            angle, such as under the vibrational load of a descent engine.
        relax_factor: Fraction of the excess above repose moved per iteration.
            Affects convergence rate, not the fixed point.
        max_iter: Iteration ceiling.  A returned count equal to ``max_iter``
            means the terrain had not fully stabilised.
        connectivity: 4 for von Neumann neighbours, 8 to include diagonals.
            Diagonal neighbours use a ``sqrt(2)`` larger ground distance.
        tolerance: Total excess below this, in metres, counts as stable.

    Returns:
        ``(relaxed_elevation, toppled_mask, iterations)``.  ``toppled_mask``
        marks cells that shed material at any point during the run, and
        ``iterations`` counts update steps actually applied — 0 when the input
        was already stable.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 (von Neumann) or 8 (Moore).")
    spacing_y, spacing_x = _spacing_pair(grid_spacing)
    if crit < 0:
        raise ValueError("crit must be non-negative.")
    if not 0 < relax_factor <= 1:
        raise ValueError("relax_factor must lie in (0, 1].")

    source = np.asarray(H)
    if source.ndim != 2:
        raise ValueError("Mass wasting expects a two-dimensional elevation grid.")
    if not np.all(np.isfinite(source)):
        raise ValueError("Elevation grid contains non-finite values.")

    # Accumulate in float64 so mass conservation is not limited by the input's
    # storage precision; the caller's dtype is restored on return.
    terrain = source.astype(np.float64, copy=True)
    directions = _build_directions(terrain.shape, spacing_y, spacing_x, connectivity)

    toppled = np.zeros(terrain.shape, dtype=bool)
    iterations = 0

    for _ in range(max_iter):
        # --- Gather: every direction is measured against the same snapshot. ---
        drops, excesses = _gather(terrain, directions, crit)
        total_excess = excesses.sum(axis=0)
        if not (total_excess > tolerance).any():
            break

        # --- Distribute outflow across downhill directions, weighted by excess. ---
        share = np.divide(
            excesses, total_excess, out=np.zeros_like(excesses), where=total_excess > 0
        )
        outflow = relax_factor * total_excess

        # --- Limit: a cell must not end up below a neighbour it just fed. ---
        # The cell keeps ``H - outflow``; neighbour d receives ``outflow * share_d``.
        # Staying above it requires ``outflow * (1 + share_d) <= drop_d`` for
        # every receiving direction, so clamp to the tightest of those bounds.
        headroom = np.divide(
            drops, 1.0 + share, out=np.full_like(drops, np.inf), where=excesses > 0
        )
        outflow = np.minimum(outflow, headroom.min(axis=0))
        np.maximum(outflow, 0.0, out=outflow)

        flux = share * outflow

        # --- Scatter: apply every flux against the snapshot in one step. ---
        deposits = np.zeros_like(terrain)
        for index, (row_shift, column_shift, _, _) in enumerate(directions):
            deposits += np.roll(flux[index], (row_shift, column_shift), axis=(0, 1))

        terrain += deposits - outflow
        toppled |= outflow > 0
        iterations += 1

    return terrain.astype(source.dtype, copy=False), toppled, iterations
