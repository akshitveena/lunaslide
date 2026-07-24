"""Invariant tests for the synchronous mass-wasting cellular automaton."""

from __future__ import annotations

import unittest

import numpy as np

from src.physics.relaxation import compute_slope, simulate_mass_wasting


def max_directional_slope(H: np.ndarray, grid_spacing: float, connectivity: int = 4) -> float:
    """Largest downhill slope to any neighbour — the quantity the CA bounds.

    Deliberately independent of ``compute_slope``: the automaton constrains
    per-neighbour drops, so verifying convergence with the same central-
    difference estimator used for feature extraction would be circular.
    """
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    worst = 0.0
    for row_shift, column_shift in offsets:
        interior = H[
            max(row_shift, 0) : H.shape[0] + min(row_shift, 0) or None,
            max(column_shift, 0) : H.shape[1] + min(column_shift, 0) or None,
        ]
        neighbour = np.roll(H, (-row_shift, -column_shift), axis=(0, 1))[
            max(row_shift, 0) : H.shape[0] + min(row_shift, 0) or None,
            max(column_shift, 0) : H.shape[1] + min(column_shift, 0) or None,
        ]
        distance = grid_spacing * np.hypot(row_shift, column_shift)
        worst = max(worst, float(np.max((interior - neighbour) / distance)))
    return worst


def cone(size: int = 101, peak: float = 300.0, gradient: float = 4.0) -> np.ndarray:
    """A radially symmetric cone, exactly centred so it is rot90-invariant."""
    centre = (size - 1) / 2.0
    axis = np.arange(size) - centre
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    return np.maximum(peak - gradient * radius, 0.0)


class TestMassConservation(unittest.TestCase):
    def test_total_elevation_is_invariant(self):
        rng = np.random.default_rng(0)
        H = 500 + rng.normal(0, 60, (64, 64))
        relaxed, _, iterations = simulate_mass_wasting(H, grid_spacing=5.0)
        self.assertGreater(iterations, 0)
        self.assertAlmostEqual(float(H.sum()), float(relaxed.sum()), places=6)

    def test_a_ramp_stable_in_the_interior_is_left_alone(self):
        # 500 m down to 0 m over 40 rows at 100 m spacing is a 0.13 slope
        # everywhere -- well below repose, so nothing should move.  But the
        # top and bottom rows differ by 500 m, which a periodic np.roll reads
        # as an adjacent cliff: the "Pac-Man" artifact.  Any movement here is
        # mass wrapping across the domain edge.
        H = np.linspace(500, 0, 40).reshape(-1, 1) @ np.ones((1, 40))
        relaxed, toppled, iterations = simulate_mass_wasting(H, grid_spacing=100.0)
        self.assertEqual(iterations, 0)
        self.assertFalse(toppled.any())
        np.testing.assert_array_equal(relaxed, H)

    def test_an_unstable_ramp_sheds_downhill_only(self):
        # The same ramp at 5 m spacing is a 2.56 slope and genuinely fails.
        # Material must pile up at the toe (last row) and never appear at the
        # crest, which is where wrapped mass would land.
        H = np.linspace(500, 0, 40).reshape(-1, 1) @ np.ones((1, 40))
        relaxed, _, iterations = simulate_mass_wasting(H, grid_spacing=5.0)
        self.assertGreater(iterations, 0)
        self.assertAlmostEqual(float(H.sum()), float(relaxed.sum()), places=6)
        self.assertLessEqual(float(relaxed[0].max()), float(H[0].max()) + 1e-9)
        self.assertGreater(float(relaxed[-1].max()), float(H[-1].max()))

    def test_conservation_holds_for_moore_connectivity(self):
        rng = np.random.default_rng(7)
        H = 200 + rng.normal(0, 40, (48, 48))
        relaxed, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, connectivity=8)
        self.assertAlmostEqual(float(H.sum()), float(relaxed.sum()), places=6)


class TestDirectionalIsotropy(unittest.TestCase):
    """The regression guard for the directional-bias artifact.

    The neighbour set is closed under 90-degree rotation and every flux is
    computed from one frozen snapshot, so relaxing a radially symmetric cone
    must commute with ``np.rot90``.  The previous sequential implementation
    mutated the terrain between directions and failed this.
    """

    def test_relaxing_a_cone_commutes_with_rotation(self):
        H = cone()
        relaxed, _, iterations = simulate_mass_wasting(H, grid_spacing=5.0, max_iter=60)
        self.assertGreater(iterations, 0)
        change = relaxed - H
        self.assertGreater(float(np.abs(change).max()), 1.0, "cone did not relax")
        np.testing.assert_allclose(change, np.rot90(change), atol=1e-9)

    def test_toppled_mask_is_rotationally_symmetric(self):
        _, toppled, _ = simulate_mass_wasting(cone(), grid_spacing=5.0, max_iter=60)
        self.assertTrue(toppled.any())
        np.testing.assert_array_equal(toppled, np.rot90(toppled))

    def test_transposing_the_terrain_transposes_the_result(self):
        rng = np.random.default_rng(3)
        H = 400 + rng.normal(0, 50, (50, 50))
        direct, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, max_iter=40)
        transposed, _, _ = simulate_mass_wasting(H.T, grid_spacing=5.0, max_iter=40)
        np.testing.assert_allclose(direct.T, transposed, atol=1e-9)


class TestStability(unittest.TestCase):
    def test_relaxation_drives_slopes_to_the_angle_of_repose(self):
        # Transport is diffusive, so iterations scale with the square of the
        # distance material has to travel; a 25-cell cone settles in ~1.7k.
        H = cone(size=25, peak=400.0, gradient=6.0)
        relaxed, _, iterations = simulate_mass_wasting(
            H, grid_spacing=5.0, crit=0.577, max_iter=5000
        )
        self.assertLess(iterations, 5000, "did not converge within the iteration ceiling")
        self.assertLessEqual(max_directional_slope(relaxed, 5.0), 0.577 + 1e-6)

    def test_slope_decreases_monotonically_even_when_truncated(self):
        # Stopping early must still leave the terrain no steeper than it began.
        H = cone(size=81, peak=400.0, gradient=6.0)
        before = max_directional_slope(H, 5.0)
        for ceiling in (10, 100, 1000):
            relaxed, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, max_iter=ceiling)
            after = max_directional_slope(relaxed, 5.0)
            with self.subTest(max_iter=ceiling):
                self.assertLessEqual(after, before + 1e-9)

    def test_limiter_keeps_an_aggressive_relax_factor_stable(self):
        # relax_factor = 1.0 exceeds the 1/connectivity bound the scheme would
        # otherwise need; the non-inversion limiter must absorb it.
        H = np.zeros((32, 32))
        H[16, 16] = 900.0
        relaxed, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, relax_factor=1.0)
        self.assertTrue(np.all(np.isfinite(relaxed)))
        self.assertAlmostEqual(float(H.sum()), float(relaxed.sum()), places=6)
        self.assertLessEqual(float(relaxed.max()), float(H.max()))
        self.assertGreaterEqual(float(relaxed.min()), float(H.min()) - 1e-9)

    def test_already_stable_terrain_is_untouched(self):
        H = np.linspace(0, 10, 30).reshape(-1, 1) @ np.ones((1, 30))
        relaxed, toppled, iterations = simulate_mass_wasting(H, grid_spacing=100.0)
        self.assertEqual(iterations, 0)
        self.assertFalse(toppled.any())
        np.testing.assert_array_equal(relaxed, H)

    def test_repeated_runs_are_identical(self):
        rng = np.random.default_rng(11)
        H = 300 + rng.normal(0, 45, (40, 40))
        first, mask_a, count_a = simulate_mass_wasting(H, grid_spacing=5.0)
        second, mask_b, count_b = simulate_mass_wasting(H, grid_spacing=5.0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(mask_a, mask_b)
        self.assertEqual(count_a, count_b)

    def test_dtype_of_the_input_is_preserved(self):
        H = cone().astype(np.float32)
        relaxed, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, max_iter=20)
        self.assertEqual(relaxed.dtype, np.float32)


class TestAnisotropicSpacing(unittest.TestCase):
    """Equirectangular patches have rows and columns at different ground scales."""

    def test_scalar_and_equal_pair_agree(self):
        H = cone(size=31, peak=200.0, gradient=5.0)
        scalar, _, _ = simulate_mass_wasting(H, grid_spacing=5.0, max_iter=30)
        pair, _, _ = simulate_mass_wasting(H, grid_spacing=(5.0, 5.0), max_iter=30)
        np.testing.assert_array_equal(scalar, pair)

    def test_slope_uses_each_axis_spacing(self):
        # 3 m of rise per column, cells 6 m apart east-west -> slope 0.5,
        # regardless of how far apart the rows are.
        H = np.ones((12, 1)) @ (np.arange(12) * 3.0).reshape(1, -1)
        np.testing.assert_allclose(compute_slope(H, (118.0, 6.0)), 0.5, atol=1e-12)

    def test_narrow_columns_make_east_west_slopes_steeper(self):
        H = np.ones((16, 1)) @ (np.arange(16) * 4.0).reshape(1, -1)
        square = compute_slope(H, 118.0).max()
        polar = compute_slope(H, (118.0, 40.0)).max()
        self.assertAlmostEqual(polar / square, 118.0 / 40.0, places=9)

    def test_transposing_terrain_and_spacing_transposes_the_result(self):
        rng = np.random.default_rng(5)
        H = 300 + rng.normal(0, 40, (40, 40))
        direct, _, _ = simulate_mass_wasting(H, grid_spacing=(118.0, 40.0), max_iter=25)
        swapped, _, _ = simulate_mass_wasting(H.T, grid_spacing=(40.0, 118.0), max_iter=25)
        np.testing.assert_allclose(direct.T, swapped, atol=1e-9)

    def test_mass_is_conserved_with_unequal_spacing(self):
        rng = np.random.default_rng(9)
        H = 200 + rng.normal(0, 50, (48, 48))
        relaxed, _, _ = simulate_mass_wasting(H, grid_spacing=(118.0, 12.0))
        self.assertAlmostEqual(float(H.sum()), float(relaxed.sum()), places=6)

    def test_rejects_malformed_spacing(self):
        H = np.zeros((8, 8))
        for spacing in ((0.0, 5.0), (5.0, -1.0), (5.0, 5.0, 5.0)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                simulate_mass_wasting(H, grid_spacing=spacing)


class TestValidation(unittest.TestCase):
    def test_rejects_non_finite_elevation(self):
        H = np.zeros((8, 8))
        H[0, 0] = np.nan
        with self.assertRaises(ValueError):
            simulate_mass_wasting(H)

    def test_rejects_bad_parameters(self):
        H = np.zeros((8, 8))
        for kwargs in ({"grid_spacing": 0.0}, {"relax_factor": 1.5}, {"crit": -1.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                simulate_mass_wasting(H, **kwargs)
        with self.assertRaises(ValueError):
            simulate_mass_wasting(H, connectivity=6)
        with self.assertRaises(ValueError):
            simulate_mass_wasting(np.zeros((4, 4, 4)))


class TestComputeSlope(unittest.TestCase):
    def test_flat_terrain_has_zero_slope(self):
        np.testing.assert_allclose(compute_slope(np.full((10, 10), 42.0), 5.0), 0.0)

    def test_uniform_ramp_matches_its_analytic_gradient(self):
        H = (np.arange(20) * 3.0).reshape(-1, 1) @ np.ones((1, 20))
        # 3 m rise per cell over a 5 m cell -> slope 0.6 everywhere.
        np.testing.assert_allclose(compute_slope(H, 5.0), 0.6, atol=1e-12)

    def test_rejects_non_positive_spacing(self):
        with self.assertRaises(ValueError):
            compute_slope(np.zeros((4, 4)), 0.0)


if __name__ == "__main__":
    unittest.main()
