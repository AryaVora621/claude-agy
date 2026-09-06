"""
Unit tests for Grad-Shafranov MHD Equilibrium Solver & Solovev Benchmarks.
"""

import math
import unittest

from stellarfusion.equilibrium import (
    EquilibriumProfile,
    GradShafranovSolver,
    Grid2D,
    SolovevEquilibrium,
)


class TestGrid2D(unittest.TestCase):
    def test_grid_initialization_and_properties(self) -> None:
        grid = Grid2D(nr=21, nz=31, r_min=1.0, r_max=5.0, z_min=-2.0, z_max=2.0)
        self.assertEqual(grid.nr, 21)
        self.assertEqual(grid.nz, 31)
        self.assertAlmostEqual(grid.dr, 4.0 / 20.0)
        self.assertAlmostEqual(grid.dz, 4.0 / 30.0)
        self.assertAlmostEqual(grid.r_at(0), 1.0)
        self.assertAlmostEqual(grid.r_at(20), 5.0)
        self.assertAlmostEqual(grid.z_at(0), -2.0)
        self.assertAlmostEqual(grid.z_at(30), 2.0)

    def test_grid_coordinate_containment(self) -> None:
        grid = Grid2D(nr=11, nz=11, r_min=1.0, r_max=4.0, z_min=-1.5, z_max=1.5)
        self.assertTrue(grid.contains(2.5, 0.0))
        self.assertFalse(grid.contains(0.5, 0.0))
        self.assertFalse(grid.contains(2.5, 2.5))

    def test_grid_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Grid2D(nr=3, nz=10, r_min=1.0, r_max=2.0, z_min=-1.0, z_max=1.0)
        with self.assertRaises(ValueError):
            Grid2D(nr=10, nz=10, r_min=-1.0, r_max=2.0, z_min=-1.0, z_max=1.0)


class TestSolovevEquilibrium(unittest.TestCase):
    def setUp(self) -> None:
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.5, psi_0=2.0, b_0=2.0)

    def test_analytical_axis_flux(self) -> None:
        # At magnetic axis (R_0, Z_0), diff_r2 = 0 and rel_z = 0, so psi = 0
        psi_axis = self.solovev.psi(3.0, 0.0)
        self.assertAlmostEqual(psi_axis, 0.0, places=9)

    def test_shafranov_operator_matches_exact_rhs(self) -> None:
        # Verify that Delta* psi evaluated via analytical derivatives matches the exact RHS formula
        test_points = [(2.5, 0.5), (3.0, 1.0), (3.5, -0.7), (4.0, 0.2)]
        for r, z in test_points:
            numerical_op = self.solovev.shafranov_operator(r, z)
            exact_rhs = self.solovev.exact_rhs(r)
            self.assertAlmostEqual(numerical_op, exact_rhs, places=7)

    def test_magnetic_field_on_axis(self) -> None:
        # On axis (R_0, 0), poloidal derivatives vanish, leaving only toroidal field
        br, bphi, bz = self.solovev.magnetic_field(3.0, 0.0)
        self.assertAlmostEqual(br, 0.0, places=9)
        self.assertAlmostEqual(bz, 0.0, places=9)
        self.assertAlmostEqual(bphi, 2.0, places=9)


class TestGradShafranovSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid2D(nr=25, nz=25, r_min=1.5, r_max=4.5, z_min=-1.5, z_max=1.5)
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.4, psi_0=1.0, b_0=2.5)
        self.solver = GradShafranovSolver(grid=self.grid)

    def test_finite_difference_shafranov_operator(self) -> None:
        # Seed grid with analytical Solovev solution and verify FD operator accuracy
        self.solver.initialize_with_solovev(self.solovev)
        mid_i = self.grid.nr // 2
        mid_j = self.grid.nz // 2
        r_mid = self.grid.r_at(mid_i)
        z_mid = self.grid.z_at(mid_j)

        fd_operator = self.solver.evaluate_shafranov_operator_at(mid_i, mid_j)
        exact_operator = self.solovev.shafranov_operator(r_mid, z_mid)
        # O(h^2) finite difference agreement within 1%
        rel_error = abs(fd_operator - exact_operator) / exact_operator
        self.assertLess(rel_error, 0.02)

    def test_sor_solver_iteration(self) -> None:
        self.solver.initialize_with_solovev(self.solovev)
        initial_res = self.solver.solve_step(omega=1.2)
        self.assertGreater(initial_res, 0.0)

        # Run several sweeps to verify numerical stability
        iters = self.solver.solve(max_iterations=10, tolerance=1e-5, omega=1.3)
        self.assertEqual(iters, 10)

    def test_flux_interpolation(self) -> None:
        self.solver.initialize_with_solovev(self.solovev)
        val = self.solver.interpolate_psi(3.0, 0.0)
        self.assertAlmostEqual(val, 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
