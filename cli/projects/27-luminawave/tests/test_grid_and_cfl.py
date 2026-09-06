"""Unit tests for 2D Yee Grid, Material Assignments, and CFL Stability."""

import math
import unittest
from luminawave.grid import (
    C0,
    EPSILON_0,
    MU_0,
    ETA_0,
    Grid2D,
    GridDimensions,
)


class TestGridAndCFL(unittest.TestCase):
    def test_physical_constants(self):
        """Verify fundamental electromagnetic physical constants."""
        self.assertAlmostEqual(C0, 299792458.0, delta=1.0)
        self.assertAlmostEqual(EPSILON_0, 8.8541878e-12, delta=1e-15)
        self.assertAlmostEqual(MU_0, 1.256637e-6, delta=1e-8)
        calc_c = 1.0 / math.sqrt(EPSILON_0 * MU_0)
        self.assertAlmostEqual(calc_c, C0, delta=1e3)
        calc_eta = math.sqrt(MU_0 / EPSILON_0)
        self.assertAlmostEqual(calc_eta, ETA_0, delta=1e-2)

    def test_grid_initialization_and_cfl(self):
        """Verify Yee grid creation and 2D CFL time step calculation."""
        grid = Grid2D(nx=60, ny=40, dx=50e-9, dy=50e-9, courant_factor=0.70)
        self.assertEqual(grid.nx, 60)
        self.assertEqual(grid.ny, 40)
        self.assertEqual(grid.total_cells, 2400)

        # 2D CFL limit: dt_max = dx / (c * sqrt(2)) for dx == dy
        expected_max_dt = 50e-9 / (C0 * math.sqrt(2.0))
        expected_dt = 0.70 * expected_max_dt
        self.assertAlmostEqual(grid.dt, expected_dt, delta=1e-18)

        # Dimensions
        self.assertAlmostEqual(grid.dims.width, 60 * 50e-9, delta=1e-15)
        self.assertAlmostEqual(grid.dims.height, 40 * 50e-9, delta=1e-15)

    def test_grid_invalid_parameters(self):
        """Verify error checking on invalid grid dimensions or steps."""
        with self.assertRaises(ValueError):
            Grid2D(nx=2, ny=20)
        with self.assertRaises(ValueError):
            Grid2D(nx=20, ny=20, dx=-1e-9)
        with self.assertRaises(ValueError):
            Grid2D(nx=20, ny=20, courant_factor=1.5)

    def test_material_setting_and_coefficients(self):
        """Verify material assignment and FDTD update coefficients."""
        grid = Grid2D(nx=40, ny=40, dx=50e-9, dy=50e-9)
        # Default is vacuum (eps_r = 1.0, sigma = 0.0)
        idx = grid.idx(10, 10)
        self.assertEqual(grid.eps_r[idx], 1.0)
        self.assertAlmostEqual(grid.ca[idx], 1.0, delta=1e-12)
        expected_cb = grid.dt / EPSILON_0
        self.assertAlmostEqual(grid.cb[idx], expected_cb, delta=1e-6)

        # Set Silicon core (n = 3.48 -> eps_r = 12.11)
        grid.set_refractive_index(15, 20, 3.48)
        idx_si = grid.idx(15, 20)
        self.assertAlmostEqual(grid.get_refractive_index(15, 20), 3.48, delta=1e-4)
        self.assertAlmostEqual(grid.eps_r[idx_si], 3.48 * 3.48, delta=1e-4)
        expected_cb_si = grid.dt / (EPSILON_0 * 3.48 * 3.48)
        self.assertAlmostEqual(grid.cb[idx_si], expected_cb_si, delta=1e-6)

    def test_geometric_shapes(self):
        """Verify drawing rectangle, circle, and ring into the dielectric grid."""
        grid = Grid2D(nx=50, ny=50)

        # Rectangle
        grid.add_rectangle(10, 10, 20, 20, eps_r=4.0)
        self.assertEqual(grid.eps_r[grid.idx(15, 15)], 4.0)
        self.assertEqual(grid.eps_r[grid.idx(5, 5)], 1.0)

        # Circle
        grid.add_circle(cx=35, cy=35, radius=5, eps_r=9.0)
        self.assertEqual(grid.eps_r[grid.idx(35, 35)], 9.0)
        self.assertEqual(grid.eps_r[grid.idx(35, 42)], 1.0)

        # Ring
        grid.add_ring(cx=25, cy=25, r_inner=6, r_outer=10, eps_r=12.0)
        self.assertEqual(grid.eps_r[grid.idx(25, 25)], 1.0)  # Center hole
        self.assertEqual(grid.eps_r[grid.idx(25, 33)], 12.0) # On the ring

    def test_energy_computation(self):
        """Verify electric and magnetic energy calculations."""
        grid = Grid2D(nx=20, ny=20, dx=1e-6, dy=1e-6)
        # Put 1 V/m in a single cell
        idx = grid.idx(10, 10)
        grid.ez[idx] = 100.0
        grid.hx[idx] = 0.5

        ue, uh, utotal = grid.total_energy()
        self.assertGreater(ue, 0.0)
        self.assertGreater(uh, 0.0)
        self.assertAlmostEqual(utotal, ue + uh, delta=1e-15)

        # Resetting fields zeroes energy
        grid.reset_fields()
        ue0, uh0, ut0 = grid.total_energy()
        self.assertEqual(ut0, 0.0)


if __name__ == "__main__":
    unittest.main()
