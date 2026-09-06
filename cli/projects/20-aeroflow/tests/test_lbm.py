"""
Unit Tests for AeroFlow D2Q9 Lattice Boltzmann Method Solver.
"""

import unittest
import math
from aeroflow.types import ObstacleMask
from aeroflow.lbm import LatticeD2Q9, LBMSolver
from aeroflow.obstacles import create_cylinder_obstacle


class TestLatticeD2Q9(unittest.TestCase):
    def test_lattice_constants(self):
        # Weights must sum to exactly 1.0
        self.assertAlmostEqual(sum(LatticeD2Q9.WEIGHTS), 1.0, places=10)

        # Opposite directions must be self-inverses
        for i in range(9):
            opp_i = LatticeD2Q9.OPPOSITE[i]
            self.assertEqual(LatticeD2Q9.OPPOSITE[opp_i], i)
            # Velocity must be reversed
            self.assertEqual(LatticeD2Q9.CX[i], -LatticeD2Q9.CX[opp_i])
            self.assertEqual(LatticeD2Q9.CY[i], -LatticeD2Q9.CY[opp_i])


class TestLBMSolver(unittest.TestCase):
    def test_lbm_initialization_and_equilibrium(self):
        nx, ny = 30, 20
        solver = LBMSolver(nx=nx, ny=ny, viscosity=0.02, inflow_velocity=0.05)
        solver.compute_macroscopic()

        # Check uniform density and velocity initially
        for idx in range(nx * ny):
            self.assertAlmostEqual(solver.rho.data[idx], 1.0, places=5)
            self.assertAlmostEqual(solver.velocity.u.data[idx], 0.05, places=5)
            self.assertAlmostEqual(solver.velocity.v.data[idx], 0.0, places=5)

    def test_lbm_mass_conservation(self):
        # In a closed periodic/bounce-back box, total mass must be conserved
        nx, ny = 25, 25
        solver = LBMSolver(
            nx=nx,
            ny=ny,
            viscosity=0.03,
            inflow_velocity=0.0,
            top_bottom_no_slip=True,
            left_right_no_slip=True,
        )

        # Add initial density bump in center
        center_idx = 12 * nx + 12
        for i in range(9):
            solver.f[i][center_idx] += 0.05

        solver.compute_macroscopic()
        initial_mass = sum(solver.rho.data)

        # Run 20 time steps
        for _ in range(20):
            solver.collide()
            solver.stream_and_bounce()
            solver.compute_macroscopic()

        final_mass = sum(solver.rho.data)
        # Relative mass conservation check
        rel_diff = abs(final_mass - initial_mass) / initial_mass
        self.assertLess(rel_diff, 1e-10)

    def test_lbm_obstacle_drag_generation(self):
        # Flow past cylinder should produce non-zero drag force
        nx, ny = 40, 25
        cyl = create_cylinder_obstacle(nx, ny, center_x=12, center_y=12, radius=4)
        solver = LBMSolver(nx=nx, ny=ny, viscosity=0.02, inflow_velocity=0.08, obstacle=cyl)

        # Run 25 steps to establish boundary momentum exchange
        solver.run_steps(25)

        cd, cl = solver.get_aerodynamic_coefficients(characteristic_length=8.0)
        # Drag must be positive
        self.assertGreater(solver.drag_force, 0.0)
        self.assertGreater(cd, 0.0)


if __name__ == "__main__":
    unittest.main()
