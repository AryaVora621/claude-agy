"""
Unit Tests for AeroFlow Eulerian Navier-Stokes Projection Solver.
"""

import unittest
import math
from aeroflow.types import ObstacleMask
from aeroflow.navier_stokes import NavierStokesSolver


class TestNavierStokesSolver(unittest.TestCase):
    def test_incompressibility_projection(self):
        # After projection step, velocity divergence must be near zero
        nx, ny = 20, 20
        solver = NavierStokesSolver(nx=nx, ny=ny, dx=1.0, dt=0.05, viscosity=0.005, inflow_velocity=1.0)

        # Set a divergence-rich synthetic velocity field
        for y in range(ny):
            for x in range(nx):
                solver.velocity.u.set(x, y, math.sin(x * 0.5) * 1.0)
                solver.velocity.v.set(x, y, math.cos(y * 0.5) * 1.0)

        init_div = solver.get_max_divergence(interior_only=True)

        # Run one full step containing advection, diffusion, pressure Poisson, and projection
        solver.step()

        # Check maximum divergence reduction
        max_div = solver.get_max_divergence(interior_only=True)
        # Should reduce initial divergence by more than half
        self.assertLess(max_div, 0.5 * init_div)
        self.assertLess(max_div, 0.25)

    def test_uniform_flow_preservation(self):
        # Uniform flow in open channel should remain stable
        nx, ny = 15, 15
        u_in = 1.2
        solver = NavierStokesSolver(nx=nx, ny=ny, dx=1.0, dt=0.02, viscosity=0.001, inflow_velocity=u_in)

        for _ in range(5):
            solver.step()

        # In mid-channel, u should remain close to u_in
        mid_u = solver.velocity.u.get(7, 7)
        self.assertGreater(mid_u, 0.8)


if __name__ == "__main__":
    unittest.main()
