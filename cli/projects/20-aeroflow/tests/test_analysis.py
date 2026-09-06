"""
Unit Tests for AeroFlow Hydrodynamic Analysis & Visualizers.
"""

import unittest
from aeroflow.types import Grid2D, VectorField2D, ObstacleMask
from aeroflow.analysis import (
    compute_stream_function,
    trace_streamline,
    compute_enstrophy,
    compute_q_criterion,
)
from aeroflow.visualizer import (
    BrailleFlowCanvas,
    render_vorticity_field,
    render_velocity_vector_field,
)


class TestAnalysis(unittest.TestCase):
    def test_enstrophy_calculation(self):
        vort = Grid2D(nx=10, ny=10, dx=1.0, initial_value=2.0)
        # Total enstrophy = 0.5 * sum(omega^2) * dx^2 = 0.5 * 100 * 4 = 200.0
        e = compute_enstrophy(vort)
        self.assertAlmostEqual(e, 200.0)

    def test_streamline_tracing(self):
        nx, ny = 20, 20
        vf = VectorField2D(nx=nx, ny=ny, dx=1.0, u_init=1.0, v_init=0.0)
        obs = ObstacleMask(nx=nx, ny=ny)

        # Trace from (2.0, 10.0)
        path = trace_streamline(vf, obs, seed_x=2.0, seed_y=10.0, max_steps=10, dt=1.0)
        self.assertGreater(len(path), 5)
        # Horizontal flow: y should remain 10.0, x should increase
        for pt in path:
            self.assertAlmostEqual(pt[1], 10.0)
        self.assertGreater(path[-1][0], path[0][0])

    def test_q_criterion_vortex_core(self):
        # Pure solid-body rotation u = -y, v = x
        # du/dx = 0, du/dy = -1, dv/dx = 1, dv/dy = 0
        # Q = -(du/dx*dv/dy - du/dy*dv/dx) = -(0 - (-1)(1)) = +1.0 > 0 (vortex core!)
        nx, ny = 15, 15
        vf = VectorField2D(nx=nx, ny=ny, dx=1.0)
        for y in range(ny):
            for x in range(nx):
                vf.u.set(x, y, -float(y - 7))
                vf.v.set(x, y, float(x - 7))

        q_grid = compute_q_criterion(vf)
        # In interior, Q should be positive
        for y in range(2, 13):
            for x in range(2, 13):
                self.assertGreater(q_grid.get(x, y), 0.5)

    def test_braille_canvas_rendering(self):
        canvas = BrailleFlowCanvas(char_width=20, char_height=10)
        canvas.set_pixel(5, 5)
        canvas.mark_solid(10, 5)
        lines = canvas.render()

        self.assertEqual(len(lines), 10)
        self.assertTrue(any("█" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
