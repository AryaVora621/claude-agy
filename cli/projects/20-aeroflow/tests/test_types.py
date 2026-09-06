"""
Unit Tests for AeroFlow Fundamental Geometric & Vector Types.
"""

import unittest
import math
from aeroflow.types import Vector2D, Grid2D, VectorField2D, ObstacleMask, FluidParams


class TestVector2D(unittest.TestCase):
    def test_vector_arithmetic(self):
        v1 = Vector2D(3.0, 4.0)
        v2 = Vector2D(1.0, 2.0)

        # Addition & Subtraction
        v_add = v1 + v2
        self.assertAlmostEqual(v_add.x, 4.0)
        self.assertAlmostEqual(v_add.y, 6.0)

        v_sub = v1 - v2
        self.assertAlmostEqual(v_sub.x, 2.0)
        self.assertAlmostEqual(v_sub.y, 2.0)

        # Scaling
        v_mul = v1 * 2.0
        self.assertAlmostEqual(v_mul.x, 6.0)
        self.assertAlmostEqual(v_mul.y, 8.0)

        v_div = v1 / 2.0
        self.assertAlmostEqual(v_div.x, 1.5)
        self.assertAlmostEqual(v_div.y, 2.0)

    def test_dot_and_cross(self):
        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)
        self.assertAlmostEqual(v1.dot(v2), 0.0)
        self.assertAlmostEqual(v1.cross_2d(v2), 1.0)

    def test_norm_and_normalization(self):
        v = Vector2D(3.0, 4.0)
        self.assertAlmostEqual(v.norm_sq(), 25.0)
        self.assertAlmostEqual(v.norm(), 5.0)

        v_norm = v.normalized()
        self.assertAlmostEqual(v_norm.norm(), 1.0)
        self.assertAlmostEqual(v_norm.x, 0.6)
        self.assertAlmostEqual(v_norm.y, 0.8)


class TestGrid2D(unittest.TestCase):
    def test_grid_indexing(self):
        g = Grid2D(nx=10, ny=8, dx=1.0, initial_value=0.0)
        g.set(3, 4, 42.0)
        self.assertAlmostEqual(g.get(3, 4), 42.0)
        self.assertAlmostEqual(g.get(0, 0), 0.0)

        # Clamping
        self.assertAlmostEqual(g.get(-5, 4), g.get(0, 4))
        self.assertAlmostEqual(g.get(15, 4), g.get(9, 4))

    def test_bilinear_interpolation(self):
        g = Grid2D(nx=4, ny=4, dx=1.0, initial_value=0.0)
        # Set corners of a cell
        g.set(1, 1, 10.0)
        g.set(2, 1, 20.0)
        g.set(1, 2, 10.0)
        g.set(2, 2, 20.0)

        # Sample at exact center (1.5, 1.5)
        val = g.sample_bilinear(1.5, 1.5)
        self.assertAlmostEqual(val, 15.0)


class TestVectorField2D(unittest.TestCase):
    def test_divergence_and_vorticity(self):
        vf = VectorField2D(nx=20, ny=20, dx=1.0)
        # Uniform shear flow: u = y, v = 0 -> du/dy = 1, div = 0, vort = -1
        for y in range(20):
            for x in range(20):
                vf.u.set(x, y, float(y))
                vf.v.set(x, y, 0.0)

        div = vf.compute_divergence()
        vort = vf.compute_vorticity()

        # In interior, divergence should be zero
        for y in range(2, 18):
            for x in range(2, 18):
                self.assertAlmostEqual(div.get(x, y), 0.0, places=5)
                # omega = dv/dx - du/dy = 0 - 1 = -1
                self.assertAlmostEqual(vort.get(x, y), -1.0, places=5)


class TestObstacleMask(unittest.TestCase):
    def test_obstacle_mask(self):
        mask = ObstacleMask(nx=15, ny=15)
        mask.set_solid(5, 5, True)
        self.assertTrue(mask.is_solid(5, 5))
        self.assertFalse(mask.is_solid(0, 0))

        mask.update_boundary_list()
        self.assertIn((5, 5), mask.boundary_nodes)


if __name__ == "__main__":
    unittest.main()
