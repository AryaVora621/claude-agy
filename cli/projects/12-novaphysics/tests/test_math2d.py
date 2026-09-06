"""
Unit tests for NovaPhysics 2D math: Vec2, Mat22, and Transform2D.
"""

import math
import unittest
from novaphysics.math2d import Vec2, Mat22, Transform2D, scalar_cross_vec


class TestMath2D(unittest.TestCase):
    def test_vec2_arithmetic(self):
        v1 = Vec2(3.0, 4.0)
        v2 = Vec2(1.0, 2.0)

        # addition / subtraction
        self.assertEqual(v1 + v2, Vec2(4.0, 6.0))
        self.assertEqual(v1 - v2, Vec2(2.0, 2.0))
        self.assertEqual(-v1, Vec2(-3.0, -4.0))

        # scalar multiplication / division
        self.assertEqual(v1 * 2.0, Vec2(6.0, 8.0))
        self.assertEqual(2.0 * v1, Vec2(6.0, 8.0))
        self.assertEqual(v1 / 2.0, Vec2(1.5, 2.0))

        # length and normalization
        self.assertAlmostEqual(v1.length(), 5.0)
        self.assertAlmostEqual(v1.length_sq(), 25.0)
        norm = v1.normalized()
        self.assertAlmostEqual(norm.length(), 1.0)
        self.assertAlmostEqual(norm.x, 0.6)
        self.assertAlmostEqual(norm.y, 0.8)

        # dot and cross products
        self.assertAlmostEqual(v1.dot(v2), 3.0 * 1.0 + 4.0 * 2.0)  # 11
        # 3*2 - 4*1 = 2
        self.assertAlmostEqual(v1.cross(v2), 2.0)

        # perpendicular and rotation
        self.assertEqual(Vec2(1.0, 0.0).perp(), Vec2(0.0, 1.0))
        rot = Vec2(1.0, 0.0).rotated(math.pi * 0.5)
        self.assertAlmostEqual(rot.x, 0.0, places=6)
        self.assertAlmostEqual(rot.y, 1.0, places=6)

    def test_mat22_operations(self):
        # 90-degree rotation matrix
        m = Mat22.from_angle(math.pi * 0.5)
        v = Vec2(1.0, 0.0)
        v_rot = m.mul_vec(v)
        self.assertAlmostEqual(v_rot.x, 0.0, places=6)
        self.assertAlmostEqual(v_rot.y, 1.0, places=6)

        # Inverse of rotation is transpose
        m_inv = m.inverted()
        v_back = m_inv.mul_vec(v_rot)
        self.assertAlmostEqual(v_back.x, 1.0, places=6)
        self.assertAlmostEqual(v_back.y, 0.0, places=6)

    def test_transform2d(self):
        t = Transform2D(Vec2(10.0, 5.0), angle=math.pi * 0.5)
        local_pt = Vec2(1.0, 0.0)
        # Rotated 90 deg -> (0, 1) + (10, 5) -> (10, 6)
        world_pt = t.transform_point(local_pt)
        self.assertAlmostEqual(world_pt.x, 10.0, places=6)
        self.assertAlmostEqual(world_pt.y, 6.0, places=6)

        # Invert transform
        back = t.inverse_transform_point(world_pt)
        self.assertAlmostEqual(back.x, 1.0, places=6)
        self.assertAlmostEqual(back.y, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
