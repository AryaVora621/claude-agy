"""
Unit tests for NovaPhysics shapes, AABB, mass, inertia, and support mappings.
"""

import math
import unittest
from novaphysics.math2d import Vec2, Transform2D
from novaphysics.shapes import AABB, Circle, Polygon, Box
from novaphysics.body import RigidBody, BodyType, Material


class TestShapesAndBodies(unittest.TestCase):
    def test_aabb_overlap_and_union(self):
        box1 = AABB(Vec2(0.0, 0.0), Vec2(2.0, 2.0))
        box2 = AABB(Vec2(1.0, 1.0), Vec2(3.0, 3.0))
        box3 = AABB(Vec2(5.0, 5.0), Vec2(6.0, 6.0))

        self.assertTrue(box1.overlaps(box2))
        self.assertFalse(box1.overlaps(box3))

        union_box = box1.union(box2)
        self.assertEqual(union_box.lower_bound, Vec2(0.0, 0.0))
        self.assertEqual(union_box.upper_bound, Vec2(3.0, 3.0))
        self.assertEqual(union_box.perimeter(), 12.0)  # 2 * (3 + 3)

    def test_circle_properties(self):
        c = Circle(radius=2.0)
        mass, inertia, com = c.compute_mass(density=1.0)
        expected_mass = math.pi * 4.0
        self.assertAlmostEqual(mass, expected_mass, places=4)
        # 0.5 * m * r^2
        self.assertAlmostEqual(inertia, 0.5 * expected_mass * 4.0, places=4)

        t = Transform2D(Vec2(5.0, 5.0))
        supp = c.support(Vec2(1.0, 0.0), t)
        self.assertEqual(supp, Vec2(7.0, 5.0))

    def test_box_and_polygon_support(self):
        # 4x2 box centered at origin
        box = Box(width=4.0, height=2.0)
        mass, inertia, com = box.compute_mass(density=1.0)
        self.assertAlmostEqual(mass, 8.0, places=4)  # 4 * 2 = 8
        # I = m * (w^2 + h^2) / 12 = 8 * (16 + 4) / 12 = 160 / 12 = 13.3333
        self.assertAlmostEqual(inertia, 13.3333, places=3)

        # Support in direction (1, 1) should be top-right corner (2, 1)
        t = Transform2D(Vec2(0.0, 0.0))
        supp = box.support(Vec2(1.0, 1.0), t)
        self.assertEqual(supp, Vec2(2.0, 1.0))

    def test_rigid_body_dynamics(self):
        box = Box(width=2.0, height=2.0)
        body = RigidBody(box, position=Vec2(0.0, 0.0), body_type=BodyType.DYNAMIC)

        self.assertGreater(body.mass, 0.0)
        self.assertGreater(body.inertia, 0.0)

        # Apply impulse at center
        body.apply_impulse(Vec2(10.0, 0.0))
        self.assertAlmostEqual(body.velocity.x, 10.0 * body.inv_mass)
        self.assertAlmostEqual(body.angular_velocity, 0.0)

        # Apply impulse off-center to generate torque and spin
        body.apply_impulse(Vec2(0.0, 10.0), world_point=Vec2(1.0, 0.0))
        self.assertGreater(body.angular_velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
