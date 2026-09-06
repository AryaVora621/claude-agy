"""
Unit tests for NovaPhysics GJK and EPA collision detection algorithms.
"""

import math
import unittest
from novaphysics.math2d import Vec2
from novaphysics.shapes import Box, Circle, Polygon
from novaphysics.body import RigidBody, BodyType
from novaphysics.gjk import gjk_intersect
from novaphysics.epa import epa_penetration
from novaphysics.contact import find_collision


class TestGjkEpa(unittest.TestCase):
    def test_gjk_intersecting_boxes(self):
        # Two 2x2 boxes overlapping by 0.5 units along x axis
        box_a = RigidBody(Box(2.0, 2.0), position=Vec2(0.0, 0.0))
        box_b = RigidBody(Box(2.0, 2.0), position=Vec2(1.5, 0.0))

        colliding, simplex = gjk_intersect(box_a, box_b)
        self.assertTrue(colliding)
        self.assertEqual(len(simplex), 3)

        # EPA should yield penetration ~ 0.5 and normal along x axis
        penetration, normal = epa_penetration(box_a, box_b, simplex)
        self.assertAlmostEqual(penetration, 0.5, places=3)
        self.assertAlmostEqual(abs(normal.x), 1.0, places=3)
        self.assertAlmostEqual(normal.y, 0.0, places=3)

    def test_gjk_separated_boxes(self):
        # Two 2x2 boxes separated by 1.0 unit gap
        box_a = RigidBody(Box(2.0, 2.0), position=Vec2(0.0, 0.0))
        box_b = RigidBody(Box(2.0, 2.0), position=Vec2(3.0, 0.0))

        colliding, simplex = gjk_intersect(box_a, box_b)
        self.assertFalse(colliding)

    def test_circle_circle_collision(self):
        # Two radius 1 circles separated by distance 1.5 (overlap = 0.5)
        c1 = RigidBody(Circle(1.0), position=Vec2(0.0, 0.0))
        c2 = RigidBody(Circle(1.0), position=Vec2(1.5, 0.0))

        manifold = find_collision(c1, c2)
        self.assertIsNotNone(manifold)
        self.assertAlmostEqual(manifold.points[0].penetration, 0.5, places=3)
        self.assertAlmostEqual(manifold.normal.x, 1.0, places=3)
        self.assertAlmostEqual(manifold.normal.y, 0.0, places=3)

    def test_circle_polygon_collision(self):
        # Circle of radius 1 colliding with 2x2 box
        poly = RigidBody(Box(2.0, 2.0), position=Vec2(0.0, 0.0))
        circ = RigidBody(Circle(1.0), position=Vec2(0.0, 1.8))  # Top edge of box at y=1.0, overlap = 0.2

        manifold = find_collision(poly, circ)
        self.assertIsNotNone(manifold)
        self.assertAlmostEqual(manifold.points[0].penetration, 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
