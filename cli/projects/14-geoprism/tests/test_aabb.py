"""
Unit tests for GeoPrism N-Dimensional AABB.
"""

import math
import unittest
from geoprism.aabb import AABB


class TestAABB(unittest.TestCase):
    def test_2d_properties(self):
        box = AABB((2.0, 3.0), (6.0, 8.0))
        self.assertEqual(box.dimensions, 2)
        self.assertAlmostEqual(box.volume(), 4.0 * 5.0)  # 20.0
        self.assertAlmostEqual(box.margin(), 4.0 + 5.0)   # 9.0
        self.assertEqual(box.center(), (4.0, 5.5))

    def test_containment_and_intersection(self):
        b1 = AABB((0.0, 0.0), (10.0, 10.0))
        b2 = AABB((2.0, 2.0), (5.0, 5.0))
        b3 = AABB((8.0, 8.0), (15.0, 15.0))
        b4 = AABB((20.0, 20.0), (25.0, 25.0))

        self.assertTrue(b1.contains_point((5.0, 5.0)))
        self.assertFalse(b1.contains_point((12.0, 5.0)))

        self.assertTrue(b1.contains_box(b2))
        self.assertFalse(b2.contains_box(b1))

        self.assertTrue(b1.intersects(b3))
        self.assertFalse(b1.intersects(b4))

        inter = b1.intersection(b3)
        self.assertIsNotNone(inter)
        self.assertEqual(inter.lower, (8.0, 8.0))
        self.assertEqual(inter.upper, (10.0, 10.0))
        self.assertAlmostEqual(inter.volume(), 4.0)

    def test_min_dist(self):
        box = AABB((0.0, 0.0), (10.0, 10.0))

        # Point inside box: min_dist = 0
        self.assertAlmostEqual(box.min_dist((5.0, 5.0)), 0.0)

        # Point directly to the right
        self.assertAlmostEqual(box.min_dist((15.0, 5.0)), 5.0)

        # Point diagonal to corner (10, 10)
        self.assertAlmostEqual(box.min_dist((13.0, 14.0)), 5.0)  # sqrt(3^2 + 4^2) = 5.0

    def test_min_max_dist(self):
        box = AABB((0.0, 0.0), (4.0, 4.0))
        point = (-1.0, 2.0)
        min_max = box.min_max_dist(point)
        min_d = box.min_dist(point)
        self.assertGreaterEqual(min_max, min_d)


if __name__ == "__main__":
    unittest.main()
