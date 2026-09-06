"""
Unit tests for Solida Constructive Solid Geometry (CSG) 3D Boolean Engine.
"""

import unittest
from solida.geometry import Vector3D
from solida.primitives import make_box, make_cylinder
from solida.csg import (
    csg_union,
    csg_difference,
    csg_intersection,
)


class TestCSG(unittest.TestCase):
    def test_csg_box_subtraction(self):
        # Base box: 10 x 10 x 10, Volume = 1000
        box_a = make_box(10, 10, 10, center=True, name="BaseBox")
        # Tool box: 4 x 4 x 20, Volume inside A = 4 * 4 * 10 = 160
        box_b = make_box(4, 4, 20, center=True, name="ToolBox")

        # Difference (A - B)
        diff = csg_difference(box_a, box_b, name="HollowBox")
        expected_diff_vol = 1000.0 - 160.0  # 840.0
        self.assertAlmostEqual(diff.volume(), expected_diff_vol, places=2)

    def test_csg_intersection(self):
        # Two offset boxes: [0, 10]^3 and [5, 15] x [0, 10] x [0, 10]
        # Overlap region: [5, 10] x [0, 10] x [0, 10] -> 5 * 10 * 10 = 500
        box1 = make_box(10, 10, 10, center=False, origin=Vector3D(0, 0, 0))
        box2 = make_box(10, 10, 10, center=False, origin=Vector3D(5, 0, 0))

        inter = csg_intersection(box1, box2, name="OverlapBox")
        self.assertAlmostEqual(inter.volume(), 500.0, places=2)

    def test_csg_union(self):
        # Two contiguous boxes: [0, 10]^3 and [10, 20] x [0, 10] x [0, 10]
        # Total union volume: 1000 + 1000 = 2000
        box1 = make_box(10, 10, 10, center=False, origin=Vector3D(0, 0, 0))
        box2 = make_box(10, 10, 10, center=False, origin=Vector3D(5, 0, 0))

        # Union of overlapping boxes: Vol(A) + Vol(B) - Vol(A & B) = 1000 + 1000 - 500 = 1500
        union_box = csg_union(box1, box2, name="UnionBox")
        self.assertAlmostEqual(union_box.volume(), 1500.0, delta=10.0)

    def test_csg_disjoint_operations(self):
        # Disjoint boxes
        b1 = make_box(2, 2, 2, center=True, origin=Vector3D(-20, 0, 0))
        b2 = make_box(2, 2, 2, center=True, origin=Vector3D(20, 0, 0))

        # Difference: A - B should still have Volume(A)
        diff = csg_difference(b1, b2)
        self.assertAlmostEqual(diff.volume(), 8.0, places=3)

        # Intersection: should have zero volume
        inter = csg_intersection(b1, b2)
        self.assertAlmostEqual(inter.volume(), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
