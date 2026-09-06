"""
Unit tests for GeoPrism Computational Geometry Engine.
"""

import math
import unittest
from geoprism.geometry import (
    orient2d, convex_hull, polygon_area, point_in_polygon,
    Triangle, delaunay_triangulation, voronoi_diagram, euclidean_dist
)


class TestGeometry(unittest.TestCase):
    def test_orient2d(self):
        p = (0.0, 0.0)
        q = (1.0, 0.0)
        r_left = (0.5, 1.0)
        r_right = (0.5, -1.0)
        r_collinear = (2.0, 0.0)

        self.assertGreater(orient2d(p, q, r_left), 0.0)
        self.assertLess(orient2d(p, q, r_right), 0.0)
        self.assertAlmostEqual(orient2d(p, q, r_collinear), 0.0)

    def test_convex_hull(self):
        # 4 corners of a square plus interior points
        points = [
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
            (2.0, 3.0), (5.0, 5.0), (8.0, 2.0), (1.0, 9.0)
        ]
        hull = convex_hull(points)
        self.assertEqual(len(hull), 4)
        hull_set = set(hull)
        self.assertEqual(hull_set, {(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)})

        # Area of hull should be 100
        area = polygon_area(hull)
        self.assertAlmostEqual(abs(area), 100.0)

    def test_polygon_area_and_pip(self):
        # Triangle
        tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        self.assertAlmostEqual(abs(polygon_area(tri)), 6.0)

        # Point in polygon
        self.assertTrue(point_in_polygon((1.0, 1.0), tri))
        self.assertTrue(point_in_polygon((0.0, 1.5), tri))  # on edge
        self.assertFalse(point_in_polygon((3.0, 3.0), tri))  # outside

    def test_delaunay_triangulation(self):
        # 4 corners of unit square
        square_pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        triangles = delaunay_triangulation(square_pts)
        self.assertEqual(len(triangles), 2)

        # Total area of triangles must equal area of square (1.0)
        total_area = sum(abs(polygon_area([t.a, t.b, t.c])) for t in triangles)
        self.assertAlmostEqual(total_area, 1.0)

        # Delaunay empty circumcircle property
        pts = [
            (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0),
            (1.0, 1.0), (0.5, 1.5), (1.5, 0.5)
        ]
        delaunay_tris = delaunay_triangulation(pts)
        self.assertGreater(len(delaunay_tris), 0)

        for t in delaunay_tris:
            for p in pts:
                if p in (t.a, t.b, t.c):
                    continue
                # No other input point should be strictly inside circumcircle
                dist_to_cc = euclidean_dist(p, t.circumcenter)
                self.assertGreaterEqual(
                    dist_to_cc,
                    t.circumradius - 1e-7,
                    f"Point {p} violated circumcircle of {t}"
                )

    def test_voronoi_diagram(self):
        pts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
        vor = voronoi_diagram(pts)

        # Should have vertices and cells
        self.assertGreater(len(vor.vertices), 0)
        self.assertEqual(len(vor.cells), len(pts))
        # Center point (1, 1) has fully enclosed cell
        center_cell = vor.cells[(1.0, 1.0)]
        self.assertGreaterEqual(len(center_cell), 3)


if __name__ == "__main__":
    unittest.main()
