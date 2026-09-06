"""
Unit tests for Solida Non-Uniform Rational B-Splines (NURBS) Curves and Surfaces.
"""

import math
import unittest
from solida.geometry import Vector3D
from solida.nurbs import (
    create_clamped_knot_vector,
    basis_function,
    basis_derivative,
    NURBSCurve,
    NURBSSurface,
)


class TestNURBS(unittest.TestCase):
    def test_knot_vector_and_partition_of_unity(self):
        # Degree 3 with 5 control points -> 5 + 3 + 1 = 9 knots
        knots = create_clamped_knot_vector(num_ctrl_pts=5, degree=3)
        self.assertEqual(len(knots), 9)
        self.assertEqual(knots[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(knots[-4:], [1.0, 1.0, 1.0, 1.0])

        # Test partition of unity: sum of basis functions must equal 1.0 everywhere
        for step in range(21):
            u = step / 20.0
            sum_n = sum(basis_function(i, 3, u, knots) for i in range(5))
            self.assertAlmostEqual(sum_n, 1.0, places=6)

    def test_linear_curve(self):
        pts = [Vector3D(0, 0, 0), Vector3D(10, 0, 0)]
        curve = NURBSCurve(degree=1, control_points=pts)

        self.assertEqual(curve.point_at(0.0), Vector3D(0, 0, 0))
        self.assertEqual(curve.point_at(1.0), Vector3D(10, 0, 0))
        self.assertEqual(curve.point_at(0.5), Vector3D(5, 0, 0))

        # Tangent vector should be (1, 0, 0) everywhere
        t = curve.tangent_at(0.5)
        self.assertAlmostEqual(t.x, 1.0)
        self.assertAlmostEqual(t.y, 0.0)
        self.assertAlmostEqual(t.z, 0.0)

        # Curvature of straight line should be 0.0
        self.assertAlmostEqual(curve.curvature_at(0.5), 0.0)

    def test_cubic_bezier_curve(self):
        # Degree 3 curve
        ctrl_pts = [
            Vector3D(0, 0, 0),
            Vector3D(0, 10, 0),
            Vector3D(10, 10, 0),
            Vector3D(10, 0, 0),
        ]
        curve = NURBSCurve(degree=3, control_points=ctrl_pts)

        # Endpoints must strictly match first and last control points
        p0 = curve.point_at(0.0)
        p1 = curve.point_at(1.0)
        self.assertAlmostEqual((p0 - ctrl_pts[0]).norm(), 0.0)
        self.assertAlmostEqual((p1 - ctrl_pts[-1]).norm(), 0.0)

        # Symmetry at u = 0.5: X should be 5.0
        pmid = curve.point_at(0.5)
        self.assertAlmostEqual(pmid.x, 5.0)

    def test_exact_circular_arc(self):
        # Exact quarter circle of radius R = 5.0 using rational weights
        r = 5.0
        arc = NURBSCurve.create_circular_arc(
            radius=r,
            start_angle_rad=0.0,
            sweep_angle_rad=math.pi / 2.0,
        )

        # Verify radius at multiple sample points
        for step in range(21):
            u = step / 20.0
            pt = arc.point_at(u)
            dist_from_origin = math.sqrt(pt.x**2 + pt.y**2)
            self.assertAlmostEqual(dist_from_origin, r, places=4)
            self.assertAlmostEqual(pt.z, 0.0)

        # Exact curvature should be 1 / R = 0.2
        k_mid = arc.curvature_at(0.5)
        self.assertAlmostEqual(k_mid, 1.0 / r, places=3)

    def test_nurbs_surface(self):
        # 3x3 control net for a bicubic planar surface
        net = [
            [Vector3D(0, 0, 0), Vector3D(5, 0, 0), Vector3D(10, 0, 0)],
            [Vector3D(0, 5, 0), Vector3D(5, 5, 2), Vector3D(10, 5, 0)],
            [Vector3D(0, 10, 0), Vector3D(5, 10, 0), Vector3D(10, 10, 0)],
        ]
        surf = NURBSSurface(degree_u=2, degree_v=2, control_points=net)

        # Corners
        self.assertEqual(surf.point_at(0.0, 0.0), Vector3D(0, 0, 0))
        self.assertEqual(surf.point_at(1.0, 1.0), Vector3D(10, 10, 0))

        # Peak at center (u=0.5, v=0.5) should have positive Z
        center_pt = surf.point_at(0.5, 0.5)
        self.assertGreater(center_pt.z, 0.0)

        # Unit normal at center
        norm = surf.normal_at(0.5, 0.5)
        self.assertAlmostEqual(norm.norm(), 1.0)

        # Gaussian curvature should exist and be finite
        k_gauss = surf.gaussian_curvature_at(0.5, 0.5)
        self.assertTrue(math.isfinite(k_gauss))


if __name__ == "__main__":
    unittest.main()
