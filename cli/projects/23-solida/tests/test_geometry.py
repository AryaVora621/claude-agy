"""
Unit tests for Solida 3D Differential Geometry and Affine Transformations.
"""

import math
import unittest
from solida.geometry import (
    Vector3D,
    Matrix4x4,
    Quaternion,
    Ray3D,
    Plane,
    BoundingBox3D,
)


class TestGeometry(unittest.TestCase):
    def test_vector_operations(self):
        v1 = Vector3D(1.0, 2.0, 3.0)
        v2 = Vector3D(4.0, 5.0, 6.0)

        # Addition, subtraction, negation
        self.assertEqual(v1 + v2, Vector3D(5.0, 7.0, 9.0))
        self.assertEqual(v2 - v1, Vector3D(3.0, 3.0, 3.0))
        self.assertEqual(-v1, Vector3D(-1.0, -2.0, -3.0))

        # Scalar multiplication and division
        self.assertEqual(v1 * 2.0, Vector3D(2.0, 4.0, 6.0))
        self.assertEqual(v1 / 2.0, Vector3D(0.5, 1.0, 1.5))

        # Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertAlmostEqual(v1.dot(v2), 32.0)

        # Cross product
        cross = v1.cross(v2)
        # (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        self.assertEqual(cross, Vector3D(-3.0, 6.0, -3.0))
        self.assertAlmostEqual(cross.dot(v1), 0.0)
        self.assertAlmostEqual(cross.dot(v2), 0.0)

        # Norm and normalization
        v3 = Vector3D(0.0, 3.0, 4.0)
        self.assertAlmostEqual(v3.norm(), 5.0)
        self.assertAlmostEqual(v3.normalized().norm(), 1.0)
        self.assertEqual(v3.normalized(), Vector3D(0.0, 0.6, 0.8))

        # Angular distance
        vx = Vector3D(1, 0, 0)
        vy = Vector3D(0, 1, 0)
        self.assertAlmostEqual(vx.angle_to(vy), math.pi / 2.0)

        # Projection & rejection
        proj = vy.project_onto(vy)
        self.assertAlmostEqual((proj - vy).norm(), 0.0)

    def test_matrix_transformations(self):
        ident = Matrix4x4.identity()
        p = Vector3D(2, 3, 4)
        self.assertEqual(ident.transform_point(p), p)

        # Translation
        trans = Matrix4x4.translation(10, -5, 2)
        p_trans = trans.transform_point(p)
        self.assertEqual(p_trans, Vector3D(12, -2, 6))

        # Scaling
        scale = Matrix4x4.scaling(2, 3, 4)
        p_scale = scale.transform_point(p)
        self.assertEqual(p_scale, Vector3D(4, 9, 16))

        # Rotations
        rot_z = Matrix4x4.rotation_z(math.pi / 2.0)
        p_rot = rot_z.transform_point(Vector3D(1, 0, 0))
        self.assertAlmostEqual(p_rot.x, 0.0)
        self.assertAlmostEqual(p_rot.y, 1.0)
        self.assertAlmostEqual(p_rot.z, 0.0)

        # Inversion
        comb = trans @ scale @ rot_z
        comb_inv = comb.inverse()
        ident_res = comb @ comb_inv
        for r in range(4):
            for c in range(4):
                expected = 1.0 if r == c else 0.0
                self.assertAlmostEqual(ident_res.data[r][c], expected, places=5)

    def test_quaternion_rotations(self):
        # 90 degree rotation about Z axis
        q = Quaternion.from_axis_angle(Vector3D(0, 0, 1), math.pi / 2.0)
        p = Vector3D(1, 0, 0)
        p_rot = q.rotate_vector(p)
        self.assertAlmostEqual(p_rot.x, 0.0)
        self.assertAlmostEqual(p_rot.y, 1.0)
        self.assertAlmostEqual(p_rot.z, 0.0)

        # Matrix conversion equivalence
        mat = q.to_matrix()
        p_mat = mat.transform_point(p)
        self.assertAlmostEqual((p_rot - p_mat).norm(), 0.0)

        # Slerp
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(Vector3D(0, 0, 1), math.pi)
        q_mid = q1.slerp(q2, 0.5)
        p_mid = q_mid.rotate_vector(p)
        self.assertAlmostEqual(p_mid.x, 0.0)
        self.assertAlmostEqual(p_mid.y, 1.0)

    def test_plane_and_ray(self):
        plane = Plane(Vector3D(0, 0, 1), -5.0)  # z = 5
        self.assertAlmostEqual(plane.signed_distance(Vector3D(2, 2, 7)), 2.0)
        self.assertAlmostEqual(plane.signed_distance(Vector3D(2, 2, 3)), -2.0)

        ray = Ray3D(Vector3D(0, 0, 0), Vector3D(0, 0, 1))
        hit = ray.intersect_plane(plane)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.x, 0.0)
        self.assertAlmostEqual(hit.y, 0.0)
        self.assertAlmostEqual(hit.z, 5.0)

        # Ray-triangle intersection (Möller-Trumbore)
        v0 = Vector3D(-1, -1, 5)
        v1 = Vector3D(1, -1, 5)
        v2 = Vector3D(0, 1, 5)
        t_hit = ray.intersect_triangle(v0, v1, v2)
        self.assertIsNotNone(t_hit)
        self.assertAlmostEqual(t_hit[0], 5.0)

    def test_bounding_box(self):
        bbox = BoundingBox3D()
        bbox.include_point(Vector3D(0, 0, 0))
        bbox.include_point(Vector3D(10, 20, 30))

        self.assertEqual(bbox.min_pt, Vector3D(0, 0, 0))
        self.assertEqual(bbox.max_pt, Vector3D(10, 20, 30))
        self.assertEqual(bbox.center(), Vector3D(5, 10, 15))
        self.assertEqual(bbox.extents(), Vector3D(10, 20, 30))
        self.assertAlmostEqual(bbox.volume(), 6000.0)

        other_box = BoundingBox3D(Vector3D(5, 5, 5), Vector3D(15, 25, 35))
        self.assertTrue(bbox.intersects(other_box))

        disjoint_box = BoundingBox3D(Vector3D(100, 100, 100), Vector3D(110, 110, 110))
        self.assertFalse(bbox.intersects(disjoint_box))


if __name__ == "__main__":
    unittest.main()
