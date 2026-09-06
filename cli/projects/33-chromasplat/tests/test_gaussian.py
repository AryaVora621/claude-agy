"""Unit tests for 3D Gaussian Splatting representation and quaternion mathematics."""

import math
import unittest

from chromasplat.gaussian import Gaussian3D, Quaternion


class TestGaussianAndQuaternion(unittest.TestCase):
    """Test suite for Quaternion and Gaussian3D primitives."""

    def test_quaternion_identity_and_normalization(self) -> None:
        q_id = Quaternion.identity()
        self.assertAlmostEqual(q_id.norm(), 1.0)
        self.assertEqual(q_id.w, 1.0)
        self.assertEqual(q_id.x, 0.0)

        q_unnorm = Quaternion(2.0, 2.0, 2.0, 2.0)
        q_norm = q_unnorm.normalized()
        self.assertAlmostEqual(q_norm.norm(), 1.0)
        self.assertAlmostEqual(q_norm.w, 0.5)

    def test_quaternion_axis_angle_and_rotation_matrix(self) -> None:
        # Rotate 90 degrees around Y axis
        q = Quaternion.from_axis_angle((0.0, 1.0, 0.0), math.pi * 0.5)
        r = q.to_rotation_matrix()

        # Check orthonormality: R * R^T = I
        for row in range(3):
            for col in range(3):
                dot = sum(r[row][k] * r[col][k] for k in range(3))
                expected = 1.0 if row == col else 0.0
                self.assertAlmostEqual(dot, expected, places=5)

        # Rotating unit X vector (1, 0, 0) around Y by 90 deg gives (0, 0, -1)
        vx, vy, vz = 1.0, 0.0, 0.0
        rx = r[0][0] * vx + r[0][1] * vy + r[0][2] * vz
        ry = r[1][0] * vx + r[1][1] * vy + r[1][2] * vz
        rz = r[2][0] * vx + r[2][1] * vy + r[2][2] * vz

        self.assertAlmostEqual(rx, 0.0, places=5)
        self.assertAlmostEqual(ry, 0.0, places=5)
        self.assertAlmostEqual(rz, -1.0, places=5)

    def test_quaternion_multiplication(self) -> None:
        q1 = Quaternion.from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.25)
        q2 = Quaternion.from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.25)
        q_prod = q1.multiply(q2)

        # Total rotation should be 90 degrees around Z
        q_expected = Quaternion.from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.5)
        self.assertAlmostEqual(q_prod.w, q_expected.w, places=5)
        self.assertAlmostEqual(q_prod.z, q_expected.z, places=5)

    def test_gaussian_covariance_positive_semidefinite(self) -> None:
        q = Quaternion.from_euler(0.3, 0.5, 0.2)
        g = Gaussian3D(
            position=(1.0, -2.0, 3.0),
            scale=(0.4, 0.8, 1.2),
            rotation=q,
        )
        cov = g.compute_covariance_3d()

        # Check symmetry: cov[i][j] == cov[j][i]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(cov[i][j], cov[j][i], places=6)

        # Check positive definiteness: v^T * cov * v > 0 for non-zero v
        test_vectors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.577, 0.577, 0.577),
            (-0.3, 0.8, 0.5),
        ]
        for v in test_vectors:
            quad = sum(v[i] * cov[i][j] * v[j] for i in range(3) for j in range(3))
            self.assertGreater(quad, 0.0)

    def test_gaussian_isotropic_and_volume(self) -> None:
        g = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.5)
        self.assertEqual(g.scale, (0.5, 0.5, 0.5))
        expected_vol = (4.0 / 3.0) * math.pi * (0.5**3)
        self.assertAlmostEqual(g.volume(), expected_vol, places=5)

    def test_gaussian_transforms(self) -> None:
        g = Gaussian3D.isotropic(position=(1.0, 2.0, 3.0), radius=1.0)
        g_trans = g.translate(2.0, -1.0, 4.0)
        self.assertEqual(g_trans.position, (3.0, 1.0, 7.0))

        g_scaled = g.scale_uniform(2.0)
        self.assertEqual(g_scaled.scale, (2.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
