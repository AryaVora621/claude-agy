"""
Unit tests for VeloSLAM Linear Algebra & Covariance Matrix Engine.
"""

import math
import unittest
from veloslam.linalg import (
    Vector, Matrix, normalize_angle, mahalanobis_distance, covariance_ellipse_2d
)


class TestLinalg(unittest.TestCase):
    def test_normalize_angle(self):
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(math.pi * 3.0), -math.pi)
        self.assertAlmostEqual(normalize_angle(-math.pi * 3.0), -math.pi)
        self.assertAlmostEqual(normalize_angle(2.0 * math.pi), 0.0)
        self.assertAlmostEqual(normalize_angle(0.5), 0.5)

    def test_vector_operations(self):
        v1 = Vector([1.0, 2.0, 3.0])
        v2 = Vector([4.0, 5.0, 6.0])

        # Addition & subtraction
        v_add = v1 + v2
        self.assertEqual(v_add.to_list(), [5.0, 7.0, 9.0])
        v_sub = v2 - v1
        self.assertEqual(v_sub.to_list(), [3.0, 3.0, 3.0])

        # Dot product & norm
        self.assertAlmostEqual(v1.dot(v2), 32.0)
        self.assertAlmostEqual(Vector([3.0, 4.0]).norm(), 5.0)

        # Outer product
        outer = v1.outer(v2)
        self.assertEqual(outer.rows, 3)
        self.assertEqual(outer.cols, 3)
        self.assertEqual(outer[0][0], 4.0)
        self.assertEqual(outer[2][2], 18.0)

    def test_matrix_operations(self):
        m1 = Matrix([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        m2 = Matrix([
            [2.0, 0.0],
            [1.0, 2.0]
        ])

        # Multiplication M @ M
        prod = m1 @ m2
        self.assertEqual(prod[0][0], 4.0)
        self.assertEqual(prod[0][1], 4.0)
        self.assertEqual(prod[1][0], 10.0)
        self.assertEqual(prod[1][1], 8.0)

        # Multiplication M @ v
        v = Vector([1.0, 2.0])
        mv = m1 @ v
        self.assertEqual(mv.to_list(), [5.0, 11.0])

        # Transpose
        tr = m1.transpose()
        self.assertEqual(tr[0][1], 3.0)
        self.assertEqual(tr[1][0], 2.0)

    def test_matrix_determinant(self):
        m2x2 = Matrix([[3.0, 8.0], [4.0, 6.0]])
        self.assertAlmostEqual(m2x2.det(), -14.0)

        m3x3 = Matrix([
            [6.0, 1.0, 1.0],
            [4.0, -2.0, 5.0],
            [2.0, 8.0, 7.0]
        ])
        # Det = 6(-14 - 40) - 1(28 - 10) + 1(32 - (-4)) = 6(-54) - 18 + 36 = -324 - 18 + 36 = -306
        self.assertAlmostEqual(m3x3.det(), -306.0)

    def test_matrix_inversion(self):
        # 2x2 analytical inverse
        m2x2 = Matrix([[4.0, 7.0], [2.0, 6.0]])
        inv2x2 = m2x2.inv()
        id2 = m2x2 @ inv2x2
        self.assertAlmostEqual(id2[0][0], 1.0, places=5)
        self.assertAlmostEqual(id2[0][1], 0.0, places=5)
        self.assertAlmostEqual(id2[1][0], 0.0, places=5)
        self.assertAlmostEqual(id2[1][1], 1.0, places=5)

        # 3x3 Gauss-Jordan inverse
        m3x3 = Matrix([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]
        ])
        inv3x3 = m3x3.inv()
        id3 = m3x3 @ inv3x3
        for r in range(3):
            for c in range(3):
                expected = 1.0 if r == c else 0.0
                self.assertAlmostEqual(id3[r][c], expected, places=5)

    def test_mahalanobis_distance(self):
        diff = Vector([2.0, -1.0])
        # Covariance S = diag(4.0, 1.0) -> S^-1 = diag(0.25, 1.0)
        s_inv = Matrix([[0.25, 0.0], [0.0, 1.0]])
        # D_M^2 = 2^2 * 0.25 + (-1)^2 * 1.0 = 1 + 1 = 2.0
        d_m2 = mahalanobis_distance(diff, s_inv)
        self.assertAlmostEqual(d_m2, 2.0)

    def test_covariance_ellipse_2d(self):
        # Diagonal covariance with variances 4.0 and 1.0 (std dev 2.0 and 1.0)
        cov = Matrix([[4.0, 0.0], [0.0, 1.0]])
        semi_maj, semi_min, ang = covariance_ellipse_2d(cov, chi2_val=1.0)
        self.assertAlmostEqual(semi_maj, 2.0)
        self.assertAlmostEqual(semi_min, 1.0)
        self.assertAlmostEqual(ang, 0.0)


if __name__ == "__main__":
    unittest.main()
