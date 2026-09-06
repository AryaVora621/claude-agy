"""
Unit tests for ZetaProof Elliptic Curve Cryptography.
"""

import unittest
from zetaproof.curve import BN254, Point, BilinearEngine


class TestEllipticCurve(unittest.TestCase):
    def test_generator_on_curve(self):
        g1 = BN254.generator_g1()
        self.assertTrue(BN254.is_on_curve(g1.x, g1.y))
        self.assertFalse(g1.is_infinity)

    def test_point_addition_and_doubling(self):
        g1 = BN254.generator_g1()
        # 2 * G1
        two_g1 = g1 + g1
        self.assertTrue(BN254.is_on_curve(two_g1.x, two_g1.y))
        self.assertNotEqual(g1, two_g1)

        # 3 * G1
        three_g1 = two_g1 + g1
        self.assertTrue(BN254.is_on_curve(three_g1.x, three_g1.y))

        # Scalar multiplication 3 * G1
        scalar_3g1 = g1 * 3
        self.assertEqual(three_g1, scalar_3g1)

    def test_identity_and_inverses(self):
        g1 = BN254.generator_g1()
        inf = Point.infinity(BN254)

        self.assertEqual(g1 + inf, g1)
        self.assertEqual(inf + g1, g1)

        # P + (-P) = O
        neg_g1 = -g1
        self.assertEqual(g1 + neg_g1, inf)

    def test_scalar_multiplication_properties(self):
        g1 = BN254.generator_g1()
        inf = Point.infinity(BN254)

        self.assertEqual(g1 * 0, inf)
        self.assertEqual(g1 * 1, g1)

        # Distributivity: (a + b) * G == a*G + b*G
        a = 5
        b = 7
        self.assertEqual(g1 * (a + b), (g1 * a) + (g1 * b))


if __name__ == "__main__":
    unittest.main()
