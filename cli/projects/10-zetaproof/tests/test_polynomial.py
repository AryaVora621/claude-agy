"""
Unit tests for ZetaProof Polynomial Calculus over F_p[X].
"""

import unittest
from zetaproof.field import FieldElement, TEST_PRIME
from zetaproof.polynomial import Polynomial, lagrange_interpolation


class TestPolynomial(unittest.TestCase):
    def test_polynomial_arithmetic(self):
        p = TEST_PRIME
        # P1(X) = 3 + 2X + X^2
        p1 = Polynomial([3, 2, 1], p)
        # P2(X) = 1 + 4X
        p2 = Polynomial([1, 4], p)

        # Sum: 4 + 6X + X^2
        sum_p = p1 + p2
        self.assertEqual(sum_p, Polynomial([4, 6, 1], p))
        self.assertEqual(sum_p.degree, 2)

        # Diff: 2 - 2X + X^2
        diff_p = p1 - p2
        self.assertEqual(diff_p, Polynomial([2, -2, 1], p))

        # Mul: (3 + 2X + X^2)(1 + 4X) = 3 + 12X + 2X + 8X^2 + X^2 + 4X^3
        #     = 3 + 14X + 9X^2 + 4X^3
        prod_p = p1 * p2
        self.assertEqual(prod_p, Polynomial([3, 14, 9, 4], p))
        self.assertEqual(prod_p.degree, 3)

    def test_evaluation(self):
        p = TEST_PRIME
        # P(X) = 5 + 3X + 2X^2
        poly = Polynomial([5, 3, 2], p)
        # At X = 0 -> 5
        self.assertEqual(poly(0), FieldElement(5, p))
        # At X = 2 -> 5 + 6 + 8 = 19
        self.assertEqual(poly(2), FieldElement(19, p))
        # At X = 10 -> 5 + 30 + 200 = 235
        self.assertEqual(poly(10), FieldElement(235, p))

    def test_polynomial_division(self):
        p = TEST_PRIME
        # Let A(X) = X^3 - 2X^2 - 4
        # Let B(X) = X - 3
        # A(X) = (X - 3)(X^2 + X + 3) + 5
        poly_a = Polynomial([-4, 0, -2, 1], p)
        poly_b = Polynomial([-3, 1], p)

        quot, rem = divmod(poly_a, poly_b)
        expected_quot = Polynomial([3, 1, 1], p)
        expected_rem = Polynomial([5], p)

        self.assertEqual(quot, expected_quot)
        self.assertEqual(rem, expected_rem)
        self.assertEqual(poly_b * quot + rem, poly_a)

    def test_vanishing_polynomial_and_exact_division(self):
        p = TEST_PRIME
        roots = [1, 2, 3, 4]
        z = Polynomial.from_roots(roots, p)
        # Check Z(r) == 0 for each root
        for r in roots:
            self.assertEqual(z(r), FieldElement.zero(p))

        # Multiply by arbitrary polynomial Q(X)
        q = Polynomial([7, 5, 2], p)
        h = z * q
        # Exact division should have zero remainder
        quot = h / z
        self.assertEqual(quot, q)

    def test_lagrange_interpolation(self):
        p = TEST_PRIME
        # Points: (1, 3), (2, 7), (3, 13)
        # Notice y = X^2 + X + 1 -> (1->3, 2->7, 3->13)
        points = [(1, 3), (2, 7), (3, 13)]
        poly = lagrange_interpolation(points, p)

        self.assertEqual(poly, Polynomial([1, 1, 1], p))
        for x, y in points:
            self.assertEqual(poly(x), FieldElement(y, p))


if __name__ == "__main__":
    unittest.main()
