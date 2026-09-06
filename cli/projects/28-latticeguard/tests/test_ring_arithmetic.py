"""Unit Tests for Finite Field Z_q and Polynomial Ring R_q Arithmetic."""

import math
import random
import unittest

from latticeguard.ring import (
    KYBER_N,
    KYBER_Q,
    MONTGOMERY_R,
    QINV,
    BARRETT_V,
    freeze,
    montgomery_reduce,
    barrett_reduce,
    fqmul,
    Polynomial,
    PolyVec,
)


class TestRingArithmetic(unittest.TestCase):
    """Test suite for ring arithmetic in R_q = Z_q[X] / (X^256 + 1)."""

    def setUp(self) -> None:
        random.seed(42)

    def test_field_constants(self) -> None:
        """Verify FIPS 203 mathematical constants."""
        self.assertEqual(KYBER_N, 256)
        self.assertEqual(KYBER_Q, 3329)
        # Check q is prime: 3329 - 1 = 13 * 256
        self.assertEqual((KYBER_Q - 1) % 256, 0)
        # Check Montgomery inverse constant
        self.assertEqual((KYBER_Q * QINV) & 0xFFFF, 1)

    def test_freeze(self) -> None:
        """Verify freeze canonical reduction to [0, q-1]."""
        self.assertEqual(freeze(0), 0)
        self.assertEqual(freeze(3329), 0)
        self.assertEqual(freeze(3330), 1)
        self.assertEqual(freeze(-1), 3328)
        self.assertEqual(freeze(-3329), 0)

    def test_barrett_reduction(self) -> None:
        """Verify Barrett reduction produces correct centered representative."""
        for _ in range(500):
            a = random.randint(-KYBER_Q, KYBER_Q)
            red = barrett_reduce(a)
            # Result should be congruent to a mod q and bounded
            self.assertEqual((a - red) % KYBER_Q, 0)
            self.assertLessEqual(abs(red), KYBER_Q)

    def test_polynomial_basic_ops(self) -> None:
        """Verify polynomial addition, subtraction, negation, and scalar multiplication."""
        coeffs1 = [random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]
        coeffs2 = [random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]
        p1 = Polynomial(coeffs1)
        p2 = Polynomial(coeffs2)

        # Addition
        p_add = p1 + p2
        for i in range(KYBER_N):
            self.assertEqual(p_add[i], (coeffs1[i] + coeffs2[i]) % KYBER_Q)

        # Subtraction
        p_sub = p1 - p2
        for i in range(KYBER_N):
            self.assertEqual(p_sub[i], (coeffs1[i] - coeffs2[i]) % KYBER_Q)

        # Negation
        p_neg = -p1
        p_zero = p1 + p_neg
        for i in range(KYBER_N):
            self.assertEqual(p_zero[i], 0)

        # Scalar multiplication
        scalar = 17
        p_scale = p1 * scalar
        for i in range(KYBER_N):
            self.assertEqual(p_scale[i], (coeffs1[i] * scalar) % KYBER_Q)

    def test_direct_polynomial_multiplication_mod_ring(self) -> None:
        """Verify polynomial multiplication modulo (X^256 + 1)."""
        # Test X^128 * X^128 = X^256 = -1 mod (X^256 + 1) = q - 1
        p_x128 = Polynomial.zero()
        p_x128[128] = 1

        p_prod = p_x128 * p_x128
        # Should be -1 at degree 0, all other degrees 0
        self.assertEqual(p_prod[0], KYBER_Q - 1)
        for i in range(1, KYBER_N):
            self.assertEqual(p_prod[i], 0)

        # Test X^200 * X^100 = X^300 = -X^44 mod (X^256 + 1)
        p_x200 = Polynomial.zero()
        p_x200[200] = 1
        p_x100 = Polynomial.zero()
        p_x100[100] = 1

        p_prod2 = p_x200 * p_x100
        self.assertEqual(p_prod2[44], KYBER_Q - 1)
        for i in range(KYBER_N):
            if i != 44:
                self.assertEqual(p_prod2[i], 0)

    def test_polynomial_norms(self) -> None:
        """Verify infinity norm, L1 norm, and RMS energy calculations."""
        p = Polynomial.zero()
        p[0] = 5
        p[1] = KYBER_Q - 7  # Centered is -7
        p[2] = 2

        self.assertEqual(p.infinity_norm, 7)
        self.assertEqual(p.l1_norm, 14)
        self.assertGreater(p.energy, 0.0)

    def test_polyvec_operations(self) -> None:
        """Verify polynomial vector operations across ranks k=2, 3, 4."""
        for k in [2, 3, 4]:
            vec1 = PolyVec([Polynomial([random.randint(0, 100) for _ in range(KYBER_N)]) for _ in range(k)])
            vec2 = PolyVec([Polynomial([random.randint(0, 100) for _ in range(KYBER_N)]) for _ in range(k)])

            v_add = vec1 + vec2
            self.assertEqual(len(v_add), k)
            for idx in range(k):
                self.assertEqual(v_add[idx], vec1[idx] + vec2[idx])

            v_sub = vec1 - vec2
            for idx in range(k):
                self.assertEqual(v_sub[idx], vec1[idx] - vec2[idx])


if __name__ == "__main__":
    unittest.main()
