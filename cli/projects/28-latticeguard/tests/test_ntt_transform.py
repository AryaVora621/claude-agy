"""Unit Tests for Number Theoretic Transform (NTT) and Polynomial Multiplication."""

import random
import unittest

from latticeguard.ring import KYBER_N, KYBER_Q, Polynomial, PolyVec
from latticeguard.ntt import (
    ZETA,
    ZETAS,
    GAMMA_TABLE,
    INV_128,
    bitrev7,
    forward_ntt,
    inverse_ntt,
    poly_ntt_multiply,
    polyvec_forward_ntt,
    polyvec_inverse_ntt,
    polyvec_ntt_dot,
    matrix_vector_mul_ntt,
)


class TestNTTTransform(unittest.TestCase):
    """Test suite for Cooley-Tukey NTT, Gentleman-Sande INTT, and NTT multiplications."""

    def setUp(self) -> None:
        random.seed(123)

    def test_roots_of_unity(self) -> None:
        """Verify order and powers of primitive root zeta = 17."""
        self.assertEqual(pow(ZETA, 128, KYBER_Q), KYBER_Q - 1)  # zeta^128 = -1 mod q
        self.assertEqual(pow(ZETA, 256, KYBER_Q), 1)            # zeta^256 = 1 mod q
        # Inverse of 128 modulo 3329
        self.assertEqual((128 * INV_128) % KYBER_Q, 1)

    def test_bitrev7(self) -> None:
        """Verify 7-bit reversal permutation."""
        self.assertEqual(bitrev7(0), 0)
        self.assertEqual(bitrev7(1), 64)
        self.assertEqual(bitrev7(64), 1)
        self.assertEqual(bitrev7(127), 127)
        # Involutive: bitrev(bitrev(x)) == x
        for i in range(128):
            self.assertEqual(bitrev7(bitrev7(i)), i)

    def test_forward_inverse_ntt_roundtrip(self) -> None:
        """Verify INTT(NTT(p)) == p for random polynomials."""
        for _ in range(50):
            coeffs = [random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]
            p = Polynomial(coeffs)
            p_hat = forward_ntt(p)
            p_rec = inverse_ntt(p_hat)
            self.assertEqual(p_rec, p)

    def test_ntt_multiplication_vs_direct_convolution(self) -> None:
        """Verify that NTT multiplication exactly matches direct O(n^2) ring multiplication."""
        for _ in range(10):
            p1 = Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)])
            p2 = Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)])

            # Direct multiplication mod (X^256 + 1)
            p_direct = p1 * p2

            # NTT multiplication
            p1_hat = forward_ntt(p1)
            p2_hat = forward_ntt(p2)
            p_prod_hat = poly_ntt_multiply(p1_hat, p2_hat)
            p_ntt = inverse_ntt(p_prod_hat)

            self.assertEqual(p_ntt, p_direct)

    def test_ntt_linearity(self) -> None:
        """Verify NTT(a + b) == NTT(a) + NTT(b)."""
        p1 = Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)])
        p2 = Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)])

        p_sum_hat = forward_ntt(p1 + p2)
        p_hat_sum = forward_ntt(p1) + forward_ntt(p2)
        self.assertEqual(p_sum_hat, p_hat_sum)

    def test_polyvec_ntt_roundtrip(self) -> None:
        """Verify vector forward and inverse NTT roundtrip."""
        k = 3
        vec = PolyVec([Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]) for _ in range(k)])
        vec_hat = polyvec_forward_ntt(vec)
        vec_rec = polyvec_inverse_ntt(vec_hat)
        self.assertEqual(vec_rec, vec)

    def test_polyvec_ntt_dot_vs_direct(self) -> None:
        """Verify dot product in NTT domain matches direct sum of products."""
        k = 2
        v1 = PolyVec([Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]) for _ in range(k)])
        v2 = PolyVec([Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]) for _ in range(k)])

        # Direct dot product
        p_direct = v1.dot(v2)

        # NTT dot product
        v1_hat = polyvec_forward_ntt(v1)
        v2_hat = polyvec_forward_ntt(v2)
        p_ntt_hat = polyvec_ntt_dot(v1_hat, v2_hat)
        p_ntt = inverse_ntt(p_ntt_hat)

        self.assertEqual(p_ntt, p_direct)


if __name__ == "__main__":
    unittest.main()
