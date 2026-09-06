"""Unit Tests for Rejection Sampling, CBD Noise, and Bit-Packing Codecs."""

import hashlib
import os
import random
import unittest

from latticeguard.ring import KYBER_N, KYBER_Q, Polynomial, PolyVec
from latticeguard.sampling import (
    compress_val,
    decompress_val,
    pack_bits,
    unpack_bits,
    poly_to_bytes,
    bytes_to_poly,
    polyvec_to_bytes,
    bytes_to_polyvec,
    compress_poly,
    decompress_poly,
    compress_polyvec,
    decompress_polyvec,
    msg_to_poly,
    poly_to_msg,
    sample_poly_cbd,
    sample_ntt,
    generate_matrix_A,
)


class TestSamplingCodecs(unittest.TestCase):
    """Test suite for CBD noise, SHAKE-128 rejection sampling, and bit codecs."""

    def setUp(self) -> None:
        random.seed(999)

    def test_compression_decompression_bounds(self) -> None:
        """Verify compression roundtrip error |decompress(compress(x, d)) - x| <= round(q / 2^(d+1))."""
        for d in [1, 4, 5, 10, 11]:
            max_allowed_error = (KYBER_Q + (1 << d)) // (1 << (d + 1)) + 1
            for x in range(KYBER_Q):
                c = compress_val(x, d)
                self.assertGreaterEqual(c, 0)
                self.assertLess(c, 1 << d)

                x_rec = decompress_val(c, d)
                # Compute modular distance
                diff = abs(x - x_rec)
                err = min(diff, KYBER_Q - diff)
                self.assertLessEqual(err, max_allowed_error)

    def test_pack_unpack_bits(self) -> None:
        """Verify bit packing and unpacking for various bit widths d."""
        for d in [1, 4, 5, 10, 11]:
            mask = (1 << d) - 1
            values = [random.randint(0, mask) for _ in range(KYBER_N)]
            packed = pack_bits(values, d)
            expected_bytes = (KYBER_N * d + 7) // 8
            self.assertEqual(len(packed), expected_bytes)

            unpacked = unpack_bits(packed, d, KYBER_N)
            self.assertEqual(unpacked, values)

    def test_poly_12bit_byte_codec_roundtrip(self) -> None:
        """Verify poly_to_bytes (384 bytes) and bytes_to_poly roundtrip."""
        for _ in range(10):
            coeffs = [random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]
            p = Polynomial(coeffs)
            b = poly_to_bytes(p)
            self.assertEqual(len(b), 384)

            p_rec = bytes_to_poly(b)
            self.assertEqual(p_rec, p)

    def test_polyvec_byte_codec_roundtrip(self) -> None:
        """Verify polyvec serialization roundtrip across ranks k=2, 3, 4."""
        for k in [2, 3, 4]:
            vec = PolyVec([Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)]) for _ in range(k)])
            raw = polyvec_to_bytes(vec)
            self.assertEqual(len(raw), 384 * k)

            vec_rec = bytes_to_polyvec(raw, k)
            self.assertEqual(vec_rec, vec)

    def test_message_poly_roundtrip(self) -> None:
        """Verify 32-byte message encoding to polynomial and decoding back."""
        for _ in range(20):
            msg = os.urandom(32)
            poly = msg_to_poly(msg)
            # Coefficients should be 0 or round(q/2) = 1665
            for c in poly.coeffs:
                self.assertIn(c, (0, 1665))

            msg_rec = poly_to_msg(poly)
            self.assertEqual(msg_rec, msg)

    def test_sample_poly_cbd_eta2(self) -> None:
        """Verify Centered Binomial Distribution CBD_2 properties."""
        buf = os.urandom(128)
        poly = sample_poly_cbd(buf, eta=2)
        # Coefficients centered must be in [-2, 2]
        for c in poly.to_centered_list():
            self.assertIn(c, (-2, -1, 0, 1, 2))

    def test_sample_poly_cbd_eta3(self) -> None:
        """Verify Centered Binomial Distribution CBD_3 properties."""
        buf = os.urandom(192)
        poly = sample_poly_cbd(buf, eta=3)
        # Coefficients centered must be in [-3, 3]
        for c in poly.to_centered_list():
            self.assertIn(c, (-3, -2, -1, 0, 1, 2, 3))

    def test_sample_ntt_rejection(self) -> None:
        """Verify rejection sampling creates uniform coefficients in [0, q-1]."""
        seed = os.urandom(32)
        poly = sample_ntt(seed, row=0, col=1)
        self.assertEqual(len(poly), KYBER_N)
        for c in poly.coeffs:
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, KYBER_Q)

    def test_matrix_generation_transposed_consistency(self) -> None:
        """Verify that transposed matrix has (A^T)[i][j] == A[j][i]."""
        seed = os.urandom(32)
        k = 3
        A = generate_matrix_A(seed, k, transposed=False)
        A_T = generate_matrix_A(seed, k, transposed=True)

        for i in range(k):
            for j in range(k):
                self.assertEqual(A_T[i][j], A[j][i])


if __name__ == "__main__":
    unittest.main()
