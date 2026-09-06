"""
Unit tests for ZetaProof Finite Field Arithmetic.
"""

import unittest
from zetaproof.field import FieldElement, BN254_SCALAR_FIELD, TEST_PRIME, inner_product


class TestFieldElement(unittest.TestCase):
    def test_basic_arithmetic(self):
        p = TEST_PRIME
        a = FieldElement(42, p)
        b = FieldElement(58, p)
        self.assertEqual(a + b, FieldElement(100, p))
        self.assertEqual(a - b, FieldElement((42 - 58) % p, p))
        self.assertEqual(a * b, FieldElement((42 * 58) % p, p))
        self.assertEqual(-a, FieldElement((-42) % p, p))

    def test_modular_inverse(self):
        p = TEST_PRIME
        for val in (1, 2, 7, 13, 999999, p - 1):
            elem = FieldElement(val, p)
            inv = elem.inverse()
            self.assertEqual(elem * inv, FieldElement(1, p))

        with self.assertRaises(ZeroDivisionError):
            FieldElement(0, p).inverse()

    def test_division(self):
        p = TEST_PRIME
        a = FieldElement(123456, p)
        b = FieldElement(789, p)
        c = a / b
        self.assertEqual(c * b, a)

    def test_tonelli_shanks_sqrt(self):
        p = TEST_PRIME
        # Check squares have valid roots
        for x in (1, 4, 9, 16, 25, 49, 144, 12345):
            val = FieldElement(x, p) ** 2
            root = val.sqrt()
            self.assertEqual(root ** 2, val)

        # Zero square root
        self.assertEqual(FieldElement(0, p).sqrt(), FieldElement(0, p))

    def test_bn254_scalar_field(self):
        p = BN254_SCALAR_FIELD
        a = FieldElement(10**18, p)
        b = FieldElement(10**18 + 7, p)
        self.assertEqual((a + b) - b, a)
        self.assertEqual((a * b) / b, a)

    def test_inner_product(self):
        p = TEST_PRIME
        vec_a = [FieldElement(1, p), FieldElement(2, p), FieldElement(3, p)]
        vec_b = [FieldElement(4, p), FieldElement(5, p), FieldElement(6, p)]
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        res = inner_product(vec_a, vec_b)
        self.assertEqual(res, FieldElement(32, p))


if __name__ == "__main__":
    unittest.main()
