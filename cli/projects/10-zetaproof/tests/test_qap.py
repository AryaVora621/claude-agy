"""
Unit tests for ZetaProof QAP Reduction from R1CS.
"""

import unittest
from zetaproof.field import FieldElement, TEST_PRIME
from zetaproof.circuit import Circuit
from zetaproof.qap import QAP


class TestQAP(unittest.TestCase):
    def test_qap_reduction_divisibility(self):
        # Equation: y = x^3 + x + 5
        # Private: x = 3 -> y = 35
        p = TEST_PRIME
        c = Circuit(p)
        x = c.private_input("x")
        y = c.public_input("y")

        x2 = x * x
        x3 = x2 * x
        c.assert_equal(x3 + x + 5, y)

        r1cs = c.to_r1cs()
        witness = c.solve_witness(
            public_inputs={"y": 35},
            private_inputs={"x": 3}
        )

        qap = QAP.from_r1cs(r1cs)
        self.assertEqual(qap.num_constraints, 3)
        self.assertEqual(qap.num_variables, r1cs.num_variables)

        # Compute QAP polynomials: A, B, C, H
        poly_a, poly_b, poly_c, quot_h = qap.compute_polynomials(witness)

        # Verify exact identity: A(X)*B(X) - C(X) == H(X)*Z(X)
        lhs = (poly_a * poly_b) - poly_c
        rhs = quot_h * qap.z_poly
        self.assertEqual(lhs, rhs)

        # Satisfiability check returns True
        self.assertTrue(qap.is_satisfied(witness))

    def test_qap_invalid_witness_fails(self):
        p = TEST_PRIME
        c = Circuit(p)
        x = c.private_input("x")
        y = c.public_input("y")
        c.assert_equal(x * x, y)

        r1cs = c.to_r1cs()
        # Invalid witness: x = 4, y = 17 (should be 16)
        witness = c.solve_witness(
            public_inputs={"y": 17},
            private_inputs={"x": 4}
        )

        qap = QAP.from_r1cs(r1cs)
        self.assertFalse(qap.is_satisfied(witness))
        with self.assertRaises(ValueError):
            qap.compute_polynomials(witness)


if __name__ == "__main__":
    unittest.main()
