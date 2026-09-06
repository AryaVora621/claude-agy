"""
Unit tests for ZetaProof Arithmetic Circuit Compiler.
"""

import unittest
from zetaproof.field import FieldElement, TEST_PRIME
from zetaproof.circuit import Circuit


class TestCircuit(unittest.TestCase):
    def test_simple_equation(self):
        # Statement: y = x^3 + x + 5
        # Private input: x = 3
        # Public input: y = 3^3 + 3 + 5 = 27 + 3 + 5 = 35
        p = TEST_PRIME
        c = Circuit(p)
        x = c.private_input("x")
        y = c.public_input("y")

        # x^2
        x2 = x * x
        # x^3
        x3 = x2 * x
        # x^3 + x + 5 == y
        c.assert_equal(x3 + x + 5, y)

        r1cs = c.to_r1cs()
        self.assertEqual(r1cs.num_constraints, 3)

        # Solve witness for x = 3, y = 35
        witness = c.solve_witness(
            public_inputs={"y": 35},
            private_inputs={"x": 3}
        )

        self.assertTrue(r1cs.is_satisfied(witness))

        # Check invalid output fails
        bad_witness = list(witness)
        bad_witness[r1cs.public_var_ids[1]] = FieldElement(36, p)
        self.assertFalse(r1cs.is_satisfied(bad_witness))

    def test_boolean_constraint(self):
        p = TEST_PRIME
        c = Circuit(p)
        b = c.private_input("b")
        c.assert_boolean(b)

        r1cs = c.to_r1cs()
        # b = 0 should satisfy
        w0 = c.solve_witness(private_inputs={"b": 0})
        self.assertTrue(r1cs.is_satisfied(w0))

        # b = 1 should satisfy
        w1 = c.solve_witness(private_inputs={"b": 1})
        self.assertTrue(r1cs.is_satisfied(w1))

        # b = 2 should fail
        w2 = list(w0)
        w2[c.var_map["b"].id] = FieldElement(2, p)
        self.assertFalse(r1cs.is_satisfied(w2))

    def test_range_check(self):
        p = TEST_PRIME
        c = Circuit(p)
        val = c.private_input("val")
        # 4-bit range check: 0 <= val < 16
        c.range_check(val, num_bits=4)

        r1cs = c.to_r1cs()

        for valid_val in (0, 1, 7, 15):
            w = c.solve_witness(private_inputs={"val": valid_val})
            self.assertTrue(r1cs.is_satisfied(w))

        # Value outside 4-bit range (e.g. 16)
        # Note: 16 will produce bit decomposition 0,0,0,0 which sums to 0 != 16
        w_invalid = c.solve_witness(private_inputs={"val": 16})
        self.assertFalse(r1cs.is_satisfied(w_invalid))


if __name__ == "__main__":
    unittest.main()
