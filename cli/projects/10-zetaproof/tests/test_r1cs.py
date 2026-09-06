"""
Unit tests for ZetaProof R1CS Matrix and Constraint System.
"""

import unittest
from zetaproof.field import FieldElement, TEST_PRIME
from zetaproof.r1cs import Constraint, R1CS


class TestR1CS(unittest.TestCase):
    def test_direct_r1cs_construction(self):
        # Enforce: x * y = z
        # Variables: [~one(0), x(1), y(2), z(3)]
        p = TEST_PRIME
        # (1 * x) * (1 * y) = (1 * z)
        c1 = Constraint(
            a={1: FieldElement.one(p)},
            b={2: FieldElement.one(p)},
            c={3: FieldElement.one(p)}
        )

        r1cs = R1CS(
            constraints=[c1],
            num_variables=4,
            num_public_inputs=2,
            variable_names=["~one", "x", "y", "z"],
            public_var_ids=[0, 3],
            p=p
        )

        # Valid witness: x = 6, y = 7, z = 42
        witness = [
            FieldElement(1, p),
            FieldElement(6, p),
            FieldElement(7, p),
            FieldElement(42, p)
        ]
        self.assertTrue(r1cs.is_satisfied(witness))

        # Invalid witness: x = 6, y = 7, z = 43
        bad_witness = list(witness)
        bad_witness[3] = FieldElement(43, p)
        self.assertFalse(r1cs.is_satisfied(bad_witness))

    def test_dense_matrix_export(self):
        p = TEST_PRIME
        c1 = Constraint(
            a={0: FieldElement(2, p), 1: FieldElement(3, p)},
            b={2: FieldElement(4, p)},
            c={3: FieldElement(5, p)}
        )
        r1cs = R1CS(
            constraints=[c1],
            num_variables=4,
            num_public_inputs=1,
            variable_names=["v0", "v1", "v2", "v3"],
            public_var_ids=[0],
            p=p
        )

        mat_a, mat_b, mat_c = r1cs.dense_matrices()
        self.assertEqual(len(mat_a), 1)
        self.assertEqual(len(mat_a[0]), 4)
        self.assertEqual(mat_a[0][0], FieldElement(2, p))
        self.assertEqual(mat_a[0][1], FieldElement(3, p))
        self.assertEqual(mat_b[0][2], FieldElement(4, p))
        self.assertEqual(mat_c[0][3], FieldElement(5, p))


if __name__ == "__main__":
    unittest.main()
