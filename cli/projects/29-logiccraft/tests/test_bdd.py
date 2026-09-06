"""Unit tests for Reduced Ordered Binary Decision Diagrams (ROBDD)."""

import unittest
from logiccraft.bdd import BDDManager, FALSE_ID, TRUE_ID


class TestROBDD(unittest.TestCase):
    """Test suite verifying ROBDD canonical properties, ITE logic, and SAT operations."""

    def setUp(self) -> None:
        self.mgr = BDDManager(["a", "b", "c", "d"])

    def test_constants_and_variables(self) -> None:
        """Verify terminal constants and single variable nodes."""
        self.assertEqual(self.mgr.constant_false, FALSE_ID)
        self.assertEqual(self.mgr.constant_true, TRUE_ID)

        va = self.mgr.var("a")
        vb = self.mgr.var("b")
        self.assertNotEqual(va, vb)
        self.assertNotEqual(va, FALSE_ID)
        self.assertNotEqual(va, TRUE_ID)

    def test_canonicity_and_identities(self) -> None:
        """Verify unique table deduplication and Boolean identities."""
        va = self.mgr.var("a")
        va2 = self.mgr.var("a")
        self.assertEqual(va, va2)

        # a & a == a
        self.assertEqual(self.mgr.and_(va, va), va)
        # a | a == a
        self.assertEqual(self.mgr.or_(va, va), va)
        # a & 1 == a
        self.assertEqual(self.mgr.and_(va, TRUE_ID), va)
        # a & 0 == 0
        self.assertEqual(self.mgr.and_(va, FALSE_ID), FALSE_ID)
        # a | 0 == a
        self.assertEqual(self.mgr.or_(va, FALSE_ID), va)
        # a | 1 == 1
        self.assertEqual(self.mgr.or_(va, TRUE_ID), TRUE_ID)
        # a ^ a == 0
        self.assertEqual(self.mgr.xor_(va, va), FALSE_ID)
        # a ^ 0 == a
        self.assertEqual(self.mgr.xor_(va, FALSE_ID), va)
        # ~~a == a
        self.assertEqual(self.mgr.not_(self.mgr.not_(va)), va)

    def test_de_morgan_laws(self) -> None:
        """Verify De Morgan's laws: ~(A & B) == ~A | ~B and ~(A | B) == ~A & ~B."""
        va = self.mgr.var("a")
        vb = self.mgr.var("b")

        # ~(a & b)
        lhs1 = self.mgr.not_(self.mgr.and_(va, vb))
        # ~a | ~b
        rhs1 = self.mgr.or_(self.mgr.not_(va), self.mgr.not_(vb))
        self.assertTrue(self.mgr.equiv(lhs1, rhs1))

        # ~(a | b)
        lhs2 = self.mgr.not_(self.mgr.or_(va, vb))
        # ~a & ~b
        rhs2 = self.mgr.and_(self.mgr.not_(va), self.mgr.not_(vb))
        self.assertTrue(self.mgr.equiv(lhs2, rhs2))

    def test_xor_and_mux_equivalence(self) -> None:
        """Verify XOR expansion: (a & ~b) | (~a & b)."""
        va = self.mgr.var("a")
        vb = self.mgr.var("b")

        xor_direct = self.mgr.xor_(va, vb)
        term1 = self.mgr.and_(va, self.mgr.not_(vb))
        term2 = self.mgr.and_(self.mgr.not_(va), vb)
        xor_expanded = self.mgr.or_(term1, term2)

        self.assertTrue(self.mgr.equiv(xor_direct, xor_expanded))

    def test_sat_counting_and_witness(self) -> None:
        """Verify SAT model count and witness generation."""
        va = self.mgr.var("a")
        vb = self.mgr.var("b")

        # Function: a & b over 4 variables (a, b, c, d)
        # 2^2 = 4 satisfying assignments
        fn = self.mgr.and_(va, vb)
        count = self.mgr.sat_count(fn, total_vars=4)
        self.assertEqual(count, 4)

        # Witness should satisfy a=True, b=True
        witness = self.mgr.any_sat(fn)
        self.assertIsNotNone(witness)
        self.assertTrue(witness["a"])
        self.assertTrue(witness["b"])

        # Contradiction: a & ~a
        unsat = self.mgr.and_(va, self.mgr.not_(va))
        self.assertEqual(self.mgr.sat_count(unsat, total_vars=4), 0)
        self.assertIsNone(self.mgr.any_sat(unsat))

    def test_expression_parser(self) -> None:
        """Verify parsing infix Boolean expressions into canonical BDD roots."""
        expr1 = self.mgr.parse_expr("(a & b) | (~a & c)")
        va = self.mgr.var("a")
        vb = self.mgr.var("b")
        vc = self.mgr.var("c")
        expected = self.mgr.or_(
            self.mgr.and_(va, vb),
            self.mgr.and_(self.mgr.not_(va), vc)
        )
        self.assertTrue(self.mgr.equiv(expr1, expected))


if __name__ == "__main__":
    unittest.main()
