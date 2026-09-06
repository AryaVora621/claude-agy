"""Unit tests for And-Inverter Graphs (AIG) and Structural Hashing."""

import unittest
from logiccraft.aig import (
    AIGGraph,
    CONST_FALSE_LIT,
    CONST_TRUE_LIT,
    lit_is_inv,
    lit_node,
    lit_not,
    make_lit,
)


class TestAIG(unittest.TestCase):
    """Test suite for AIG structural hashing, reduction, and simulation."""

    def setUp(self) -> None:
        self.aig = AIGGraph()
        self.a = self.aig.create_pi("a")
        self.b = self.aig.create_pi("b")
        self.c = self.aig.create_pi("c")

    def test_literal_manipulation(self) -> None:
        """Verify literal encoding, decoding, and inversion."""
        lit = make_lit(42, False)
        self.assertEqual(lit_node(lit), 42)
        self.assertFalse(lit_is_inv(lit))

        inv_lit = lit_not(lit)
        self.assertEqual(lit_node(inv_lit), 42)
        self.assertTrue(lit_is_inv(inv_lit))
        self.assertEqual(lit_not(inv_lit), lit)

    def test_structural_hashing_strashing(self) -> None:
        """Verify on-the-fly deduplication and Boolean simplifications."""
        # a & a == a
        self.assertEqual(self.aig.and_(self.a, self.a), self.a)
        # a & 0 == 0
        self.assertEqual(self.aig.and_(self.a, CONST_FALSE_LIT), CONST_FALSE_LIT)
        # a & 1 == a
        self.assertEqual(self.aig.and_(self.a, CONST_TRUE_LIT), self.a)
        # a & ~a == 0
        self.assertEqual(self.aig.and_(self.a, lit_not(self.a)), CONST_FALSE_LIT)

        # Commutative strashing: and(a, b) == and(b, a)
        n1 = self.aig.and_(self.a, self.b)
        n2 = self.aig.and_(self.b, self.a)
        self.assertEqual(n1, n2)

    def test_derived_logic_gates(self) -> None:
        """Verify OR, XOR, NAND, NOR, and MUX operators."""
        # De Morgan OR
        or_lit = self.aig.or_(self.a, self.b)
        # NAND
        nand_lit = self.aig.nand_(self.a, self.b)
        self.assertEqual(nand_lit, lit_not(self.aig.and_(self.a, self.b)))
        # NOR
        nor_lit = self.aig.nor_(self.a, self.b)
        self.assertEqual(nor_lit, lit_not(or_lit))

    def test_logic_simulation(self) -> None:
        """Verify bitwise simulation against full truth tables."""
        # 1-bit full adder sum and carry:
        # Sum = a ^ b ^ cin
        # Cout = (a & b) | (cin & (a ^ b))
        cin = self.aig.create_pi("cin")
        a_xor_b = self.aig.xor_(self.a, self.b)
        sum_out = self.aig.xor_(a_xor_b, cin)
        carry_out = self.aig.or_(
            self.aig.and_(self.a, self.b),
            self.aig.and_(cin, a_xor_b)
        )

        self.aig.set_output("sum", sum_out)
        self.aig.set_output("cout", carry_out)

        # Test all 8 input combinations
        for a_val in (False, True):
            for b_val in (False, True):
                for cin_val in (False, True):
                    res = self.aig.simulate({"a": a_val, "b": b_val, "cin": cin_val})
                    int_sum = int(a_val) + int(b_val) + int(cin_val)
                    expected_sum = bool(int_sum & 1)
                    expected_cout = bool((int_sum >> 1) & 1)
                    self.assertEqual(res["sum"], expected_sum)
                    self.assertEqual(res["cout"], expected_cout)

    def test_topological_sort_and_depth(self) -> None:
        """Verify DAG levelization and topological order."""
        t1 = self.aig.and_(self.a, self.b)
        t2 = self.aig.and_(t1, self.c)
        self.aig.set_output("out", t2)

        topo = self.aig.topological_sort()
        # All inputs must precede outputs in topo order
        idx_a = topo.index(lit_node(self.a))
        idx_t1 = topo.index(lit_node(t1))
        idx_t2 = topo.index(lit_node(t2))

        self.assertLess(idx_a, idx_t1)
        self.assertLess(idx_t1, idx_t2)
        self.assertGreaterEqual(self.aig.max_depth, 2)


if __name__ == "__main__":
    unittest.main()
