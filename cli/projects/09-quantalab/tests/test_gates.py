"""
QuantaLab: Unit Tests for Quantum Gates and Unitarity Invariants.
"""

import unittest
import math
import cmath
from quantalab.types import StateVector
from quantalab.gates import (
    Gate, UnitaryMatrix,
    apply_gate_1q, apply_cnot, apply_swap, apply_toffoli, apply_fredkin
)


class TestQuantumGates(unittest.TestCase):

    def test_unitarity(self):
        for g in (Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.T):
            self.assertTrue(g.is_unitary())
            # U * U^dagger = I
            prod = g.matmul(g.dagger())
            self.assertAlmostEqual(prod.m00.real, 1.0)
            self.assertAlmostEqual(prod.m11.real, 1.0)
            self.assertAlmostEqual(abs(prod.m01), 0.0)

    def test_pauli_x(self):
        # X |0> = |1>
        sv = StateVector.zero(1)
        apply_gate_1q(sv, Gate.X, 0)
        self.assertAlmostEqual(sv.amplitudes[0].real, 0.0)
        self.assertAlmostEqual(sv.amplitudes[1].real, 1.0)

        # X |1> = |0>
        apply_gate_1q(sv, Gate.X, 0)
        self.assertAlmostEqual(sv.amplitudes[0].real, 1.0)

    def test_hadamard_reversibility(self):
        # H * H = I
        sv = StateVector.zero(2)
        apply_gate_1q(sv, Gate.H, 0)
        apply_gate_1q(sv, Gate.H, 0)
        self.assertAlmostEqual(sv.amplitudes[0].real, 1.0)
        self.assertAlmostEqual(sv.amplitudes[1].real, 0.0)

    def test_cnot_truth_table(self):
        # CNOT |00> -> |00>
        # CNOT |01> -> |01> (q1=0, q0=1: control q1 is 0, target q0 unchanged)
        # CNOT |10> -> |11> (q1=1, q0=0: control q1 is 1, target q0 flipped to 1)
        # CNOT |11> -> |10>
        for init_b, expected_b in [("00", "00"), ("01", "01"), ("10", "11"), ("11", "10")]:
            sv = StateVector.from_basis(2, init_b)
            apply_cnot(sv, control=1, target=0)
            expected_sv = StateVector.from_basis(2, expected_b)
            self.assertAlmostEqual(sv.fidelity(expected_sv), 1.0)

    def test_swap_gate(self):
        sv = StateVector.from_basis(2, "10")
        apply_swap(sv, 0, 1)
        expected = StateVector.from_basis(2, "01")
        self.assertAlmostEqual(sv.fidelity(expected), 1.0)

    def test_toffoli_gate(self):
        # Target flips only when both controls are 1
        for init_b, expected_b in [("000", "000"), ("110", "111"), ("111", "110"), ("101", "101")]:
            sv = StateVector.from_basis(3, init_b)
            apply_toffoli(sv, c1=2, c2=1, target=0)
            expected_sv = StateVector.from_basis(3, expected_b)
            self.assertAlmostEqual(sv.fidelity(expected_sv), 1.0)


if __name__ == "__main__":
    unittest.main()
