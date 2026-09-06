"""
QuantaLab: Unit Tests for Quantum Algorithms (QFT, Grover, Teleportation, Superdense, Deutsch-Jozsa, Shor).
"""

import unittest
import math
from quantalab.types import StateVector
from quantalab.circuit import QuantumCircuit
from quantalab.algorithms import (
    build_qft_circuit, build_iqft_circuit, build_grover_circuit,
    run_quantum_teleportation, run_superdense_coding,
    run_deutsch_jozsa, run_shor_factoring
)


class TestQuantumAlgorithms(unittest.TestCase):

    def test_qft_and_iqft_reversibility(self):
        # QFT followed by IQFT must return initial state with fidelity 1.0
        n_qubits = 3
        qft = build_qft_circuit(n_qubits)
        iqft = build_iqft_circuit(n_qubits)

        init_state = StateVector.from_basis(n_qubits, "101")
        qft_res = qft.simulate(initial_state=init_state)
        final_res = iqft.simulate(initial_state=qft_res.state)

        fidelity = final_res.state.fidelity(init_state)
        self.assertAlmostEqual(fidelity, 1.0, places=5)

    def test_grover_search_2_qubits(self):
        # In a 2-qubit space (4 items), 1 Grover iteration finds the marked item with 100% probability!
        for target in ("00", "01", "10", "11"):
            circ = build_grover_circuit(2, target)
            sim = circ.simulate()
            probs = sim.probabilities()
            self.assertGreater(probs.get(target, 0.0), 0.99)

    def test_grover_search_3_qubits(self):
        # In a 3-qubit space (8 items), target item should have highest probability (>90%)
        target = "101"
        circ = build_grover_circuit(3, target)
        sim = circ.simulate()
        probs = sim.probabilities()
        self.assertGreater(probs.get(target, 0.0), 0.90)

    def test_quantum_teleportation(self):
        # Teleport unknown state |psi> = 0.6|0> + 0.8|1>
        bob_state, fidelity = run_quantum_teleportation(0.6, 0.8)
        self.assertAlmostEqual(fidelity, 1.0, places=5)

    def test_superdense_coding(self):
        # Test all 4 classical 2-bit combinations
        for b0, b1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            rec_b0, rec_b1 = run_superdense_coding(b0, b1)
            self.assertEqual((rec_b0, rec_b1), (b0, b1))

    def test_deutsch_jozsa(self):
        res_const = run_deutsch_jozsa(n_qubits=3, is_constant=True)
        self.assertEqual(res_const, "constant")

        res_balanced = run_deutsch_jozsa(n_qubits=3, is_constant=False)
        self.assertEqual(res_balanced, "balanced")

    def test_shor_factoring_15(self):
        p, q = run_shor_factoring(15)
        self.assertEqual(p * q, 15)
        self.assertTrue(set([p, q]) == {3, 5})

    def test_shor_factoring_21(self):
        p, q = run_shor_factoring(21)
        self.assertEqual(p * q, 21)
        self.assertTrue(set([p, q]) == {3, 7})


if __name__ == "__main__":
    unittest.main()
