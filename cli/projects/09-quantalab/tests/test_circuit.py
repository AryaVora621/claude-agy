"""
QuantaLab: Unit Tests for QuantumCircuit Composition, Depth, and Simulation.
"""

import unittest
import math
from quantalab.circuit import QuantumCircuit
from quantalab.types import StateVector


class TestQuantumCircuit(unittest.TestCase):

    def test_bell_state_generation(self):
        # Create Bell state (|00> + |11>) / sqrt(2)
        circ = QuantumCircuit(2)
        circ.h(0).cx(0, 1)

        sim = circ.simulate()
        state = sim.state
        self.assertTrue(state.is_normalized())

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(state.amplitudes[0].real, inv_sqrt2)
        self.assertAlmostEqual(state.amplitudes[1].real, 0.0)
        self.assertAlmostEqual(state.amplitudes[2].real, 0.0)
        self.assertAlmostEqual(state.amplitudes[3].real, inv_sqrt2)

    def test_ghz_state_generation(self):
        # 3-qubit GHZ state (|000> + |111>) / sqrt(2)
        circ = QuantumCircuit(3)
        circ.h(0).cx(0, 1).cx(1, 2)

        sim = circ.simulate()
        state = sim.state
        self.assertTrue(state.is_normalized())

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(state.amplitudes[0].real, inv_sqrt2)
        self.assertAlmostEqual(state.amplitudes[7].real, inv_sqrt2)

    def test_circuit_depth_and_gate_count(self):
        circ = QuantumCircuit(3)
        circ.h(0).h(1).cx(0, 1).h(2)

        self.assertEqual(circ.gate_count, 4)
        # Qubit 0: H, CX -> depth 2
        # Qubit 1: H, CX -> depth 2
        # Qubit 2: H -> depth 1
        self.assertEqual(circ.depth, 2)

    def test_sampling(self):
        circ = QuantumCircuit(2, 2)
        circ.h(0).cx(0, 1).measure_all()

        samples = circ.sample(shots=500)
        # Should only observe '00' and '11'
        self.assertTrue(set(samples.keys()).issubset({"00", "11"}))
        self.assertGreater(samples.get("00", 0), 100)
        self.assertGreater(samples.get("11", 0), 100)


if __name__ == "__main__":
    unittest.main()
