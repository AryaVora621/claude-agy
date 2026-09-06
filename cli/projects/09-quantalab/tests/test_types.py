"""
QuantaLab: Unit Tests for StateVector and Complex Linear Algebra Primitives.
"""

import unittest
import math
import cmath
from quantalab.types import StateVector, BasisState


class TestStateVector(unittest.TestCase):

    def test_zero_state_initialization(self):
        sv = StateVector.zero(3)
        self.assertEqual(sv.n_qubits, 3)
        self.assertEqual(sv.dim, 8)
        self.assertEqual(sv.amplitudes[0], 1.0 + 0.0j)
        for i in range(1, 8):
            self.assertEqual(sv.amplitudes[i], 0.0 + 0.0j)
        self.assertTrue(sv.is_normalized())

    def test_basis_state(self):
        sv = StateVector.from_basis(3, "011")
        self.assertEqual(sv.amplitudes[3], 1.0 + 0.0j)
        self.assertEqual(sv.amplitudes[0], 0.0 + 0.0j)

    def test_inner_product_and_fidelity(self):
        sv1 = StateVector.from_basis(2, "00")
        sv2 = StateVector.from_basis(2, "00")
        sv3 = StateVector.from_basis(2, "11")

        self.assertAlmostEqual(sv1.inner(sv2).real, 1.0)
        self.assertAlmostEqual(sv1.fidelity(sv2), 1.0)
        self.assertAlmostEqual(sv1.inner(sv3).real, 0.0)
        self.assertAlmostEqual(sv1.fidelity(sv3), 0.0)

    def test_tensor_product(self):
        s0 = StateVector.from_basis(1, "0")
        s1 = StateVector.from_basis(1, "1")
        s01 = s0.tensor(s1)

        self.assertEqual(s01.n_qubits, 2)
        self.assertEqual(s01.dim, 4)
        self.assertAlmostEqual(s01.amplitudes[1].real, 1.0)  # |01> is index 1

    def test_measurement_collapse(self):
        # Prepare superposition (|0> + |1>) / sqrt(2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        sv = StateVector([inv_sqrt2, inv_sqrt2], 1)

        # Force outcome 0
        bit0, collapsed0 = sv.measure_qubit(0, random_val=0.1)
        self.assertEqual(bit0, 0)
        self.assertAlmostEqual(collapsed0.amplitudes[0].real, 1.0)
        self.assertAlmostEqual(collapsed0.amplitudes[1].real, 0.0)

        # Force outcome 1
        bit1, collapsed1 = sv.measure_qubit(0, random_val=0.9)
        self.assertEqual(bit1, 1)
        self.assertAlmostEqual(collapsed1.amplitudes[0].real, 0.0)
        self.assertAlmostEqual(collapsed1.amplitudes[1].real, 1.0)

    def test_bloch_coordinates(self):
        # |0> has Bloch vector (0, 0, 1)
        s0 = StateVector.from_basis(1, 0)
        x, y, z = s0.bloch_coordinates(0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 1.0)

        # |1> has Bloch vector (0, 0, -1)
        s1 = StateVector.from_basis(1, 1)
        x, y, z = s1.bloch_coordinates(0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, -1.0)


if __name__ == "__main__":
    unittest.main()
