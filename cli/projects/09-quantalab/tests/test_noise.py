"""
QuantaLab: Unit Tests for Density Matrix and Quantum Noise Channels.
"""

import unittest
from quantalab.types import StateVector
from quantalab.noise import DensityMatrix, NoiseChannel


class TestQuantumNoise(unittest.TestCase):

    def test_density_matrix_pure_state(self):
        # Pure state |0>
        sv = StateVector.zero(1)
        dm = DensityMatrix.from_state_vector(sv)

        self.assertAlmostEqual(dm.trace().real, 1.0)
        self.assertAlmostEqual(dm.purity(), 1.0)  # Pure state has purity 1.0

    def test_bit_flip_decoherence(self):
        sv = StateVector.zero(1)
        dm = DensityMatrix.from_state_vector(sv)

        # Apply 50% bit flip noise
        kraus = NoiseChannel.bit_flip(p=0.5)
        dm.apply_single_qubit_kraus(kraus, target_qubit=0)

        # Should be mixed state with purity 0.5 (maximally mixed)
        self.assertAlmostEqual(dm.trace().real, 1.0)
        self.assertAlmostEqual(dm.purity(), 0.5)

    def test_amplitude_damping(self):
        # Excited state |1>
        sv = StateVector.from_basis(1, "1")
        dm = DensityMatrix.from_state_vector(sv)

        # Decay towards ground state
        kraus = NoiseChannel.amplitude_damping(gamma=0.8)
        dm.apply_single_qubit_kraus(kraus, target_qubit=0)

        # Population of |0> should be 0.8, population of |1> should be 0.2
        self.assertAlmostEqual(dm.matrix[0][0].real, 0.8)
        self.assertAlmostEqual(dm.matrix[1][1].real, 0.2)
        self.assertAlmostEqual(dm.trace().real, 1.0)


if __name__ == "__main__":
    unittest.main()
