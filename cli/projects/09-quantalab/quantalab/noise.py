"""
QuantaLab: Quantum Noise Channels & Density Matrix Simulator.
Features:
1. DensityMatrix: 2^N x 2^N complex matrix tracking mixed quantum states.
2. Purity, fidelity, and trace preservation invariants.
3. Kraus Operator formalism: Bit-Flip, Phase-Flip, Depolarizing, and Amplitude Damping channels.
"""

import math
import cmath
from typing import List, Tuple, Sequence, Optional
from .types import StateVector
from .gates import Gate, UnitaryMatrix


class DensityMatrix:
    """
    Mixed quantum state representation for N qubits via a 2^N x 2^N Hermitian positive semi-definite matrix rho.
    Tr(rho) = 1, Tr(rho^2) <= 1.
    """
    __slots__ = ("dim", "n_qubits", "matrix")

    def __init__(self, matrix: List[List[complex]], n_qubits: Optional[int] = None):
        self.matrix = matrix
        self.dim = len(matrix)
        if self.dim == 0:
            raise ValueError("DensityMatrix cannot be empty")

        if n_qubits is None:
            n = int(math.log2(self.dim))
            if (1 << n) != self.dim:
                raise ValueError(f"Matrix dimension {self.dim} is not a power of 2")
            self.n_qubits = n
        else:
            self.n_qubits = n_qubits

    @classmethod
    def from_state_vector(cls, state: StateVector) -> "DensityMatrix":
        """Construct pure state density matrix rho = |psi><psi|."""
        dim = state.dim
        amps = state.amplitudes
        matrix = []
        for i in range(dim):
            row = []
            a_i = amps[i]
            for j in range(dim):
                row.append(a_i * amps[j].conjugate())
            matrix.append(row)
        return cls(matrix, state.n_qubits)

    def trace(self) -> complex:
        """Calculate the trace Tr(rho) = sum(rho_ii)."""
        return sum(self.matrix[i][i] for i in range(self.dim))

    def purity(self) -> float:
        """
        Calculate purity gamma = Tr(rho^2).
        gamma = 1.0 for pure states, gamma = 1/2^n for maximally mixed states.
        """
        total = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                total += (self.matrix[i][j] * self.matrix[j][i]).real
        return total

    def apply_single_qubit_kraus(self, kraus_ops: Sequence[UnitaryMatrix], target_qubit: int) -> None:
        """
        Apply a set of single-qubit Kraus operators {E_k} to target qubit:
        rho' = sum_k E_k rho E_k^dagger.
        """
        if not (0 <= target_qubit < self.n_qubits):
            raise IndexError(f"Target qubit {target_qubit} out of range")

        new_matrix = [[0.0 + 0.0j for _ in range(self.dim)] for _ in range(self.dim)]
        step = 1 << (target_qubit + 1)
        half = 1 << target_qubit

        for k_op in kraus_ops:
            m00, m01 = k_op.m00, k_op.m01
            m10, m11 = k_op.m10, k_op.m11

            # Transform rows: M * rho
            temp = [[0.0 + 0.0j for _ in range(self.dim)] for _ in range(self.dim)]
            for base in range(0, self.dim, step):
                for i in range(half):
                    r0 = base + i
                    r1 = r0 + half
                    for c in range(self.dim):
                        val0 = self.matrix[r0][c]
                        val1 = self.matrix[r1][c]
                        temp[r0][c] = m00 * val0 + m01 * val1
                        temp[r1][c] = m10 * val0 + m11 * val1

            # Transform cols: (M * rho) * M^dagger
            m00_c, m01_c = m00.conjugate(), m01.conjugate()
            m10_c, m11_c = m10.conjugate(), m11.conjugate()
            for r in range(self.dim):
                for base in range(0, self.dim, step):
                    for j in range(half):
                        c0 = base + j
                        c1 = c0 + half
                        val0 = temp[r][c0]
                        val1 = temp[r][c1]
                        # Col ops: v0 * m00_c + v1 * m01_c
                        new_matrix[r][c0] += val0 * m00_c + val1 * m01_c
                        new_matrix[r][c1] += val0 * m10_c + val1 * m11_c

        self.matrix = new_matrix


class NoiseChannel:
    """Standard quantum decoherence and noise channels."""

    @staticmethod
    def bit_flip(p: float) -> List[UnitaryMatrix]:
        """Bit-flip channel: E0 = sqrt(1-p)*I, E1 = sqrt(p)*X."""
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in [0, 1]")
        e0 = UnitaryMatrix(math.sqrt(1.0 - p), 0.0, 0.0, math.sqrt(1.0 - p))
        e1 = UnitaryMatrix(0.0, math.sqrt(p), math.sqrt(p), 0.0)
        return [e0, e1]

    @staticmethod
    def phase_flip(p: float) -> List[UnitaryMatrix]:
        """Phase-flip channel: E0 = sqrt(1-p)*I, E1 = sqrt(p)*Z."""
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in [0, 1]")
        e0 = UnitaryMatrix(math.sqrt(1.0 - p), 0.0, 0.0, math.sqrt(1.0 - p))
        e1 = UnitaryMatrix(math.sqrt(p), 0.0, 0.0, -math.sqrt(p))
        return [e0, e1]

    @staticmethod
    def depolarizing(p: float) -> List[UnitaryMatrix]:
        """Depolarizing channel: E0 = sqrt(1 - 3p/4)*I, E1..3 = sqrt(p/4)*{X, Y, Z}."""
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be in [0, 1]")
        w0 = math.sqrt(max(0.0, 1.0 - 0.75 * p))
        w1 = math.sqrt(p / 4.0)
        e0 = UnitaryMatrix(w0, 0.0, 0.0, w0)
        e1 = UnitaryMatrix(0.0, w1, w1, 0.0)
        e2 = UnitaryMatrix(0.0, -1.0j * w1, 1.0j * w1, 0.0)
        e3 = UnitaryMatrix(w1, 0.0, 0.0, -w1)
        return [e0, e1, e2, e3]

    @staticmethod
    def amplitude_damping(gamma: float) -> List[UnitaryMatrix]:
        """Amplitude damping (T1 energy relaxation): E0 = [[1, 0], [0, sqrt(1-gamma)]], E1 = [[0, sqrt(gamma)], [0, 0]]."""
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Gamma must be in [0, 1]")
        e0 = UnitaryMatrix(1.0, 0.0, 0.0, math.sqrt(1.0 - gamma))
        e1 = UnitaryMatrix(0.0, math.sqrt(gamma), 0.0, 0.0)
        return [e0, e1]
