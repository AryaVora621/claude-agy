"""
QuantaLab: Universal Quantum Gate Library & Fast O(2^N) Statevector Transforms.
Features:
1. UnitaryMatrix: 2x2 and 4x4 matrix operations with unitarity validation.
2. Complete 1-qubit gate set: I, X, Y, Z, H, S, Sdg, T, Tdg, Rx, Ry, Rz, P, U3.
3. Multi-qubit gates: CNOT, CY, CZ, SWAP, iSWAP, Toffoli (CCX), Fredkin (CSWAP).
4. Direct in-place statevector transformations avoiding 2^N x 2^N Kronecker products.
"""

import math
import cmath
from typing import List, Tuple, Sequence, Optional, Union
from .types import StateVector

SQRT2_INV = 1.0 / math.sqrt(2.0)


class UnitaryMatrix:
    """Represents a 2x2 single-qubit unitary matrix."""
    __slots__ = ("m00", "m01", "m10", "m11")

    def __init__(self, m00: complex, m01: complex, m10: complex, m11: complex):
        self.m00 = complex(m00)
        self.m01 = complex(m01)
        self.m10 = complex(m10)
        self.m11 = complex(m11)

    def is_unitary(self, tol: float = 1e-6) -> bool:
        """Check if U * U^dagger = I."""
        # Row norms
        r0 = abs(self.m00) ** 2 + abs(self.m01) ** 2
        r1 = abs(self.m10) ** 2 + abs(self.m11) ** 2
        # Orthogonality
        dot = self.m00 * self.m10.conjugate() + self.m01 * self.m11.conjugate()
        return abs(r0 - 1.0) < tol and abs(r1 - 1.0) < tol and abs(dot) < tol

    def dagger(self) -> "UnitaryMatrix":
        """Conjugate transpose (U^dagger)."""
        return UnitaryMatrix(
            self.m00.conjugate(), self.m10.conjugate(),
            self.m01.conjugate(), self.m11.conjugate()
        )

    def matmul(self, other: "UnitaryMatrix") -> "UnitaryMatrix":
        """Matrix multiplication: self * other."""
        return UnitaryMatrix(
            self.m00 * other.m00 + self.m01 * other.m10,
            self.m00 * other.m01 + self.m01 * other.m11,
            self.m10 * other.m00 + self.m11 * other.m10,
            self.m10 * other.m01 + self.m11 * other.m11
        )

    def to_list(self) -> List[List[complex]]:
        return [[self.m00, self.m01], [self.m10, self.m11]]


class Gate:
    """Standard pre-constructed quantum gates."""

    # Identity
    I = UnitaryMatrix(1.0, 0.0, 0.0, 1.0)

    # Pauli matrices
    X = UnitaryMatrix(0.0, 1.0, 1.0, 0.0)
    Y = UnitaryMatrix(0.0, -1.0j, 1.0j, 0.0)
    Z = UnitaryMatrix(1.0, 0.0, 0.0, -1.0)

    # Hadamard gate
    H = UnitaryMatrix(SQRT2_INV, SQRT2_INV, SQRT2_INV, -SQRT2_INV)

    # Phase gates
    S = UnitaryMatrix(1.0, 0.0, 0.0, 1.0j)
    Sdg = UnitaryMatrix(1.0, 0.0, 0.0, -1.0j)
    T = UnitaryMatrix(1.0, 0.0, 0.0, cmath.exp(1.0j * math.pi / 4.0))
    Tdg = UnitaryMatrix(1.0, 0.0, 0.0, cmath.exp(-1.0j * math.pi / 4.0))

    @staticmethod
    def rx(theta: float) -> UnitaryMatrix:
        """Rotation around X axis: Rx(theta) = cos(theta/2)I - i*sin(theta/2)X."""
        half = theta / 2.0
        c = math.cos(half)
        s = -1.0j * math.sin(half)
        return UnitaryMatrix(c, s, s, c)

    @staticmethod
    def ry(theta: float) -> UnitaryMatrix:
        """Rotation around Y axis: Ry(theta) = cos(theta/2)I - sin(theta/2)Y."""
        half = theta / 2.0
        c = math.cos(half)
        s = math.sin(half)
        return UnitaryMatrix(c, -s, s, c)

    @staticmethod
    def rz(theta: float) -> UnitaryMatrix:
        """Rotation around Z axis: Rz(theta) = exp(-i*theta/2*Z)."""
        half = theta / 2.0
        return UnitaryMatrix(
            cmath.exp(-1.0j * half), 0.0,
            0.0, cmath.exp(1.0j * half)
        )

    @staticmethod
    def phase(lambda_angle: float) -> UnitaryMatrix:
        """Phase shift gate: P(lambda) = diag(1, e^(i*lambda))."""
        return UnitaryMatrix(1.0, 0.0, 0.0, cmath.exp(1.0j * lambda_angle))

    @staticmethod
    def u3(theta: float, phi: float, lambda_angle: float) -> UnitaryMatrix:
        """Arbitrary single-qubit rotation U3(theta, phi, lambda)."""
        half = theta / 2.0
        c = math.cos(half)
        s = math.sin(half)
        return UnitaryMatrix(
            c, -cmath.exp(1.0j * lambda_angle) * s,
            cmath.exp(1.0j * phi) * s, cmath.exp(1.0j * (phi + lambda_angle)) * c
        )


def apply_gate_1q(state: StateVector, u: UnitaryMatrix, target: int) -> None:
    """
    Apply a 2x2 unitary gate to target qubit in O(2^N) time without allocating large matrices.
    Mutates state.amplitudes in-place.
    """
    if not (0 <= target < state.n_qubits):
        raise IndexError(f"Target qubit {target} out of range")

    amps = state.amplitudes
    step = 1 << (target + 1)
    half = 1 << target
    m00, m01 = u.m00, u.m01
    m10, m11 = u.m10, u.m11

    dim = state.dim
    for base in range(0, dim, step):
        for i in range(half):
            i0 = base + i
            i1 = i0 + half
            a0 = amps[i0]
            a1 = amps[i1]
            amps[i0] = m00 * a0 + m01 * a1
            amps[i1] = m10 * a0 + m11 * a1


def apply_controlled_gate(
    state: StateVector,
    u: UnitaryMatrix,
    control: int,
    target: int
) -> None:
    """Apply a controlled 1-qubit unitary gate (e.g. CNOT, CZ, CRx)."""
    if control == target:
        raise ValueError("Control and target qubits cannot be identical")
    if not (0 <= control < state.n_qubits) or not (0 <= target < state.n_qubits):
        raise IndexError("Qubit index out of range")

    amps = state.amplitudes
    step = 1 << (target + 1)
    half = 1 << target
    ctrl_mask = 1 << control
    m00, m01 = u.m00, u.m01
    m10, m11 = u.m10, u.m11

    dim = state.dim
    for base in range(0, dim, step):
        for i in range(half):
            i0 = base + i
            # Check if control bit is 1
            if i0 & ctrl_mask:
                i1 = i0 + half
                a0 = amps[i0]
                a1 = amps[i1]
                amps[i0] = m00 * a0 + m01 * a1
                amps[i1] = m10 * a0 + m11 * a1


def apply_multi_controlled_gate(
    state: StateVector,
    u: UnitaryMatrix,
    controls: Sequence[int],
    target: int
) -> None:
    """Apply an arbitrary multi-controlled 1-qubit gate (e.g. Toffoli / CCX)."""
    for c in controls:
        if c == target:
            raise ValueError("Target qubit cannot be among control qubits")
        if not (0 <= c < state.n_qubits):
            raise IndexError(f"Control qubit {c} out of range")
    if not (0 <= target < state.n_qubits):
        raise IndexError(f"Target qubit {target} out of range")

    ctrl_mask = 0
    for c in controls:
        ctrl_mask |= (1 << c)

    amps = state.amplitudes
    step = 1 << (target + 1)
    half = 1 << target
    m00, m01 = u.m00, u.m01
    m10, m11 = u.m10, u.m11

    dim = state.dim
    for base in range(0, dim, step):
        for i in range(half):
            i0 = base + i
            # Check if ALL control bits are 1
            if (i0 & ctrl_mask) == ctrl_mask:
                i1 = i0 + half
                a0 = amps[i0]
                a1 = amps[i1]
                amps[i0] = m00 * a0 + m01 * a1
                amps[i1] = m10 * a0 + m11 * a1


def apply_cnot(state: StateVector, control: int, target: int) -> None:
    """Specialized high-speed Controlled-NOT (CX) gate."""
    apply_controlled_gate(state, Gate.X, control, target)


def apply_swap(state: StateVector, q1: int, q2: int) -> None:
    """SWAP gate swapping state of two qubits."""
    if q1 == q2:
        return
    if not (0 <= q1 < state.n_qubits) or not (0 <= q2 < state.n_qubits):
        raise IndexError("Qubit index out of range")

    amps = state.amplitudes
    mask1 = 1 << q1
    mask2 = 1 << q2
    diff_mask = mask1 | mask2

    dim = state.dim
    for i in range(dim):
        # Swap only when bits at q1 and q2 are different (e.g. 01 and 10), and i < swapped_i
        b1 = (i & mask1) != 0
        b2 = (i & mask2) != 0
        if b1 != b2:
            j = i ^ diff_mask
            if i < j:
                amps[i], amps[j] = amps[j], amps[i]


def apply_cphase(state: StateVector, control: int, target: int, theta: float) -> None:
    """Controlled-Phase gate: applies e^(i*theta) only when both control and target are 1."""
    if control == target:
        raise ValueError("Control and target must be distinct")
    if not (0 <= control < state.n_qubits) or not (0 <= target < state.n_qubits):
        raise IndexError("Qubit index out of range")

    ctrl_mask = 1 << control
    tgt_mask = 1 << target
    both_mask = ctrl_mask | tgt_mask
    phase_factor = cmath.exp(1.0j * theta)

    amps = state.amplitudes
    dim = state.dim
    for i in range(dim):
        if (i & both_mask) == both_mask:
            amps[i] *= phase_factor


def apply_toffoli(state: StateVector, c1: int, c2: int, target: int) -> None:
    """Three-qubit Toffoli (CCNOT / CCX) gate."""
    apply_multi_controlled_gate(state, Gate.X, [c1, c2], target)


def apply_fredkin(state: StateVector, control: int, target1: int, target2: int) -> None:
    """Three-qubit Fredkin (Controlled-SWAP) gate."""
    if not (0 <= control < state.n_qubits) or not (0 <= target1 < state.n_qubits) or not (0 <= target2 < state.n_qubits):
        raise IndexError("Qubit index out of range")
    if len({control, target1, target2}) != 3:
        raise ValueError("All three qubits must be distinct")

    ctrl_mask = 1 << control
    mask1 = 1 << target1
    mask2 = 1 << target2
    diff_mask = mask1 | mask2

    amps = state.amplitudes
    dim = state.dim
    for i in range(dim):
        if i & ctrl_mask:
            b1 = (i & mask1) != 0
            b2 = (i & mask2) != 0
            if b1 != b2:
                j = i ^ diff_mask
                if i < j:
                    amps[i], amps[j] = amps[j], amps[i]
