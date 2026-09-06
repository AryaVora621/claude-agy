"""
QuantaLab: Quantum State Vector & Complex Linear Algebra Primitives.
Features:
1. StateVector: N-qubit quantum state representation with 2^N complex amplitudes.
2. Tensor products, inner products, and quantum state fidelity.
3. Projective measurement with non-destructive sampling and state collapse (Born Rule).
4. Partial trace and reduced density matrices.
5. Pauli expectation values and Bloch sphere coordinates (x, y, z).
"""

import math
import cmath
import random
from typing import List, Tuple, Dict, Sequence, Optional, Union


class BasisState:
    """Helper for formatting and interpreting quantum computational basis states."""

    @staticmethod
    def to_binary(index: int, n_qubits: int) -> str:
        """Format an integer index into an N-qubit bitstring (e.g. 3 with 3 qubits -> '011')."""
        return f"{index:0{n_qubits}b}"

    @staticmethod
    def from_binary(bitstring: str) -> int:
        """Convert a bitstring (e.g. '011') into an integer index (3)."""
        return int(bitstring, 2)


class StateVector:
    """
    N-qubit pure quantum state vector in the 2^N dimensional Hilbert space.
    Amplitudes are ordered according to standard computational basis:
    |00...0>, |00...1>, ..., |11...1>.
    """
    __slots__ = ("n_qubits", "dim", "amplitudes")

    def __init__(self, amplitudes: Sequence[complex], n_qubits: Optional[int] = None):
        self.amplitudes = list(amplitudes)
        self.dim = len(self.amplitudes)
        if self.dim == 0:
            raise ValueError("StateVector cannot be empty")

        if n_qubits is None:
            n = int(math.log2(self.dim))
            if (1 << n) != self.dim:
                raise ValueError(f"Amplitude length {self.dim} is not a power of 2")
            self.n_qubits = n
        else:
            if (1 << n_qubits) != self.dim:
                raise ValueError(f"Provided n_qubits={n_qubits} does not match amplitude count {self.dim}")
            self.n_qubits = n_qubits

    @classmethod
    def zero(cls, n_qubits: int) -> "StateVector":
        """Initialize the ground state |00...0> for N qubits."""
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        dim = 1 << n_qubits
        amps = [0.0 + 0.0j] * dim
        amps[0] = 1.0 + 0.0j
        return cls(amps, n_qubits)

    @classmethod
    def from_basis(cls, n_qubits: int, basis_state: Union[int, str]) -> "StateVector":
        """Initialize a basis state |k> by integer index or bitstring."""
        if isinstance(basis_state, str):
            basis_state = BasisState.from_binary(basis_state)
        dim = 1 << n_qubits
        if not (0 <= basis_state < dim):
            raise IndexError(f"Basis state {basis_state} out of bounds for {n_qubits} qubits")
        amps = [0.0 + 0.0j] * dim
        amps[basis_state] = 1.0 + 0.0j
        return cls(amps, n_qubits)

    def copy(self) -> "StateVector":
        """Create a deep copy of the state vector."""
        return StateVector(list(self.amplitudes), self.n_qubits)

    def norm_squared(self) -> float:
        """Calculate the sum of squared magnitudes of all amplitudes."""
        return sum(abs(a) ** 2 for a in self.amplitudes)

    def norm(self) -> float:
        """Calculate the Euclidean norm of the state vector."""
        return math.sqrt(self.norm_squared())

    def is_normalized(self, tol: float = 1e-6) -> bool:
        """Verify if the state vector satisfies the unit normalization condition."""
        return abs(self.norm_squared() - 1.0) < tol

    def normalize(self) -> "StateVector":
        """Normalize the state vector in-place and return self."""
        n = self.norm()
        if n < 1e-15:
            raise ValueError("Cannot normalize zero-vector")
        inv_n = 1.0 / n
        self.amplitudes = [a * inv_n for a in self.amplitudes]
        return self

    def inner(self, other: "StateVector") -> complex:
        """
        Compute the Dirac inner product <self | other> = sum(a_i^* * b_i).
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return sum(self.amplitudes[i].conjugate() * other.amplitudes[i] for i in range(self.dim))

    def fidelity(self, other: "StateVector") -> float:
        """
        Calculate state fidelity F = |<self | other>|^2.
        F = 1.0 indicates identical quantum states up to global phase.
        """
        overlap = self.inner(other)
        return abs(overlap) ** 2

    def tensor(self, other: "StateVector") -> "StateVector":
        """
        Compute the Kronecker tensor product |self> (x) |other>.
        New qubit count is self.n_qubits + other.n_qubits.
        """
        new_n = self.n_qubits + other.n_qubits
        new_amps = []
        for a in self.amplitudes:
            for b in other.amplitudes:
                new_amps.append(a * b)
        return StateVector(new_amps, new_n)

    def probabilities(self) -> List[float]:
        """Return the probability distribution P(i) = |a_i|^2 across all computational basis states."""
        return [abs(a) ** 2 for a in self.amplitudes]

    def marginal_probability(self, qubit: int, outcome: int) -> float:
        """
        Calculate the marginal probability of observing 'outcome' (0 or 1) on 'qubit'.
        Qubit index 0 is the least significant bit (rightmost).
        """
        if not (0 <= qubit < self.n_qubits):
            raise IndexError(f"Qubit index {qubit} out of range (0 to {self.n_qubits - 1})")
        if outcome not in (0, 1):
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")

        mask = 1 << qubit
        total_p = 0.0
        for i in range(self.dim):
            bit = 1 if (i & mask) else 0
            if bit == outcome:
                total_p += abs(self.amplitudes[i]) ** 2
        return total_p

    def measure_qubit(self, qubit: int, random_val: Optional[float] = None) -> Tuple[int, "StateVector"]:
        """
        Perform a projective measurement on 'qubit', collapsing the state vector (Born Rule).
        Returns (measured_bit, collapsed_state).
        """
        if not (0 <= qubit < self.n_qubits):
            raise IndexError(f"Qubit index {qubit} out of range")

        p0 = self.marginal_probability(qubit, 0)
        p1 = 1.0 - p0

        # Choose outcome
        r = random.random() if random_val is None else random_val
        outcome = 0 if r < p0 else 1
        p_outcome = p0 if outcome == 0 else p1

        if p_outcome < 1e-15:
            # Degenerate case, collapse to valid non-zero component
            outcome = 1 - outcome
            p_outcome = 1.0 - p_outcome

        # Collapse state: zero out non-matching amplitudes and renormalize
        mask = 1 << qubit
        norm_factor = 1.0 / math.sqrt(p_outcome)
        new_amps = []
        for i in range(self.dim):
            bit = 1 if (i & mask) else 0
            if bit == outcome:
                new_amps.append(self.amplitudes[i] * norm_factor)
            else:
                new_amps.append(0.0 + 0.0j)

        return outcome, StateVector(new_amps, self.n_qubits)

    def sample_measurements(self, shots: int = 1024) -> Dict[str, int]:
        """
        Perform Monte Carlo sampling of full-register measurements over 'shots' iterations.
        Does not mutate the underlying quantum state.
        """
        probs = self.probabilities()
        indices = list(range(self.dim))
        samples = random.choices(indices, weights=probs, k=shots)

        counts: Dict[str, int] = {}
        for s in samples:
            bitstring = BasisState.to_binary(s, self.n_qubits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return dict(sorted(counts.items()))

    def partial_trace_single_qubit(self, target_qubit: int) -> List[List[complex]]:
        """
        Compute the 2x2 reduced density matrix rho for 'target_qubit' by tracing out all other qubits.
        rho[i][j] = sum_{other states} a_{i, other} * a_{j, other}^*
        """
        if not (0 <= target_qubit < self.n_qubits):
            raise IndexError(f"Qubit index {target_qubit} out of range")

        rho = [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
        mask = 1 << target_qubit

        for i in range(self.dim):
            bit_i = 1 if (i & mask) else 0
            amp_i = self.amplitudes[i]
            if abs(amp_i) < 1e-15:
                continue

            for j in range(self.dim):
                # Only states differing at most by the target qubit contribute to the partial trace
                if (i ^ j) in (0, mask):
                    bit_j = 1 if (j & mask) else 0
                    amp_j = self.amplitudes[j]
                    rho[bit_i][bit_j] += amp_i * amp_j.conjugate()

        return rho

    def bloch_coordinates(self, target_qubit: int = 0) -> Tuple[float, float, float]:
        """
        Calculate the (x, y, z) Bloch sphere coordinates for a single qubit subsystem:
        x = <X> = 2 * Re(rho_01)
        y = <Y> = 2 * Im(rho_10) = -2 * Im(rho_01)
        z = <Z> = rho_00 - rho_11
        Returns tuple (x, y, z) where x^2 + y^2 + z^2 <= 1.0.
        """
        rho = self.partial_trace_single_qubit(target_qubit)
        x = 2.0 * rho[0][1].real
        y = 2.0 * rho[1][0].imag
        z = rho[0][0].real - rho[1][1].real
        return (x, y, z)

    def to_ket_string(self, threshold: float = 1e-4) -> str:
        """Format the state vector into Dirac ket notation (e.g. '0.707|00> + 0.707|11>')."""
        terms = []
        for i, a in enumerate(self.amplitudes):
            if abs(a) >= threshold:
                bs = BasisState.to_binary(i, self.n_qubits)
                if abs(a.imag) < 1e-6:
                    coeff = f"{a.real:+.3f}"
                elif abs(a.real) < 1e-6:
                    coeff = f"{a.imag:+.3f}j"
                else:
                    coeff = f"({a.real:+.3f}{a.imag:+.3f}j)"
                terms.append(f"{coeff}|{bs}>")

        if not terms:
            return "0"
        res = " ".join(terms)
        if res.startswith("+"):
            res = res[1:]
        return res
