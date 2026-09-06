"""
QuantaLab: Quantum Circuit Builder, Execution Engine, and Circuit Depth Analyzer.
Features:
1. QuantumCircuit: Fluent API for composing multi-qubit gate sequences.
2. Classical register tracking and projective mid-circuit measurements.
3. Accurate circuit depth and critical path calculation.
4. Fast statevector simulation engine with exact state output and Monte Carlo sampling.
"""

import time
import math
from typing import List, Dict, Tuple, Optional, Sequence, Union, Any
from .types import StateVector, BasisState
from .gates import (
    Gate, UnitaryMatrix,
    apply_gate_1q, apply_controlled_gate, apply_multi_controlled_gate,
    apply_cnot, apply_swap, apply_cphase, apply_toffoli, apply_fredkin
)


class CircuitOp:
    """Base class for an instruction in a quantum circuit."""
    __slots__ = ("name", "qubits", "clbits", "params")

    def __init__(
        self,
        name: str,
        qubits: Sequence[int],
        clbits: Sequence[int] = (),
        params: Sequence[float] = ()
    ):
        self.name = name
        self.qubits = list(qubits)
        self.clbits = list(clbits)
        self.params = list(params)


class SimulationResult:
    """Encapsulates the output of a quantum circuit statevector simulation."""
    __slots__ = ("state", "clbits", "execution_time_ms", "n_qubits")

    def __init__(self, state: StateVector, clbits: List[int], execution_time_ms: float):
        self.state = state
        self.clbits = clbits
        self.execution_time_ms = execution_time_ms
        self.n_qubits = state.n_qubits

    def probabilities(self, threshold: float = 1e-4) -> Dict[str, float]:
        """Return non-zero basis state probabilities as a dictionary."""
        probs = self.state.probabilities()
        res = {}
        for i, p in enumerate(probs):
            if p >= threshold:
                res[BasisState.to_binary(i, self.n_qubits)] = p
        return res

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Perform Monte Carlo sampling on the output state."""
        return self.state.sample_measurements(shots)

    def most_frequent(self, shots: int = 1024) -> str:
        """Return the bitstring outcome with the highest sample count."""
        samples = self.sample(shots)
        return max(samples.items(), key=lambda item: item[1])[0]


class QuantumCircuit:
    """
    Quantum Circuit representing a sequence of quantum gates and measurement operations.
    Supports fluent chaining: circ.h(0).cx(0, 1).measure(0, 0).
    """

    def __init__(self, n_qubits: int, n_clbits: int = 0):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.ops: List[CircuitOp] = []

    def _validate_qubits(self, *qubits: int) -> None:
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise IndexError(f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit")

    def _validate_clbit(self, clbit: int) -> None:
        if not (0 <= clbit < self.n_clbits):
            raise IndexError(f"Classical bit index {clbit} out of range for {self.n_clbits} clbits")

    # -------------------------------------------------------------------------
    # 1-Qubit Gate Operations
    # -------------------------------------------------------------------------

    def h(self, qubit: int) -> "QuantumCircuit":
        """Apply Hadamard gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("h", [qubit]))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-X (NOT) gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("x", [qubit]))
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Y gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("y", [qubit]))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Z gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("z", [qubit]))
        return self

    def s(self, qubit: int) -> "QuantumCircuit":
        """Apply Phase (S) gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("s", [qubit]))
        return self

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """Apply S-dagger gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("sdg", [qubit]))
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        """Apply T (pi/8) gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("t", [qubit]))
        return self

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """Apply T-dagger gate."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("tdg", [qubit]))
        return self

    def rx(self, qubit: int, theta: float) -> "QuantumCircuit":
        """Apply rotation around X axis by theta."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("rx", [qubit], params=[theta]))
        return self

    def ry(self, qubit: int, theta: float) -> "QuantumCircuit":
        """Apply rotation around Y axis by theta."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("ry", [qubit], params=[theta]))
        return self

    def rz(self, qubit: int, theta: float) -> "QuantumCircuit":
        """Apply rotation around Z axis by theta."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("rz", [qubit], params=[theta]))
        return self

    def p(self, qubit: int, lambda_angle: float) -> "QuantumCircuit":
        """Apply Phase Shift gate P(lambda)."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("p", [qubit], params=[lambda_angle]))
        return self

    def u3(self, qubit: int, theta: float, phi: float, lambda_angle: float) -> "QuantumCircuit":
        """Apply arbitrary single-qubit unitary U3(theta, phi, lambda)."""
        self._validate_qubits(qubit)
        self.ops.append(CircuitOp("u3", [qubit], params=[theta, phi, lambda_angle]))
        return self

    # -------------------------------------------------------------------------
    # 2-Qubit & Multi-Qubit Gate Operations
    # -------------------------------------------------------------------------

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Apply Controlled-NOT (CNOT) gate."""
        self._validate_qubits(control, target)
        if control == target:
            raise ValueError("Control and target must be distinct")
        self.ops.append(CircuitOp("cx", [control, target]))
        return self

    cnot = cx

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Apply Controlled-Y gate."""
        self._validate_qubits(control, target)
        if control == target:
            raise ValueError("Control and target must be distinct")
        self.ops.append(CircuitOp("cy", [control, target]))
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Apply Controlled-Z gate."""
        self._validate_qubits(control, target)
        if control == target:
            raise ValueError("Control and target must be distinct")
        self.ops.append(CircuitOp("cz", [control, target]))
        return self

    def cp(self, control: int, target: int, theta: float) -> "QuantumCircuit":
        """Apply Controlled-Phase gate."""
        self._validate_qubits(control, target)
        if control == target:
            raise ValueError("Control and target must be distinct")
        self.ops.append(CircuitOp("cp", [control, target], params=[theta]))
        return self

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        """Apply SWAP gate between q1 and q2."""
        self._validate_qubits(q1, q2)
        if q1 == q2:
            raise ValueError("SWAP qubits must be distinct")
        self.ops.append(CircuitOp("swap", [q1, q2]))
        return self

    def ccx(self, c1: int, c2: int, target: int) -> "QuantumCircuit":
        """Apply Toffoli (CCX) gate."""
        self._validate_qubits(c1, c2, target)
        if len({c1, c2, target}) != 3:
            raise ValueError("Toffoli qubits must be distinct")
        self.ops.append(CircuitOp("ccx", [c1, c2, target]))
        return self

    toffoli = ccx

    def cswap(self, control: int, t1: int, t2: int) -> "QuantumCircuit":
        """Apply Fredkin (Controlled-SWAP) gate."""
        self._validate_qubits(control, t1, t2)
        if len({control, t1, t2}) != 3:
            raise ValueError("Fredkin qubits must be distinct")
        self.ops.append(CircuitOp("cswap", [control, t1, t2]))
        return self

    fredkin = cswap

    def mcz(self, controls: Sequence[int], target: int) -> "QuantumCircuit":
        """Apply multi-controlled Z gate."""
        all_qs = list(controls) + [target]
        self._validate_qubits(*all_qs)
        if len(set(all_qs)) != len(all_qs):
            raise ValueError("Control and target qubits must all be distinct")
        self.ops.append(CircuitOp("mcz", all_qs))
        return self

    # -------------------------------------------------------------------------
    # Measurements & Barrier
    # -------------------------------------------------------------------------

    def measure(self, qubit: int, clbit: int) -> "QuantumCircuit":
        """Measure qubit into classical bit."""
        self._validate_qubits(qubit)
        self._validate_clbit(clbit)
        self.ops.append(CircuitOp("measure", [qubit], [clbit]))
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits in order into corresponding classical bits."""
        if self.n_clbits < self.n_qubits:
            self.n_clbits = self.n_qubits
        for q in range(self.n_qubits):
            self.ops.append(CircuitOp("measure", [q], [q]))
        return self

    def barrier(self, *qubits: int) -> "QuantumCircuit":
        """Insert execution barrier for compiler optimization and visualization."""
        target_qs = list(qubits) if qubits else list(range(self.n_qubits))
        self._validate_qubits(*target_qs)
        self.ops.append(CircuitOp("barrier", target_qs))
        return self

    # -------------------------------------------------------------------------
    # Circuit Properties & Analysis
    # -------------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Compute the circuit depth (critical path length)."""
        qubit_depths = [0] * self.n_qubits
        for op in self.ops:
            if op.name == "barrier":
                continue
            max_d = max(qubit_depths[q] for q in op.qubits)
            new_d = max_d + 1
            for q in op.qubits:
                qubit_depths[q] = new_d
        return max(qubit_depths) if qubit_depths else 0

    @property
    def gate_count(self) -> int:
        """Count total gates excluding barriers."""
        return sum(1 for op in self.ops if op.name not in ("barrier", "measure"))

    def count_ops(self) -> Dict[str, int]:
        """Return counts of each gate and operation type."""
        counts: Dict[str, int] = {}
        for op in self.ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    # -------------------------------------------------------------------------
    # Simulation Runner
    # -------------------------------------------------------------------------

    def simulate(self, initial_state: Optional[StateVector] = None) -> SimulationResult:
        """
        Execute the circuit on a quantum statevector in O(2^N) per gate.
        Returns a SimulationResult with final state and classical registers.
        """
        t0 = time.perf_counter()
        if initial_state is None:
            state = StateVector.zero(self.n_qubits)
        else:
            if initial_state.n_qubits != self.n_qubits:
                raise ValueError(f"State dimension mismatch: {initial_state.n_qubits} vs {self.n_qubits}")
            state = initial_state.copy()

        clbits = [0] * self.n_clbits

        for op in self.ops:
            name = op.name
            qs = op.qubits

            if name == "h":
                apply_gate_1q(state, Gate.H, qs[0])
            elif name == "x":
                apply_gate_1q(state, Gate.X, qs[0])
            elif name == "y":
                apply_gate_1q(state, Gate.Y, qs[0])
            elif name == "z":
                apply_gate_1q(state, Gate.Z, qs[0])
            elif name == "s":
                apply_gate_1q(state, Gate.S, qs[0])
            elif name == "sdg":
                apply_gate_1q(state, Gate.Sdg, qs[0])
            elif name == "t":
                apply_gate_1q(state, Gate.T, qs[0])
            elif name == "tdg":
                apply_gate_1q(state, Gate.Tdg, qs[0])
            elif name == "rx":
                apply_gate_1q(state, Gate.rx(op.params[0]), qs[0])
            elif name == "ry":
                apply_gate_1q(state, Gate.ry(op.params[0]), qs[0])
            elif name == "rz":
                apply_gate_1q(state, Gate.rz(op.params[0]), qs[0])
            elif name == "p":
                apply_gate_1q(state, Gate.phase(op.params[0]), qs[0])
            elif name == "u3":
                apply_gate_1q(state, Gate.u3(op.params[0], op.params[1], op.params[2]), qs[0])
            elif name == "cx":
                apply_cnot(state, qs[0], qs[1])
            elif name == "cy":
                apply_controlled_gate(state, Gate.Y, qs[0], qs[1])
            elif name == "cz":
                apply_controlled_gate(state, Gate.Z, qs[0], qs[1])
            elif name == "cp":
                apply_cphase(state, qs[0], qs[1], op.params[0])
            elif name == "swap":
                apply_swap(state, qs[0], qs[1])
            elif name == "ccx":
                apply_toffoli(state, qs[0], qs[1], qs[2])
            elif name == "cswap":
                apply_fredkin(state, qs[0], qs[1], qs[2])
            elif name == "mcz":
                apply_multi_controlled_gate(state, Gate.Z, qs[:-1], qs[-1])
            elif name == "measure":
                q = qs[0]
                cb = op.clbits[0]
                measured_bit, collapsed_state = state.measure_qubit(q)
                clbits[cb] = measured_bit
                state = collapsed_state
            elif name == "barrier":
                pass
            else:
                raise ValueError(f"Unknown operation '{name}'")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SimulationResult(state, clbits, elapsed_ms)

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Simulate and sample measurement outcomes."""
        has_measures = any(op.name == "measure" for op in self.ops)
        if not has_measures:
            res = self.simulate()
            return res.sample(shots)

        counts: Dict[str, int] = {}
        for _ in range(shots):
            res = self.simulate()
            bitstring = "".join(str(b) for b in res.clbits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return dict(sorted(counts.items()))
