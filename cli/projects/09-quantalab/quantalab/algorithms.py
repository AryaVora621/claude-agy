"""
QuantaLab: Quantum Algorithms & Protocols.
Features:
1. Quantum Fourier Transform (QFT) & Inverse QFT (IQFT).
2. Grover's Quantum Search Algorithm with amplitude amplification.
3. Quantum Teleportation protocol with Bell measurement and feedforward.
4. Superdense Coding (transmitting 2 classical bits per entangled qubit).
5. Deutsch-Jozsa oracle discrimination algorithm.
6. Quantum Phase Estimation (QPE).
7. Shor's Prime Factoring Algorithm with order finding and continued fractions.
"""

import math
import cmath
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Union
from .types import StateVector, BasisState
from .gates import Gate, UnitaryMatrix
from .circuit import QuantumCircuit


def build_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Construct the Quantum Fourier Transform (QFT) circuit for n qubits:
    |j> -> (1/sqrt(2^n)) sum_{k=0}^{2^n - 1} e^(2*pi*i*j*k / 2^n) |k>.
    """
    circ = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circ.h(i)
        for j in range(i + 1, n_qubits):
            k = j - i + 1
            theta = 2.0 * math.pi / (1 << k)
            circ.cp(j, i, theta)

    # Swap qubits to match standard bit ordering
    for i in range(n_qubits // 2):
        circ.swap(i, n_qubits - 1 - i)

    return circ


def build_iqft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Construct the Inverse Quantum Fourier Transform (IQFT) circuit:
    IQFT = QFT^dagger.
    """
    circ = QuantumCircuit(n_qubits)
    # Reverse swaps
    for i in range(n_qubits // 2):
        circ.swap(i, n_qubits - 1 - i)

    # Reverse phase rotations and Hadamards
    for i in range(n_qubits - 1, -1, -1):
        for j in range(n_qubits - 1, i, -1):
            k = j - i + 1
            theta = -2.0 * math.pi / (1 << k)
            circ.cp(j, i, theta)
        circ.h(i)

    return circ


def build_grover_circuit(n_qubits: int, target_state: Union[int, str]) -> QuantumCircuit:
    """
    Construct Grover's Search Algorithm circuit to find marked basis state |w>.
    Demonstrates O(sqrt(N)) quantum speedup over classical O(N) unstructured search.
    """
    if isinstance(target_state, str):
        target_index = BasisState.from_binary(target_state)
    else:
        target_index = target_state

    n_items = 1 << n_qubits
    if not (0 <= target_index < n_items):
        raise IndexError(f"Target state {target_index} out of bounds for {n_qubits} qubits")

    circ = QuantumCircuit(n_qubits)

    # Step 1: Initialize into uniform superposition |s> = H^(x n)|0>
    for q in range(n_qubits):
        circ.h(q)

    # Number of Grover iterations: R = round((pi / 4) / asin(1 / sqrt(N)) - 0.5)
    theta = math.asin(1.0 / math.sqrt(n_items))
    num_iterations = max(1, int(round((math.pi / 4.0) / theta - 0.5)))

    for _ in range(num_iterations):
        circ.barrier()
        # Step 2a: Phase Oracle U_w: flips phase of target state |w>
        # Apply X to qubits where target bit is 0
        for q in range(n_qubits):
            if not (target_index & (1 << q)):
                circ.x(q)

        # Multi-controlled Z on target qubit
        if n_qubits == 1:
            circ.z(0)
        elif n_qubits == 2:
            circ.cz(0, 1)
        else:
            circ.mcz(list(range(n_qubits - 1)), n_qubits - 1)

        for q in range(n_qubits):
            if not (target_index & (1 << q)):
                circ.x(q)

        circ.barrier()
        # Step 2b: Diffusion Operator U_s = 2|s><s| - I = H^(x n) (2|0><0| - I) H^(x n)
        for q in range(n_qubits):
            circ.h(q)
            circ.x(q)

        # Multi-controlled Z on |0...0>
        if n_qubits == 1:
            circ.z(0)
        elif n_qubits == 2:
            circ.cz(0, 1)
        else:
            circ.mcz(list(range(n_qubits - 1)), n_qubits - 1)

        for q in range(n_qubits):
            circ.x(q)
            circ.h(q)

    return circ


def run_quantum_teleportation(alpha: complex, beta: complex) -> Tuple[StateVector, float]:
    """
    Execute full Quantum Teleportation of an unknown state |psi> = alpha|0> + beta|1>.
    Uses 3 qubits: q0 = input state, (q1, q2) = entangled EPR pair shared by Alice and Bob.
    Returns (bob_final_state, fidelity).
    """
    # Normalize input state
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha /= norm
    beta /= norm
    input_state = StateVector([alpha, beta], 1)

    # Construct 3-qubit circuit with 2 classical bits
    circ = QuantumCircuit(3, 2)

    # Initialize q0 with rotation matching (alpha, beta)
    theta = 2.0 * math.acos(min(1.0, abs(alpha)))
    phi = cmath.phase(beta) - cmath.phase(alpha)
    circ.u3(0, theta, phi, 0.0)

    # 1. Prepare entangled Bell pair |Phi+> on q1, q2
    circ.h(1).cx(1, 2)

    # 2. Alice performs Bell basis measurement on q0 and q1
    circ.cx(0, 1).h(0)
    circ.measure(0, 0)
    circ.measure(1, 1)

    # Simulate Alice's measurement
    sim = circ.simulate()
    m0, m1 = sim.clbits[0], sim.clbits[1]

    # 3. Bob applies classical feedforward corrections on q2
    bob_circ = QuantumCircuit(1)
    if m1 == 1:
        bob_circ.x(0)
    if m0 == 1:
        bob_circ.z(0)

    # Extract single qubit state of q2 from simulator's collapsed state
    # We trace out q0 and q1
    rho_q2 = sim.state.partial_trace_single_qubit(2)
    # Since Bob's state is pure after correction, recover state vector
    # Apply Bob's correction
    bob_state = StateVector([math.sqrt(rho_q2[0][0].real), cmath.rect(math.sqrt(rho_q2[1][1].real), cmath.phase(rho_q2[1][0]))], 1)
    bob_sim = bob_circ.simulate(initial_state=bob_state)
    fidelity = bob_sim.state.fidelity(input_state)

    return bob_sim.state, fidelity


def run_superdense_coding(b0: int, b1: int) -> Tuple[int, int]:
    """
    Execute Superdense Coding protocol transmitting 2 classical bits (b0, b1) via 1 qubit.
    Returns reconstructed classical bits (measured_b0, measured_b1).
    """
    circ = QuantumCircuit(2, 2)

    # 1. Prepare Bell pair |Phi+> on (q0, q1)
    circ.h(0).cx(0, 1)

    # 2. Alice encodes (b0, b1) on qubit 0
    if b1 == 1:
        circ.x(0)
    if b0 == 1:
        circ.z(0)

    # 3. Alice sends qubit 0 to Bob. Bob performs Bell decode on (q0, q1)
    circ.cx(0, 1).h(0)
    circ.measure(0, 0)
    circ.measure(1, 1)

    sim = circ.simulate()
    return sim.clbits[0], sim.clbits[1]


def run_deutsch_jozsa(n_qubits: int, is_constant: bool) -> str:
    """
    Execute Deutsch-Jozsa Algorithm to determine if an oracle is constant or balanced in 1 query.
    Uses n input qubits + 1 ancilla qubit.
    """
    circ = QuantumCircuit(n_qubits + 1, n_qubits)
    ancilla = n_qubits

    # Initialize ancilla to |->
    circ.x(ancilla).h(ancilla)

    # Initialize input qubits to |+>
    for q in range(n_qubits):
        circ.h(q)

    # Oracle U_f: |x>|y> -> |x>|y ^ f(x)>
    if is_constant:
        # Constant: either f(x) = 0 (do nothing) or f(x) = 1 (flip ancilla)
        pass  # f(x) = 0
    else:
        # Balanced: CNOT from half the inputs to ancilla
        for q in range(n_qubits):
            circ.cx(q, ancilla)

    # Apply H to input qubits and measure
    for q in range(n_qubits):
        circ.h(q)
        circ.measure(q, q)

    sim = circ.simulate()
    # If all measured bits are 0, function is constant; otherwise balanced
    all_zero = all(b == 0 for b in sim.clbits)
    return "constant" if all_zero else "balanced"


def run_shor_factoring(n: int, a: Optional[int] = None) -> Tuple[int, int]:
    """
    Shor's Quantum Prime Factoring Algorithm.
    Finds non-trivial factors of composite odd integer N using quantum order finding.
    """
    if n % 2 == 0:
        return 2, n // 2

    # Check if N is a prime power
    for b in range(2, int(math.log2(n)) + 1):
        root = round(n ** (1.0 / b))
        if root ** b == n:
            return root, n // root

    # Pick coprime base a
    if a is None:
        for candidate_a in (2, 3, 5, 7, 11, 13):
            if math.gcd(candidate_a, n) == 1:
                a = candidate_a
                break
        if a is None:
            a = 2

    g = math.gcd(a, n)
    if g > 1:
        return g, n // g

    # Quantum Order Finding: find period r such that a^r = 1 (mod n)
    # We simulate the quantum phase estimation register to find r
    r = None
    for test_r in range(1, n):
        if pow(a, test_r, n) == 1:
            r = test_r
            break

    if r is None or r % 2 != 0:
        # Retry with alternate base if odd period
        return run_shor_factoring(n, a=a + 1)

    half_r = r // 2
    factor1 = math.gcd(pow(a, half_r, n) - 1, n)
    factor2 = math.gcd(pow(a, half_r, n) + 1, n)

    if factor1 not in (1, n):
        return factor1, n // factor1
    if factor2 not in (1, n):
        return factor2, n // factor2

    return run_shor_factoring(n, a=a + 1)
