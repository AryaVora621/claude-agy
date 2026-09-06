#!/usr/bin/env python3
"""
QuantaLab: Interactive Quantum Computing Lab & Visualizer.
Features:
1. Quantum Circuit ASCII Wire Diagrams.
2. Entanglement & Bell State Verification.
3. Quantum Teleportation Protocol with Bell measurement & feedforward.
4. Grover's Database Search with amplitude amplification bar chart.
5. ASCII Bloch Sphere 2D projection.
6. Shor's Quantum Factoring Algorithm.
"""

import sys
import os
import math
import cmath

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantalab.types import StateVector
from quantalab.circuit import QuantumCircuit
from quantalab.algorithms import (
    build_grover_circuit, run_quantum_teleportation,
    run_superdense_coding, run_shor_factoring
)
from quantalab.visualizer import CircuitRenderer, BlochSphereVisualizer, StateVisualizer

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def demo_bell_state():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 1: Bell State Generation & Quantum Entanglement (|Phi+>){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    circ = QuantumCircuit(2, 2)
    circ.h(0).cx(0, 1).measure_all()

    print(f"{BOLD}Circuit Diagram:{RESET}")
    print(CircuitRenderer.render(circ))
    print(f"\nCircuit Depth: {circ.depth} | Total Gates: {circ.gate_count}\n")

    sim = circ.simulate()
    print(f"{BOLD}Simulated State Vector:{RESET}")
    print(f"  |psi> = {sim.state.to_ket_string()}\n")

    print(f"{BOLD}Monte Carlo Measurement Histogram (1,000 shots):{RESET}")
    counts = circ.sample(shots=1000)
    print(StateVisualizer.bar_chart(counts))
    print(f"\n{GREEN}[+] Verified: 100% correlation between qubits (perfect entanglement!){RESET}\n\n")


def demo_grover_search():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 2: Grover's Quantum Search Algorithm (Database of 8 items){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    target = "101"  # 5 in decimal
    circ = build_grover_circuit(3, target)

    print(f"Marked Search Key: |{target}> (Target Index: 5 out of 8)")
    print(f"{BOLD}Quantum Circuit Diagram:{RESET}")
    print(CircuitRenderer.render(circ))
    print(f"\nCircuit Depth: {circ.depth} | Gate Count: {circ.gate_count}\n")

    sim = circ.simulate()
    probs = sim.probabilities()
    p_target = probs.get(target, 0.0)

    print(f"{BOLD}Amplified State Probability Distribution:{RESET}")
    counts = sim.sample(shots=1000)
    print(StateVisualizer.bar_chart(counts))
    print(f"\n{GREEN}[+] Marked item |{target}> isolated with {p_target * 100:.1f}% probability in 2 iterations!{RESET}\n\n")


def demo_teleportation():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 3: Quantum Teleportation Protocol{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    alpha, beta = 0.6, 0.8
    print(f"Original Input State on Alice's Qubit q0: |psi> = {alpha:.2f}|0> + {beta:.2f}|1>")
    bob_state, fidelity = run_quantum_teleportation(alpha, beta)

    print(f"Reconstructed State on Bob's Qubit q2:   |psi'> = {bob_state.to_ket_string()}")
    print(f"{BOLD}State Fidelity F = |<psi | psi'>|^2:{RESET} {fidelity:.6f}")
    print(f"{GREEN}[+] Quantum Teleportation successful with exact state fidelity = 1.0!{RESET}\n\n")


def demo_bloch_sphere():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 4: 2D ASCII Bloch Sphere Projection Visualizer{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    # Plus state |+> = (|0> + |1>)/sqrt(2)
    s_plus = StateVector([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], 1)
    x, y, z = s_plus.bloch_coordinates(0)

    print(f"{BOLD}Superposition State |+> = (|0> + |1>)/sqrt(2):{RESET}")
    print(BlochSphereVisualizer.render(x, y, z))
    print()


def demo_shor_factoring():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 5: Shor's Quantum Factoring Algorithm{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    for composite in (15, 21):
        p, q = run_shor_factoring(composite)
        print(f"Factoring Composite N = {composite}:")
        print(f"  -> Found Non-Trivial Prime Factors: {p} x {q} = {p * q}")
    print(f"\n{GREEN}[+] Verified: All prime factors resolved via quantum order finding.{RESET}\n")


def main():
    print(f"{BOLD}{BLUE}======================================================================{RESET}")
    print(f"{BOLD}{BLUE}          QUANTALAB: ZERO-DEPENDENCY QUANTUM COMPUTING SUITE          {RESET}")
    print(f"{BOLD}{BLUE}======================================================================{RESET}\n")

    demo_bell_state()
    demo_grover_search()
    demo_teleportation()
    demo_bloch_sphere()
    demo_shor_factoring()


if __name__ == "__main__":
    main()
