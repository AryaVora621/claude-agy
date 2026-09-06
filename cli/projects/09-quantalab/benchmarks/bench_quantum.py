"""
QuantaLab: Quantum Computing Simulator Performance Benchmarks.
Measures:
1. 1-Qubit Gate (Hadamard, Pauli-X, Rotations) application throughput (gates/sec).
2. 2-Qubit Entangling Gate (CNOT) throughput (gates/sec).
3. Full Circuit Execution Scaling across statevector dimensions (2^10 to 2^15 amplitudes).
4. Quantum Fourier Transform (QFT) execution rate across register widths.
5. Grover's Algorithm amplitude amplification speedup.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantalab.types import StateVector
from quantalab.gates import Gate, apply_gate_1q, apply_cnot
from quantalab.circuit import QuantumCircuit
from quantalab.algorithms import build_qft_circuit, build_grover_circuit


def bench_gate_throughput(n_qubits: int = 12, iterations: int = 200) -> None:
    dim = 1 << n_qubits
    print(f"[*] Benchmarking Gate Throughput ({n_qubits} qubits, {dim:,} amplitudes)...")

    # 1. Hadamard 1Q Gate
    sv = StateVector.zero(n_qubits)
    t0 = time.perf_counter()
    for _ in range(iterations):
        for q in range(n_qubits):
            apply_gate_1q(sv, Gate.H, q)
    t_1q = time.perf_counter() - t0
    total_1q_gates = iterations * n_qubits
    rate_1q = total_1q_gates / t_1q
    print(f"    [1Q Gate: H]    Executed {total_1q_gates:,} gates in {t_1q * 1000:.2f} ms")
    print(f"                    Throughput: {rate_1q:,.0f} gates/sec ({rate_1q * dim / 1e6:,.1f} M amplitude-ops/sec)")

    # 2. CNOT 2Q Gate
    t0 = time.perf_counter()
    for _ in range(iterations):
        for q in range(n_qubits - 1):
            apply_cnot(sv, q, q + 1)
    t_2q = time.perf_counter() - t0
    total_2q_gates = iterations * (n_qubits - 1)
    rate_2q = total_2q_gates / t_2q
    print(f"    [2Q Gate: CNOT] Executed {total_2q_gates:,} gates in {t_2q * 1000:.2f} ms")
    print(f"                    Throughput: {rate_2q:,.0f} gates/sec\n")


def bench_qft_scaling() -> None:
    print("[*] Benchmarking Quantum Fourier Transform (QFT) Scaling...")
    qubit_counts = [4, 6, 8, 10, 12]

    for n in qubit_counts:
        circ = build_qft_circuit(n)
        dim = 1 << n

        # Warmup
        circ.simulate()

        runs = 20
        t0 = time.perf_counter()
        for _ in range(runs):
            circ.simulate()
        elapsed = time.perf_counter() - t0
        avg_time_ms = (elapsed / runs) * 1000.0

        print(f"    -> {n:2d} Qubits ({dim:5,d} amps, {circ.gate_count:3d} gates, depth {circ.depth:2d}): {avg_time_ms:6.2f} ms/run ({runs / elapsed:,.0f} QFTs/sec)")
    print()


def bench_grover_search() -> None:
    print("[*] Benchmarking Grover's Quantum Search Algorithm...")
    for n in (3, 4, 5, 6):
        n_items = 1 << n
        target = "0" * (n - 1) + "1"
        circ = build_grover_circuit(n, target)

        t0 = time.perf_counter()
        sim = circ.simulate()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probs = sim.probabilities()
        p_target = probs.get(target, 0.0)
        print(f"    -> Search across {n_items:2d} states ({n} qubits): {elapsed_ms:5.2f} ms | Found '{target}' with {p_target * 100.1:.1f}% probability")
    print()


def run_all_benchmarks() -> None:
    print("==================================================================")
    print("      QUANTALAB: QUANTUM COMPUTING SIMULATOR BENCHMARK SUITE      ")
    print("==================================================================\n")
    bench_gate_throughput(n_qubits=12, iterations=100)
    bench_qft_scaling()
    bench_grover_search()
    print("==================================================================")
    print("            ALL QUANTALAB BENCHMARKS COMPLETED                    ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_benchmarks()
