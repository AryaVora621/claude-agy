"""
ZetaProof: Performance Benchmark Suite.
Measures:
1. Finite Field F_p arithmetic operations per second (add, mul, inv).
2. Polynomial multiplication, division, and Lagrange interpolation throughput.
3. R1CS constraint compilation and QAP reduction scaling.
4. zk-SNARK Setup, Prover, and Verifier execution latencies.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zetaproof.field import FieldElement, BN254_SCALAR_FIELD, TEST_PRIME
from zetaproof.polynomial import Polynomial, lagrange_interpolation
from zetaproof.circuit import Circuit
from zetaproof.r1cs import R1CS
from zetaproof.qap import QAP
from zetaproof.snark import Snark
from zetaproof.circuits_zoo import build_mimc_hash_circuit, compute_mimc_hash


def bench_field_arithmetic():
    print("=" * 70)
    print(" 1. FINITE FIELD ARITHMETIC BENCHMARK (BN254 254-BIT SCALAR FIELD)")
    print("=" * 70)

    p = BN254_SCALAR_FIELD
    a = FieldElement(12345678901234567890, p)
    b = FieldElement(98765432109876543210, p)

    n_ops = 200_000

    # Addition
    t0 = time.perf_counter()
    curr = a
    for _ in range(n_ops):
        curr = curr + b
    dt_add = time.perf_counter() - t0
    ops_add = n_ops / dt_add

    # Multiplication
    t0 = time.perf_counter()
    curr = a
    for _ in range(n_ops):
        curr = curr * b
    dt_mul = time.perf_counter() - t0
    ops_mul = n_ops / dt_mul

    # Modular Inversion
    n_inv = 10_000
    t0 = time.perf_counter()
    curr = a
    for _ in range(n_inv):
        curr = curr.inverse()
    dt_inv = time.perf_counter() - t0
    ops_inv = n_inv / dt_inv

    print(f"  Field Addition:        {ops_add:>12,.0f} ops/sec  ({dt_add*1000/n_ops*1e3:.2f} ns/op)")
    print(f"  Field Multiplication:  {ops_mul:>12,.0f} ops/sec  ({dt_mul*1000/n_ops*1e3:.2f} ns/op)")
    print(f"  Modular Inversion:     {ops_inv:>12,.0f} ops/sec  ({dt_inv*1000/n_inv:.2f} us/op)")
    print()


def bench_polynomial_calculus():
    print("=" * 70)
    print(" 2. POLYNOMIAL CALCULUS BENCHMARK (F_p[X])")
    print("=" * 70)

    p = TEST_PRIME

    # Polynomial Multiplication (degree 32 x degree 32)
    deg = 32
    poly1 = Polynomial([i + 1 for i in range(deg + 1)], p)
    poly2 = Polynomial([i + 2 for i in range(deg + 1)], p)

    n_mul = 1_000
    t0 = time.perf_counter()
    for _ in range(n_mul):
        _ = poly1 * poly2
    dt_mul = time.perf_counter() - t0
    print(f"  Poly Multiplication (deg {deg}):   {n_mul / dt_mul:>8,.0f} muls/sec ({dt_mul*1000/n_mul:.3f} ms/op)")

    # Polynomial Long Division (degree 64 / degree 32)
    poly_big = poly1 * poly2
    n_div = 1_000
    t0 = time.perf_counter()
    for _ in range(n_div):
        _ = divmod(poly_big, poly1)
    dt_div = time.perf_counter() - t0
    print(f"  Poly Long Division (64 / 32):    {n_div / dt_div:>8,.0f} divs/sec ({dt_div*1000/n_div:.3f} ms/op)")

    # Lagrange Interpolation across 16 points
    n_pts = 16
    points = [(i + 1, (i + 1) ** 3 + 7) for i in range(n_pts)]
    n_interp = 200
    t0 = time.perf_counter()
    for _ in range(n_interp):
        _ = lagrange_interpolation(points, p)
    dt_interp = time.perf_counter() - t0
    print(f"  Lagrange Interpolation ({n_pts} pts): {n_interp / dt_interp:>8,.0f} interps/sec ({dt_interp*1000/n_interp:.3f} ms/op)")
    print()


def bench_snark_pipeline():
    print("=" * 70)
    print(" 3. GROTH16 ZK-SNARK PIPELINE LATENCY (MIMC HASH PREIMAGE)")
    print("=" * 70)

    p = BN254_SCALAR_FIELD

    for rounds in (2, 4, 8):
        circuit, constants = build_mimc_hash_circuit(rounds=rounds, p=p)
        r1cs = circuit.to_r1cs()

        # 1. Setup Time
        t0 = time.perf_counter()
        pk, vk = Snark.setup(circuit)
        dt_setup = time.perf_counter() - t0

        # Witness generation
        secret_x = 424242
        pub_h = compute_mimc_hash(secret_x, rounds=rounds, p=p, constants=constants)
        witness = circuit.solve_witness(
            public_inputs={"hash": pub_h},
            private_inputs={"preimage": secret_x}
        )

        # 2. Prover Time
        n_proofs = 10
        t0 = time.perf_counter()
        for _ in range(n_proofs):
            proof = Snark.prove(pk, witness)
        dt_prove = (time.perf_counter() - t0) / n_proofs

        # 3. Verifier Time
        public_inputs = [1, pub_h]
        n_verifs = 50
        t0 = time.perf_counter()
        for _ in range(n_verifs):
            valid = Snark.verify(vk, public_inputs, proof)
        dt_verify = (time.perf_counter() - t0) / n_verifs

        print(f"  MiMC-{rounds} Rounds (Constraints: {r1cs.num_constraints:>2}, Wires: {r1cs.num_variables:>2}):")
        print(f"    Trusted Setup:  {dt_setup * 1000:.2f} ms")
        print(f"    Prover Time:    {dt_prove * 1000:.2f} ms per proof ({1.0 / dt_prove:.1f} proofs/sec)")
        print(f"    Verifier Time:  {dt_verify * 1000:.3f} ms per verification ({1.0 / dt_verify:,.0f} verifs/sec)")
        print(f"    Status:         {'VERIFIED' if valid else 'FAILED'}")
        print()


def main():
    print("\n" + "=" * 70)
    print("       ZETAPROOF: ZERO-KNOWLEDGE PROOF ENGINE PERFORMANCE BENCHMARKS  ")
    print("=" * 70 + "\n")

    bench_field_arithmetic()
    bench_polynomial_calculus()
    bench_snark_pipeline()


if __name__ == "__main__":
    main()
