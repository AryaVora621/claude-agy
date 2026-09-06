"""Performance Microbenchmarks for LatticeGuard (FIPS 203 ML-KEM).

Measures:
1. Low-level Ring Arithmetic: Montgomery reduction, Barrett reduction
2. NTT Engine: Forward Cooley-Tukey NTT, Inverse Gentleman-Sande INTT, NTT Multiplication vs Direct O(n^2)
3. Sampling & Codecs: CBD_2, CBD_3, SHAKE-128 Rejection Sampling, Bit Serialization
4. Cryptographic Key Encapsulation: KeyGen, Encaps, Decaps across ML-KEM-512, 768, 1024
5. Decapsulation Timing Uniformity: Honest Decaps vs Constant-Time Implicit Rejection
6. Sub-Pixel Visualizer: BrailleLatticeCanvas frame rendering throughput
"""

from __future__ import annotations
import os
import random
import time
from typing import Callable, Tuple

from latticeguard.ring import (
    KYBER_N,
    KYBER_Q,
    montgomery_reduce,
    barrett_reduce,
    Polynomial,
    PolyVec,
)
from latticeguard.ntt import (
    forward_ntt,
    inverse_ntt,
    poly_ntt_multiply,
)
from latticeguard.sampling import (
    sample_poly_cbd,
    sample_ntt,
    poly_to_bytes,
    bytes_to_poly,
    compress_polyvec,
    decompress_polyvec,
)
from latticeguard.kem import MLKEM512, MLKEM768, MLKEM1024
from latticeguard.visualizer import BrailleLatticeCanvas


def bench_fn(fn: Callable[[], None], min_duration: float = 0.5) -> Tuple[int, float, float]:
    """Benchmark function execution over min_duration seconds.

    Returns: (iterations, total_time, ops_per_sec)
    """
    # Warmup
    for _ in range(5):
        fn()

    iterations = 0
    t0 = time.perf_counter()
    while True:
        fn()
        iterations += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= min_duration and iterations >= 10:
            break

    ops_per_sec = iterations / elapsed
    return iterations, elapsed, ops_per_sec


def run_benchmarks() -> None:
    print("=" * 78)
    print("LATTICEGUARD: FIPS 203 ML-KEM POST-QUANTUM PERFORMANCE BENCHMARKS")
    print("=" * 78)

    # 1. Ring & Modular Reductions
    print("\n[1] Low-Level Modular Arithmetic")
    print("-" * 78)
    test_vals = [random.randint(-100000, 100000) for _ in range(1000)]

    def run_montgomery() -> None:
        for v in test_vals:
            montgomery_reduce(v)

    _, _, ops = bench_fn(run_montgomery, min_duration=0.3)
    mops = (ops * 1000) / 1e6
    print(f"  Montgomery Reduction (R=2^16)  : {mops:8.2f} MOps/s")

    def run_barrett() -> None:
        for v in test_vals:
            barrett_reduce(v)

    _, _, ops = bench_fn(run_barrett, min_duration=0.3)
    bops = (ops * 1000) / 1e6
    print(f"  Barrett Reduction (v=20159)    : {bops:8.2f} MOps/s")

    # 2. NTT Transforms & Multiplication
    print("\n[2] Number Theoretic Transform (NTT) Engine")
    print("-" * 78)
    p_test = Polynomial([random.randint(0, KYBER_Q - 1) for _ in range(KYBER_N)])
    p_hat = forward_ntt(p_test)

    _, t_fwd, ops_fwd = bench_fn(lambda: forward_ntt(p_test), min_duration=0.4)
    us_fwd = (1.0 / ops_fwd) * 1e6
    print(f"  Forward NTT (Cooley-Tukey)     : {ops_fwd:8.1f} transforms/s ({us_fwd:6.2f} us/transform)")

    _, t_inv, ops_inv = bench_fn(lambda: inverse_ntt(p_hat), min_duration=0.4)
    us_inv = (1.0 / ops_inv) * 1e6
    print(f"  Inverse NTT (Gentleman-Sande)  : {ops_inv:8.1f} transforms/s ({us_inv:6.2f} us/transform)")

    p1_hat = forward_ntt(p_test)
    p2_hat = forward_ntt(p_test)
    _, _, ops_ntt_mul = bench_fn(lambda: poly_ntt_multiply(p1_hat, p2_hat), min_duration=0.4)
    us_ntt_mul = (1.0 / ops_ntt_mul) * 1e6
    print(f"  NTT Domain Multiplication      : {ops_ntt_mul:8.1f} products/s   ({us_ntt_mul:6.2f} us/product)")

    # Compare against direct classical O(n^2) ring multiplication
    _, _, ops_direct_mul = bench_fn(lambda: p_test * p_test, min_duration=0.4)
    us_direct = (1.0 / ops_direct_mul) * 1e6
    speedup = us_direct / (us_fwd * 2 + us_ntt_mul + us_inv)
    print(f"  Direct O(n^2) Multiplication   : {ops_direct_mul:8.1f} products/s   ({us_direct:6.2f} us/product)")
    print(f"  >> NTT Acceleration Speedup    : {speedup:8.1f}x faster than O(n^2)")

    # 3. Sampling & Codecs
    print("\n[3] Noise Sampling & Bit-Packing Codecs")
    print("-" * 78)
    buf128 = os.urandom(128)
    _, _, ops_cbd2 = bench_fn(lambda: sample_poly_cbd(buf128, 2), min_duration=0.3)
    print(f"  CBD_2 Noise Sampling           : {ops_cbd2:8.1f} polys/s")

    buf192 = os.urandom(192)
    _, _, ops_cbd3 = bench_fn(lambda: sample_poly_cbd(buf192, 3), min_duration=0.3)
    print(f"  CBD_3 Noise Sampling           : {ops_cbd3:8.1f} polys/s")

    seed_rho = os.urandom(32)
    _, _, ops_rej = bench_fn(lambda: sample_ntt(seed_rho, 0, 1), min_duration=0.3)
    print(f"  SHAKE-128 Rejection Sampling   : {ops_rej:8.1f} polys/s")

    raw_384 = poly_to_bytes(p_test)
    _, _, ops_pack = bench_fn(lambda: poly_to_bytes(p_test), min_duration=0.3)
    mb_pack = (ops_pack * 384) / (1024 * 1024)
    print(f"  12-bit Byte Serialization      : {ops_pack:8.1f} packs/s   ({mb_pack:6.2f} MB/s)")

    _, _, ops_unpack = bench_fn(lambda: bytes_to_poly(raw_384), min_duration=0.3)
    mb_unpack = (ops_unpack * 384) / (1024 * 1024)
    print(f"  12-bit Byte Deserialization    : {ops_unpack:8.1f} unpacks/s ({mb_unpack:6.2f} MB/s)")

    # 4. End-to-End ML-KEM Cryptographic Operations
    print("\n[4] Full NIST FIPS 203 ML-KEM Key Encapsulation Operations")
    print("-" * 78)
    for kem in [MLKEM512, MLKEM768, MLKEM1024]:
        # KeyGen
        _, _, ops_kg = bench_fn(lambda: kem.keygen(), min_duration=0.4)
        ms_kg = (1.0 / ops_kg) * 1000

        # Pre-generate keypair for Encaps/Decaps
        kp = kem.keygen()
        _, _, ops_enc = bench_fn(lambda: kem.encaps(kp.ek), min_duration=0.4)
        ms_enc = (1.0 / ops_enc) * 1000

        # Pre-generate ciphertext for Decaps
        res = kem.encaps(kp.ek)
        _, _, ops_dec = bench_fn(lambda: kem.decaps(kp.dk, res.ciphertext), min_duration=0.4)
        ms_dec = (1.0 / ops_dec) * 1000

        # Corrupted ciphertext for implicit rejection timing test
        bad_ct = bytearray(res.ciphertext)
        bad_ct[0] ^= 0x01
        _, _, ops_dec_rej = bench_fn(lambda: kem.decaps(kp.dk, bytes(bad_ct)), min_duration=0.4)
        ms_dec_rej = (1.0 / ops_dec_rej) * 1000

        print(f"  {kem.name:<12} | KeyGen: {ms_kg:6.2f} ms ({ops_kg:5.1f} ops/s) | "
              f"Encaps: {ms_enc:6.2f} ms ({ops_enc:5.1f} ops/s) | "
              f"Decaps: {ms_dec:6.2f} ms ({ops_dec:5.1f} ops/s) | "
              f"Reject: {ms_dec_rej:6.2f} ms")

    # 5. Visualizer Throughput
    print("\n[5] Sub-Pixel Braille Visualizer")
    print("-" * 78)
    canvas = BrailleLatticeCanvas(char_width=80, char_height=12)

    def run_render() -> None:
        canvas.clear()
        canvas.plot_polynomial(p_test, centered=True)
        canvas.render(show_border=False)

    _, _, fps = bench_fn(run_render, min_duration=0.3)
    print(f"  BrailleLatticeCanvas Rendering : {fps:8.1f} FPS")

    print("\n" + "=" * 78)
    print("BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmarks()
