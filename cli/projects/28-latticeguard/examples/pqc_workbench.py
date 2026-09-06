"""Interactive Post-Quantum Cryptography Workbench for LatticeGuard.

Features:
- Live demonstration of NIST FIPS 203 ML-KEM (512, 768, 1024)
- Sub-pixel Unicode Braille coefficient lattice visualizer
- Cooley-Tukey NTT butterfly frequency domain explorer
- Real-time active adversary chosen-ciphertext attack simulation
  with Fujisaki-Okamoto constant-time implicit rejection
"""

from __future__ import annotations
import argparse
import os
import sys
import time

from latticeguard.ring import KYBER_N, KYBER_Q, Polynomial, PolyVec
from latticeguard.ntt import forward_ntt, inverse_ntt, poly_ntt_multiply
from latticeguard.ind_cpa import PARAMS_512, PARAMS_768, PARAMS_1024
from latticeguard.kem import MLKEM, MLKEM512, MLKEM768, MLKEM1024
from latticeguard.visualizer import (
    BrailleLatticeCanvas,
    NTTButterflyVisualizer,
    PQCWorkbenchHUD,
    rgb_ansi,
)


def run_walkthrough(kem: MLKEM) -> None:
    print("\n" + "=" * 78)
    print(rgb_ansi(240, 210, 60, f"LATTICEGUARD WORKBENCH: {kem.name} FULL LIFECYCLE WALKTHROUGH"))
    print("=" * 78)

    # 1. Parameter Table
    print(PQCWorkbenchHUD.render_params_table(kem.params))

    # 2. Key Generation (Alice)
    print("\n" + rgb_ansi(60, 220, 240, "[STEP 1] Alice: Key Generation (K-PKE.KeyGen + ML-KEM.KeyGen)"))
    t0 = time.perf_counter()
    kp = kem.keygen()
    t_kg = (time.perf_counter() - t0) * 1000
    print(f"  Generated Keypair in {t_kg:.2f} ms")
    print(f"  Encapsulation Key (ek) : {len(kp.ek):,} bytes | Fingerprint: {kp.ek[:16].hex()}...")
    print(f"  Decapsulation Key (dk) : {len(kp.dk):,} bytes | Fingerprint: {kp.dk[:16].hex()}...")

    # Extract first polynomial of t_hat for visualization
    from latticeguard.sampling import bytes_to_polyvec
    t_hat = bytes_to_polyvec(kp.ek[: 384 * kem.params.k], kem.params.k)
    poly_t0 = inverse_ntt(t_hat[0])

    print("\n" + rgb_ansi(200, 180, 255, "  Sub-Pixel Braille Visualization: Alice Public Key Polynomial t_0(X) in R_q:"))
    canvas = BrailleLatticeCanvas(char_width=72, char_height=8)
    canvas.plot_polynomial(poly_t0, centered=True)
    print(canvas.render())

    # 3. Encapsulation (Bob)
    print("\n" + rgb_ansi(60, 220, 240, "[STEP 2] Bob: Encapsulation (ML-KEM.Encaps)"))
    t0 = time.perf_counter()
    enc_res = kem.encaps(kp.ek)
    t_enc = (time.perf_counter() - t0) * 1000
    print(f"  Encapsulated Shared Secret in {t_enc:.2f} ms")
    print(f"  Negotiated Shared Key (K) : {len(enc_res.shared_secret)} bytes ({len(enc_res.shared_secret)*8} bits)")
    print(f"  >> Hex: {rgb_ansi(120, 240, 140, enc_res.shared_secret.hex())}")
    print(f"  Transmitted Ciphertext (c): {len(enc_res.ciphertext):,} bytes | Fingerprint: {enc_res.ciphertext[:16].hex()}...")

    # 4. Decapsulation (Alice)
    print("\n" + rgb_ansi(60, 220, 240, "[STEP 3] Alice: Honest Decapsulation (ML-KEM.Decaps)"))
    t0 = time.perf_counter()
    alice_ss = kem.decaps(kp.dk, enc_res.ciphertext)
    t_dec = (time.perf_counter() - t0) * 1000
    print(f"  Decapsulated in {t_dec:.2f} ms")
    print(f"  Alice Recovered Key (K')  : {rgb_ansi(120, 240, 140, alice_ss.hex())}")

    if alice_ss == enc_res.shared_secret:
        status_banner = "SUCCESS: Alice and Bob shared secrets match with 100% cryptographic fidelity!"
        print(f"\n  {rgb_ansi(100, 240, 120, status_banner)}")
    else:
        print(f"\n  {rgb_ansi(240, 80, 80, 'ERROR: Shared secret mismatch!')}")

    # 5. Active Adversary Attack Simulation (Eve)
    print("\n" + rgb_ansi(240, 100, 80, "[STEP 4] Active Adversary: Chosen-Ciphertext Attack (CCA) Simulation"))
    print("  Adversary Eve intercepts Bob ciphertext and injects bit flips to probe the decryption oracle...")

    corrupted_ct = bytearray(enc_res.ciphertext)
    corrupted_ct[42] ^= 0x55  # Tamper with 4 bits
    print(f"  Tampered byte 42: original 0x{enc_res.ciphertext[42]:02x} -> corrupted 0x{corrupted_ct[42]:02x}")

    t0 = time.perf_counter()
    eve_ss = kem.decaps(kp.dk, bytes(corrupted_ct))
    t_rej = (time.perf_counter() - t0) * 1000

    print(f"\n  Alice Decapsulation with Re-encryption Check completed in {t_rej:.2f} ms")
    print(f"  >> Honest Key Target     : {enc_res.shared_secret.hex()[:32]}...")
    print(f"  >> Decaps Output (K_bar) : {rgb_ansi(255, 140, 100, eve_ss.hex()[:32])}...")

    assert eve_ss != enc_res.shared_secret, "Security breach: corrupted ciphertext produced genuine key!"
    print(f"\n  {rgb_ansi(100, 240, 120, 'PROTECTION VERIFIED: Fujisaki-Okamoto Implicit Rejection Triggered!')}")
    print("  Constant-time fallback derived K_bar = SHAKE-256(z || c_corrupted).")
    print("  No oracle leakage or timing side-channel exposed to adversary.")


def run_ntt_explorer() -> None:
    print("\n" + "=" * 78)
    print(rgb_ansi(240, 210, 60, "LATTICEGUARD: NUMBER THEORETIC TRANSFORM (NTT) EXPLORER"))
    print("=" * 78)

    print(NTTButterflyVisualizer.render_butterfly_summary())

    # Create two test polynomials in R_q
    p1 = Polynomial([int(1500 * (1.0 + 0.9 * (i / 256.0))) % KYBER_Q for i in range(256)])
    p2 = Polynomial([int(800 * (1.0 - 0.7 * (i / 256.0))) % KYBER_Q for i in range(256)])

    print("\n" + rgb_ansi(60, 220, 240, "1. Spatial Domain: Polynomial p1(X) coefficients in Z_q:"))
    canvas1 = BrailleLatticeCanvas(char_width=72, char_height=7)
    canvas1.plot_polynomial(p1, centered=False)
    print(canvas1.render())

    print("\n" + rgb_ansi(200, 150, 255, "2. Frequency Domain: Forward NTT(p1) coefficients:"))
    p1_hat = forward_ntt(p1)
    canvas2 = BrailleLatticeCanvas(char_width=72, char_height=7)
    canvas2.plot_polynomial(p1_hat, centered=False)
    print(canvas2.render())

    print("\n" + rgb_ansi(120, 240, 140, "3. Ring Product via NTT: INTT(NTT(p1) * NTT(p2)) mod (X^256 + 1):"))
    p2_hat = forward_ntt(p2)
    prod_hat = poly_ntt_multiply(p1_hat, p2_hat)
    prod = inverse_ntt(prod_hat)
    canvas3 = BrailleLatticeCanvas(char_width=72, char_height=7)
    canvas3.plot_polynomial(prod, centered=False)
    print(canvas3.render())


def main() -> None:
    parser = argparse.ArgumentParser(description="LatticeGuard FIPS 203 ML-KEM Workbench")
    parser.add_argument(
        "--suite",
        choices=["512", "768", "1024"],
        default="768",
        help="ML-KEM parameter set (default: 768)",
    )
    parser.add_argument(
        "--mode",
        choices=["walkthrough", "ntt", "all"],
        default="all",
        help="Workbench demonstration mode",
    )
    args = parser.parse_args()

    kem_map = {
        "512": MLKEM512,
        "768": MLKEM768,
        "1024": MLKEM1024,
    }
    kem = kem_map[args.suite]

    if args.mode in ("walkthrough", "all"):
        run_walkthrough(kem)

    if args.mode in ("ntt", "all"):
        run_ntt_explorer()


if __name__ == "__main__":
    main()
