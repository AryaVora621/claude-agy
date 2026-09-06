# LatticeGuard: NIST FIPS 203 ML-KEM Post-Quantum Cryptography Engine

A mathematically rigorous, zero-dependency, pure Python 3.10+ standard library implementation of **NIST FIPS 203** (Module-Lattice-Based Key-Encapsulation Mechanism / ML-KEM, formerly CRYSTALS-Kyber).

LatticeGuard implements the post-quantum cryptographic stack from first principles:
- **Ring Arithmetic**: Polynomial ring $R_q = \mathbb{Z}_q[X] / (X^{256} + 1)$ with prime modulus $q = 3329$
- **Fast NTT Engine**: 7-stage Cooley-Tukey and Gentleman-Sande Number Theoretic Transform butterfly network with precomputed roots of unity ($\zeta = 17 \pmod{3329}$)
- **Montgomery & Barrett Reduction**: High-throughput modular reduction algorithms without division
- **Noise Sampling**: Centered Binomial Distribution ($CBD_\eta$) for $\eta \in \{2, 3\}$ and SHAKE-128 rejection sampling
- **Standardized Parameter Sets**: Full compliance with **ML-KEM-512**, **ML-KEM-768**, and **ML-KEM-1024**
- **IND-CCA2 Security**: Public key encryption (K-PKE) coupled with the Fujisaki-Okamoto (FO) transform and constant-time implicit rejection against chosen-ciphertext adaptive attacks
- **Sub-Pixel Telemetry**: 2x4 Unicode Braille lattice visualizer (`U+2800..U+28FF`) and real-time ANSI terminal telemetry HUD

---

## Mathematical Architecture

### 1. The Cyclotomic Quotient Ring $R_q$

Let $n = 256$ and $q = 3329$. LatticeGuard operates over the cyclotomic polynomial quotient ring:

$$R_q = \mathbb{Z}_q[X] / (X^{256} + 1)$$

Polynomials have degree at most 255:

$$f(X) = \sum_{i=0}^{255} f_i X^i, \quad f_i \in \mathbb{Z}_q = \{0, 1, \dots, q-1\}$$

Multiplication of two polynomials $a(X), b(X) \in R_q$ corresponds to negative-wrapped cyclic convolution:

$$(a \cdot b)_k = \sum_{i+j=k} a_i b_j - \sum_{i+j=k+256} a_i b_j \pmod q$$

### 2. Number Theoretic Transform (NTT)

The prime modulus $q = 3329$ satisfies $q - 1 = 3328 = 13 \times 256$. The element $\zeta = 17$ is a primitive 256-th root of unity:

$$\zeta^{128} \equiv -1 \pmod{3329}, \quad \zeta^{256} \equiv 1 \pmod{3329}$$

In $\mathbb{Z}_q[X]$, the polynomial $X^{256} + 1$ factors into 128 quadratic polynomials:

$$X^{256} + 1 = \prod_{i=0}^{127} \left( X^2 - \zeta^{2 \cdot \text{bitrev}_7(i) + 1} \right) \pmod q$$

Forward NTT transforms polynomials from $R_q$ into 128 pairs of degree-1 polynomials:
- **Cooley-Tukey butterfly** splits blocks of length $128 \to 64 \to 32 \to 16 \to 8 \to 4 \to 2$
- **BaseCaseMultiply** multiplies degree-1 polynomial pairs modulo $(X^2 - \gamma_i)$:

$$(a_0 + a_1 X) \cdot (b_0 + b_1 X) \equiv (a_0 b_0 + a_1 b_1 \gamma_i) + (a_0 b_1 + a_1 b_0) X \pmod{X^2 - \gamma_i}$$

- **Gentleman-Sande butterfly** performs inverse NTT with exact normalization factor $128^{-1} \equiv 3303 \pmod{3329}$.
- Complexity: Reduces ring multiplication from $O(n^2) = 65,536$ operations to $O(n \log n) = 1,792$ butterfly operations (36.6x theoretical reduction).

### 3. Modular Reduction

1. **Montgomery Reduction ($R = 2^{16} = 65536$)**:
   With $q^{-1} \equiv 62209 \pmod{2^{16}}$:
   $$u \leftarrow (a \cdot 62209) \bmod 2^{16}, \quad t \leftarrow (a - u \cdot 3329) \gg 16$$
2. **Barrett Reduction**:
   With precomputed multiplier $v = \lfloor 2^{26} / 3329 + 1/2 \rfloor = 20159$:
   $$t \leftarrow \left\lfloor \frac{a \cdot 20159 + 2^{25}}{2^{26}} \right\rfloor \cdot 3329, \quad \text{Barrett}(a) = a - t$$

### 4. Module Learning With Errors (M-LWE)

In rank-$k$ Module-LWE:
- Matrix $\mathbf{A} \in R_q^{k \times k}$ generated via SHAKE-128 rejection sampling from public seed $\rho$
- Secret vector $\mathbf{s} \in R_q^k$ sampled from Centered Binomial Distribution $CBD_{\eta_1}$
- Error vector $\mathbf{e} \in R_q^k$ sampled from $CBD_{\eta_1}$
- Public Key: $\mathbf{t} = \mathbf{A} \mathbf{s} + \mathbf{e} \in R_q^k$ in the NTT domain

### 5. Fujisaki-Okamoto Transform with Constant-Time Implicit Rejection

To achieve IND-CCA2 security against adaptive chosen-ciphertext attacks, decapsulation re-encrypts the decrypted candidate message $m'$ using derived coins $r' = G(m' \parallel h)$ where $h = \text{SHA3-256}(ek)$:
- If $c == c'$, return honest shared secret $K' = G(m' \parallel h)$
- If $c \ne c'$, return pseudorandom key $K_{\text{bar}} = \text{SHAKE-256}(z \parallel c)$ using secret seed $z$
- Constant-time comparison prevents timing side-channels and oracle exploration.

---

## FIPS 203 Parameter Sets

| Parameter | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|:---|:---|:---|:---|
| Security Category | NIST Level 1 (AES-128) | NIST Level 3 (AES-192) | NIST Level 5 (AES-256) |
| Module Rank ($k$) | 2 | 3 | 4 |
| Secret Noise ($\eta_1$) | 3 | 2 | 2 |
| Encryption Noise ($\eta_2$) | 2 | 2 | 2 |
| Vector Compression ($d_u$) | 10 bits | 10 bits | 11 bits |
| Scalar Compression ($d_v$) | 4 bits | 4 bits | 5 bits |
| Public Key Size ($ek$) | 800 bytes | 1,184 bytes | 1,568 bytes |
| Secret Key Size ($dk$) | 1,632 bytes | 2,400 bytes | 3,168 bytes |
| Ciphertext Size ($c$) | 768 bytes | 1,088 bytes | 1,568 bytes |
| Shared Secret Key ($K$) | 32 bytes (256 bits) | 32 bytes (256 bits) | 32 bytes (256 bits) |

---

## Performance Benchmarks

Measured on Apple Silicon (pure Python standard library, zero C extensions or native dependencies):

```
==============================================================================
LATTICEGUARD: FIPS 203 ML-KEM POST-QUANTUM PERFORMANCE BENCHMARKS
==============================================================================

[1] Low-Level Modular Arithmetic
------------------------------------------------------------------------------
  Montgomery Reduction (R=2^16)  :    11.36 MOps/s
  Barrett Reduction (v=20159)    :    11.15 MOps/s

[2] Number Theoretic Transform (NTT) Engine
------------------------------------------------------------------------------
  Forward NTT (Cooley-Tukey)     :   9,254.6 transforms/s (108.05 us/transform)
  Inverse NTT (Gentleman-Sande)  :   9,148.6 transforms/s (109.31 us/transform)
  NTT Domain Multiplication      :  22,052.4 products/s   ( 45.35 us/product)
  Direct O(n^2) Multiplication   :     172.6 products/s   (5,793.83 us/product)
  >> NTT Acceleration Speedup    :      15.6x faster than O(n^2)

[3] Noise Sampling & Bit-Packing Codecs
------------------------------------------------------------------------------
  CBD_2 Noise Sampling           :  22,163.8 polys/s
  CBD_3 Noise Sampling           :  16,459.6 polys/s
  SHAKE-128 Rejection Sampling   :  23,185.5 polys/s
  12-bit Byte Serialization      :  39,748.2 packs/s   (14.56 MB/s)
  12-bit Byte Deserialization    :  35,052.4 unpacks/s (12.84 MB/s)

[4] Full NIST FIPS 203 ML-KEM Key Encapsulation Operations
------------------------------------------------------------------------------
  ML-KEM-512   | KeyGen:   1.50 ms (668.7 ops/s) | Encaps:   2.29 ms (435.9 ops/s) | Decaps:   3.57 ms (280.4 ops/s) | Reject:   3.57 ms
  ML-KEM-768   | KeyGen:   2.50 ms (400.6 ops/s) | Encaps:   3.62 ms (276.3 ops/s) | Decaps:   5.35 ms (187.0 ops/s) | Reject:   5.34 ms
  ML-KEM-1024  | KeyGen:   3.81 ms (262.4 ops/s) | Encaps:   5.31 ms (188.3 ops/s) | Decaps:   7.78 ms (128.5 ops/s) | Reject:   7.65 ms

[5] Sub-Pixel Braille Visualizer
------------------------------------------------------------------------------
  BrailleLatticeCanvas Rendering :   1,144.3 FPS
```

Notice the decapsulation timing uniformity: ML-KEM-768 Honest Decapsulation (5.35 ms) matches Constant-Time Implicit Rejection (5.34 ms) to within 0.2%, preventing oracle side-channel leakage.

---

## Directory Structure

```
projects/28-latticeguard/
├── latticeguard/
│   ├── __init__.py          # Public package exports
│   ├── ring.py              # Finite field Z_q, Montgomery/Barrett reductions, Polynomial, PolyVec
│   ├── ntt.py               # 7-stage Cooley-Tukey NTT, Gentleman-Sande INTT, BaseCaseMultiply
│   ├── sampling.py          # SHAKE-128 rejection sampling, CBD_2, CBD_3, bit-packing codecs
│   ├── ind_cpa.py           # Module-LWE IND-CPA Public Key Encryption (K-PKE)
│   ├── kem.py               # NIST FIPS 203 ML-KEM with Fujisaki-Okamoto implicit rejection
│   └── visualizer.py        # 2x4 sub-pixel Braille visualizer, NTT butterfly, telemetry HUD
├── tests/
│   ├── test_ring_arithmetic.py   # Field constants, reductions, polynomial norms, arithmetic
│   ├── test_ntt_transform.py     # Roots of unity, bit reversal, forward/inverse roundtrip, linearity
│   ├── test_sampling_codecs.py   # Compression bounds, bit codecs, CBD distributions, rejection
│   ├── test_ind_cpa.py           # K-PKE encryption/decryption roundtrip across all parameter sets
│   ├── test_kem_fips203.py       # ML-KEM encapsulation, honest decaps, implicit rejection, guards
│   └── test_visualizer.py        # Braille canvas rasterization, HUD cards, NTT diagrams
├── benchmarks/
│   └── bench_latticeguard.py     # Complete performance microbenchmark suite
├── examples/
│   └── pqc_workbench.py          # Interactive CLI post-quantum cryptography workbench
├── PLAN.md                  # Comprehensive algebraic specification and derivations
├── TASK_QUEUE.md            # Project milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # Documentation and architectural guide
```

---

## Quickstart & Usage

### 1. Key Encapsulation (ML-KEM-768)

```python
from latticeguard import MLKEM768

# Alice generates keypair
kp = MLKEM768.keygen()
ek = kp.ek  # 1,184 bytes encapsulation key (public)
dk = kp.dk  # 2,400 bytes decapsulation key (private)

# Bob encapsulates shared secret using Alice's public key
enc = MLKEM768.encaps(ek)
shared_secret_bob = enc.shared_secret  # 32 bytes (256 bits)
ciphertext = enc.ciphertext            # 1,088 bytes transmitted to Alice

# Alice decapsulates ciphertext using private key
shared_secret_alice = MLKEM768.decaps(dk, ciphertext)

assert shared_secret_alice == shared_secret_bob
print("Post-quantum shared secret established:", shared_secret_alice.hex())
```

### 2. Adversarial Tamper Simulation

```python
# Attacker Eve corrupts ciphertext byte 15
corrupted_ct = bytearray(ciphertext)
corrupted_ct[15] ^= 0x01

# Alice decapsulates corrupted ciphertext
eve_secret = MLKEM768.decaps(dk, bytes(corrupted_ct))

# Fujisaki-Okamoto implicit rejection triggers:
# Returns K_bar = SHAKE-256(z || c) without throwing exception or leaking timing
assert eve_secret != shared_secret_bob
print("Implicit rejection fallback key:", eve_secret.hex())
```

### 3. Running Tests and Benchmarks

```bash
# Run complete test suite (33 tests)
python3 -m unittest discover -s tests -v

# Run performance microbenchmarks
python3 benchmarks/bench_latticeguard.py

# Launch interactive PQC workbench
python3 examples/pqc_workbench.py --suite 768 --mode all
```

---

## License

MIT License. Designed and authored from first principles for post-quantum security research, education, and high-assurance cryptographic experimentation.
