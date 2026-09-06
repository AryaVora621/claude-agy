# PLAN.md: LatticeGuard Post-Quantum Module-Lattice Engine (FIPS 203 ML-KEM)

## 1. Executive Summary & Objective

The objective of **LatticeGuard** is to provide a complete, mathematically rigorous, zero-dependency implementation of NIST FIPS 203 (Module-Lattice-Based Key-Encapsulation Mechanism / ML-KEM, formerly CRYSTALS-Kyber) in pure Python 3.10+.

LatticeGuard implements the full post-quantum cryptographic stack from first principles:
- Ring arithmetic over $R_q = \mathbb{Z}_q[X]/(X^{256} + 1)$ with prime modulus $q = 3329$.
- Fast $O(n \log n)$ Number Theoretic Transform (NTT) using precomputed Montgomery roots of unity and Cooley-Tukey / Gentleman-Sande butterfly networks.
- Barrett modular reduction and Montgomery modular multiplication.
- Centered Binomial Distribution ($CBD_\eta$) and SHAKE-128 rejection sampling for uniform matrix expansion.
- All three standardized security parameter sets: ML-KEM-512, ML-KEM-768, and ML-KEM-1024.
- IND-CPA Public Key Encryption (PKE) and the Fujisaki-Okamoto (FO) transform for IND-CCA2 security with constant-time implicit rejection.
- Real-time terminal telemetry HUD and sub-pixel Unicode Braille lattice coefficient visualizer.

---

## 2. Mathematical Foundation

### 2.1 The Polynomial Ring $R_q$

Let $n = 256$ and $q = 3329$. The underlying algebraic structure is the cyclotomic polynomial quotient ring:

$$R_q = \mathbb{Z}_q[X] / (X^{256} + 1)$$

Elements of $R_q$ are polynomials of degree less than 256:

$$f(X) = \sum_{i=0}^{255} f_i X^i, \quad f_i \in \mathbb{Z}_q = \{0, 1, \dots, q-1\}$$

Multiplication of two polynomials $a(X), b(X) \in R_q$ is performed modulo $X^{256} + 1$ and modulo $q$, which corresponds to negative-wrapped cyclic convolution:

$$(a \cdot b)_k = \sum_{i+j=k} a_i b_j - \sum_{i+j=k+256} a_i b_j \pmod q$$

### 2.2 Prime Modulus and Roots of Unity

The modulus $q = 3329$ satisfies:

$$q - 1 = 3328 = 13 \times 256 = 13 \times 2^8$$

Thus, $\mathbb{Z}_q^*$ has an element of order 256. Specifically:

$$\zeta = 17 \implies \zeta^{128} \equiv -1 \equiv 3328 \pmod{3329}, \quad \zeta^{256} \equiv 1 \pmod{3329}$$

Over $\mathbb{Z}_q$, the polynomial $X^{256} + 1$ factors into 128 quadratic polynomials:

$$X^{256} + 1 = \prod_{i=0}^{127} \left( X^2 - \zeta^{2 \cdot \text{BitRev}_7(i) + 1} \right) \pmod q$$

### 2.3 Number Theoretic Transform (NTT)

By the Chinese Remainder Theorem:

$$R_q \cong \prod_{i=0}^{127} \mathbb{Z}_q[X] / \left( X^2 - \zeta^{2 \cdot \text{BitRev}_7(i) + 1} \right)$$

The forward NTT maps a polynomial $f \in R_q$ to 128 pairs of coefficients in Montgomery representation:

$$\hat{f} = \text{NTT}(f)$$

Multiplication in the NTT domain is performed component-wise on 128 degree-1 polynomials:

$$(a_0 + a_1 X) \cdot (b_0 + b_1 X) \equiv (a_0 b_0 + a_1 b_1 \zeta^{2k+1}) + (a_0 b_1 + a_1 b_0) X \pmod{X^2 - \zeta^{2k+1}}$$

This requires only 4 field multiplications per pair, reducing polynomial multiplication complexity from $O(n^2)$ ($65,536$ ops) to $O(n \log n)$ ($256 \times 7 = 1,792$ butterfly ops + 512 base multiplications).

### 2.4 Modular Reductions

1. **Montgomery Reduction**:
   Computes $t \cdot R^{-1} \pmod q$ for $R = 2^{16} = 65536$.
   Precomputed constant: $q^{-1} \equiv 62209 \pmod{2^{16}}$.
   For an integer $a \in [-2^{31}, 2^{31}-1]$:
   $$u \leftarrow (a \cdot q^{-1}) \bmod 2^{16}$$
   $$t \leftarrow (a - u \cdot q) \gg 16$$

2. **Barrett Reduction**:
   Reduces $a \in [-2^{31}, 2^{31}-1]$ to range $[-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$ without division:
   $$v = \left\lfloor \frac{2^{26}}{q} + \frac{1}{2} \right\rfloor = 20159$$
   $$t = \left\lfloor \frac{a \cdot v + 2^{25}}{2^{26}} \right\rfloor \cdot q$$
   $$\text{Barrett}(a) = a - t$$

---

## 3. Module Learning With Errors (M-LWE)

In Module-LWE, the secret is a vector of polynomials $\mathbf{s} \in R_q^k$, and the public key is:

$$\mathbf{t} = \mathbf{A} \mathbf{s} + \mathbf{e} \in R_q^k$$

where:
- $\mathbf{A} \in R_q^{k \times k}$ is a pseudorandom matrix generated via SHAKE-128 rejection sampling from public seed $\rho$.
- $\mathbf{s} \leftarrow CBD_{\eta_1}(R_q^k)$ is the secret noise vector.
- $\mathbf{e} \leftarrow CBD_{\eta_1}(R_q^k)$ is the error noise vector.

### Centered Binomial Distribution ($CBD_\eta$)

Given $2\eta$ random bits $(a_0, \dots, a_{\eta-1}, b_0, \dots, b_{\eta-1})$:

$$x = \sum_{i=0}^{\eta-1} a_i - \sum_{i=0}^{\eta-1} b_i \in [-\eta, \eta]$$

Variance: $\sigma^2 = \frac{\eta}{2}$.

---

## 4. FIPS 203 ML-KEM Parameter Sets

| Parameter | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|:---|:---|:---|:---|
| Security Category | NIST 1 (AES-128) | NIST 3 (AES-192) | NIST 5 (AES-256) |
| Module Rank $k$ | 2 | 3 | 4 |
| Secret Noise $\eta_1$ | 3 | 2 | 2 |
| Encryption Noise $\eta_2$ | 2 | 2 | 2 |
| Compression $d_u$ | 10 | 10 | 11 |
| Compression $d_v$ | 4 | 4 | 5 |
| Public Key Size | 800 bytes | 1,184 bytes | 1,568 bytes |
| Ciphertext Size | 768 bytes | 1,088 bytes | 1,568 bytes |
| Shared Secret Size | 32 bytes | 32 bytes | 32 bytes |

---

## 5. Architectural Components

1. `latticeguard/ring.py`:
   - `FieldElement`: Montgomery arithmetic over $\mathbb{Z}_q$.
   - `Polynomial`: Degree-255 polynomial in $R_q$.
   - Barrett and Montgomery reductions.
2. `latticeguard/ntt.py`:
   - Precomputed Montgomery root tables $\zeta_{\text{table}}$.
   - `forward_ntt(poly)`: Cooley-Tukey in-place butterfly network.
   - `inverse_ntt(poly)`: Gentleman-Sande in-place butterfly network.
   - `poly_ntt_multiply(p1, p2)`: Base multiplication in NTT domain.
3. `latticeguard/sampling.py`:
   - `generate_matrix_A(seed, k, transposed)`: Rejection sampling from SHAKE-128.
   - `sample_cbd(buf, eta)`: Centered binomial distribution sampling.
   - `compress(val, d)` and `decompress(val, d)`: Bit packing and rounding.
4. `latticeguard/ind_cpa.py`:
   - `cpa_keygen(seed)`: Public matrix $\mathbf{A}$, secrets $\mathbf{s}, \mathbf{e}$, public key $\mathbf{t} = \mathbf{A} \mathbf{s} + \mathbf{e}$.
   - `cpa_encrypt(pk, msg, coins)`: $\mathbf{u} = \mathbf{A}^T \mathbf{r} + \mathbf{e}_1$, $v = \mathbf{t}^T \mathbf{r} + e_2 + \text{encode}(msg)$.
   - `cpa_decrypt(sk, c)`: $m = \text{decode}(v - \mathbf{s}^T \mathbf{u})$.
5. `latticeguard/kem.py`:
   - `MLKEM`: Unified class parameterizing ML-KEM-512, 768, 1024.
   - `keygen()`: $(ek, dk)$ keypair generation.
   - `encaps(ek)`: $(c, K)$ ciphertext and shared key encapsulation.
   - `decaps(dk, c)`: Shared key decapsulation with Fujisaki-Okamoto re-encryption and implicit rejection.
6. `latticeguard/visualizer.py`:
   - `BrailleLatticeCanvas`: 2x4 sub-pixel Unicode Braille mapping of polynomial coefficients.
   - `NTTButterflyVisualizer`: Terminal ASCII visualization of 7-stage NTT butterfly stages.
   - `PQCWorkbenchHUD`: Telemetry dashboard showing key sizes, noise distributions, and security parameters.
