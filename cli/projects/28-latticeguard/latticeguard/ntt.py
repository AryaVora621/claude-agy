"""Number Theoretic Transform (NTT) Engine for FIPS 203 ML-KEM.

Implements:
1. Precomputed roots of unity zeta and bit-reversal indexing for R_q = Z_q[X] / (X^256 + 1)
2. Cooley-Tukey forward NTT with negative-wrapped convolution decomposition
3. Gentleman-Sande inverse NTT (INTT) with exact 128^(-1) normalization
4. BaseCaseMultiply for degree-1 factor pairs in the NTT domain
5. Fast vector and matrix-vector polynomial products in the NTT domain
"""

from __future__ import annotations
from typing import List, Sequence, Tuple

from .ring import KYBER_N, KYBER_Q, Polynomial, PolyVec, freeze


ZETA: int = 17


def bitrev7(n: int) -> int:
    """Compute 7-bit reversal of integer n (0 <= n < 128)."""
    res = 0
    for _ in range(7):
        res = (res << 1) | (n & 1)
        n >>= 1
    return res


# Precompute 128 powers of zeta in bit-reversed order for Cooley-Tukey NTT
ZETAS: List[int] = [pow(ZETA, bitrev7(i), KYBER_Q) for i in range(128)]

# Precompute gamma multipliers for BaseCaseMultiply:
# gamma_i = zeta^(2 * bitrev7(i) + 1) mod q
GAMMA_TABLE: List[int] = [pow(ZETA, 2 * bitrev7(i) + 1, KYBER_Q) for i in range(128)]

# Inverse of 128 modulo 3329: 128 * 3303 = 1 mod 3329
INV_128: int = pow(128, -1, KYBER_Q)


def forward_ntt(poly: Polynomial) -> Polynomial:
    """Compute in-place forward Number Theoretic Transform (Cooley-Tukey butterfly).

    Transforms polynomial f in R_q to NTT domain representation f_hat.
    """
    f = list(poly.coeffs)
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % KYBER_Q
                f[j + length] = (f[j] - t) % KYBER_Q
                f[j] = (f[j] + t) % KYBER_Q
            start += 2 * length
        length //= 2

    return Polynomial(f)


def inverse_ntt(poly: Polynomial) -> Polynomial:
    """Compute inverse Number Theoretic Transform (Gentleman-Sande butterfly).

    Transforms NTT domain representation f_hat back to standard polynomial f in R_q.
    """
    f = list(poly.coeffs)
    k = 127
    length = 2
    while length <= 128:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % KYBER_Q
                f[j + length] = (zeta * (f[j + length] - t)) % KYBER_Q
            start += 2 * length
        length *= 2

    for j in range(256):
        f[j] = (f[j] * INV_128) % KYBER_Q

    return Polynomial(f)


def base_case_multiply(a0: int, a1: int, b0: int, b1: int, gamma: int) -> Tuple[int, int]:
    """Multiply two degree-1 polynomials (a0 + a1*X) and (b0 + b1*X) mod (X^2 - gamma)."""
    c0 = (a0 * b0 + a1 * b1 * gamma) % KYBER_Q
    c1 = (a0 * b1 + a1 * b0) % KYBER_Q
    return c0, c1


def poly_ntt_multiply(p1_hat: Polynomial, p2_hat: Polynomial) -> Polynomial:
    """Multiply two polynomials in the NTT domain using 128 base case products."""
    c_hat = [0] * KYBER_N
    for i in range(128):
        gamma = GAMMA_TABLE[i]
        c0, c1 = base_case_multiply(
            p1_hat[2 * i], p1_hat[2 * i + 1],
            p2_hat[2 * i], p2_hat[2 * i + 1],
            gamma
        )
        c_hat[2 * i] = c0
        c_hat[2 * i + 1] = c1
    return Polynomial(c_hat)


def polyvec_forward_ntt(vec: PolyVec) -> PolyVec:
    """Transform each polynomial in vector to the NTT domain."""
    return PolyVec([forward_ntt(p) for p in vec.vec])


def polyvec_inverse_ntt(vec: PolyVec) -> PolyVec:
    """Transform each polynomial in vector from NTT domain back to standard ring."""
    return PolyVec([inverse_ntt(p) for p in vec.vec])


def polyvec_ntt_dot(v1_hat: PolyVec, v2_hat: PolyVec) -> Polynomial:
    """Compute dot product of two polynomial vectors in the NTT domain."""
    if v1_hat.k != v2_hat.k:
        raise ValueError(f"Vector rank mismatch: {v1_hat.k} vs {v2_hat.k}")
    res = Polynomial.zero()
    for p1, p2 in zip(v1_hat.vec, v2_hat.vec):
        res = res + poly_ntt_multiply(p1, p2)
    return res


def matrix_vector_mul_ntt(matrix_A_hat: List[List[Polynomial]], vec_s_hat: PolyVec) -> PolyVec:
    """Multiply matrix of polynomials A_hat (k x k) by vector s_hat (k) in NTT domain."""
    k = vec_s_hat.k
    res_elements: List[Polynomial] = []
    for row in range(k):
        row_vec = PolyVec(matrix_A_hat[row])
        acc = polyvec_ntt_dot(row_vec, vec_s_hat)
        res_elements.append(acc)
    return PolyVec(res_elements)
