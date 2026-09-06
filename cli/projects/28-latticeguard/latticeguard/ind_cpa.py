"""Module-LWE IND-CPA Public Key Encryption (PKE) Scheme.

Implements the NIST FIPS 203 K-PKE scheme:
1. K-PKE.KeyGen: Generates public matrix A, secret vector s, error vector e,
   and computes public key t = A*s + e in the NTT domain.
2. K-PKE.Encrypt: Encrypts 32-byte message m using public key (t, rho) and random coins r.
   Computes u = A^T * r + e1 and v = t^T * r + e2 + encode(m), compresses to ciphertext.
3. K-PKE.Decrypt: Decrypts ciphertext (c1, c2) using secret key s.
   Computes w = v - s^T * u and decodes back to 32-byte message m.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from typing import NamedTuple, Tuple

from .ring import KYBER_N, KYBER_Q, Polynomial, PolyVec
from .ntt import (
    forward_ntt,
    inverse_ntt,
    matrix_vector_mul_ntt,
    polyvec_forward_ntt,
    polyvec_inverse_ntt,
    polyvec_ntt_dot,
)
from .sampling import (
    compress_poly,
    compress_polyvec,
    decompress_poly,
    decompress_polyvec,
    generate_matrix_A,
    msg_to_poly,
    poly_to_bytes,
    poly_to_msg,
    polyvec_to_bytes,
    sample_poly_cbd,
)


@dataclass(frozen=True)
class MLKEMParams:
    """Standardized FIPS 203 ML-KEM parameter set."""
    name: str
    k: int         # Module rank (2, 3, or 4)
    eta1: int      # CBD parameter for secret and keygen error
    eta2: int      # CBD parameter for encryption noise
    du: int        # Compression bit width for vector u
    dv: int        # Compression bit width for scalar v

    @property
    def pk_bytes_len(self) -> int:
        """Public key byte length: 384 * k + 32."""
        return 384 * self.k + 32

    @property
    def sk_cpa_bytes_len(self) -> int:
        """CPA secret key byte length: 384 * k."""
        return 384 * self.k

    @property
    def ct_bytes_len(self) -> int:
        """Ciphertext byte length: (256 * du * k) / 8 + (256 * dv) / 8."""
        return (32 * self.du * self.k) + (32 * self.dv)

    @property
    def c1_bytes_len(self) -> int:
        """Compressed vector u byte length: 32 * du * k."""
        return 32 * self.du * self.k


# Standard NIST FIPS 203 parameter sets
PARAMS_512 = MLKEMParams(name="ML-KEM-512", k=2, eta1=3, eta2=2, du=10, dv=4)
PARAMS_768 = MLKEMParams(name="ML-KEM-768", k=3, eta1=2, eta2=2, du=10, dv=4)
PARAMS_1024 = MLKEMParams(name="ML-KEM-1024", k=4, eta1=2, eta2=2, du=11, dv=5)


def _prf(seed: bytes, nonce: int, n_bytes: int) -> bytes:
    """Pseudo-random function PRF_eta(seed, nonce) using SHAKE-256."""
    shake = hashlib.shake_256()
    shake.update(seed)
    shake.update(bytes([nonce]))
    return shake.digest(n_bytes)


class CPAKeyPair(NamedTuple):
    """Container for IND-CPA public and secret keys."""
    public_key: bytes
    secret_key: bytes
    params: MLKEMParams


def cpa_keygen(params: MLKEMParams, seed_d: bytes | None = None) -> CPAKeyPair:
    """Generate IND-CPA key pair (ek_PKE, dk_PKE) according to FIPS 203 Algorithm 12.

    Args:
        params: MLKEMParams parameter set (512, 768, or 1024).
        seed_d: Optional 32-byte seed. If None, generated from os.urandom(32).

    Returns:
        CPAKeyPair containing serialized public key and secret key.
    """
    if seed_d is None:
        seed_d = os.urandom(32)
    elif len(seed_d) != 32:
        raise ValueError(f"Seed d must be exactly 32 bytes, got {len(seed_d)}")

    # G(d || k) -> (rho, sigma)
    g_hash = hashlib.sha3_512(seed_d + bytes([params.k])).digest()
    rho = g_hash[:32]
    sigma = g_hash[32:]

    # Expand public matrix A_hat in NTT domain
    A_hat = generate_matrix_A(rho, params.k, transposed=False)

    # Sample secret vector s from CBD_eta1
    s_elements = []
    for i in range(params.k):
        buf = _prf(sigma, i, 64 * params.eta1)
        s_elements.append(sample_poly_cbd(buf, params.eta1))
    s_vec = PolyVec(s_elements)

    # Sample error vector e from CBD_eta1
    e_elements = []
    for i in range(params.k):
        buf = _prf(sigma, params.k + i, 64 * params.eta1)
        e_elements.append(sample_poly_cbd(buf, params.eta1))
    e_vec = PolyVec(e_elements)

    # Transform s and e to NTT domain
    s_hat = polyvec_forward_ntt(s_vec)
    e_hat = polyvec_forward_ntt(e_vec)

    # Compute t_hat = A_hat * s_hat + e_hat in NTT domain
    t_hat = matrix_vector_mul_ntt(A_hat, s_hat) + e_hat

    # Encode public key ek_PKE = ByteEncode_12(t_hat) || rho
    pk_bytes = polyvec_to_bytes(t_hat) + rho

    # Encode secret key dk_PKE = ByteEncode_12(s_hat)
    sk_bytes = polyvec_to_bytes(s_hat)

    return CPAKeyPair(public_key=pk_bytes, secret_key=sk_bytes, params=params)


def cpa_encrypt(
    params: MLKEMParams,
    public_key: bytes,
    msg_32bytes: bytes,
    coins_32bytes: bytes | None = None,
) -> bytes:
    """Encrypt 32-byte message under public key according to FIPS 203 Algorithm 13.

    Args:
        params: MLKEMParams parameter set.
        public_key: Serialized public key (384*k + 32 bytes).
        msg_32bytes: 32-byte plaintext message.
        coins_32bytes: Optional 32-byte random coin seed for deterministic encryption.

    Returns:
        Serialized ciphertext byte array.
    """
    if len(public_key) != params.pk_bytes_len:
        raise ValueError(
            f"Invalid public key length: {len(public_key)} (expected {params.pk_bytes_len})"
        )
    if len(msg_32bytes) != 32:
        raise ValueError(f"Message must be exactly 32 bytes, got {len(msg_32bytes)}")

    if coins_32bytes is None:
        coins_32bytes = os.urandom(32)
    elif len(coins_32bytes) != 32:
        raise ValueError(f"Coins must be exactly 32 bytes, got {len(coins_32bytes)}")

    # Unpack t_hat and rho from public key
    t_hat_bytes = public_key[: 384 * params.k]
    rho = public_key[384 * params.k :]

    from .sampling import bytes_to_polyvec
    t_hat = bytes_to_polyvec(t_hat_bytes, params.k)

    # Generate transposed matrix A^T in NTT domain
    A_T_hat = generate_matrix_A(rho, params.k, transposed=True)

    # Sample random vector r from CBD_eta1
    r_elements = []
    for i in range(params.k):
        buf = _prf(coins_32bytes, i, 64 * params.eta1)
        r_elements.append(sample_poly_cbd(buf, params.eta1))
    r_vec = PolyVec(r_elements)

    # Sample error vector e1 from CBD_eta2
    e1_elements = []
    for i in range(params.k):
        buf = _prf(coins_32bytes, params.k + i, 64 * params.eta2)
        e1_elements.append(sample_poly_cbd(buf, params.eta2))
    e1_vec = PolyVec(e1_elements)

    # Sample error scalar e2 from CBD_eta2
    e2_buf = _prf(coins_32bytes, 2 * params.k, 64 * params.eta2)
    e2 = sample_poly_cbd(e2_buf, params.eta2)

    # Transform r to NTT domain
    r_hat = polyvec_forward_ntt(r_vec)

    # Compute u = INTT(A_T_hat * r_hat) + e1
    u = polyvec_inverse_ntt(matrix_vector_mul_ntt(A_T_hat, r_hat)) + e1_vec

    # Encode message m to polynomial mu
    mu = msg_to_poly(msg_32bytes)

    # Compute v = INTT(t_hat^T * r_hat) + e2 + mu
    v = inverse_ntt(polyvec_ntt_dot(t_hat, r_hat)) + e2 + mu

    # Compress u and v
    c1 = compress_polyvec(u, params.du)
    c2 = compress_poly(v, params.dv)

    return c1 + c2


def cpa_decrypt(
    params: MLKEMParams,
    secret_key: bytes,
    ciphertext: bytes,
) -> bytes:
    """Decrypt ciphertext using secret key according to FIPS 203 Algorithm 14.

    Args:
        params: MLKEMParams parameter set.
        secret_key: Serialized CPA secret key (384*k bytes).
        ciphertext: Serialized ciphertext.

    Returns:
        32-byte decrypted message.
    """
    if len(secret_key) != params.sk_cpa_bytes_len:
        raise ValueError(
            f"Invalid secret key length: {len(secret_key)} (expected {params.sk_cpa_bytes_len})"
        )
    if len(ciphertext) != params.ct_bytes_len:
        raise ValueError(
            f"Invalid ciphertext length: {len(ciphertext)} (expected {params.ct_bytes_len})"
        )

    from .sampling import bytes_to_polyvec
    s_hat = bytes_to_polyvec(secret_key, params.k)

    # Decompress u and v
    c1_len = params.c1_bytes_len
    c1_bytes = ciphertext[:c1_len]
    c2_bytes = ciphertext[c1_len:]

    u = decompress_polyvec(c1_bytes, params.k, params.du)
    v = decompress_poly(c2_bytes, params.dv)

    # Transform u to NTT domain and compute w = v - INTT(s_hat^T * u_hat)
    u_hat = polyvec_forward_ntt(u)
    inner = inverse_ntt(polyvec_ntt_dot(s_hat, u_hat))
    w = v - inner

    # Decode w to 32-byte message
    return poly_to_msg(w)
