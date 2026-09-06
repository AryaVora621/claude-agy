"""LatticeGuard: NIST FIPS 203 ML-KEM Post-Quantum Cryptography Engine.

Pure Python 3.10+ standard library implementation of Module-Lattice-Based
Key-Encapsulation Mechanism (ML-KEM, formerly CRYSTALS-Kyber).

Features:
- Ring arithmetic over R_q = Z_q[X] / (X^256 + 1) with prime modulus q = 3329
- Fast O(n log n) Number Theoretic Transform (NTT) and inverse NTT
- Montgomery reduction (R = 2^16) and Barrett reduction
- Centered Binomial Distribution (CBD_eta) and SHAKE-128 rejection sampling
- NIST standardized parameter sets: ML-KEM-512, ML-KEM-768, ML-KEM-1024
- IND-CPA Public Key Encryption and Fujisaki-Okamoto IND-CCA2 transform
- Constant-time implicit rejection against chosen-ciphertext attacks
- Sub-pixel Unicode Braille lattice coefficient visualizer and telemetry HUD
"""

from __future__ import annotations

from .ring import (
    KYBER_N,
    KYBER_Q,
    MONTGOMERY_R,
    BARRETT_V,
    QINV,
    freeze,
    barrett_reduce,
    montgomery_reduce,
    Polynomial,
    PolyVec,
)

from .ntt import (
    ZETA,
    ZETAS,
    GAMMA_TABLE,
    INV_128,
    bitrev7,
    forward_ntt,
    inverse_ntt,
    poly_ntt_multiply,
    polyvec_forward_ntt,
    polyvec_inverse_ntt,
    polyvec_ntt_dot,
    matrix_vector_mul_ntt,
)

from .sampling import (
    compress_val,
    decompress_val,
    pack_bits,
    unpack_bits,
    poly_to_bytes,
    bytes_to_poly,
    polyvec_to_bytes,
    bytes_to_polyvec,
    compress_poly,
    decompress_poly,
    compress_polyvec,
    decompress_polyvec,
    msg_to_poly,
    poly_to_msg,
    sample_ntt,
    sample_poly_cbd,
    generate_matrix_A,
)

from .ind_cpa import (
    MLKEMParams,
    PARAMS_512,
    PARAMS_768,
    PARAMS_1024,
    CPAKeyPair,
    cpa_keygen,
    cpa_encrypt,
    cpa_decrypt,
)

from .kem import (
    KEMKeyPair,
    EncapsResult,
    MLKEM,
    MLKEM512,
    MLKEM768,
    MLKEM1024,
)

__all__ = [
    # Ring
    "KYBER_N",
    "KYBER_Q",
    "MONTGOMERY_R",
    "BARRETT_V",
    "QINV",
    "freeze",
    "barrett_reduce",
    "montgomery_reduce",
    "Polynomial",
    "PolyVec",
    # NTT
    "ZETA",
    "ZETAS",
    "GAMMA_TABLE",
    "INV_128",
    "bitrev7",
    "forward_ntt",
    "inverse_ntt",
    "poly_ntt_multiply",
    "polyvec_forward_ntt",
    "polyvec_inverse_ntt",
    "polyvec_ntt_dot",
    "matrix_vector_mul_ntt",
    # Sampling & Codecs
    "compress_val",
    "decompress_val",
    "pack_bits",
    "unpack_bits",
    "poly_to_bytes",
    "bytes_to_poly",
    "polyvec_to_bytes",
    "bytes_to_polyvec",
    "compress_poly",
    "decompress_poly",
    "compress_polyvec",
    "decompress_polyvec",
    "msg_to_poly",
    "poly_to_msg",
    "sample_ntt",
    "sample_poly_cbd",
    "generate_matrix_A",
    # IND-CPA
    "MLKEMParams",
    "PARAMS_512",
    "PARAMS_768",
    "PARAMS_1024",
    "CPAKeyPair",
    "cpa_keygen",
    "cpa_encrypt",
    "cpa_decrypt",
    # KEM
    "KEMKeyPair",
    "EncapsResult",
    "MLKEM",
    "MLKEM512",
    "MLKEM768",
    "MLKEM1024",
]
