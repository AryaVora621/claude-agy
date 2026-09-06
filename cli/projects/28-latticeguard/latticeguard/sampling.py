"""Rejection Sampling, Centered Binomial Distributions, and Bit-Packing Codecs.

Implements:
1. Rejection sampling from SHAKE-128 stream for uniform NTT matrix A
2. Centered Binomial Distribution (CBD_eta) noise sampling for eta in {2, 3}
3. Modulus compression and decompression algorithms
4. Bit-level serialization codecs for 12-bit polynomials and d-bit compressed vectors
"""

from __future__ import annotations
import hashlib
from typing import List, Sequence, Tuple

from .ring import KYBER_N, KYBER_Q, Polynomial, PolyVec, freeze


def compress_val(x: int, d: int) -> int:
    """Compress field element x in [0, q-1] down to d bits."""
    return (((x << d) + (KYBER_Q >> 1)) // KYBER_Q) & ((1 << d) - 1)


def decompress_val(y: int, d: int) -> int:
    """Decompress d-bit integer y back to field representative in [0, q-1]."""
    return ((y * KYBER_Q) + (1 << (d - 1))) >> d


def pack_bits(values: Sequence[int], d: int) -> bytes:
    """Pack an array of integers each of width d bits into a byte string."""
    total_bits = len(values) * d
    total_bytes = (total_bits + 7) // 8
    buf = bytearray(total_bytes)
    bit_pos = 0
    mask = (1 << d) - 1

    for v in values:
        v_masked = v & mask
        for bit_idx in range(d):
            bit = (v_masked >> bit_idx) & 1
            if bit:
                byte_idx = (bit_pos + bit_idx) >> 3
                sub_bit = (bit_pos + bit_idx) & 7
                buf[byte_idx] |= (1 << sub_bit)
        bit_pos += d

    return bytes(buf)


def unpack_bits(buf: bytes, d: int, count: int = KYBER_N) -> List[int]:
    """Unpack count d-bit integers from a byte string."""
    values = [0] * count
    bit_pos = 0

    for i in range(count):
        val = 0
        for bit_idx in range(d):
            byte_idx = (bit_pos + bit_idx) >> 3
            sub_bit = (bit_pos + bit_idx) & 7
            bit = (buf[byte_idx] >> sub_bit) & 1
            val |= (bit << bit_idx)
        values[i] = val
        bit_pos += d

    return values


def poly_to_bytes(poly: Polynomial) -> bytes:
    """Serialize 256 12-bit coefficients into 384 bytes."""
    coeffs = [freeze(c) for c in poly.coeffs]
    raw = bytearray(384)
    for i in range(128):
        c0 = coeffs[2 * i]
        c1 = coeffs[2 * i + 1]
        raw[3 * i + 0] = c0 & 0xFF
        raw[3 * i + 1] = ((c0 >> 8) & 0x0F) | ((c1 & 0x0F) << 4)
        raw[3 * i + 2] = (c1 >> 4) & 0xFF
    return bytes(raw)


def bytes_to_poly(buf: bytes) -> Polynomial:
    """Deserialize 384 bytes into a polynomial with 256 coefficients."""
    coeffs = [0] * KYBER_N
    for i in range(128):
        b0 = buf[3 * i + 0]
        b1 = buf[3 * i + 1]
        b2 = buf[3 * i + 2]
        coeffs[2 * i + 0] = b0 | ((b1 & 0x0F) << 8)
        coeffs[2 * i + 1] = (b1 >> 4) | (b2 << 4)
    return Polynomial(coeffs)


def polyvec_to_bytes(vec: PolyVec) -> bytes:
    """Serialize a PolyVec of length k into k * 384 bytes."""
    return b"".join(poly_to_bytes(p) for p in vec.vec)


def bytes_to_polyvec(buf: bytes, k: int) -> PolyVec:
    """Deserialize k * 384 bytes into a PolyVec of length k."""
    elements: List[Polynomial] = []
    chunk_size = 384
    for i in range(k):
        sub_buf = buf[i * chunk_size : (i + 1) * chunk_size]
        elements.append(bytes_to_poly(sub_buf))
    return PolyVec(elements)


def compress_poly(poly: Polynomial, d: int) -> bytes:
    """Compress and pack polynomial coefficients down to d bits each."""
    compressed_coeffs = [compress_val(c, d) for c in poly.coeffs]
    return pack_bits(compressed_coeffs, d)


def decompress_poly(buf: bytes, d: int) -> Polynomial:
    """Unpack and decompress d-bit integers back into polynomial coefficients."""
    unpacked = unpack_bits(buf, d, KYBER_N)
    decompressed = [decompress_val(y, d) for y in unpacked]
    return Polynomial(decompressed)


def compress_polyvec(vec: PolyVec, d: int) -> bytes:
    """Compress and pack each polynomial in a PolyVec."""
    return b"".join(compress_poly(p, d) for p in vec.vec)


def decompress_polyvec(buf: bytes, k: int, d: int) -> PolyVec:
    """Unpack and decompress a PolyVec from packed bytes."""
    poly_bytes = (KYBER_N * d) // 8
    elements: List[Polynomial] = []
    for i in range(k):
        sub = buf[i * poly_bytes : (i + 1) * poly_bytes]
        elements.append(decompress_poly(sub, d))
    return PolyVec(elements)


def msg_to_poly(msg_32bytes: bytes) -> Polynomial:
    """Convert 32-byte message (256 bits) to polynomial via 1-bit decompression."""
    coeffs = [0] * KYBER_N
    for i in range(256):
        byte_idx = i >> 3
        bit_idx = i & 7
        bit = (msg_32bytes[byte_idx] >> bit_idx) & 1
        # Decompress 1 bit to 0 or round(q / 2) = 1665
        coeffs[i] = decompress_val(bit, 1)
    return Polynomial(coeffs)


def poly_to_msg(poly: Polynomial) -> bytes:
    """Convert polynomial back to 32-byte message via 1-bit compression."""
    bits = [compress_val(c, 1) for c in poly.coeffs]
    return pack_bits(bits, 1)


def sample_ntt(seed_rho: bytes, row: int, col: int) -> Polynomial:
    """Rejection sample 256 uniform coefficients modulo q in the NTT domain.

    Uses SHAKE-128 sponge initialized with seed_rho || col || row.
    """
    coeffs = [0] * KYBER_N
    count = 0
    buf_len = 504

    while count < KYBER_N:
        shake = hashlib.shake_128()
        shake.update(seed_rho)
        shake.update(bytes([col, row]))
        buf = shake.digest(buf_len)

        pos = 0
        count = 0
        while pos + 3 <= len(buf) and count < KYBER_N:
            b0 = buf[pos]
            b1 = buf[pos + 1]
            b2 = buf[pos + 2]
            pos += 3

            d1 = b0 | ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) | (b2 << 4)

            if d1 < KYBER_Q and count < KYBER_N:
                coeffs[count] = d1
                count += 1
            if d2 < KYBER_Q and count < KYBER_N:
                coeffs[count] = d2
                count += 1

        buf_len += 168

    return Polynomial(coeffs)


def sample_poly_cbd(buf: bytes, eta: int) -> Polynomial:
    """Sample polynomial with coefficients from Centered Binomial Distribution CBD_eta."""
    coeffs = [0] * KYBER_N

    if eta == 2:
        # 128 bytes needed for 256 coefficients (4 bits per coefficient)
        if len(buf) < 128:
            raise ValueError(f"CBD2 requires at least 128 bytes, got {len(buf)}")
        for i in range(256):
            byte_idx = i >> 1
            shift = 4 * (i & 1)
            val = (buf[byte_idx] >> shift) & 0x0F
            a = (val & 1) + ((val >> 1) & 1)
            b = ((val >> 2) & 1) + ((val >> 3) & 1)
            coeffs[i] = freeze(a - b)

    elif eta == 3:
        # 192 bytes needed for 256 coefficients (6 bits per coefficient)
        if len(buf) < 192:
            raise ValueError(f"CBD3 requires at least 192 bytes, got {len(buf)}")
        for i in range(0, 256, 4):
            byte_idx = (i * 3) >> 2
            chunk = buf[byte_idx] | (buf[byte_idx + 1] << 8) | (buf[byte_idx + 2] << 16)
            for j in range(4):
                val = (chunk >> (6 * j)) & 0x3F
                a = (val & 1) + ((val >> 1) & 1) + ((val >> 2) & 1)
                b = ((val >> 3) & 1) + ((val >> 4) & 1) + ((val >> 5) & 1)
                coeffs[i + j] = freeze(a - b)
    else:
        raise ValueError(f"Unsupported eta parameter: {eta} (expected 2 or 3)")

    return Polynomial(coeffs)


def generate_matrix_A(seed_rho: bytes, k: int, transposed: bool = False) -> List[List[Polynomial]]:
    """Generate k x k matrix of polynomials in the NTT domain via rejection sampling."""
    matrix: List[List[Polynomial]] = []
    for i in range(k):
        row: List[Polynomial] = []
        for j in range(k):
            if transposed:
                p = sample_ntt(seed_rho, j, i)
            else:
                p = sample_ntt(seed_rho, i, j)
            row.append(p)
        matrix.append(row)
    return matrix
