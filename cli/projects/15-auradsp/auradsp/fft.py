"""
AuraDSP: Fast Fourier Transform & Spectral Analysis Engine.
Implements Cooley-Tukey Radix-2 DIT FFT, IFFT, Discrete Fourier Transform (DFT),
Parseval energy validation, and standard spectral windowing functions.
"""

import cmath
import math
from typing import List, Union

ComplexList = List[complex]
NumericSignal = List[Union[float, complex]]


def is_power_of_two(n: int) -> bool:
    """Returns True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Returns the smallest power of two greater than or equal to n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def dft(x: NumericSignal) -> ComplexList:
    """
    Computes the Discrete Fourier Transform (DFT) via direct O(N^2) summation.
    Reference baseline for verifying FFT accuracy.
    """
    n = len(x)
    output: ComplexList = []
    factor = -2j * math.pi / n
    for k in range(n):
        s = 0.0 + 0.0j
        for t in range(n):
            angle = factor * k * t
            s += x[t] * cmath.exp(angle)
        output.append(s)
    return output


def idft(x: ComplexList) -> ComplexList:
    """
    Computes the Inverse Discrete Fourier Transform (IDFT) via direct O(N^2) summation.
    """
    n = len(x)
    output: ComplexList = []
    factor = 2j * math.pi / n
    for t in range(n):
        s = 0.0 + 0.0j
        for k in range(n):
            angle = factor * k * t
            s += x[k] * cmath.exp(angle)
        output.append(s / n)
    return output


def fft(x: NumericSignal, pad_to_pow2: bool = False) -> ComplexList:
    """
    Computes the Fast Fourier Transform (FFT) using the Cooley-Tukey Radix-2
    Decimation-In-Time (DIT) algorithm with iterative butterfly networks.
    Time Complexity: O(N log N).
    """
    n = len(x)
    if n == 0:
        return []

    if not is_power_of_two(n):
        if pad_to_pow2:
            target_len = next_power_of_two(n)
            padded = list(x) + [0.0] * (target_len - n)
            return fft(padded, pad_to_pow2=False)
        else:
            raise ValueError(f"FFT input length ({n}) must be a power of two. Set pad_to_pow2=True to zero-pad.")

    # Number of bits for bit-reversal
    num_bits = (n - 1).bit_length()

    # Pre-allocate buffer with bit-reversal permutation
    a: ComplexList = [0j] * n
    for i in range(n):
        rev = 0
        val = i
        for _ in range(num_bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        a[rev] = complex(x[i])

    # Iterative Cooley-Tukey butterfly stages
    len_block = 2
    while len_block <= n:
        half = len_block // 2
        # Twiddle factor step
        w_step = cmath.exp(-2j * math.pi / len_block)

        # Precompute twiddle factors for this block size
        twiddles = [1.0 + 0.0j] * half
        w = 1.0 + 0.0j
        for j in range(1, half):
            w *= w_step
            twiddles[j] = w

        for start in range(0, n, len_block):
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half] * twiddles[j]
                a[start + j] = u + v
                a[start + j + half] = u - v

        len_block <<= 1

    return a


def ifft(x: ComplexList) -> ComplexList:
    """
    Computes the Inverse Fast Fourier Transform (IFFT) via conjugate symmetry:
    IFFT(X) = (1/N) * conj(FFT(conj(X)))
    Time Complexity: O(N log N).
    """
    n = len(x)
    if n == 0:
        return []

    if not is_power_of_two(n):
        raise ValueError(f"IFFT input length ({n}) must be a power of two.")

    # Take complex conjugate of input
    conj_in = [z.conjugate() for z in x]
    # Perform forward FFT
    transformed = fft(conj_in)
    # Conjugate and divide by N
    return [z.conjugate() / n for z in transformed]


# Window Functions for Spectral Leakage Suppression

def hann_window(n: int) -> List[float]:
    """Hann (Hanning) window: 0.5 - 0.5 * cos(2 * pi * n / (N - 1))."""
    if n <= 1:
        return [1.0] * n
    denom = n - 1
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / denom) for i in range(n)]


def hamming_window(n: int) -> List[float]:
    """Hamming window: 0.54 - 0.46 * cos(2 * pi * n / (N - 1))."""
    if n <= 1:
        return [1.0] * n
    denom = n - 1
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / denom) for i in range(n)]


def blackman_window(n: int) -> List[float]:
    """Blackman window: 0.42 - 0.5 * cos(...) + 0.08 * cos(...)."""
    if n <= 1:
        return [1.0] * n
    denom = n - 1
    return [
        0.42
        - 0.50 * math.cos(2.0 * math.pi * i / denom)
        + 0.08 * math.cos(4.0 * math.pi * i / denom)
        for i in range(n)
    ]


def apply_window(signal: List[float], window: List[float]) -> List[float]:
    """Applies a windowing function point-wise to an audio signal."""
    if len(signal) != len(window):
        raise ValueError(f"Signal length ({len(signal)}) must match window length ({len(window)})")
    return [s * w for s, w in zip(signal, window)]


# Spectral Metrics & Analysis

def magnitude_spectrum(x: ComplexList) -> List[float]:
    """
    Returns the positive frequency magnitude spectrum |X[k]| for 0 <= k <= N/2.
    """
    n = len(x)
    half = n // 2 + 1
    return [abs(x[k]) for k in range(half)]


def power_spectrum_db(x: ComplexList, min_db: float = -120.0) -> List[float]:
    """
    Returns the magnitude spectrum in decibels (dB): 20 * log10(magnitude).
    Clamped to min_db to avoid log(0) singularities.
    """
    mags = magnitude_spectrum(x)
    n = len(x)
    scale = 2.0 / n if n > 0 else 1.0  # Normalize amplitude

    db_vals: List[float] = []
    for m in mags:
        val = m * scale
        if val > 1e-12:
            db = 20.0 * math.log10(val)
        else:
            db = min_db
        db_vals.append(max(min_db, db))
    return db_vals


def fft_frequencies(n: int, sample_rate: float) -> List[float]:
    """
    Returns frequency values (Hz) for bins 0 <= k <= N/2.
    """
    half = n // 2 + 1
    bin_width = sample_rate / n
    return [k * bin_width for k in range(half)]
