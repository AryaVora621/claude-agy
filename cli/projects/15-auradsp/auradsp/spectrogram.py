"""
AuraDSP: Short-Time Fourier Transform (STFT) & Spectral Analysis Engine.
Computes time-frequency distributions, decibel spectrogram matrices,
spectral centroid, and spectral flatness metrics.
"""

import math
from typing import List, Tuple, Optional, Callable
from auradsp.fft import fft, hann_window, apply_window, magnitude_spectrum, power_spectrum_db, fft_frequencies


def stft(
    signal: List[float],
    frame_size: int = 512,
    hop_size: int = 256,
    window_fn: Optional[Callable[[int], List[float]]] = None
) -> List[List[complex]]:
    """
    Computes the Short-Time Fourier Transform (STFT) over overlapping sliding windows.
    Parameters:
      - signal: 1D audio sample list
      - frame_size: Length of each FFT frame (must be power of two)
      - hop_size: Step advance between consecutive frames
      - window_fn: Windowing function (defaults to Hann window)
    Returns:
      List of complex spectrum frames: frames[t][k]
    """
    if len(signal) < frame_size:
        return []

    if window_fn is None:
        window = hann_window(frame_size)
    else:
        window = window_fn(frame_size)

    frames: List[List[complex]] = []
    num_samples = len(signal)
    start = 0

    while start + frame_size <= num_samples:
        segment = signal[start:start + frame_size]
        windowed = apply_window(segment, window)
        spectrum = fft(windowed)
        frames.append(spectrum)
        start += hop_size

    return frames


def spectrogram_matrix(
    signal: List[float],
    sample_rate: float = 44100.0,
    frame_size: int = 512,
    hop_size: int = 256,
    min_db: float = -80.0
) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Computes a 2D magnitude spectrogram normalized to [0.0, 1.0].
    Returns:
      (matrix: List[List[float]] of shape [time_steps][num_freq_bins],
       times: List[float] timestamps in seconds,
       freqs: List[float] bin frequencies in Hz)
    """
    frames = stft(signal, frame_size=frame_size, hop_size=hop_size)
    if not frames:
        return [], [], []

    half_bins = frame_size // 2 + 1
    freqs = fft_frequencies(frame_size, sample_rate)
    times = [(i * hop_size) / sample_rate for i in range(len(frames))]

    matrix: List[List[float]] = []
    db_range = abs(min_db)

    for frame in frames:
        db_vals = power_spectrum_db(frame, min_db=min_db)
        # Normalize dB from [min_db, 0] to [0.0, 1.0]
        row: List[float] = []
        for db in db_vals:
            norm_val = (db - min_db) / db_range
            row.append(max(0.0, min(1.0, norm_val)))
        matrix.append(row)

    return matrix, times, freqs


def spectral_centroid(spectrum_mags: List[float], freqs: List[float]) -> float:
    """
    Computes the spectral centroid (center of mass / perceptual brightness of sound) in Hz.
    """
    sum_weighted = 0.0
    sum_mags = 0.0
    for mag, f in zip(spectrum_mags, freqs):
        sum_weighted += mag * f
        sum_mags += mag
    return sum_weighted / sum_mags if sum_mags > 1e-12 else 0.0


def spectral_flatness(spectrum_mags: List[float]) -> float:
    """
    Computes spectral flatness (Wiener entropy): ratio of geometric mean to arithmetic mean.
    Values near 0 indicate tonal sounds; values near 1 indicate white noise.
    """
    if not spectrum_mags:
        return 0.0

    n = len(spectrum_mags)
    log_sum = 0.0
    arithmetic_sum = 0.0

    for m in spectrum_mags:
        clamped = max(1e-12, m)
        log_sum += math.log(clamped)
        arithmetic_sum += clamped

    geometric_mean = math.exp(log_sum / n)
    arithmetic_mean = arithmetic_sum / n
    return geometric_mean / arithmetic_mean if arithmetic_mean > 1e-12 else 0.0
