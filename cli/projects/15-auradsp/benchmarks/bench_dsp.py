"""
AuraDSP Performance Benchmarking Suite.
Evaluates Cooley-Tukey FFT speedup vs naive DFT, biquad filter throughput,
polyphonic sound synthesis rate, STFT spectral decomposition, and 16-bit WAV codec speed.
Pure Python standard library.
"""

import math
import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auradsp.fft import fft, ifft, dft
from auradsp.filters import BiquadFilter, CascadeFilter
from auradsp.synth import Oscillator, ADSREnvelope, PolyphonicSynth
from auradsp.spectrogram import spectrogram_matrix
from auradsp.wav import encode_wav_bytes, decode_wav_bytes


def benchmark_fft_vs_dft() -> None:
    print("================================================================================")
    print("  Benchmark 1: Cooley-Tukey Radix-2 FFT vs Naive DFT Algorithmic Speedup")
    print("================================================================================")
    print(f" {'N':>6} | {'Naive DFT (ms)':>15} | {'Radix-2 FFT (ms)':>17} | {'Speedup':>10} | {'Ops (N log N)':>14}")
    print("-" * 75)

    sizes = [64, 128, 256, 512, 1024]
    for n in sizes:
        signal = [math.sin(0.1 * i) for i in range(n)]

        # Time Naive DFT (fewer reps for larger N)
        reps_dft = 100 if n <= 256 else (20 if n <= 512 else 5)
        t0 = time.perf_counter()
        for _ in range(reps_dft):
            dft(signal)
        t_dft = (time.perf_counter() - t0) / reps_dft * 1000.0

        # Time Radix-2 FFT
        reps_fft = 200
        t0 = time.perf_counter()
        for _ in range(reps_fft):
            fft(signal)
        t_fft = (time.perf_counter() - t0) / reps_fft * 1000.0

        speedup = t_dft / t_fft if t_fft > 0 else float("inf")
        ops = int(n * math.log2(n))

        print(f" {n:6d} | {t_dft:15.3f} | {t_fft:17.3f} | {speedup:9.1f}x | {ops:14,d}")

    # Scale test for larger power-of-two FFTs
    print("\n  High-Resolution FFT Scaling (Pure Python):")
    large_sizes = [2048, 4096, 8192]
    for n in large_sizes:
        signal = [math.sin(0.05 * i) for i in range(n)]
        reps = 30 if n <= 4096 else 10
        t0 = time.perf_counter()
        for _ in range(reps):
            fft(signal)
        t_fft = (time.perf_counter() - t0) / reps * 1000.0
        rate = n / (t_fft / 1000.0)
        print(f"   N = {n:5d} : {t_fft:6.2f} ms per FFT ({rate:12,.0f} samples/sec)")
    print()


def benchmark_filters() -> None:
    print("================================================================================")
    print("  Benchmark 2: Digital Biquad IIR Filter Processing Throughput")
    print("================================================================================")
    sample_rate = 44100.0
    n_samples = 44100 * 2  # 2 seconds of audio = 88,200 samples
    signal = [math.sin(0.05 * i) for i in range(n_samples)]

    # Single 2nd-order low-pass biquad
    lpf = BiquadFilter.low_pass(sample_rate, 1000.0, q=0.7071)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        lpf.reset()
        lpf.process(signal)
    t_single = (time.perf_counter() - t0) / reps
    rate_single = n_samples / t_single

    # 4th-order cascade Butterworth
    cascade = CascadeFilter.butterworth_4th_order_low_pass(sample_rate, 1000.0)
    t0 = time.perf_counter()
    for _ in range(reps):
        cascade.reset()
        cascade.process(signal)
    t_cascade = (time.perf_counter() - t0) / reps
    rate_cascade = n_samples / t_cascade

    print(f"  Signal Length        : {n_samples:,d} samples ({n_samples / sample_rate:.1f}s audio)")
    print(f"  2nd-Order Biquad     : {rate_single:12,.0f} samples/sec ({rate_single / sample_rate:5.1f}x Real-Time)")
    print(f"  4th-Order Cascade    : {rate_cascade:12,.0f} samples/sec ({rate_cascade / sample_rate:5.1f}x Real-Time)")
    print()


def benchmark_synthesis() -> None:
    print("================================================================================")
    print("  Benchmark 3: Sound Synthesis & ADSR Modulation Throughput")
    print("================================================================================")
    sample_rate = 44100.0
    synth = PolyphonicSynth(sample_rate=sample_rate)
    env = ADSREnvelope(0.02, 0.1, 0.6, 0.1, sample_rate=sample_rate)

    durations = 1.0  # 1 second = 44,100 samples
    waveforms = ["sine", "saw", "square", "triangle"]

    for wf in waveforms:
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            synth.synthesize_note(440.0, durations, waveform=wf, envelope=env)
        elapsed = (time.perf_counter() - t0) / reps
        rate = 44100 / elapsed
        print(f"  Waveform '{wf:<8}': {rate:12,.0f} samples/sec ({rate / sample_rate:5.1f}x Real-Time)")

    # Polyphonic 4-voice chord synthesis
    t0 = time.perf_counter()
    reps = 10
    chord_freqs = [261.63, 329.63, 392.00, 493.88]  # Cmaj7
    for _ in range(reps):
        synth.synthesize_chord(chord_freqs, durations, waveform="saw")
    elapsed_chord = (time.perf_counter() - t0) / reps
    rate_chord = 44100 / elapsed_chord
    print(f"  4-Voice Chord Synth : {rate_chord:12,.0f} samples/sec ({rate_chord / sample_rate:5.1f}x Real-Time)")
    print()


def benchmark_stft_spectrogram() -> None:
    print("================================================================================")
    print("  Benchmark 4: STFT Sliding Window Spectrogram Decomposition")
    print("================================================================================")
    sample_rate = 44100.0
    duration = 2.0  # 2 seconds
    n_samples = int(duration * sample_rate)
    signal = [math.sin(0.03 * i) + 0.5 * math.sin(0.07 * i) for i in range(n_samples)]

    configs = [
        (256, 128),
        (512, 256),
        (1024, 512),
    ]

    for frame_size, hop_size in configs:
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            spectrogram_matrix(signal, sample_rate=sample_rate, frame_size=frame_size, hop_size=hop_size)
        elapsed = (time.perf_counter() - t0) / reps
        num_frames = 1 + (n_samples - frame_size) // hop_size
        fps = num_frames / elapsed
        rt_factor = duration / elapsed
        print(f"  Frame {frame_size:4d}, Hop {hop_size:4d} : {num_frames:5d} frames in {elapsed * 1000.0:6.2f} ms ({fps:7,.1f} frames/sec, {rt_factor:4.1f}x Real-Time)")
    print()


def benchmark_wav_codec() -> None:
    print("================================================================================")
    print("  Benchmark 5: 16-Bit Linear PCM RIFF WAV Binary Codec")
    print("================================================================================")
    sample_rate = 44100
    n_samples = sample_rate * 5  # 5 seconds = 220,500 samples
    signal = [0.8 * math.sin(0.05 * i) for i in range(n_samples)]

    # Benchmark WAV encoding
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        wav_bytes = encode_wav_bytes(signal, sample_rate)
    t_enc = (time.perf_counter() - t0) / reps
    mb_size = len(wav_bytes) / (1024.0 * 1024.0)
    mb_rate_enc = mb_size / t_enc

    # Benchmark WAV decoding
    t0 = time.perf_counter()
    for _ in range(reps):
        decode_wav_bytes(wav_bytes)
    t_dec = (time.perf_counter() - t0) / reps
    mb_rate_dec = mb_size / t_dec

    print(f"  Data Payload         : {len(wav_bytes):,d} bytes ({mb_size:.2f} MB, {n_samples:,d} samples)")
    print(f"  16-bit PCM Encode    : {mb_rate_enc:8.2f} MB/sec ({n_samples / t_enc:12,.0f} samples/sec)")
    print(f"  16-bit PCM Decode    : {mb_rate_dec:8.2f} MB/sec ({n_samples / t_dec:12,.0f} samples/sec)")
    print()


def main() -> None:
    print("\nStarting AuraDSP Audio Signal Processing Benchmarks...\n")
    benchmark_fft_vs_dft()
    benchmark_filters()
    benchmark_synthesis()
    benchmark_stft_spectrogram()
    benchmark_wav_codec()
    print("All AuraDSP benchmarks completed successfully.\n")


if __name__ == "__main__":
    main()
