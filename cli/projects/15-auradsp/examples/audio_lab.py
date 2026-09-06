"""
AuraDSP: Interactive Audio Digital Signal Processing & Synthesis Laboratory.
Demonstrates FFT spectral decomposition, RBJ biquad IIR filtering, polyphonic sound synthesis,
16-bit WAV file generation, and real-time Unicode Braille spectrogram waterfall visualization.
"""

import math
import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auradsp.fft import fft, ifft, magnitude_spectrum, power_spectrum_db, fft_frequencies
from auradsp.filters import BiquadFilter, CascadeFilter
from auradsp.synth import Oscillator, ADSREnvelope, LFO, PolyphonicSynth, midi_to_freq
from auradsp.wav import write_wav
from auradsp.spectrogram import stft, spectrogram_matrix, spectral_centroid
from auradsp.visualizer import render_braille_waterfall, render_spectrum_bars


def print_banner() -> None:
    banner = r"""
================================================================================
  AURADSP : Audio Digital Signal Processing, Spectral Analysis & Synthesis
  Pure Python 3 Standard Library - Zero External Dependencies
================================================================================
"""
    print(banner)


def demo_fft_spectral_analysis() -> None:
    print("--- 1. Cooley-Tukey Radix-2 FFT Spectral Decomposition ---")
    sample_rate = 44100.0
    n = 1024  # 1024-point FFT

    # Construct dual-tone signal: Concert A (440 Hz) + E5 harmonic (660 Hz)
    dt = 1.0 / sample_rate
    signal = [
        0.6 * math.sin(2.0 * math.pi * 440.0 * i * dt) +
        0.4 * math.sin(2.0 * math.pi * 660.0 * i * dt)
        for i in range(n)
    ]

    t0 = time.perf_counter()
    spectrum = fft(signal)
    t_fft = (time.perf_counter() - t0) * 1e6

    freqs = fft_frequencies(n, sample_rate)
    db_vals = power_spectrum_db(spectrum)

    # Reconstruct via IFFT
    t0 = time.perf_counter()
    reconstructed = ifft(spectrum)
    t_ifft = (time.perf_counter() - t0) * 1e6

    max_err = max(abs(s - r.real) for s, r in zip(signal, reconstructed))

    print(f"FFT Execution Time   : {t_fft:.2f} us for {n}-point transform")
    print(f"IFFT Execution Time  : {t_ifft:.2f} us")
    print(f"Max Reconstruction Err: {max_err:.2e} (Bit-exact mathematical inverse)")

    print("\nInstantaneous Spectrum Bar Chart (0 Hz - 2.5 kHz):")
    # Sub-select frequencies up to 2500 Hz for high-detail view
    cutoff_bin = int(2500.0 / (sample_rate / n))
    print(render_spectrum_bars(db_vals[:cutoff_bin], freqs[:cutoff_bin], num_bars=45, bar_height=7))
    print()


def demo_biquad_filtering() -> None:
    print("--- 2. Robert Bristow-Johnson Digital Biquad IIR Filter Design ---")
    sample_rate = 44100.0
    cutoff_hz = 1200.0

    # Low-Pass Butterworth Filter
    lpf = BiquadFilter.low_pass(sample_rate, cutoff_hz=cutoff_hz, q=1.0 / math.sqrt(2.0))
    p1, p2 = lpf.poles()
    print(f"Low-Pass Biquad Filter (Cutoff = {cutoff_hz} Hz, Q = 0.7071):")
    print(f"  Normalized Transfer Function:")
    print(f"    H(z) = ({lpf.b0:.4f} + {lpf.b1:.4f}*z^-1 + {lpf.b2:.4f}*z^-2) / (1 + {lpf.a1:.4f}*z^-1 + {lpf.a2:.4f}*z^-2)")
    print(f"  Complex Poles: {p1.real:.4f}{p1.imag:+.4f}j, {p2.real:.4f}{p2.imag:+.4f}j (Magnitudes: {abs(p1):.4f}, {abs(p2):.4f})")
    print(f"  BIBO Stability: {lpf.is_stable()} (Poles reside strictly within unit circle)")

    # Evaluate frequency response across log scale
    eval_freqs = [100.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 10000.0]
    mags_db, phases = lpf.frequency_response(eval_freqs)
    print("\n  Frequency Response Curve:")
    for f, db, ph in zip(eval_freqs, mags_db, phases):
        bar_len = max(0, int((db + 40.0) / 40.0 * 25))
        bar = "#" * bar_len
        print(f"    {f:7.0f} Hz: {db:6.2f} dB | {ph:6.1f} deg | {bar}")
    print()


def demo_sound_synthesis_and_wav() -> None:
    print("--- 3. Polyphonic Sound Synthesis & 16-Bit RIFF WAV Generation ---")
    sample_rate = 44100.0
    synth = PolyphonicSynth(sample_rate=sample_rate)

    # Create Arpeggiated Progression: C4, E4, G4, B4, C5 (Cmaj7 Arpeggio)
    notes = [60, 64, 67, 71, 72]  # MIDI notes
    note_duration = 0.25  # 250 ms per note
    total_audio: List[float] = []

    # ADSR: Crisp plucky synth envelope
    env = ADSREnvelope(attack_time=0.01, decay_time=0.12, sustain_level=0.4, release_time=0.1, sample_rate=sample_rate)
    vibrato = LFO(rate_hz=5.5, depth=4.0, sample_rate=sample_rate)

    for midi_note in notes:
        freq = midi_to_freq(midi_note)
        tone = synth.synthesize_note(
            frequency=freq,
            duration=note_duration,
            waveform="saw",
            envelope=env,
            vibrato_lfo=vibrato,
            amplitude=0.75
        )
        total_audio.extend(tone)

    # Apply Low-Pass Filter sweep for warm analog feel
    lpf = BiquadFilter.low_pass(sample_rate, cutoff_hz=2200.0, q=1.2)
    filtered_audio = lpf.process(total_audio)

    # Export to .wav file
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_wav = os.path.join(out_dir, "arpeggio_cmaj7.wav")
    write_wav(out_wav, filtered_audio, sample_rate=int(sample_rate))

    file_size_kb = os.path.getsize(out_wav) / 1024.0
    print(f"Synthesized Cmaj7 Arpeggio ({len(notes)} notes, {len(filtered_audio)} samples):")
    print(f"  Duration  : {len(filtered_audio) / sample_rate:.2f} seconds")
    print(f"  Output WAV: {os.path.abspath(out_wav)}")
    print(f"  File Size : {file_size_kb:.1f} KB (16-bit 44.1 kHz linear PCM)")
    print()


def demo_braille_waterfall_spectrogram() -> None:
    print("--- 4. Short-Time Fourier Transform & Unicode Braille Waterfall ---")
    sample_rate = 22050.0
    duration = 1.0
    n_samples = int(duration * sample_rate)

    # Generate an ascending frequency chirp (200 Hz to 4000 Hz) plus a steady bass pedal
    chirp_signal: List[float] = []
    dt = 1.0 / sample_rate
    for i in range(n_samples):
        t = i * dt
        # Linear frequency chirp: f(t) = f0 + (f1 - f0) * (t / T)
        instant_freq = 200.0 + (3500.0 * (t / duration))
        val = 0.6 * math.sin(2.0 * math.pi * instant_freq * t)
        # Steady bass tone at 400 Hz
        val += 0.4 * math.sin(2.0 * math.pi * 400.0 * t)
        chirp_signal.append(val)

    # Compute STFT matrix
    t0 = time.perf_counter()
    matrix, times, freqs = spectrogram_matrix(
        chirp_signal,
        sample_rate=sample_rate,
        frame_size=256,
        hop_size=128,
        min_db=-70.0
    )
    t_stft = (time.perf_counter() - t0) * 1e3

    print(f"STFT Spectral Decomposition: {len(matrix)} time frames x {len(matrix[0])} frequency bins in {t_stft:.2f} ms")
    print(f"Time Range : 0.00s to {times[-1]:.2f}s")
    print(f"Freq Range : 0 Hz to {freqs[-1]:.0f} Hz")

    print("\nTerminal Braille Waterfall Spectrogram (Time -> X, Freq -> Y):")
    print(render_braille_waterfall(matrix, width_chars=60, height_chars=16, threshold=0.25, use_color=True, border=True))
    print()


def main() -> None:
    print_banner()
    demo_fft_spectral_analysis()
    demo_biquad_filtering()
    demo_sound_synthesis_and_wav()
    demo_braille_waterfall_spectrogram()
    print("AuraDSP audio signal processing laboratory demonstration completed successfully.")


if __name__ == "__main__":
    main()
