# AuraDSP: Audio Digital Signal Processing, Spectral Analysis & Synthesis Engine

[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib%20only)-success)]()

AuraDSP is a zero-dependency, portfolio-grade digital signal processing (DSP) and sound synthesis engine implemented entirely from first principles in the pure Python standard library. It provides high-performance Cooley-Tukey Radix-2 Fast Fourier Transforms (FFT), Robert Bristow-Johnson digital biquad IIR filters with Z-plane stability analysis, polyphonic sound synthesis with 4-stage exponential ADSR envelopes, a binary 16-bit RIFF WAV codec, and a high-resolution sub-pixel Unicode Braille spectrogram waterfall visualizer with 24-bit TrueColor ANSI output.

---

## Key Features & Theoretical Foundations

### 1. Cooley-Tukey Radix-2 Decimation-In-Time (DIT) FFT
* **Iterative In-Place Butterfly Network**: Replaces recursive divide-and-conquer call stacks with an iterative in-place butterfly computation using precomputed twiddle factors ($W_N^k = e^{-i 2\pi k / N}$).
* **Bit-Reversal Permutation**: Initial permutation achieved via bitwise integer reversal in $O(N)$ time.
* **Exact Inverse FFT (IFFT)**: Reconstructs time-domain signals from complex frequency spectra with zero mathematical drift ($\le 2.4 \times 10^{-14}$ error) via complex conjugate symmetry:
  $$\text{IFFT}(X) = \frac{1}{N} \left( \text{FFT}(X^*) \right)^*$$
* **Parseval's Energy Conservation**: Verified conservation of signal energy between time and frequency domains: $\sum |x[n]|^2 = \frac{1}{N} \sum |X[k]|^2$.
* **Spectral Windowing**: Hann, Hamming, and Blackman windowing functions for sidelobe suppression and spectral leakage elimination.

### 2. Robert Bristow-Johnson (RBJ) Digital Biquad IIR Filters
* **Direct Form II Transposed Topology**: Numerically robust second-order difference equation minimizing round-off noise and internal state register saturation:
  $$y[n] = b_0 x[n] + s_1[n-1]$$
  $$s_1[n] = b_1 x[n] - a_1 y[n] + s_2[n-1]$$
  $$s_2[n] = b_2 x[n] - a_2 y[n]$$
* **Filter Types**: Low-Pass, High-Pass, Band-Pass, Notch (band-stop), and Peaking EQ with customizable center frequency ($f_0$), quality factor ($Q$), and gain ($dB$).
* **Z-Plane Pole-Zero Stability**: Analytical quadratic solver computing complex transfer function poles and zeros, validating bounded-input bounded-output (BIBO) stability: $|p_{1,2}| < 1.0$.
* **4th-Order Butterworth Cascades**: Two cascaded biquad stages with Butterworth Q factors ($Q_1 = 0.5412$, $Q_2 = 1.3066$) producing a maximally flat 24 dB/octave attenuation curve.
* **Windowed-Sinc FIR Filtering**: Linear-phase Finite Impulse Response filter utilizing Blackman-windowed sinc kernels.

### 3. Polyphonic Sound Synthesis & ADSR Envelopes
* **Continuous-Phase Oscillators**: Phase-continuous generation for Sine, Sawtooth, Square (with variable Pulse-Width Modulation), Triangle, and Gaussian/Uniform White Noise waveforms.
* **4-Stage Exponential ADSR Generator**: Non-linear natural attack, exponential decay, configurable sustain level, and exponential release envelope.
* **Low-Frequency Oscillator (LFO)**: Pitch vibrato and amplitude tremolo modulation.
* **Polyphonic Voicing & Limiter**: Multi-oscillator voice mixing with hyperbolic tangent ($tanh$) soft-saturation limiting to prevent digital clipping.

### 4. Binary 16-Bit Linear PCM RIFF WAV Codec
* **Pure Python Bit-Level Serialization**: Custom binary packing and unpacking using standard library `struct`.
* **Standard RIFF Header**: Encodes 44-byte canonical WAV headers (`RIFF`, `WAVE`, `fmt ` sub-chunk with audio format 1, 16-bit depth, and `data` chunk).
* **Quantization & Dithering**: Symmetric $[-32767, 32767]$ integer mapping with full preservation of sample rate and channel geometry.

### 5. Short-Time Fourier Transform (STFT) & Braille Waterfall Spectrogram
* **Sliding-Window Decomposition**: Computes overlapping windowed FFT frames with configurable frame length and hop size.
* **Spectral Metrics**:
  * **Spectral Centroid**: Sound brightness / perceptual center of mass in Hz: $\frac{\sum f_k |X[k]|}{\sum |X[k]|}$.
  * **Spectral Flatness (Wiener Entropy)**: Ratio of geometric mean to arithmetic mean: $\frac{\exp(\frac{1}{N}\sum \ln |X[k]|^2)}{\frac{1}{N}\sum |X[k]|^2}$. Distinguishes pure tones ($\approx 0.0$) from white noise ($\approx 0.8$).
* **Sub-Pixel Braille Waterfall Visualizer**: Uses Unicode Braille patterns (`U+2800..U+28FF`) providing 2x4 sub-pixel resolution (120x64 pixels in a 60x16 character grid) with 24-bit TrueColor thermal heat color ramps (Navy -> Violet -> Magenta -> Coral -> Amber -> White).

---

## Performance Benchmarks

Executed on Apple Silicon (Python 3.13 standard library, single-threaded):

### 1. Cooley-Tukey Radix-2 FFT vs Naive DFT ($O(N \log N)$ vs $O(N^2)$)
| Transform Size ($N$) | Naive DFT (ms) | Radix-2 FFT (ms) | Speedup | Operations ($N \log_2 N$) |
|:-------------------:|:--------------:|:----------------:|:-------:|:-------------------------:|
| 64 | 0.383 ms | 0.038 ms | **10.1x** | 384 |
| 128 | 1.521 ms | 0.081 ms | **18.7x** | 896 |
| 256 | 6.153 ms | 0.183 ms | **33.7x** | 2,048 |
| 512 | 25.287 ms | 0.442 ms | **57.2x** | 4,608 |
| 1024 | 107.770 ms | 1.012 ms | **106.5x** | 10,240 |
| 2048 | -- | 2.260 ms | -- | 906,067 samples/s |
| 4096 | -- | 4.960 ms | -- | 824,976 samples/s |

### 2. Filter & Synthesis Throughput
| Component | Throughput (samples/sec) | Real-Time Factor (44.1 kHz) |
|:---|:---:|:---:|
| 2nd-Order Biquad IIR Filter | **9,947,125 samples/s** | **225.6x** |
| 4th-Order Cascade Butterworth | **4,995,078 samples/s** | **113.3x** |
| Sine Wave Oscillator | **3,045,748 samples/s** | **69.1x** |
| Sawtooth Wave Oscillator | **3,258,366 samples/s** | **73.9x** |
| 4-Voice Polyphonic Chord | **732,758 samples/s** | **16.6x** |
| 16-Bit WAV Binary Encoder | **11,542,423 samples/s** (22.0 MB/s) | **261.7x** |
| 16-Bit WAV Binary Decoder | **31,006,262 samples/s** (59.2 MB/s) | **703.1x** |

---

## Quick Start Guide

### 1. Fast Fourier Transform & Spectral Analysis
```python
import math
from auradsp.fft import fft, ifft, magnitude_spectrum, power_spectrum_db, fft_frequencies

sample_rate = 44100.0
n = 1024

# Dual-tone signal: 440 Hz (A4) + 880 Hz (A5)
dt = 1.0 / sample_rate
signal = [
    0.6 * math.sin(2.0 * math.pi * 440.0 * i * dt) +
    0.4 * math.sin(2.0 * math.pi * 880.0 * i * dt)
    for i in range(n)
]

# Forward FFT
spectrum = fft(signal)

# Inverse FFT reconstruction
reconstructed = ifft(spectrum)
max_error = max(abs(s - r.real) for s, r in zip(signal, reconstructed))
print(f"Max reconstruction error: {max_error:.2e}")  # < 1e-14
```

### 2. Audio Filtering with RBJ Biquad
```python
from auradsp.filters import BiquadFilter

# Design 1.2 kHz Low-Pass Butterworth Filter
lpf = BiquadFilter.low_pass(sample_rate=44100.0, cutoff_hz=1200.0, q=0.7071)
print(f"Filter stable: {lpf.is_stable()}")

# Process audio buffer
filtered_signal = lpf.process(signal)
```

### 3. Sound Synthesis & WAV Export
```python
from auradsp.synth import PolyphonicSynth, ADSREnvelope, midi_to_freq
from auradsp.wav import write_wav

synth = PolyphonicSynth(sample_rate=44100.0)
env = ADSREnvelope(attack_time=0.01, decay_time=0.1, sustain_level=0.5, release_time=0.2)

# Synthesize C Major Chord: C4 (60), E4 (64), G4 (67)
frequencies = [midi_to_freq(m) for m in [60, 64, 67]]
chord_audio = synth.synthesize_chord(frequencies, duration=1.0, waveform="saw", envelope=env)

# Export to 16-bit linear PCM WAV
write_wav("c_major.wav", chord_audio, sample_rate=44100)
```

### 4. Interactive Laboratory & Spectrogram Waterfall
```bash
python3 examples/audio_lab.py
```

---

## Directory Structure

```
projects/15-auradsp/
├── auradsp/
│   ├── __init__.py           # Package exports
│   ├── fft.py               # Cooley-Tukey Radix-2 FFT/IFFT, twiddle factors, spectral windows
│   ├── filters.py           # RBJ Biquad IIR, Butterworth cascade, FIR windowed-sinc
│   ├── synth.py             # Oscillators, ADSR envelope, LFO vibrato, polyphonic synth
│   ├── wav.py               # Pure Python binary 16-bit linear PCM RIFF WAV codec
│   ├── spectrogram.py       # STFT sliding window, dB spectrogram, centroid, flatness
│   └── visualizer.py        # Sub-pixel Unicode Braille waterfall & ASCII spectrum visualizer
├── tests/
│   ├── __init__.py
│   ├── test_fft.py          # 8 tests: FFT accuracy, IFFT inverse, Parseval conservation
│   ├── test_filters.py      # 7 tests: LPF, HPF, BPF, Notch, Peaking EQ, stability, FIR
│   ├── test_synth.py        # 4 tests: MIDI conversion, oscillators, ADSR, polyphony
│   ├── test_wav.py          # 3 tests: Mono/stereo roundtrip, file I/O
│   └── test_spectrogram.py  # 6 tests: STFT dimensions, centroid, flatness, Braille render
├── benchmarks/
│   └── bench_dsp.py         # Full benchmarking suite (FFT, filters, synth, STFT, WAV)
├── examples/
│   └── audio_lab.py         # Interactive laboratory demonstration
├── output/
│   └── arpeggio_cmaj7.wav   # Generated 16-bit WAV audio sample
├── PLAN.md                  # Comprehensive architectural specification
├── CHECKPOINT_LAST.md       # Operational milestone checkpoint
└── README.md                # Showcase documentation
```

---

## Verification & Testing

Run the full unit test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Run the performance benchmarks:
```bash
python3 benchmarks/bench_dsp.py
```
