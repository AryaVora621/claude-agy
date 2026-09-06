# Architectural Design Plan: AuraDSP (Audio Digital Signal Processing & Spectral Engine)

## Executive Summary
AuraDSP is a zero-dependency, portfolio-grade Digital Signal Processing (DSP), spectral analysis, and sound synthesis engine implemented entirely in the pure Python 3 standard library.

The system provides industrial-grade digital audio processing:
1. Fast Fourier Transform (Cooley-Tukey Radix-2 FFT and IFFT in O(N log N)).
2. Spectral windowing functions (Hann, Hamming, Blackman, Flat-Top).
3. Robert Bristow-Johnson Audio EQ Cookbook digital biquad IIR filter design (Low-Pass, High-Pass, Band-Pass, Notch, Peaking EQ, Shelving) with Direct Form II transposed realization and complex Z-plane stability analysis.
4. Polyphonic synthesizer engine with continuous-phase oscillators, exponential 4-stage ADSR envelope generators, and LFO modulation.
5. Uncompressed 16-bit linear PCM RIFF WAV audio file codec for generating playable audio files.
6. Short-Time Fourier Transform (STFT) engine and high-resolution Unicode Braille waterfall spectrogram visualizer in 24-bit ANSI TrueColor.

---

## Technical Specifications & Mathematical Formulations

### 1. Cooley-Tukey Radix-2 Fast Fourier Transform (`auradsp/fft.py`)
- **Forward DFT**:
  $$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N} \quad (0 \le k < N)$$
- **Decimation-In-Time (DIT) Radix-2 Decomposition**:
  For $N = 2^m$, split into even and odd index sub-sequences:
  $$X[k] = E[k] + W_N^k O[k]$$
  $$X[k + N/2] = E[k] - W_N^k O[k]$$
  where $W_N^k = e^{-i 2\pi k / N} = \cos(2\pi k / N) - i \sin(2\pi k / N)$ (twiddle factors).
- **In-Place Bit-Reversal Permutation**:
  Replaces recursive call overhead with an iterative butterfly network for peak throughput.
- **Inverse FFT (IFFT)**:
  Exact signal reconstruction using complex conjugate symmetry:
  $$x[n] = \frac{1}{N} \left( \text{FFT}(X^*[k]) \right)^*$$
- **Window Functions**:
  - Hann: $w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)$
  - Hamming: $w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)$
  - Blackman: $w[n] = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right) + 0.08 \cos\left(\frac{4\pi n}{N-1}\right)$

---

### 2. Digital Biquad Filter Design (`auradsp/filters.py`)
- **Second-Order IIR Difference Equation & Transfer Function**:
  $$H(z) = \frac{Y(z)}{X(z)} = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$
- **Direct Form II Transposed Topology**:
  Minimizes numerical roundoff noise in floating-point arithmetic:
  $$y[n] = \frac{b_0}{a_0} x[n] + s_1[n-1]$$
  $$s_1[n] = \frac{b_1}{a_0} x[n] - \frac{a_1}{a_0} y[n] + s_2[n-1]$$
  $$s_2[n] = \frac{b_2}{a_0} x[n] - \frac{a_2}{a_0} y[n]$$
- **RBJ Audio EQ Cookbook Formulations**:
  Given normalized cutoff frequency $\omega_0 = 2\pi f_0 / f_s$, resonance $Q$ or bandwidth, and gain $A = 10^{\text{gain\_db} / 40}$:
  - **Low-Pass Filter (LPF)**:
    $b_0 = (1 - \cos\omega_0)/2, \quad b_1 = 1 - \cos\omega_0, \quad b_2 = (1 - \cos\omega_0)/2$
    $a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha \quad (\alpha = \sin\omega_0 / (2Q))$
  - **High-Pass Filter (HPF)**:
    $b_0 = (1 + \cos\omega_0)/2, \quad b_1 = -(1 + \cos\omega_0), \quad b_2 = (1 + \cos\omega_0)/2$
    $a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha$
  - **Band-Pass Filter (BPF)**:
    $b_0 = \alpha, \quad b_1 = 0, \quad b_2 = -\alpha$
  - **Notch Filter (Band-Reject)**:
    $b_0 = 1, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1$
  - **Peaking EQ**:
    $b_0 = 1 + \alpha A, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1 - \alpha A$
    $a_0 = 1 + \alpha / A, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha / A$
- **Z-Plane Pole-Zero Stability**:
  Roots of denominator polynomial $a_0 z^2 + a_1 z + a_2 = 0$ must lie strictly within the unit circle $|p| < 1$.
- **Complex Frequency Response Evaluation**:
  Evaluates magnitude $|H(e^{i \omega})|$ and phase $\theta(\omega)$ across 0 to Nyquist frequency.

---

### 3. Sound Synthesis & RIFF WAV Audio Codec (`auradsp/synth.py`, `auradsp/wav.py`)
- **Continuous-Phase Oscillators**:
  - Sine: $\sin(2\pi \phi)$
  - Sawtooth: $2(\phi - \lfloor \phi + 0.5 \rfloor)$
  - Square: $\text{sgn}(\sin(2\pi \phi) - \text{duty})$ (variable pulse-width modulation)
  - Triangle: $2 |\text{saw}(\phi)| - 1$
  - White Noise: Uniform random sampling $[-1.0, 1.0]$
- **Exponential 4-Stage ADSR Envelope Generator**:
  - Attack: Linear or exponential rise to 1.0 over duration $t_A$
  - Decay: Exponential drop toward sustain level $S$ over duration $t_D$
  - Sustain: Constant level $S \in [0.0, 1.0]$ held during key-down
  - Release: Exponential decay from sustain to 0.0 over duration $t_R$
- **Low-Frequency Oscillator (LFO)**:
  Modulates frequency (vibrato), amplitude (tremolo), or filter cutoff (wah-wah).
- **Audio File Codec (`auradsp/wav.py`)**:
  Binary encoding and decoding of uncompressed RIFF/WAVE containers:
  - 44.1 kHz, 16-bit signed integer little-endian PCM (`<h`), mono or stereo.
  - Generates verifiable, playable `.wav` files directly to disk.

---

### 4. Short-Time Fourier Transform & Braille Spectrogram (`auradsp/spectrogram.py`, `auradsp/visualizer.py`)
- **STFT Spectral Decomposition**:
  Divides continuous audio signals into overlapping frames of size $N$ (e.g. 512, 1024) stepped by hop size $H$ (e.g. 256).
  Applies window function $w[n]$ and runs FFT per frame, generating a 2D time-frequency matrix $S[t, k]$.
- **Logarithmic Frequency & Decibel Scaling**:
  Magnitude to decibels: $dB = 20 \log_{10}(\max(|X[k]|, 10^{-6}))$.
  Bin compression onto logarithmic or Mel-like frequency scale matching human auditory perception.
- **Unicode Braille Waterfall Visualizer**:
  Uses 2x4 sub-pixel Braille characters (`U+2800..U+28FF`) with 24-bit TrueColor ANSI color ramps (dark purple -> cyan -> yellow -> white) to render spectrograms directly in standard terminal output.

---

## Directory & File Structure

```text
projects/15-auradsp/
├── auradsp/
│   ├── __init__.py           # Package exports
│   ├── fft.py                # Cooley-Tukey Radix-2 FFT, IFFT, and Spectral Windows
│   ├── filters.py            # Digital Biquad IIR filters (RBJ EQ Cookbook) & Z-plane analysis
│   ├── synth.py              # Polyphonic Synthesizer, Oscillators, ADSR, and LFO
│   ├── wav.py                # Binary 16-bit PCM RIFF WAV audio file codec
│   ├── spectrogram.py        # Short-Time Fourier Transform (STFT) & spectral features
│   └── visualizer.py         # Terminal Braille waterfall spectrogram & ANSI heatmaps
├── tests/
│   ├── test_fft.py           # FFT / IFFT roundtrip, Parseval's theorem, spectral windows
│   ├── test_filters.py       # Biquad LPF/HPF/BPF/Notch frequency response, stability
│   ├── test_synth.py         # Oscillator waveforms, ADSR phases, polyphonic mixing
│   └── test_wav.py           # RIFF WAV binary header, 16-bit PCM read/write roundtrip
├── examples/
│   └── audio_lab.py          # Interactive audio synthesizer and terminal spectrogram lab
├── benchmarks/
│   └── bench_dsp.py          # FFT throughput, biquad filtering speed, and audio generation
├── PLAN.md                   # Architectural design specification
├── TASK_QUEUE.md             # Task execution status
├── CHECKPOINT_LAST.md        # Work unit tracking
└── README.md                 # System documentation & benchmarks
```

---

## Verification & Done Criteria
1. FFT vs DFT exact numerical equivalence within $10^{-10}$ tolerance.
2. IFFT(FFT(signal)) reconstructs original signal with zero distortion.
3. Parseval's energy conservation theorem holds in frequency domain: $\sum |x[n]|^2 = \frac{1}{N} \sum |X[k]|^2$.
4. Digital biquad filters match theoretical cutoff attenuation (e.g. -3dB at cutoff frequency for Butterworth LPF/HPF).
5. Filter poles reside inside the unit circle $|z| < 1$ for guaranteed BIBO stability.
6. WAV codec writes valid 44.1kHz 16-bit RIFF audio that can be decoded and verified bit-for-bit.
7. STFT correctly identifies tone frequencies (e.g. 440 Hz concert A).
8. Interactive laboratory renders clear Braille spectrogram waterfalls and exports playable WAV files.
9. All unit tests pass 100% and integrate into master `showcase.py`.
