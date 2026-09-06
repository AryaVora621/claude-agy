"""SignalScope Digital Signal Processing (DSP) Engine.

First-principles audio DSP library implementing Cooley-Tukey Radix-2 FFT,
spectral windowing (Hanning, Hamming, Blackman), multi-waveform generation,
frequency and amplitude modulation, resonant biquad filtering, edge-triggering,
and 16-bit PCM WAV audio exporting using the Python standard library.
"""

from __future__ import annotations
import math
import cmath
import struct
import wave
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


def next_power_of_two(n: int) -> int:
    """Return smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fft(samples: List[complex]) -> List[complex]:
    """In-place Cooley-Tukey Radix-2 decimation-in-time Fast Fourier Transform."""
    n = len(samples)
    if n <= 1:
        return list(samples)

    # Pad to power of 2 if necessary
    p2 = next_power_of_two(n)
    if p2 != n:
        samples = samples + [0j] * (p2 - n)
        n = p2

    # Bit-reversal permutation
    a = list(samples)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Iterative Cooley-Tukey butterfly computation
    length = 2
    while length <= n:
        half = length // 2
        w_step = cmath.exp(-2j * math.pi / length)
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = w * a[i + k + half]
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= w_step
        length <<= 1

    return a


def apply_window(samples: List[float], window_type: str = "Hanning") -> List[float]:
    """Apply spectral windowing function to reduce spectral leakage."""
    n = len(samples)
    if n <= 1:
        return list(samples)

    out = [0.0] * n
    denom = n - 1

    if window_type == "Hanning":
        for i in range(n):
            w = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / denom))
            out[i] = samples[i] * w
    elif window_type == "Hamming":
        for i in range(n):
            w = 0.54 - 0.46 * math.cos(2.0 * math.pi * i / denom)
            out[i] = samples[i] * w
    elif window_type == "Blackman":
        for i in range(n):
            w = (0.42
                 - 0.5 * math.cos(2.0 * math.pi * i / denom)
                 + 0.08 * math.cos(4.0 * math.pi * i / denom))
            out[i] = samples[i] * w
    else:  # Rectangular
        out = list(samples)

    return out


@dataclass
class ChannelParams:
    """Configurable parameters for a synthesizer channel."""
    waveform: str = "Sine"  # "Sine", "Square", "Triangle", "Sawtooth", "Noise"
    frequency: float = 440.0  # Hz
    amplitude: float = 1.0    # 0.0 to 1.0
    phase: float = 0.0        # Degrees
    duty_cycle: float = 0.5   # For square wave PWM (0.05 to 0.95)
    offset_dc: float = 0.0    # Volts DC offset


@dataclass
class BiquadFilter:
    """Direct Form I 2nd-order IIR Biquad Filter."""
    filter_type: str = "Lowpass"  # "Lowpass", "Highpass", "Bandpass", "Bypass"
    cutoff: float = 2000.0        # Hz
    q_factor: float = 0.707       # Resonance Q factor
    sample_rate: float = 44100.0

    # Filter coefficients
    b0: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    # Delay state registers
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0

    def __post_init__(self):
        self.recompute()

    def recompute(self) -> None:
        """Calculate biquad coefficients based on Robert Bristow-Johnson Audio EQ Cookbook."""
        if self.filter_type == "Bypass" or self.cutoff <= 10.0:
            self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
            self.a1, self.a2 = 0.0, 0.0
            return

        nyquist = self.sample_rate * 0.5
        fc = max(20.0, min(nyquist * 0.95, self.cutoff))
        w0 = 2.0 * math.pi * fc / self.sample_rate
        alpha = math.sin(w0) / (2.0 * max(0.1, self.q_factor))
        cos_w0 = math.cos(w0)

        if self.filter_type == "Lowpass":
            b0 = (1.0 - cos_w0) * 0.5
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) * 0.5
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif self.filter_type == "Highpass":
            b0 = (1.0 + cos_w0) * 0.5
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) * 0.5
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        elif self.filter_type == "Bandpass":
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha
        else:
            self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
            self.a1, self.a2 = 0.0, 0.0
            return

        inv_a0 = 1.0 / a0
        self.b0 = b0 * inv_a0
        self.b1 = b1 * inv_a0
        self.b2 = b2 * inv_a0
        self.a1 = a1 * inv_a0
        self.a2 = a2 * inv_a0

    def process_sample(self, x: float) -> float:
        """Process a single audio sample through the filter."""
        if self.filter_type == "Bypass":
            return x
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

    def reset_state(self) -> None:
        """Clear filter memory registers."""
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0


class SignalEngine:
    """Multi-channel audio and DSP waveform synthesis engine."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.ch1 = ChannelParams(waveform="Sine", frequency=440.0, amplitude=1.0, phase=0.0)
        self.ch2 = ChannelParams(waveform="Sine", frequency=880.0, amplitude=0.5, phase=90.0)

        # Modulation configuration
        self.fm_enabled = False
        self.fm_depth = 100.0  # Hz frequency deviation
        self.am_enabled = False
        self.am_depth = 0.5    # Modulation index (0.0 to 1.0)

        # Noise generator state
        self._noise_state = 123456789

        # Filter
        self.filter = BiquadFilter(filter_type="Bypass", cutoff=3000.0, q_factor=1.0, sample_rate=sample_rate)

        # Trigger Engine
        self.trigger_mode = "Auto"   # "Auto", "Normal", "Single"
        self.trigger_slope = "Rising"  # "Rising", "Falling"
        self.trigger_level = 0.0       # Volts
        self.trigger_channel = 1       # 1 or 2

    def _pseudo_random(self) -> float:
        """Linear Congruential Generator for deterministic white noise without seed pollution."""
        self._noise_state = (1103515245 * self._noise_state + 12345) & 0x7FFFFFFF
        return (self._noise_state / 0x7FFFFFFF) * 2.0 - 1.0

    def evaluate_channel_at(self, ch: ChannelParams, t: float, mod_freq_offset: float = 0.0) -> float:
        """Compute the instantaneous amplitude of a channel at time t (seconds)."""
        freq = max(1.0, ch.frequency + mod_freq_offset)
        phase_rad = math.radians(ch.phase)
        theta = (2.0 * math.pi * freq * t + phase_rad) % (2.0 * math.pi)

        val = 0.0
        if ch.waveform == "Sine":
            val = math.sin(theta)
        elif ch.waveform == "Square":
            duty_rad = 2.0 * math.pi * max(0.05, min(0.95, ch.duty_cycle))
            val = 1.0 if theta < duty_rad else -1.0
        elif ch.waveform == "Triangle":
            # Normalized triangle from -1 to +1
            phase_norm = theta / (2.0 * math.pi)
            if phase_norm < 0.5:
                val = 4.0 * phase_norm - 1.0
            else:
                val = 3.0 - 4.0 * phase_norm
        elif ch.waveform == "Sawtooth":
            phase_norm = theta / (2.0 * math.pi)
            val = 2.0 * phase_norm - 1.0
        elif ch.waveform == "Noise":
            val = self._pseudo_random()

        return ch.offset_dc + ch.amplitude * val

    def generate_buffers(self, num_samples: int, t_start: float = 0.0) -> Tuple[List[float], List[float], List[float]]:
        """Generate time-series sample arrays for Channel 1, Channel 2, and Combined Mix."""
        buf1 = [0.0] * num_samples
        buf2 = [0.0] * num_samples
        mix = [0.0] * num_samples

        dt = 1.0 / self.sample_rate

        for i in range(num_samples):
            t = t_start + i * dt

            # Channel 2 base sample
            s2 = self.evaluate_channel_at(self.ch2, t)
            buf2[i] = s2

            # Channel 1 with optional FM from Channel 2
            mod_freq = (s2 * self.fm_depth) if self.fm_enabled else 0.0
            s1 = self.evaluate_channel_at(self.ch1, t, mod_freq_offset=mod_freq)

            # Optional AM modulation from Channel 2
            if self.am_enabled:
                am_factor = 1.0 + self.am_depth * s2
                s1 *= am_factor

            buf1[i] = s1

            # Sum and run through biquad filter
            combined = s1 + s2
            filtered = self.filter.process_sample(combined)
            mix[i] = filtered

        return buf1, buf2, mix

    def find_trigger_index(self, buffer: List[float], level: float, slope: str = "Rising") -> int:
        """Find the sample index where the waveform crosses the trigger threshold."""
        n = len(buffer)
        if n < 4:
            return 0

        # Scan for edge transition
        for i in range(1, n - 1):
            prev_s = buffer[i - 1]
            curr_s = buffer[i]

            if slope == "Rising":
                if prev_s <= level and curr_s > level:
                    return i
            else:  # Falling
                if prev_s >= level and curr_s < level:
                    return i

        return 0

    def compute_spectrum(self, samples: List[float], window_type: str = "Hanning") -> Tuple[List[float], List[float]]:
        """Compute frequency spectrum in decibels (dB) via Radix-2 FFT.

        Returns:
            frequencies (Hz), magnitudes (dB normalized to 0 dB full-scale).
        """
        n = len(samples)
        p2 = next_power_of_two(n)
        # Apply window
        windowed = apply_window(samples, window_type)
        padded = [complex(s, 0.0) for s in windowed]
        if p2 > n:
            padded += [0j] * (p2 - n)

        fft_res = fft(padded)
        half_n = p2 // 2

        freqs = [0.0] * half_n
        mags_db = [0.0] * half_n
        df = self.sample_rate / p2

        ref_power = 1e-12
        for i in range(half_n):
            freqs[i] = i * df
            mag = abs(fft_res[i]) / half_n
            # Convert to decibels with -100 dB noise floor clamp
            val_db = 20.0 * math.log10(max(1e-5, mag))
            mags_db[i] = max(-100.0, val_db)

        return freqs, mags_db

    def analyze_signal_metrics(self, samples: List[float]) -> Dict[str, float]:
        """Compute key waveform metrics: Peak-to-Peak (Vpp), RMS voltage, and estimated frequency."""
        if not samples:
            return {"vpp": 0.0, "vrms": 0.0, "vmax": 0.0, "vmin": 0.0, "freq": 0.0}

        vmax = max(samples)
        vmin = min(samples)
        vpp = vmax - vmin

        sum_sq = sum(s * s for s in samples)
        vrms = math.sqrt(sum_sq / len(samples))

        # Estimate dominant fundamental frequency via zero-crossing count
        crossings = 0
        mean_v = sum(samples) / len(samples)
        for i in range(1, len(samples)):
            if (samples[i - 1] - mean_v) * (samples[i] - mean_v) < 0:
                crossings += 1

        duration = len(samples) / self.sample_rate
        freq_est = (crossings * 0.5) / duration if duration > 0 else 0.0

        return {
            "vpp": vpp,
            "vrms": vrms,
            "vmax": vmax,
            "vmin": vmin,
            "freq": freq_est
        }

    def export_wav(self, file_path: str, duration_sec: float = 2.0) -> None:
        """Synthesize and export current audio mix to a 16-bit PCM mono WAV file."""
        total_samples = int(duration_sec * self.sample_rate)
        _, _, mix = self.generate_buffers(total_samples, t_start=0.0)

        # Normalize to prevent digital clipping
        peak = max(max(abs(s) for s in mix), 1e-6)
        scale = 32000.0 / peak if peak > 1.0 else 32000.0

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)        # Mono
            wf.setsampwidth(2)        # 16-bit
            wf.setframerate(self.sample_rate)

            # Pack 16-bit little-endian PCM integers
            raw_data = bytearray()
            for s in mix:
                clamped = max(-32767, min(32767, int(s * scale)))
                raw_data.extend(struct.pack("<h", clamped))

            wf.writeframes(raw_data)
