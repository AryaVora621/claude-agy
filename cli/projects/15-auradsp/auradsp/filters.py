"""
AuraDSP: Digital Biquad IIR & FIR Filter Design Engine.
Implements Robert Bristow-Johnson (RBJ) Audio EQ Cookbook second-order IIR biquad filters
(Low-Pass, High-Pass, Band-Pass, Notch, Peaking EQ, Shelving) using Direct Form II Transposed
topology, Z-plane pole-zero stability analysis, filter cascading, and windowed-sinc FIR filters.
"""

import cmath
import math
from typing import List, Tuple, Optional


class BiquadFilter:
    """
    Digital Second-Order IIR Biquad Filter implemented via Direct Form II Transposed topology.
    Transfer function:
      H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    """
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "s1", "s2", "sample_rate")

    def __init__(
        self,
        b0: float,
        b1: float,
        b2: float,
        a0: float,
        a1: float,
        a2: float,
        sample_rate: float = 44100.0
    ) -> None:
        if abs(a0) < 1e-15:
            raise ValueError("Leading denominator coefficient a0 cannot be zero")

        # Normalize by a0 so denominator is (1 + a1*z^-1 + a2*z^-2)
        inv_a0 = 1.0 / a0
        self.b0 = b0 * inv_a0
        self.b1 = b1 * inv_a0
        self.b2 = b2 * inv_a0
        self.a1 = a1 * inv_a0
        self.a2 = a2 * inv_a0
        self.sample_rate = sample_rate

        # Direct Form II Transposed state registers
        self.s1 = 0.0
        self.s2 = 0.0

    def reset(self) -> None:
        """Clears filter internal delay state."""
        self.s1 = 0.0
        self.s2 = 0.0

    def process_sample(self, x: float) -> float:
        """
        Processes a single input sample through Direct Form II Transposed state update.
        y[n]  = b0 * x[n] + s1[n-1]
        s1[n] = b1 * x[n] - a1 * y[n] + s2[n-1]
        s2[n] = b2 * x[n] - a2 * y[n]
        """
        y = self.b0 * x + self.s1
        self.s1 = self.b1 * x - self.a1 * y + self.s2
        self.s2 = self.b2 * x - self.a2 * y
        return y

    def process(self, signal: List[float]) -> List[float]:
        """Filters an entire sequence of audio samples."""
        output = [0.0] * len(signal)
        for i, sample in enumerate(signal):
            output[i] = self.process_sample(sample)
        return output

    def poles(self) -> Tuple[complex, complex]:
        """
        Computes the two complex poles (roots of z^2 + a1*z + a2 = 0).
        """
        disc = cmath.sqrt(self.a1 * self.a1 - 4.0 * self.a2)
        p1 = (-self.a1 + disc) / 2.0
        p2 = (-self.a1 - disc) / 2.0
        return p1, p2

    def zeros(self) -> Tuple[complex, complex]:
        """
        Computes the two complex zeros (roots of b0*z^2 + b1*z + b2 = 0).
        """
        if abs(self.b0) < 1e-15:
            if abs(self.b1) > 1e-15:
                return complex(-self.b2 / self.b1), 0j
            return 0j, 0j
        disc = cmath.sqrt(self.b1 * self.b1 - 4.0 * self.b0 * self.b2)
        z1 = (-self.b1 + disc) / (2.0 * self.b0)
        z2 = (-self.b1 - disc) / (2.0 * self.b0)
        return z1, z2

    def is_stable(self, margin: float = 1e-5) -> bool:
        """
        Returns True if all poles reside strictly inside the Z-plane unit circle (|p| < 1.0).
        Guarantees Bounded-Input Bounded-Output (BIBO) numerical stability.
        """
        p1, p2 = self.poles()
        return abs(p1) < (1.0 - margin) and abs(p2) < (1.0 - margin)

    def response_at(self, freq_hz: float) -> complex:
        """
        Evaluates complex frequency response H(e^(i*omega)) at a given frequency in Hz.
        """
        omega = 2.0 * math.pi * freq_hz / self.sample_rate
        z1 = cmath.exp(-1j * omega)
        z2 = cmath.exp(-2j * omega)
        num = self.b0 + self.b1 * z1 + self.b2 * z2
        den = 1.0 + self.a1 * z1 + self.a2 * z2
        return num / den if abs(den) > 1e-15 else complex(float("inf"))

    def frequency_response(
        self,
        frequencies: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluates magnitude (dB) and phase (degrees) across a list of frequencies.
        Returns: (magnitude_db, phase_degrees).
        """
        mags_db: List[float] = []
        phases_deg: List[float] = []

        for f in frequencies:
            h = self.response_at(f)
            mag = abs(h)
            db = 20.0 * math.log10(max(1e-12, mag))
            phase = math.degrees(cmath.phase(h))
            mags_db.append(db)
            phases_deg.append(phase)

        return mags_db, phases_deg

    # --- Robert Bristow-Johnson (RBJ) Filter Factories ---

    @classmethod
    def low_pass(
        cls,
        sample_rate: float,
        cutoff_hz: float,
        q: float = 1.0 / math.sqrt(2.0)
    ) -> "BiquadFilter":
        """Standard 2nd-order Low-Pass Filter (LPF). Default Q=0.7071 (Butterworth response)."""
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = (1.0 - cos_w0) * 0.5
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return cls(b0, b1, b2, a0, a1, a2, sample_rate)

    @classmethod
    def high_pass(
        cls,
        sample_rate: float,
        cutoff_hz: float,
        q: float = 1.0 / math.sqrt(2.0)
    ) -> "BiquadFilter":
        """Standard 2nd-order High-Pass Filter (HPF)."""
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = (1.0 + cos_w0) * 0.5
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return cls(b0, b1, b2, a0, a1, a2, sample_rate)

    @classmethod
    def band_pass(
        cls,
        sample_rate: float,
        center_hz: float,
        q: float = 1.0
    ) -> "BiquadFilter":
        """Band-Pass Filter with constant 0 dB peak gain."""
        w0 = 2.0 * math.pi * center_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return cls(b0, b1, b2, a0, a1, a2, sample_rate)

    @classmethod
    def notch(
        cls,
        sample_rate: float,
        center_hz: float,
        q: float = 10.0
    ) -> "BiquadFilter":
        """Notch (Band-Stop) filter rejecting narrow band around center frequency."""
        w0 = 2.0 * math.pi * center_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return cls(b0, b1, b2, a0, a1, a2, sample_rate)

    @classmethod
    def peaking_eq(
        cls,
        sample_rate: float,
        center_hz: float,
        gain_db: float,
        q: float = 1.0
    ) -> "BiquadFilter":
        """Parametric Peaking EQ filter boosting or cutting at center frequency."""
        w0 = 2.0 * math.pi * center_hz / sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        a = 10.0 ** (gain_db / 40.0)  # Amplitude factor
        alpha = sin_w0 / (2.0 * q)

        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a
        return cls(b0, b1, b2, a0, a1, a2, sample_rate)


class CascadeFilter:
    """
    Cascades multiple BiquadFilters in series for higher-order attenuation
    (e.g., 4th-order 24 dB/octave Butterworth filter).
    """

    def __init__(self, stages: List[BiquadFilter]) -> None:
        self.stages = stages

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def process_sample(self, x: float) -> float:
        val = x
        for stage in self.stages:
            val = stage.process_sample(val)
        return val

    def process(self, signal: List[float]) -> List[float]:
        current = signal
        for stage in self.stages:
            current = stage.process(current)
        return current

    @classmethod
    def butterworth_4th_order_low_pass(
        cls,
        sample_rate: float,
        cutoff_hz: float
    ) -> "CascadeFilter":
        """
        Creates a 4th-order (24 dB/octave) Butterworth Low-Pass Filter
        by cascading two biquad stages with Q factors: Q1 = 0.5412, Q2 = 1.3065.
        """
        q1 = 0.5411961
        q2 = 1.3065630
        s1 = BiquadFilter.low_pass(sample_rate, cutoff_hz, q=q1)
        s2 = BiquadFilter.low_pass(sample_rate, cutoff_hz, q=q2)
        return cls([s1, s2])


class FIRFilter:
    """
    Finite Impulse Response (FIR) Filter with windowed-sinc impulse response.
    """

    def __init__(self, kernel: List[float]) -> None:
        self.kernel = kernel
        self.m = len(kernel)
        self.buffer = [0.0] * self.m
        self.head = 0

    def reset(self) -> None:
        self.buffer = [0.0] * self.m
        self.head = 0

    def process_sample(self, x: float) -> float:
        self.buffer[self.head] = x
        out = 0.0
        buf_len = self.m
        idx = self.head
        for k in range(buf_len):
            out += self.kernel[k] * self.buffer[idx]
            idx = (idx - 1) if idx > 0 else (buf_len - 1)
        self.head = (self.head + 1) % buf_len
        return out

    def process(self, signal: List[float]) -> List[float]:
        return [self.process_sample(s) for s in signal]

    @classmethod
    def windowed_sinc_low_pass(
        cls,
        sample_rate: float,
        cutoff_hz: float,
        num_taps: int = 65
    ) -> "FIRFilter":
        """
        Designs a low-pass FIR filter using the windowed-sinc method with a Blackman window.
        num_taps should be odd for symmetric linear phase.
        """
        if num_taps % 2 == 0:
            num_taps += 1

        fc = cutoff_hz / sample_rate
        m = num_taps - 1
        half_m = m / 2.0
        kernel = [0.0] * num_taps

        for i in range(num_taps):
            if i == half_m:
                sinc = 2.0 * math.pi * fc
            else:
                n = i - half_m
                sinc = math.sin(2.0 * math.pi * fc * n) / n
            # Blackman window
            w = 0.42 - 0.5 * math.cos(2.0 * math.pi * i / m) + 0.08 * math.cos(4.0 * math.pi * i / m)
            kernel[i] = sinc * w

        # Normalize gain to 0 dB at DC (sum of kernel = 1.0)
        kernel_sum = sum(kernel)
        if abs(kernel_sum) > 1e-12:
            kernel = [k / kernel_sum for k in kernel]

        return cls(kernel)
