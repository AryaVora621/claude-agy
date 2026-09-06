"""Optical Excitation Sources for Electromagnetic Simulations.

Provides:
1. Broadband Gaussian pulses, continuous-wave (CW) sinusoids, modulated wavepackets, and Ricker wavelets
2. Soft, hard, and mode-profile spatial line injectors
3. Fundamental TE/TM dielectric waveguide transverse mode profiles
4. Analytical Fourier spectral envelopes for S-parameter normalization
"""

from __future__ import annotations
import math
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from .grid import C0, EPSILON_0, Grid2D


class SourceWaveform(Enum):
    """Temporal waveform excitation types."""
    GAUSSIAN_PULSE = auto()
    CONTINUOUS_WAVE = auto()
    MODULATED_GAUSSIAN = auto()
    RICKER_WAVELET = auto()


class InjectionMode(Enum):
    """Boundary injection mechanism into the FDTD grid."""
    SOFT = auto()   # Adds current density J_z, allows backward reflected waves to pass
    HARD = auto()   # Clamps Ez field directly to waveform value


class OpticalSource:
    """Configurable electromagnetic optical source for Yee grids."""

    def __init__(
        self,
        waveform: SourceWaveform = SourceWaveform.MODULATED_GAUSSIAN,
        wavelength: float = 1.55e-6,        # 1.55 um telecom wavelength
        bandwidth_fraction: float = 0.20,   # 20% fractional optical bandwidth
        amplitude: float = 1.0,             # Peak electric field (V/m)
        injection_mode: InjectionMode = InjectionMode.SOFT,
        tau: Optional[float] = None,        # Optional explicit pulse duration (s)
        t0: Optional[float] = None,         # Optional explicit pulse delay (s)
        t_ramp: Optional[float] = None,     # Optional explicit CW ramp time (s)
    ) -> None:
        """Initialize optical source.

        Args:
            waveform: Temporal profile (Gaussian, CW, Modulated, Ricker).
            wavelength: Central optical wavelength in meters (default: 1.55 um).
            bandwidth_fraction: Fractional bandwidth Delta f / f0 for pulses.
            amplitude: Peak amplitude of electric field (V/m).
            injection_mode: Soft (additive) or Hard (clamped) injection.
            tau: Optional override for pulse standard deviation.
            t0: Optional override for peak pulse delay.
            t_ramp: Optional override for CW ramp time.
        """
        self.waveform = waveform
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.injection_mode = injection_mode

        # Central optical frequency and period
        self.frequency = C0 / wavelength
        self.period = 1.0 / self.frequency

        # Pulse width and time delay
        # Bandwidth Delta f = bandwidth_fraction * f0
        delta_f = max(1e9, bandwidth_fraction * self.frequency)
        self.tau = tau if tau is not None else (1.0 / (math.pi * delta_f))
        self.t0 = t0 if t0 is not None else (4.5 * self.tau)

        # CW ramp-up time to suppress startup transients
        self.t_ramp = t_ramp if t_ramp is not None else (3.0 * self.period)

    def evaluate(self, t: float) -> float:
        """Evaluate temporal waveform amplitude at physical time t (seconds)."""
        if self.waveform == SourceWaveform.GAUSSIAN_PULSE:
            arg = (t - self.t0) / self.tau
            return self.amplitude * math.exp(-0.5 * arg * arg)

        elif self.waveform == SourceWaveform.CONTINUOUS_WAVE:
            ramp = 1.0
            if t < self.t_ramp and self.t_ramp > 0:
                ramp = 0.5 * (1.0 - math.cos(math.pi * t / self.t_ramp))
            return self.amplitude * ramp * math.sin(2.0 * math.pi * self.frequency * t)

        elif self.waveform == SourceWaveform.MODULATED_GAUSSIAN:
            arg = (t - self.t0) / self.tau
            envelope = math.exp(-0.5 * arg * arg)
            carrier = math.sin(2.0 * math.pi * self.frequency * (t - self.t0))
            return self.amplitude * envelope * carrier

        elif self.waveform == SourceWaveform.RICKER_WAVELET:
            arg = math.pi * self.frequency * (t - self.t0)
            arg2 = arg * arg
            return self.amplitude * (1.0 - 2.0 * arg2) * math.exp(-arg2)

        return 0.0

    def spectral_amplitude(self, freq: float) -> float:
        """Analytical Fourier transform magnitude |S(f)| for pulse normalization."""
        if self.waveform in (SourceWaveform.GAUSSIAN_PULSE, SourceWaveform.MODULATED_GAUSSIAN):
            # Fourier transform of Gaussian: sqrt(2*pi)*tau * exp(-2*pi^2*tau^2*(f - f0)^2)
            f_center = self.frequency if self.waveform == SourceWaveform.MODULATED_GAUSSIAN else 0.0
            diff = freq - f_center
            factor = 2.0 * (math.pi * self.tau * diff) ** 2
            return self.amplitude * self.tau * math.sqrt(2.0 * math.pi) * math.exp(-0.5 * factor)
        elif self.waveform == SourceWaveform.RICKER_WAVELET:
            f_ratio = freq / self.frequency
            return self.amplitude * (2.0 / (math.sqrt(math.pi) * self.frequency)) * (f_ratio ** 2) * math.exp(-(f_ratio ** 2))
        return 1.0


class PointSource:
    """Single-node spatial optical source injector."""

    def __init__(self, source: OpticalSource, x: int, y: int) -> None:
        self.source = source
        self.x = x
        self.y = y

    def inject(self, grid: Grid2D, t: float) -> None:
        """Inject field value into grid at time t."""
        if not grid.in_bounds(self.x, self.y):
            return
        idx = grid.idx(self.x, self.y)
        val = self.source.evaluate(t)

        if self.source.injection_mode == InjectionMode.HARD:
            grid.ez[idx] = val
        else:
            # Soft injection: adds current density contribution to Ez
            grid.ez[idx] += val * grid.cb[idx]


class WaveguideModeSource:
    """Transverse line source with cosine fundamental dielectric waveguide profile."""

    def __init__(
        self,
        source: OpticalSource,
        x: int,
        y_start: int,
        y_end: int,
        is_vertical: bool = True,
    ) -> None:
        """Initialize transverse line source along a waveguide cross-section.

        Args:
            source: Underlying OpticalSource temporal generator.
            x: Spatial coordinate (x-line if vertical, y-line if horizontal).
            y_start: Beginning of waveguide cross-section.
            y_end: End of waveguide cross-section.
            is_vertical: If True, source is a vertical line at column x.
        """
        self.source = source
        self.coord = x
        self.start = min(y_start, y_end)
        self.end = max(y_start, y_end)
        self.is_vertical = is_vertical

        # Precompute spatial mode profile (cosine peak at center, zero at edges)
        width = max(1, self.end - self.start)
        self.weights: List[float] = []
        for i in range(self.start, self.end + 1):
            pos = (i - self.start) / width
            # Cosine fundamental transverse profile
            profile = math.cos(math.pi * (pos - 0.5))
            self.weights.append(max(0.0, profile))

    def inject(self, grid: Grid2D, t: float) -> None:
        """Inject guided mode profile into the 2D Yee grid."""
        val = self.source.evaluate(t)
        if abs(val) < 1e-15:
            return

        for i, weight in enumerate(self.weights):
            y = self.start + i
            if self.is_vertical:
                gx, gy = self.coord, y
            else:
                gx, gy = y, self.coord

            if not grid.in_bounds(gx, gy):
                continue

            idx = grid.idx(gx, gy)
            src_val = val * weight

            if self.source.injection_mode == InjectionMode.HARD:
                grid.ez[idx] = src_val
            else:
                grid.ez[idx] += src_val * grid.cb[idx]
