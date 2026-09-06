"""Electromagnetic Field and Discrete Fourier Transform (DFT) Flux Monitors.

Provides:
1. PointTimeMonitor for recording time-domain waveforms at specific nodes
2. LineDFTMonitor for on-the-fly complex phasor accumulation and Poynting flux integration
3. SParameterAnalyzer for calculating transmission (S21), reflection (S11), and insertion loss
4. Resonator Quality Factor (Q) and Free Spectral Range (FSR) spectral characterization
"""

from __future__ import annotations
import cmath
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .grid import C0, Grid2D


@dataclass
class ResonanceMetrics:
    """Characterization metrics of an optical resonance peak or dip."""
    resonance_wavelength: float    # Central resonance wavelength (m)
    resonance_frequency: float     # Central resonance frequency (Hz)
    fwhm_bandwidth_hz: float       # Full Width at Half Maximum bandwidth (Hz)
    quality_factor: float          # Optical cavity Q-factor (Q = f0 / delta_f)
    extinction_ratio_db: float     # Peak-to-notch ratio in decibels


class PointTimeMonitor:
    """Time-domain electromagnetic field recorder at a fixed spatial coordinate."""

    def __init__(self, name: str, x: int, y: int) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.time_history: List[float] = []
        self.ez_history: List[float] = []
        self.hx_history: List[float] = []
        self.hy_history: List[float] = []

    def sample(self, grid: Grid2D, step: int, t: float) -> None:
        """Sample fields at the target coordinate."""
        if not grid.in_bounds(self.x, self.y):
            return
        idx = grid.idx(self.x, self.y)
        self.time_history.append(t)
        self.ez_history.append(grid.ez[idx])
        self.hx_history.append(grid.hx[idx])
        self.hy_history.append(grid.hy[idx])

    @property
    def peak_ez(self) -> float:
        """Peak absolute electric field observed."""
        if not self.ez_history:
            return 0.0
        return max(abs(v) for v in self.ez_history)

    @property
    def rms_ez(self) -> float:
        """Root-mean-square electric field amplitude."""
        if not self.ez_history:
            return 0.0
        return math.sqrt(sum(v * v for v in self.ez_history) / len(self.ez_history))


class LineDFTMonitor:
    """On-the-fly Discrete Fourier Transform (DFT) Poynting flux aperture monitor.

    Accumulates complex phasors E_z(f) and H(f) at each time step without storing
    massive time-domain histories, then integrates the time-averaged Poynting vector:
    P(f) = 0.5 * Re(integral E_z(f) * H*(f) dl)
    """

    def __init__(
        self,
        name: str,
        coord: int,
        start: int,
        end: int,
        frequencies: Sequence[float],
        is_vertical: bool = True,
    ) -> None:
        """Initialize line DFT flux monitor.

        Args:
            name: Identifier for the monitor aperture.
            coord: Fixed coordinate (x-line if vertical, y-line if horizontal).
            start: Start coordinate along cross-section.
            end: End coordinate along cross-section.
            frequencies: List of optical frequencies to monitor (Hz).
            is_vertical: If True, aperture is a vertical column (energy flows horizontally along x).
        """
        self.name = name
        self.coord = coord
        self.start = min(start, end)
        self.end = max(start, end)
        self.frequencies = list(frequencies)
        self.is_vertical = is_vertical

        self.num_freqs = len(self.frequencies)
        self.num_pts = self.end - self.start + 1

        # Complex phasor accumulation buffers: [freq_idx][point_idx]
        self.dft_ez: List[List[complex]] = [
            [complex(0.0, 0.0)] * self.num_pts for _ in range(self.num_freqs)
        ]
        self.dft_h: List[List[complex]] = [
            [complex(0.0, 0.0)] * self.num_pts for _ in range(self.num_freqs)
        ]

    def sample(self, grid: Grid2D, step: int, t: float) -> None:
        """Accumulate on-the-fly DFT Fourier coefficients at time t."""
        dt = grid.dt
        two_pi = 2.0 * math.pi

        for f_idx, freq in enumerate(self.frequencies):
            phase = two_pi * freq * t
            # Kernel e^(-i * 2 * pi * f * t) * dt
            kernel = cmath.exp(complex(0.0, -phase)) * dt

            for p_idx in range(self.num_pts):
                pos = self.start + p_idx
                if self.is_vertical:
                    gx, gy = self.coord, pos
                    if not grid.in_bounds(gx, gy):
                        continue
                    idx = grid.idx(gx, gy)
                    # Transverse magnetic field for x-directed flux is Hy
                    h_val = grid.hy[idx]
                else:
                    gx, gy = pos, self.coord
                    if not grid.in_bounds(gx, gy):
                        continue
                    idx = grid.idx(gx, gy)
                    # Transverse magnetic field for y-directed flux is -Hx
                    h_val = -grid.hx[idx]

                ez_val = grid.ez[idx]
                self.dft_ez[f_idx][p_idx] += ez_val * kernel
                self.dft_h[f_idx][p_idx] += h_val * kernel

    def compute_flux(self, grid: Grid2D) -> List[float]:
        """Compute time-averaged Poynting vector flux P(f) for all monitored frequencies.

        P(f) = 0.5 * Re( integral E_z(f) * H_transverse*(f) dl )

        Returns:
            List of flux values in Watts/meter corresponding to self.frequencies.
        """
        dl = grid.dy if self.is_vertical else grid.dx
        flux_spectrum: List[float] = []

        for f_idx in range(self.num_freqs):
            total_p = 0.0
            for p_idx in range(self.num_pts):
                ez = self.dft_ez[f_idx][p_idx]
                h_conj = self.dft_h[f_idx][p_idx].conjugate()
                # Poynting vector: 0.5 * Re(E x H*)
                s_point = 0.5 * (ez * h_conj).real
                total_p += s_point * dl
            flux_spectrum.append(max(0.0, total_p))

        return flux_spectrum


class SParameterAnalyzer:
    """Calculates S-parameters, insertion loss, and resonance Q-factors from DFT monitors."""

    @staticmethod
    def compute_s_parameters(
        in_flux: Sequence[float],
        out_flux: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """Compute power transmission ratio S21 and insertion loss in dB.

        Args:
            in_flux: Incident power spectrum at input port.
            out_flux: Transmitted power spectrum at output port.

        Returns:
            Tuple of (s21_linear, insertion_loss_db).
        """
        s21: List[float] = []
        il_db: List[float] = []

        for p_in, p_out in zip(in_flux, out_flux):
            if p_in > 1e-20:
                ratio = min(1.0, max(0.0, p_out / p_in))
                loss = -10.0 * math.log10(max(1e-12, ratio))
            else:
                ratio = 0.0
                loss = 120.0
            s21.append(ratio)
            il_db.append(loss)

        return s21, il_db

    @staticmethod
    def extract_resonance(
        frequencies: Sequence[float],
        transmission: Sequence[float],
        is_drop_port: bool = False,
    ) -> Optional[ResonanceMetrics]:
        """Extract resonance wavelength, 3dB FWHM bandwidth, Q-factor, and extinction ratio.

        Args:
            frequencies: Monitored frequency points (Hz).
            transmission: Measured transmission spectrum (linear ratio).
            is_drop_port: If True, resonance is a transmission peak; if False, a notch/dip.

        Returns:
            ResonanceMetrics object if resonance found, else None.
        """
        if len(frequencies) < 5 or len(transmission) != len(frequencies):
            return None

        # Find resonance extremum (min for through port, max for drop port)
        if is_drop_port:
            target_idx = max(range(len(transmission)), key=lambda i: transmission[i])
            peak_val = transmission[target_idx]
            half_power = peak_val * 0.5
        else:
            target_idx = min(range(len(transmission)), key=lambda i: transmission[i])
            dip_val = transmission[target_idx]
            max_val = max(transmission)
            half_power = (dip_val + max_val) * 0.5

        res_freq = frequencies[target_idx]
        res_lambda = C0 / res_freq if res_freq > 0 else 0.0

        # Find 3dB crossing points to measure FWHM
        f_left = res_freq
        f_right = res_freq

        # Search left
        for i in range(target_idx - 1, -1, -1):
            if is_drop_port and transmission[i] <= half_power:
                f_left = frequencies[i]
                break
            elif not is_drop_port and transmission[i] >= half_power:
                f_left = frequencies[i]
                break

        # Search right
        for i in range(target_idx + 1, len(transmission)):
            if is_drop_port and transmission[i] <= half_power:
                f_right = frequencies[i]
                break
            elif not is_drop_port and transmission[i] >= half_power:
                f_right = frequencies[i]
                break

        bandwidth = abs(f_right - f_left)
        if bandwidth <= 0.0:
            bandwidth = max(1e9, abs(frequencies[-1] - frequencies[0]) / len(frequencies))

        q_factor = res_freq / bandwidth if bandwidth > 0 else 0.0

        # Extinction ratio
        t_max = max(transmission)
        t_min = min(transmission)
        if t_min > 1e-15:
            extinction = 10.0 * math.log10(t_max / t_min)
        else:
            extinction = 60.0

        return ResonanceMetrics(
            resonance_wavelength=res_lambda,
            resonance_frequency=res_freq,
            fwhm_bandwidth_hz=bandwidth,
            quality_factor=q_factor,
            extinction_ratio_db=extinction,
        )
