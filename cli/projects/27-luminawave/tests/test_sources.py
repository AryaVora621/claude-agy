"""Unit tests for Optical Excitation Sources and Waveforms."""

import math
import unittest
from luminawave.grid import C0, Grid2D
from luminawave.sources import (
    InjectionMode,
    OpticalSource,
    PointSource,
    SourceWaveform,
    WaveguideModeSource,
)


class TestOpticalSources(unittest.TestCase):
    def test_gaussian_pulse_generation(self):
        """Verify Gaussian pulse timing and peak amplitude."""
        src = OpticalSource(
            waveform=SourceWaveform.GAUSSIAN_PULSE,
            wavelength=1.55e-6,
            amplitude=10.0,
        )
        # At t = 0, amplitude should be very close to zero due to t0 delay
        val_0 = src.evaluate(0.0)
        self.assertLess(abs(val_0), 1e-3)

        # At t = t0, amplitude should equal peak amplitude
        val_peak = src.evaluate(src.t0)
        self.assertAlmostEqual(val_peak, 10.0, delta=1e-4)

    def test_continuous_wave_ramp(self):
        """Verify CW smooth startup ramp and oscillation frequency."""
        src = OpticalSource(
            waveform=SourceWaveform.CONTINUOUS_WAVE,
            wavelength=1.0e-6,
            amplitude=5.0,
        )
        # Initial value at t = 0 is 0
        self.assertAlmostEqual(src.evaluate(0.0), 0.0, delta=1e-6)

        # After ramp time, peak value approaches amplitude
        t_after = src.t_ramp + 0.25 * src.period
        val = src.evaluate(t_after)
        self.assertGreater(abs(val), 3.0)

    def test_modulated_gaussian_carrier(self):
        """Verify modulated wavepacket zero crossings and envelope."""
        src = OpticalSource(
            waveform=SourceWaveform.MODULATED_GAUSSIAN,
            wavelength=1.55e-6,
            amplitude=1.0,
        )
        # At t = t0, sin(0) is 0
        self.assertAlmostEqual(src.evaluate(src.t0), 0.0, delta=1e-6)
        # At t = t0 + T/4, should have maximum carrier
        val_quarter = src.evaluate(src.t0 + 0.25 * src.period)
        self.assertGreater(abs(val_quarter), 0.8)

    def test_ricker_wavelet(self):
        """Verify Ricker wavelet peak and zero crossings."""
        src = OpticalSource(
            waveform=SourceWaveform.RICKER_WAVELET,
            wavelength=1.55e-6,
            amplitude=1.0,
        )
        # Peak occurs at t = t0
        val_peak = src.evaluate(src.t0)
        self.assertAlmostEqual(val_peak, 1.0, delta=1e-4)

    def test_point_source_and_mode_source_injection(self):
        """Verify field injection into Yee grid."""
        grid = Grid2D(nx=30, ny=30, dx=50e-9, dy=50e-9)
        src_opt = OpticalSource(
            waveform=SourceWaveform.GAUSSIAN_PULSE,
            amplitude=50.0,
            injection_mode=InjectionMode.HARD,
        )
        pt_src = PointSource(src_opt, 15, 15)

        pt_src.inject(grid, src_opt.t0)
        self.assertAlmostEqual(grid.ez[grid.idx(15, 15)], 50.0, delta=1e-3)

        # Waveguide mode line source
        mode_src = WaveguideModeSource(src_opt, x=20, y_start=10, y_end=20, is_vertical=True)
        mode_src.inject(grid, src_opt.t0)
        # Center of mode line should have non-zero value
        self.assertGreater(abs(grid.ez[grid.idx(20, 15)]), 1.0)


if __name__ == "__main__":
    unittest.main()
