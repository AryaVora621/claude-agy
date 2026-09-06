"""Unit tests for Electromagnetic Field and DFT Flux Monitors."""

import math
import unittest
from luminawave.grid import C0, Grid2D
from luminawave.fdtd import FDTDSimulator
from luminawave.monitors import (
    LineDFTMonitor,
    PointTimeMonitor,
    ResonanceMetrics,
    SParameterAnalyzer,
)
from luminawave.sources import (
    InjectionMode,
    OpticalSource,
    PointSource,
    SourceWaveform,
)


class TestMonitorsAndDFT(unittest.TestCase):
    def test_point_time_monitor(self):
        """Verify field recording and statistics in PointTimeMonitor."""
        grid = Grid2D(nx=40, ny=40)
        sim = FDTDSimulator(grid, enable_pml=False)

        pt_mon = PointTimeMonitor("Probe1", 20, 20)
        sim.add_monitor(pt_mon)

        src_opt = OpticalSource(
            waveform=SourceWaveform.CONTINUOUS_WAVE,
            wavelength=1.55e-6,
            amplitude=5.0,
            injection_mode=InjectionMode.HARD,
        )
        sim.add_source(PointSource(src_opt, 20, 20))

        sim.run(30)

        self.assertEqual(len(pt_mon.time_history), 30)
        self.assertEqual(len(pt_mon.ez_history), 30)
        self.assertGreater(pt_mon.peak_ez, 0.0)
        self.assertGreater(pt_mon.rms_ez, 0.0)

    def test_line_dft_monitor_phasor_accumulation(self):
        """Verify on-the-fly phasor accumulation and Poynting flux integration."""
        grid = Grid2D(nx=50, ny=40, dx=50e-9, dy=50e-9, courant_factor=0.5)
        sim = FDTDSimulator(grid, enable_pml=True)

        freq0 = C0 / 1.55e-6
        freqs = [freq0 * 0.95, freq0, freq0 * 1.05]

        dft_mon = LineDFTMonitor("Aperture", coord=35, start=10, end=30, frequencies=freqs, is_vertical=True)
        sim.add_monitor(dft_mon)

        # Inject CW source at x=15
        src_opt = OpticalSource(
            waveform=SourceWaveform.CONTINUOUS_WAVE,
            wavelength=1.55e-6,
            amplitude=10.0,
            injection_mode=InjectionMode.SOFT,
        )
        sim.add_source(PointSource(src_opt, 15, 20))

        sim.run(60)

        flux = dft_mon.compute_flux(grid)
        self.assertEqual(len(flux), 3)
        # Power should be non-negative
        for val in flux:
            self.assertGreaterEqual(val, 0.0)

    def test_s_parameter_calculations(self):
        """Verify transmission S21 and insertion loss formulas."""
        in_flux = [10.0, 10.0, 10.0]
        out_flux = [5.0, 10.0, 0.1]

        s21, il_db = SParameterAnalyzer.compute_s_parameters(in_flux, out_flux)

        # 50% power -> 3 dB loss
        self.assertAlmostEqual(s21[0], 0.5, delta=1e-4)
        self.assertAlmostEqual(il_db[0], 3.0103, delta=1e-2)

        # 100% power -> 0 dB loss
        self.assertAlmostEqual(s21[1], 1.0, delta=1e-4)
        self.assertAlmostEqual(il_db[1], 0.0, delta=1e-2)

        # 1% power -> 20 dB loss
        self.assertAlmostEqual(s21[2], 0.01, delta=1e-4)
        self.assertAlmostEqual(il_db[2], 20.0, delta=1e-2)

    def test_resonance_extraction(self):
        """Verify cavity Q-factor and FWHM bandwidth extraction."""
        # Simulated resonance dip at 193.4 THz (1550 nm)
        f_center = 193.4e12
        bandwidth = 0.5e12  # 500 GHz
        freqs = [f_center + (i - 10) * 0.1e12 for i in range(21)]

        # Lorentzian dip transmission
        transmission = []
        for f in freqs:
            delta = (f - f_center) / (0.5 * bandwidth)
            t = delta * delta / (1.0 + delta * delta)
            transmission.append(t)

        metrics = SParameterAnalyzer.extract_resonance(freqs, transmission, is_drop_port=False)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics.resonance_frequency, f_center, delta=1e11)
        expected_q = f_center / bandwidth
        self.assertAlmostEqual(metrics.quality_factor, expected_q, delta=150)


if __name__ == "__main__":
    unittest.main()
