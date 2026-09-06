"""Unit tests for FDTD Maxwell Equation Physics and Wave Propagation."""

import math
import unittest
from luminawave.grid import C0, Grid2D
from luminawave.fdtd import FDTDSimulator
from luminawave.sources import InjectionMode, OpticalSource, PointSource, SourceWaveform


class TestFDTDPhysics(unittest.TestCase):
    def test_wave_propagation_speed_in_vacuum(self):
        """Verify electromagnetic pulse wave propagation speed matches c0."""
        nx, ny = 100, 50
        dx = 50e-9
        grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.5)
        sim = FDTDSimulator(grid, enable_pml=False)

        # Inject single impulse / pulse at x = 20, y = 25
        x_src = 20
        y_src = 25
        # Inject compact Gaussian pulse with explicit fast timing
        src_opt = OpticalSource(
            waveform=SourceWaveform.GAUSSIAN_PULSE,
            wavelength=1.55e-6,
            amplitude=100.0,
            injection_mode=InjectionMode.HARD,
            tau=3.0 * grid.dt,
            t0=8.0 * grid.dt,
        )
        src = PointSource(src_opt, x_src, y_src)
        sim.add_source(src)

        # Run 45 steps
        n_steps = 45
        sim.run(n_steps)

        # Find peak position of the pulse along y = y_src
        peak_x = x_src
        peak_val = 0.0
        for x in range(x_src, nx - 1):
            val = abs(grid.ez[grid.idx(x, y_src)])
            if val > peak_val:
                peak_val = val
                peak_x = x

        measured_cells = peak_x - x_src
        # Time since peak injection is (45 - 8) * dt
        travel_time = (n_steps - 8) * grid.dt
        expected_cells = C0 * travel_time / dx
        self.assertAlmostEqual(measured_cells, expected_cells, delta=2.0)

    def test_phase_velocity_slowdown_in_dielectric(self):
        """Verify wave slows down in dielectric medium by factor of n (v = c0 / n)."""
        nx, ny = 100, 40
        dx = 50e-9
        # Create grid with vacuum on top half, dielectric n=2.0 on bottom half
        grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.5)
        # Top half: vacuum (n=1)
        # Bottom half: n=2.0 (eps_r = 4.0)
        grid.add_rectangle(0, 0, nx - 1, 19, eps_r=4.0)

        sim = FDTDSimulator(grid, enable_pml=False)

        # Launch pulse simultaneously in vacuum (y=30) and dielectric (y=10)
        src_opt = OpticalSource(
            waveform=SourceWaveform.GAUSSIAN_PULSE,
            wavelength=1.55e-6,
            amplitude=10.0,
            injection_mode=InjectionMode.HARD,
            tau=3.0 * grid.dt,
            t0=8.0 * grid.dt,
        )
        src_vac = PointSource(src_opt, 20, 30)
        src_diel = PointSource(src_opt, 20, 10)
        sim.add_source(src_vac)
        sim.add_source(src_diel)

        sim.run(60)

        # Find peak position in vacuum vs dielectric
        vac_peak_x = 20
        vac_max = 0.0
        diel_peak_x = 20
        diel_max = 0.0
        for x in range(20, nx - 1):
            v_val = abs(grid.ez[grid.idx(x, 30)])
            if v_val > vac_max:
                vac_max = v_val
                vac_peak_x = x
            d_val = abs(grid.ez[grid.idx(x, 10)])
            if d_val > diel_max:
                diel_max = d_val
                diel_peak_x = x

        vac_dist = vac_peak_x - 20
        diel_dist = diel_peak_x - 20

        # Dielectric distance should be approximately half of vacuum distance (n = 2)
        self.assertGreater(vac_dist, diel_dist)
        ratio = vac_dist / max(1, diel_dist)
        self.assertAlmostEqual(ratio, 2.0, delta=0.5)

    def test_numerical_stability(self):
        """Verify simulator remains bounded and stable across 200 time steps."""
        grid = Grid2D(nx=50, ny=50, courant_factor=0.70)
        sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)
        src_opt = OpticalSource(waveform=SourceWaveform.CONTINUOUS_WAVE, amplitude=1.0)
        sim.add_source(PointSource(src_opt, 25, 25))

        sim.run(200)
        self.assertTrue(sim.is_stable())
        max_ez, max_h = sim.get_max_fields()
        self.assertFalse(math.isnan(max_ez))
        self.assertFalse(math.isinf(max_ez))


if __name__ == "__main__":
    unittest.main()
