"""Unit tests for Perfectly Matched Layer (PML) Boundary Absorption."""

import math
import unittest
from luminawave.grid import Grid2D
from luminawave.pml import PMLBoundary
from luminawave.sources import OpticalSource, PointSource, SourceWaveform, InjectionMode
from luminawave.fdtd import FDTDSimulator


class TestPMLAbsorption(unittest.TestCase):
    def test_pml_profile_construction(self):
        """Verify PML polynomial grading and impedance matching."""
        grid = Grid2D(nx=60, ny=60, dx=50e-9, dy=50e-9)
        pml = PMLBoundary(grid, thickness=8, m_order=3, r0=1e-6)

        self.assertEqual(pml.thickness, 8)
        # Verify conductivity is 0 in the interior and maximum at outer wall
        self.assertEqual(pml.sigma_x[30], 0.0)
        self.assertEqual(pml.sigma_y[30], 0.0)
        self.assertGreater(pml.sigma_x[0], pml.sigma_x[7])
        self.assertGreater(pml.sigma_x[59], pml.sigma_x[52])

        # Verify boundary cell detection
        self.assertTrue(pml.is_in_pml(2, 30))
        self.assertTrue(pml.is_in_pml(30, 2))
        self.assertTrue(pml.is_in_pml(58, 30))
        self.assertFalse(pml.is_in_pml(30, 30))

    def test_pml_absorption_reflection_attenuation(self):
        """Verify that a pulse propagating into the PML is absorbed rather than reflected back."""
        # Setup two simulators: one with PML, one without PML (hard PEC reflection)
        nx, ny = 70, 70
        grid_pml = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.5)
        sim_pml = FDTDSimulator(grid_pml, pml_thickness=10, enable_pml=True)

        grid_pec = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.5)
        sim_pec = FDTDSimulator(grid_pec, enable_pml=False)

        # Inject a compact Gaussian pulse at center with fast timing
        src_opt = OpticalSource(
            waveform=SourceWaveform.GAUSSIAN_PULSE,
            wavelength=1.55e-6,
            amplitude=100.0,
            injection_mode=InjectionMode.SOFT,
            tau=3.0 * grid_pml.dt,
            t0=8.0 * grid_pml.dt,
        )
        src_pml = PointSource(src_opt, 35, 35)
        src_pec = PointSource(src_opt, 35, 35)
        sim_pml.add_source(src_pml)
        sim_pec.add_source(src_pec)

        # Run 160 steps: pulse leaves center, strikes boundaries, and absorbs in PML while reflecting in PEC
        sim_pml.run(160)
        sim_pec.run(160)

        # In PML grid, total energy inside the interior should be significantly smaller than in PEC grid
        ue_pml, uh_pml, u_tot_pml = grid_pml.total_energy()
        ue_pec, uh_pec, u_tot_pec = grid_pec.total_energy()

        # PEC boundary reflects all energy back, PML absorbs >85% of outgoing wave energy
        self.assertLess(u_tot_pml, u_tot_pec * 0.15)


if __name__ == "__main__":
    unittest.main()
