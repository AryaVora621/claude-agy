"""Unit tests for Silicon Photonic Devices and Waveguide Circuits."""

import math
import unittest
from luminawave.grid import Grid2D
from luminawave.fdtd import FDTDSimulator
from luminawave.photonics import (
    INDEX_SILICON,
    PhotonicCircuitBuilder,
    PhotonicPort,
)
from luminawave.sources import (
    InjectionMode,
    OpticalSource,
    SourceWaveform,
    WaveguideModeSource,
)


class TestPhotonicDevices(unittest.TestCase):
    def test_waveguide_tir_confinement(self):
        """Verify Total Internal Reflection (TIR) confinement in a silicon strip waveguide."""
        grid = Grid2D(nx=80, ny=40, dx=50e-9, dy=50e-9, courant_factor=0.6)
        # Create horizontal straight silicon waveguide (n=3.48) in air (n=1)
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x1=0, y1=20, x2=79, y2=20, width=6, n_core=INDEX_SILICON
        )
        sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

        # Inject guided mode at x = 12 with short ramp
        src_opt = OpticalSource(
            waveform=SourceWaveform.CONTINUOUS_WAVE,
            wavelength=1.55e-6,
            amplitude=10.0,
            injection_mode=InjectionMode.SOFT,
            t_ramp=0.5 * (1.55e-6 / 299792458.0),
        )
        mode_src = WaveguideModeSource(src_opt, x=12, y_start=17, y_end=23, is_vertical=True)
        sim.add_source(mode_src)

        sim.run(250)

        # Measure energy density inside the core vs cladding at x = 35
        core_field = abs(grid.ez[grid.idx(35, 20)])
        cladding_field = abs(grid.ez[grid.idx(35, 5)])  # 15 cells away in air

        # Optical mode should be strongly guided in core compared to far cladding
        self.assertGreater(core_field, cladding_field)

    def test_waveguide_bend_creation(self):
        """Verify 90-degree waveguide bend geometry construction."""
        grid = Grid2D(nx=60, ny=60)
        PhotonicCircuitBuilder.add_waveguide_bend(
            grid, cx=30, cy=30, radius=18, width=6, start_angle_deg=0, end_angle_deg=90
        )
        # Check that material at (30 + 18, 30) has silicon index
        self.assertAlmostEqual(grid.get_refractive_index(48, 30), INDEX_SILICON, delta=0.1)
        # Check that center (30, 30) is cladding
        self.assertEqual(grid.eps_r[grid.idx(30, 30)], 1.0)

    def test_directional_coupler_builder(self):
        """Verify 2x2 directional coupler port configuration."""
        grid = Grid2D(nx=90, ny=60)
        inputs, outputs = PhotonicCircuitBuilder.build_directional_coupler(
            grid, y_center=30, waveguide_width=6, coupling_gap=3, coupling_length=25
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(inputs[0].name, "In_Top")
        self.assertEqual(outputs[0].name, "Out_Bar")
        self.assertEqual(outputs[1].name, "Out_Cross")

    def test_ring_resonator_builder(self):
        """Verify micro-ring resonator geometry and bus coupling."""
        grid = Grid2D(nx=80, ny=80)
        in_p, through_p = PhotonicCircuitBuilder.build_ring_resonator(
            grid, cx=40, cy=50, radius=16, ring_width=5, bus_y=25, bus_width=5
        )
        self.assertEqual(in_p.name, "In_Bus")
        self.assertEqual(through_p.name, "Through_Bus")
        # Center of ring should be air/cladding
        self.assertEqual(grid.eps_r[grid.idx(40, 50)], 1.0)
        # Ring rim should be silicon
        self.assertAlmostEqual(grid.get_refractive_index(40, 50 + 16), INDEX_SILICON, delta=0.1)

    def test_mach_zehnder_builder(self):
        """Verify Mach-Zehnder Interferometer (MZI) arms."""
        grid = Grid2D(nx=90, ny=60)
        in_p, out_p = PhotonicCircuitBuilder.build_mach_zehnder(
            grid, y_center=30, arm_spacing=16, arm_length=30
        )
        self.assertEqual(in_p.name, "In")
        self.assertEqual(out_p.name, "Out")
        # Center between arms should be air
        self.assertEqual(grid.eps_r[grid.idx(45, 30)], 1.0)
        # Both arm positions should be silicon
        self.assertAlmostEqual(grid.get_refractive_index(45, 38), INDEX_SILICON, delta=0.1)
        self.assertAlmostEqual(grid.get_refractive_index(45, 22), INDEX_SILICON, delta=0.1)

    def test_photonic_crystal_waveguide(self):
        """Verify 2D periodic rod lattice and line defect waveguide."""
        grid = Grid2D(nx=60, ny=50)
        in_p, out_p = PhotonicCircuitBuilder.build_photonic_crystal_waveguide(
            grid, pitch=8, rod_radius=2, defect_row=3
        )
        self.assertEqual(in_p.name, "In_PBG")
        # Defect row (row 3: y = 24) should be empty (cladding)
        self.assertEqual(grid.eps_r[grid.idx(24, 24)], 1.0)
        # Adjacent rod location (row 2: x=16, y=16) should have rod material
        self.assertAlmostEqual(grid.get_refractive_index(16, 16), INDEX_SILICON, delta=0.1)


if __name__ == "__main__":
    unittest.main()
