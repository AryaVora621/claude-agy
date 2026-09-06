"""Interactive Silicon Photonics Workbench.

Simulates and visualizes 2D Maxwell FDTD wave propagation through nanophotonic
integrated circuits (PICs) directly inside the terminal:
1. Micro-Ring Resonator (Cavity buildup, resonance drop, Q-factor)
2. 2x2 Directional Coupler (Evanescent wave power splitting to Cross and Bar ports)
3. 90-Degree Low-Loss Waveguide Bend (Total internal reflection around tight bend)
4. Photonic Bandgap Crystal Waveguide (2D dielectric rod lattice line defect)
5. Mach-Zehnder Interferometer (Power splitting, twin arm propagation, recombination)

Zero external dependencies: 100% Python standard library.
Uses 2x4 sub-pixel Unicode Braille characters (U+2800..U+28FF) with 24-bit TrueColor ANSI.
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

from luminawave.grid import C0, Grid2D
from luminawave.pml import PMLBoundary
from luminawave.sources import (
    InjectionMode,
    OpticalSource,
    PointSource,
    SourceWaveform,
    WaveguideModeSource,
)
from luminawave.fdtd import FDTDSimulator
from luminawave.photonics import INDEX_SILICON, PhotonicCircuitBuilder, PhotonicPort
from luminawave.monitors import LineDFTMonitor, PointTimeMonitor, SParameterAnalyzer
from luminawave.visualizer import BrailleEMCanvas, PhotonicsWorkbenchHUD


def run_ring_resonator_demo(steps: int = 240, animate: bool = True, fps_target: float = 30.0) -> None:
    """Run Silicon Micro-Ring Resonator electromagnetic simulation."""
    nx, ny = 84, 84
    dx = 50e-9
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.65)

    # Build ring resonator (radius = 18 cells = 900 nm, bus at y=25)
    in_port, through_port = PhotonicCircuitBuilder.build_ring_resonator(
        grid, cx=42, cy=55, radius=18, ring_width=5, bus_y=25, bus_width=5
    )

    sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

    # Modulated Gaussian pulse spanning telecom C-band (1500 nm - 1600 nm)
    f0 = C0 / 1.55e-6
    src_opt = OpticalSource(
        waveform=SourceWaveform.MODULATED_GAUSSIAN,
        wavelength=1.55e-6,
        amplitude=10.0,
        tau=4.0 * grid.dt,
        t0=10.0 * grid.dt,
    )
    mode_src = WaveguideModeSource(
        src_opt, x=in_port.x, y_start=in_port.y - 3, y_end=in_port.y + 3, is_vertical=True
    )
    sim.add_source(mode_src)

    # Setup DFT flux monitors at input and through ports
    test_freqs = [f0 * (1.0 + 0.005 * k) for k in range(-15, 16)]
    in_mon = LineDFTMonitor("In_DFT", coord=in_port.x + 4, start=20, end=30, frequencies=test_freqs)
    through_mon = LineDFTMonitor("Through_DFT", coord=through_port.x - 4, start=20, end=30, frequencies=test_freqs)
    sim.add_monitor(in_mon)
    sim.add_monitor(through_mon)

    # Point monitor inside the resonant ring cavity
    cavity_mon = PointTimeMonitor("Cavity", x=42, y=73)
    sim.add_monitor(cavity_mon)

    print("\033[2J\033[H", end="")  # Clear terminal
    frame_interval = 1.0 / max(1.0, fps_target)
    step_chunk = 4

    hud = PhotonicsWorkbenchHUD(sim)

    for current_step in range(0, steps, step_chunk):
        sim.run(step_chunk)

        if animate:
            hud_str = hud.render(width_chars=68, height_rows=20)
            print("\033[H" + hud_str, end="", flush=True)
            time.sleep(frame_interval)

    # Analyze S-parameters post-simulation
    print("\n" + "=" * 70)
    print(" Post-Simulation S-Parameter Spectral Analysis")
    print("=" * 70)
    in_flux = in_mon.compute_flux(sim.grid)
    through_flux = through_mon.compute_flux(sim.grid)
    s21, il_db = SParameterAnalyzer.compute_s_parameters(in_flux, through_flux)

    metrics = SParameterAnalyzer.extract_resonance(test_freqs, s21, is_drop_port=False)
    if metrics:
        print(f" Detected Resonance Wavelength: {metrics.resonance_wavelength * 1e9:.2f} nm")
        print(f" Resonant Extinction Ratio   : {metrics.extinction_ratio_db:.2f} dB")
        if metrics.quality_factor:
            print(f" Optical Quality Factor (Q)   : {metrics.quality_factor:.1f}")

    # Transmission Spectrum Sparkline
    wavelengths = [(C0 / f) * 1e9 for f in test_freqs]
    sparkline = BrailleEMCanvas.plot_spectrum_sparkline(s21, width_chars=50, height_rows=5)
    print(f"\n Transmission Spectrum T(lambda):\n{sparkline}")
    print(f"  [{wavelengths[-1]:.1f} nm ... {wavelengths[0]:.1f} nm]")
    print("=" * 70 + "\n")


def run_directional_coupler_demo(steps: int = 240, animate: bool = True, fps_target: float = 30.0) -> None:
    """Run 2x2 Directional Coupler evanescent power transfer simulation."""
    nx, ny = 96, 64
    dx = 50e-9
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.65)

    # Build directional coupler with 150 nm gap (3 cells)
    inputs, outputs = PhotonicCircuitBuilder.build_directional_coupler(
        grid, y_center=32, waveguide_width=6, coupling_gap=3, coupling_length=30
    )

    sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

    # Launch continuous wave into top input port
    src_opt = OpticalSource(
        waveform=SourceWaveform.CONTINUOUS_WAVE,
        wavelength=1.55e-6,
        amplitude=10.0,
        t_ramp=1.0 * (1.55e-6 / C0),
    )
    mode_src = WaveguideModeSource(
        src_opt, x=inputs[0].x, y_start=inputs[0].y - 3, y_end=inputs[0].y + 3, is_vertical=True
    )
    sim.add_source(mode_src)

    bar_mon = PointTimeMonitor("BarPort", x=outputs[0].x - 4, y=outputs[0].y)
    cross_mon = PointTimeMonitor("CrossPort", x=outputs[1].x - 4, y=outputs[1].y)
    sim.add_monitor(bar_mon)
    sim.add_monitor(cross_mon)

    print("\033[2J\033[H", end="")
    frame_interval = 1.0 / max(1.0, fps_target)
    step_chunk = 4
    hud = PhotonicsWorkbenchHUD(sim)

    for current_step in range(0, steps, step_chunk):
        sim.run(step_chunk)

        if animate:
            hud_str = hud.render(width_chars=76, height_rows=18)
            print("\033[H" + hud_str, end="", flush=True)
            time.sleep(frame_interval)

    # Power splitting telemetry
    bar_rms = bar_mon.rms_ez
    cross_rms = cross_mon.rms_ez
    tot_rms = math.sqrt(bar_rms ** 2 + cross_rms ** 2) if (bar_rms + cross_rms) > 0 else 1.0
    bar_pct = (bar_rms ** 2 / (tot_rms ** 2)) * 100.0 if tot_rms > 0 else 0.0
    cross_pct = (cross_rms ** 2 / (tot_rms ** 2)) * 100.0 if tot_rms > 0 else 0.0

    print("\n" + "=" * 70)
    print(" 2x2 Directional Coupler Splitting Telemetry")
    print("=" * 70)
    print(f" Bar Port (Through) Power Fraction : {bar_pct:.1f}%")
    print(f" Cross Port (Coupled) Power Fraction: {cross_pct:.1f}%")
    print(f" Coupling Power Ratio (Cross/Bar)   : {cross_rms/max(1e-9, bar_rms):.3f}")
    print("=" * 70 + "\n")


def run_waveguide_bend_demo(steps: int = 220, animate: bool = True, fps_target: float = 30.0) -> None:
    """Run 90-Degree Low-Loss Waveguide Bend simulation."""
    nx, ny = 76, 76
    dx = 50e-9
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.65)

    # Horizontal input waveguide
    PhotonicCircuitBuilder.add_straight_waveguide(grid, x1=0, y1=24, x2=38, y2=24, width=6)
    # 90-degree circular bend
    PhotonicCircuitBuilder.add_waveguide_bend(
        grid, cx=38, cy=42, radius=18, width=6, start_angle_deg=270, end_angle_deg=360
    )
    # Vertical output waveguide
    PhotonicCircuitBuilder.add_straight_waveguide(grid, x1=56, y1=42, x2=56, y2=75, width=6)

    sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

    src_opt = OpticalSource(
        waveform=SourceWaveform.CONTINUOUS_WAVE,
        wavelength=1.55e-6,
        amplitude=10.0,
        t_ramp=1.0 * (1.55e-6 / C0),
    )
    mode_src = WaveguideModeSource(src_opt, x=12, y_start=21, y_end=27, is_vertical=True)
    sim.add_source(mode_src)

    out_mon = PointTimeMonitor("BendOut", x=56, y=65)
    sim.add_monitor(out_mon)

    print("\033[2J\033[H", end="")
    frame_interval = 1.0 / max(1.0, fps_target)
    step_chunk = 4
    hud = PhotonicsWorkbenchHUD(sim)

    for current_step in range(0, steps, step_chunk):
        sim.run(step_chunk)

        if animate:
            hud_str = hud.render(width_chars=72, height_rows=18)
            print("\033[H" + hud_str, end="", flush=True)
            time.sleep(frame_interval)

    print("\n" + "=" * 70)
    print(" 90-Degree Bend Propagation Telemetry")
    print("=" * 70)
    print(f" Measured Output Peak Field: {out_mon.peak_ez:.3f} V/m")
    print(f" Measured Output RMS Field : {out_mon.rms_ez:.3f} V/m")
    print("=" * 70 + "\n")


def run_photonic_crystal_demo(steps: int = 220, animate: bool = True, fps_target: float = 30.0) -> None:
    """Run 2D Photonic Bandgap (PBG) Crystal Line-Defect Waveguide simulation."""
    nx, ny = 88, 64
    dx = 50e-9
    grid = Grid2D(nx=nx, ny=ny, dx=dx, dy=dx, courant_factor=0.65)

    in_port, out_port = PhotonicCircuitBuilder.build_photonic_crystal_waveguide(
        grid, pitch=8, rod_radius=2, defect_row=4, n_rod=INDEX_SILICON
    )

    sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

    src_opt = OpticalSource(
        waveform=SourceWaveform.CONTINUOUS_WAVE,
        wavelength=1.55e-6,
        amplitude=10.0,
        t_ramp=1.0 * (1.55e-6 / C0),
    )
    mode_src = WaveguideModeSource(
        src_opt, x=in_port.x, y_start=in_port.y - 4, y_end=in_port.y + 4, is_vertical=True
    )
    sim.add_source(mode_src)

    out_mon = PointTimeMonitor("PBG_Out", x=out_port.x - 4, y=out_port.y)
    sim.add_monitor(out_mon)

    print("\033[2J\033[H", end="")
    frame_interval = 1.0 / max(1.0, fps_target)
    step_chunk = 4
    hud = PhotonicsWorkbenchHUD(sim)

    for current_step in range(0, steps, step_chunk):
        sim.run(step_chunk)

        if animate:
            hud_str = hud.render(width_chars=76, height_rows=18)
            print("\033[H" + hud_str, end="", flush=True)
            time.sleep(frame_interval)

    print("\n" + "=" * 70)
    print(" Photonic Crystal Waveguide Telemetry")
    print("=" * 70)
    print(f" PBG Defect Guided Peak Field: {out_mon.peak_ez:.3f} V/m")
    print(f" PBG Defect Guided RMS Field : {out_mon.rms_ez:.3f} V/m")
    print("=" * 70 + "\n")


def main() -> None:
    """Command-line entry point for Silicon Photonics Workbench."""
    parser = argparse.ArgumentParser(
        description="LuminaWave: Interactive Silicon Photonics 2D FDTD Maxwell Workbench"
    )
    parser.add_argument(
        "--device",
        choices=["ring", "coupler", "bend", "pbg"],
        default="ring",
        help="Select photonic circuit preset (ring, coupler, bend, pbg)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of FDTD leapfrog time steps to simulate (default: 240)",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Disable live terminal animation (fast headless execution)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target animation frame rate (default: 30 FPS)",
    )

    args = parser.parse_args()
    animate = not args.no_anim

    if args.device == "ring":
        run_ring_resonator_demo(steps=args.steps, animate=animate, fps_target=args.fps)
    elif args.device == "coupler":
        run_directional_coupler_demo(steps=args.steps, animate=animate, fps_target=args.fps)
    elif args.device == "bend":
        run_waveguide_bend_demo(steps=args.steps, animate=animate, fps_target=args.fps)
    elif args.device == "pbg":
        run_photonic_crystal_demo(steps=args.steps, animate=animate, fps_target=args.fps)


if __name__ == "__main__":
    main()
