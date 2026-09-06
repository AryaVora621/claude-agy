"""Performance Microbenchmarks for LuminaWave 2D FDTD Maxwell Engine.

Measures:
1. Pure Python 2D Yee Leapfrog Engine Throughput (MegaCells/sec, MC/s) across grid dimensions
2. Berenger Split-Field PML Absorbing Boundary Overhead
3. On-The-Fly DFT Phasor Accumulation & Line Flux Sampling Throughput
4. Silicon Photonic Integrated Circuit (PIC) Simulation Speed
5. Sub-Pixel Braille Terminal Renderer Frame Rate (FPS)
"""

from __future__ import annotations
import math
import sys
import time
from typing import Callable, Dict, List, Tuple

from luminawave.grid import Grid2D
from luminawave.pml import PMLBoundary
from luminawave.sources import (
    InjectionMode,
    OpticalSource,
    PointSource,
    SourceWaveform,
    WaveguideModeSource,
)
from luminawave.fdtd import FDTDSimulator
from luminawave.photonics import INDEX_SILICON, PhotonicCircuitBuilder
from luminawave.monitors import LineDFTMonitor, PointTimeMonitor
from luminawave.visualizer import BrailleEMCanvas, PhotonicsWorkbenchHUD


def run_benchmark(
    name: str,
    func: Callable[[], None],
    iterations: int = 1,
    warmup: int = 1,
) -> Tuple[float, float]:
    """Execute benchmark with warmup and timing statistics.

    Returns:
        Tuple of (elapsed_seconds, iterations_per_second).
    """
    for _ in range(warmup):
        func()

    start_time = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start_time
    rate = iterations / elapsed if elapsed > 0 else float("inf")
    return elapsed, rate


def bench_leapfrog_throughput() -> Dict[str, float]:
    """Benchmark raw 2D FDTD leapfrog time-stepping across multiple grid sizes."""
    results: Dict[str, float] = {}
    sizes = [(50, 50, 200), (100, 100, 100), (150, 150, 50)]

    for nx, ny, steps in sizes:
        grid = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.70)
        sim = FDTDSimulator(grid, enable_pml=False)

        # Warmup and timed run
        start = time.perf_counter()
        sim.run(steps)
        elapsed = time.perf_counter() - start

        total_cells_stepped = nx * ny * steps
        mc_per_sec = (total_cells_stepped / elapsed) / 1e6
        label = f"{nx}x{ny} ({steps} steps)"
        results[label] = mc_per_sec

    return results


def bench_pml_overhead() -> Tuple[float, float, float]:
    """Measure computational cost added by Berenger split-field PML boundary updates."""
    nx, ny, steps = 80, 80, 100

    # Baseline: no PML
    grid_pec = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.70)
    sim_pec = FDTDSimulator(grid_pec, enable_pml=False)
    t0 = time.perf_counter()
    sim_pec.run(steps)
    t_pec = time.perf_counter() - t0

    # With PML (thickness = 10 cells)
    grid_pml = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.70)
    sim_pml = FDTDSimulator(grid_pml, pml_thickness=10, enable_pml=True)
    t1 = time.perf_counter()
    sim_pml.run(steps)
    t_pml = time.perf_counter() - t1

    overhead_pct = ((t_pml - t_pec) / max(1e-9, t_pec)) * 100.0
    return t_pec, t_pml, overhead_pct


def bench_dft_monitor_throughput() -> Tuple[float, float]:
    """Measure on-the-fly phasor accumulation overhead with multiple frequency bins."""
    nx, ny, steps = 100, 50, 120
    grid = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9)
    sim = FDTDSimulator(grid, enable_pml=False)

    # 40 discrete frequency bins across optical C-band
    f0 = 299792458.0 / 1.55e-6
    frequencies = [f0 + (k - 20) * 1e11 for k in range(40)]
    mon = LineDFTMonitor(
        name="Out_DFT", coord=80, start=10, end=40, frequencies=frequencies, is_vertical=True
    )
    sim.add_monitor(mon)

    start = time.perf_counter()
    sim.run(steps)
    elapsed = time.perf_counter() - start

    phasor_updates = len(frequencies) * (40 - 10 + 1) * steps
    updates_per_sec = phasor_updates / elapsed if elapsed > 0 else 0.0
    return elapsed, updates_per_sec


def bench_silicon_pic_simulation() -> Tuple[float, float]:
    """Benchmark end-to-end silicon micro-ring resonator simulation."""
    nx, ny, steps = 90, 90, 150
    grid = Grid2D(nx=nx, ny=ny, dx=50e-9, dy=50e-9, courant_factor=0.70)

    # Construct micro-ring resonator
    in_port, through_port = PhotonicCircuitBuilder.build_ring_resonator(
        grid, cx=45, cy=55, radius=18, ring_width=5, bus_y=25, bus_width=5
    )

    sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)
    src_opt = OpticalSource(
        waveform=SourceWaveform.MODULATED_GAUSSIAN,
        wavelength=1.55e-6,
        amplitude=10.0,
        tau=4.0 * grid.dt,
        t0=10.0 * grid.dt,
    )
    src = WaveguideModeSource(
        src_opt, x=in_port.x, y_start=in_port.y - 3, y_end=in_port.y + 3, is_vertical=True
    )
    sim.add_source(src)

    start = time.perf_counter()
    sim.run(steps)
    elapsed = time.perf_counter() - start

    cells_total = nx * ny * steps
    mc_per_sec = (cells_total / elapsed) / 1e6
    return elapsed, mc_per_sec


def bench_braille_renderer() -> Tuple[float, float]:
    """Measure Unicode 2x4 sub-pixel Braille rendering frame rate."""
    grid = Grid2D(nx=80, ny=40, dx=50e-9, dy=50e-9)
    # Populate with synthetic optical wave pattern
    for y in range(40):
        for x in range(80):
            idx = grid.idx(x, y)
            grid.ez[idx] = math.sin(x * 0.2) * math.cos(y * 0.3)

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        _ = BrailleEMCanvas.render_field(
            grid, width_chars=40, height_rows=20, max_val=1.0, show_structure=False, use_color=True
        )
    elapsed = time.perf_counter() - start

    fps = iterations / elapsed if elapsed > 0 else 0.0
    return elapsed, fps


def main() -> None:
    """Run all benchmarks and print formatted performance report."""
    print("=" * 72)
    print(" LuminaWave: 2D FDTD Nanophotonics Performance Benchmark")
    print(" Standard Library Pure Python (Zero Dependencies)")
    print("=" * 72)
    print()

    # 1. Leapfrog Engine Throughput
    print("[1/5] Measuring 2D Yee Leapfrog Engine Throughput...")
    leapfrog_results = bench_leapfrog_throughput()
    for label, mc_s in leapfrog_results.items():
        print(f"  Grid {label:<22} : {mc_s:6.2f} MegaCells/sec (MC/s)")
    print()

    # 2. PML Overhead
    print("[2/5] Measuring Berenger Split-Field PML Absorbing Boundary Overhead...")
    t_pec, t_pml, overhead = bench_pml_overhead()
    print(f"  PEC Boundary (80x80, 100 steps)  : {t_pec*1000:6.2f} ms")
    print(f"  PML Boundary (10 cells, 80x80)   : {t_pml*1000:6.2f} ms")
    print(f"  PML Update Overhead Ratio        : +{overhead:5.1f}%")
    print()

    # 3. DFT Monitor Phasor Updates
    print("[3/5] Measuring On-The-Fly DFT Phasor Accumulation...")
    t_dft, updates_s = bench_dft_monitor_throughput()
    print(f"  40-frequency DFT Line Monitor    : {t_dft*1000:6.2f} ms")
    print(f"  Phasor Accumulation Throughput   : {updates_s/1e6:6.2f} MUpdates/sec")
    print()

    # 4. Silicon Photonic Circuit Simulation
    print("[4/5] Measuring Silicon Micro-Ring Resonator Simulation...")
    t_pic, mc_pic = bench_silicon_pic_simulation()
    print(f"  Ring Resonator (90x90, 150 steps): {t_pic*1000:6.2f} ms")
    print(f"  Effective Simulation Throughput  : {mc_pic:6.2f} MegaCells/sec")
    print()

    # 5. Braille Terminal Visualizer
    print("[5/5] Measuring Sub-Pixel Unicode Braille Visualizer...")
    t_vis, fps = bench_braille_renderer()
    print(f"  50 Frames TrueColor Braille      : {t_vis*1000:6.2f} ms")
    print(f"  Terminal Frame Rate              : {fps:6.1f} FPS")
    print()

    print("=" * 72)
    print(" Benchmark Completed Successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
