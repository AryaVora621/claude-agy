"""
Performance Microbenchmarks for StellarFusion Engine.

Measures throughput of:
  1. Analytical Solovev equilibrium evaluations (flux, derivatives, Shafranov operator)
  2. Grad-Shafranov finite-difference SOR relaxation sweeps
  3. 3D magnetic field vector evaluations
  4. Symplectic Boris particle pusher integration steps
  5. Poincaré field line RK4 trajectory steps
  6. Sub-pixel Braille canvas rasterization & rendering FPS
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

# Ensure parent directory is in sys.path for standalone invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stellarfusion.equilibrium import (
    EquilibriumProfile,
    GradShafranovSolver,
    Grid2D,
    SolovevEquilibrium,
)
from stellarfusion.magnetic import MagneticFieldEvaluator
from stellarfusion.particles import (
    BorisParticlePusher,
    ParticleSpecies,
    ParticleState,
)
from stellarfusion.poincare import PoincareFieldTracer
from stellarfusion.visualizer import BrailleCanvas, TokamakVisualizer


def run_benchmark(name: str, fn: Callable[[], int], target_duration: float = 1.0) -> Tuple[float, int]:
    """Execute benchmark function for approximately target_duration seconds."""
    # Warmup
    fn()

    start_time = time.perf_counter()
    total_ops = 0
    while True:
        ops = fn()
        total_ops += ops
        elapsed = time.perf_counter() - start_time
        if elapsed >= target_duration:
            break

    throughput = total_ops / elapsed
    return throughput, total_ops


def main() -> None:
    print("======================================================================")
    print("         STELLARFUSION : HIGH-PERFORMANCE MICROBENCHMARKS             ")
    print("======================================================================")

    solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.6, psi_0=1.5, b_0=2.5)
    evaluator = MagneticFieldEvaluator(solovev)

    # 1. Solovev Analytical Evaluation Throughput
    def bench_solovev() -> int:
        count = 1000
        for i in range(count):
            r = 2.0 + (i % 20) * 0.1
            z = -1.0 + (i % 20) * 0.1
            _ = solovev.shafranov_operator(r, z)
            _ = solovev.magnetic_field(r, z)
        return count

    rate_solovev, _ = run_benchmark("Solovev Analytical Evals", bench_solovev)
    print(f" 1. Solovev Equilibrium Evals   : {rate_solovev:12,.1f} evals/sec")

    # 2. Grad-Shafranov SOR Relaxation Sweeps
    grid = Grid2D(nr=31, nz=31, r_min=1.5, r_max=4.5, z_min=-1.5, z_max=1.5)
    solver = GradShafranovSolver(grid=grid)
    solver.initialize_with_solovev(solovev)

    def bench_sor() -> int:
        count = 20
        for _ in range(count):
            solver.solve_step(omega=1.3)
        return count

    rate_sor, _ = run_benchmark("GS SOR Sweeps", bench_sor)
    print(f" 2. Grad-Shafranov SOR Sweeps    : {rate_sor:12,.1f} sweeps/sec (31x31 grid)")

    # 3. 3D Magnetic Field Evaluations
    def bench_magnetic() -> int:
        count = 1000
        for i in range(count):
            r = 2.0 + (i % 25) * 0.1
            z = -1.2 + (i % 25) * 0.1
            field = evaluator.evaluate_at(r, z)
            _ = field.magnitude
        return count

    rate_mag, _ = run_benchmark("Magnetic Field Evals", bench_magnetic)
    print(f" 3. 3D Magnetic Field Vector     : {rate_mag:12,.1f} evals/sec")

    # 4. Symplectic Boris Particle Pusher
    pusher = BorisParticlePusher(evaluator, species=ParticleSpecies.DEUTERIUM)
    state = ParticleState(x=3.2, y=0.0, z=0.1, vx=1e5, vy=5e5, vz=2e5)

    def bench_boris() -> int:
        count = 1000
        nonlocal state
        dt = 1e-8
        for _ in range(count):
            state = pusher.step(state, dt)
        return count

    rate_boris, _ = run_benchmark("Boris Pusher Steps", bench_boris)
    print(f" 4. Symplectic Boris Pusher      : {rate_boris:12,.1f} steps/sec")

    # 5. Poincaré RK4 Field Line Tracer
    tracer = PoincareFieldTracer(evaluator)

    def bench_poincare() -> int:
        count = 500
        r, z, phi = 3.3, 0.0, 0.0
        dphi = 0.05
        for _ in range(count):
            r, z, phi = tracer.rk4_step(r, z, phi, dphi)
        return count

    rate_poincare, _ = run_benchmark("Poincare RK4 Steps", bench_poincare)
    print(f" 5. Poincaré RK4 Field Steps     : {rate_poincare:12,.1f} steps/sec")

    # 6. Braille Canvas Rasterization & Rendering
    canvas = BrailleCanvas(char_width=70, char_height=30)
    viz = TokamakVisualizer(canvas)

    def bench_visualizer() -> int:
        canvas.clear()
        viz.draw_flux_contours(r_axis=3.0, z_axis=0.0, a_minor=0.9, kappa=1.6, num_surfaces=5)
        viz.draw_vacuum_vessel(r_axis=3.0, z_axis=0.0, a_wall=1.1, kappa_wall=1.7)
        _ = canvas.render_to_string()
        return 1

    fps_viz, _ = run_benchmark("Braille Visualizer FPS", bench_visualizer)
    print(f" 6. Braille Canvas FPS (70x30)   : {fps_viz:12,.1f} FPS")

    print("======================================================================")
    print("                     ALL BENCHMARKS COMPLETED                         ")
    print("======================================================================")


if __name__ == "__main__":
    main()
