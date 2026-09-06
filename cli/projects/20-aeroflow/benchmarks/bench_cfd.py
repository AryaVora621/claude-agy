"""
AeroFlow: Computational Fluid Dynamics Performance Benchmarks.
Measures Lattice Boltzmann MLUPS (Mega Lattice Updates per Second),
Navier-Stokes Red-Black Gauss-Seidel Poisson solver throughput,
semi-Lagrangian advection, and Braille flow visualization FPS.
"""

import sys
import os
import time
import math

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aeroflow.types import Grid2D, VectorField2D, ObstacleMask
from aeroflow.lbm import LBMSolver
from aeroflow.navier_stokes import NavierStokesSolver
from aeroflow.obstacles import create_cylinder_obstacle, create_naca_airfoil_obstacle
from aeroflow.visualizer import render_vorticity_field, render_velocity_vector_field


def bench_lbm_throughput():
    print("\n--- 1. Lattice Boltzmann D2Q9 Collision & Streaming Throughput ---")
    nx, ny = 80, 40
    cyl = create_cylinder_obstacle(nx, ny, center_x=20, center_y=20, radius=5)
    solver = LBMSolver(nx=nx, ny=ny, viscosity=0.02, inflow_velocity=0.08, obstacle=cyl)

    steps = 100
    t0 = time.perf_counter()
    solver.run_steps(steps)
    elapsed = time.perf_counter() - t0

    total_site_updates = nx * ny * steps
    mlups = (total_site_updates / elapsed) / 1e6
    us_per_step = (elapsed / steps) * 1000.0

    print(f"LBM D2Q9 ({nx}x{ny}): {steps} steps ({total_site_updates:,} lattice site updates) in {elapsed*1000:.2f} ms")
    print(f"Throughput: {mlups:.3f} MLUPS ({us_per_step:.2f} ms/step)")


def bench_navier_stokes_projection():
    print("\n--- 2. Navier-Stokes Eulerian Projection & Red-Black Poisson Solver ---")
    nx, ny = 50, 30
    cyl = create_cylinder_obstacle(nx, ny, center_x=15, center_y=15, radius=4)
    solver = NavierStokesSolver(nx=nx, ny=ny, dx=1.0, dt=0.05, viscosity=0.002, inflow_velocity=1.0, obstacle=cyl)

    steps = 50
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.step()
    elapsed = time.perf_counter() - t0

    steps_per_sec = steps / elapsed
    ms_per_step = (elapsed / steps) * 1000.0

    print(f"Navier-Stokes Projection ({nx}x{ny}): {steps} full time steps in {elapsed*1000:.2f} ms")
    print(f"Throughput: {steps_per_sec:,.1f} steps/sec ({ms_per_step:.2f} ms/step)")


def bench_semi_lagrangian_advection():
    print("\n--- 3. Semi-Lagrangian Advection Backtracing ---")
    nx, ny = 100, 60
    vf = VectorField2D(nx=nx, ny=ny, dx=1.0, u_init=1.0, v_init=0.2)
    obs = ObstacleMask(nx=nx, ny=ny)
    solver = NavierStokesSolver(nx=nx, ny=ny, dx=1.0, dt=0.05, obstacle=obs)

    trials = 100
    t0 = time.perf_counter()
    for _ in range(trials):
        solver.advect()
    elapsed = time.perf_counter() - t0

    total_cells = nx * ny * trials
    cells_sec = total_cells / elapsed
    print(f"Semi-Lagrangian Bilinear Advection: {trials} passes ({total_cells:,} cell samples) in {elapsed*1000:.2f} ms")
    print(f"Throughput: {cells_sec:,.0f} cell advections/sec")


def bench_naca_rasterization():
    print("\n--- 4. NACA 4-Digit Airfoil Polygon Geometry Rasterization ---")
    trials = 500
    t0 = time.perf_counter()
    for _ in range(trials):
        create_naca_airfoil_obstacle(nx=80, ny=40, chord=30.0, code="2412", lead_x=20.0, lead_y=20.0, angle_of_attack_deg=8.0)
    elapsed = time.perf_counter() - t0
    evals_sec = trials / elapsed
    us_per_eval = (elapsed / trials) * 1e6

    print(f"NACA 2412 Airfoil Generation: {trials} airfoils rasterized in {elapsed*1000:.2f} ms")
    print(f"Throughput: {evals_sec:,.0f} airfoils/sec ({us_per_eval:.2f} µs/raster)")


def bench_braille_flow_visualizer():
    print("\n--- 5. Sub-Pixel Unicode Braille Flow Visualizer Rendering ---")
    nx, ny = 80, 40
    vort = Grid2D(nx=nx, ny=ny, dx=1.0)
    for y in range(ny):
        for x in range(nx):
            vort.set(x, y, math.sin(x * 0.2) * math.cos(y * 0.2))
    obs = create_cylinder_obstacle(nx, ny, center_x=20, center_y=20, radius=5)

    frames = 100
    t0 = time.perf_counter()
    for _ in range(frames):
        render_vorticity_field(vort, obs, char_width=70, char_height=22)
    elapsed = time.perf_counter() - t0
    fps = frames / elapsed

    print(f"Braille Vorticity Visualizer: {frames} full frames rendered in {elapsed*1000:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")


def run_all_benchmarks():
    print("=" * 76)
    print("      AEROFLOW: COMPUTATIONAL FLUID DYNAMICS BENCHMARK SUITE")
    print("=" * 76)
    bench_lbm_throughput()
    bench_navier_stokes_projection()
    bench_semi_lagrangian_advection()
    bench_naca_rasterization()
    bench_braille_flow_visualizer()
    print("\n" + "=" * 76)
    print("      ALL AEROFLOW BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    run_all_benchmarks()
