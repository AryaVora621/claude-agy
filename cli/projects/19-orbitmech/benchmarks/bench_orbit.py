"""
OrbitMech: Astrodynamics & Interplanetary Trajectory Optimization Benchmarks.
Measures Kepler solver throughput, Universal variable propagation,
Lambert targeting, Symplectic/RK45 integration, and Braille 3D projections.
"""

import sys
import os
import time
import math
import random

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
    SUN,
)
from orbitmech.kepler import (
    solve_kepler_equation,
    state_to_orbital_elements,
    orbital_elements_to_state,
    propagate_keplerian,
)
from orbitmech.lambert import (
    propagate_universal,
    solve_lambert,
)
from orbitmech.propagator import (
    PerturbationConfig,
    propagate_symplectic_verlet,
    propagate_rk45,
)
from orbitmech.maneuvers import hohmann_transfer
from orbitmech.visualizer import render_orbit_scene, Camera3D


def bench_kepler_solver():
    print("\n--- 1. Danby Kepler Equation Solver Throughput ---")
    trials = 100_000
    random.seed(42)
    m_list = [random.uniform(0.0, 2.0 * math.pi) for _ in range(trials)]
    e_list = [random.uniform(0.01, 0.95) for _ in range(trials)]

    t0 = time.perf_counter()
    for m, e in zip(m_list, e_list):
        solve_kepler_equation(m, e)
    elapsed = time.perf_counter() - t0
    solves_sec = trials / elapsed
    us_per_solve = (elapsed / trials) * 1e6
    print(f"Danby Kepler Equation: {trials:,} roots solved in {elapsed*1000:.2f} ms")
    print(f"Throughput: {solves_sec:,.0f} solves/sec ({us_per_solve:.2f} µs/solve)")


def bench_state_elements_conversion():
    print("\n--- 2. State Vector <-> Orbital Elements Transformations ---")
    trials = 50_000
    elements = ClassicalOrbitalElements(
        a=7000.0,
        e=0.05,
        i=math.radians(28.5),
        raan=math.radians(45.0),
        arg_peri=math.radians(30.0),
        true_anomaly=math.radians(15.0),
        mu=EARTH.mu,
    )
    state = orbital_elements_to_state(elements)

    t0 = time.perf_counter()
    for _ in range(trials):
        state_to_orbital_elements(state, mu=EARTH.mu)
    elapsed = time.perf_counter() - t0
    conversions_sec = trials / elapsed
    print(f"State -> COE: {trials:,} conversions in {elapsed*1000:.2f} ms ({conversions_sec:,.0f} conversions/sec)")


def bench_universal_propagation():
    print("\n--- 3. Universal Variable Conic Propagation ---")
    trials = 20_000
    elements = ClassicalOrbitalElements(
        a=8000.0,
        e=0.15,
        i=math.radians(45.0),
        raan=0.0,
        arg_peri=0.0,
        true_anomaly=0.0,
        mu=EARTH.mu,
    )
    state = orbital_elements_to_state(elements)

    t0 = time.perf_counter()
    for _ in range(trials):
        propagate_universal(state, dt=1800.0, mu=EARTH.mu)
    elapsed = time.perf_counter() - t0
    props_sec = trials / elapsed
    us_per_prop = (elapsed / trials) * 1e6
    print(f"Universal Propagation: {trials:,} steps in {elapsed*1000:.2f} ms ({props_sec:,.0f} propagations/sec, {us_per_prop:.2f} µs/step)")


def bench_lambert_solver():
    print("\n--- 4. Lambert Boundary Value Targeting Solver ---")
    trials = 2_000
    r1 = Vector3(7000.0, 0.0, 0.0)
    r2 = Vector3(0.0, 8500.0, 1000.0)
    tof = 2400.0  # 40 minutes

    t0 = time.perf_counter()
    for _ in range(trials):
        solve_lambert(r1, r2, tof, mu=EARTH.mu)
    elapsed = time.perf_counter() - t0
    solves_sec = trials / elapsed
    us_per_solve = (elapsed / trials) * 1e6
    print(f"Lambert Targeting: {trials:,} transfers solved in {elapsed*1000:.2f} ms ({solves_sec:,.0f} solves/sec, {us_per_solve:.2f} µs/solve)")


def bench_numerical_integration():
    print("\n--- 5. Symplectic Störmer-Verlet & Adaptive RK45 Integrators ---")
    elements = ClassicalOrbitalElements(
        a=7200.0,
        e=0.08,
        i=math.radians(51.6),
        raan=0.0,
        arg_peri=0.0,
        true_anomaly=0.0,
        mu=EARTH.mu,
    )
    state = orbital_elements_to_state(elements)
    # Propagate 20 orbits (~33 hours)
    duration = 20.0 * elements.period

    # Symplectic Verlet
    t0 = time.perf_counter()
    res_verlet = propagate_symplectic_verlet(
        initial_state=state,
        duration=duration,
        dt=10.0,
        config=PerturbationConfig(enable_j2=True),
    )
    elapsed_verlet = time.perf_counter() - t0
    steps_sec = res_verlet.step_count / elapsed_verlet
    print(f"Symplectic Verlet (with J2): {res_verlet.step_count:,} steps in {elapsed_verlet*1000:.2f} ms ({steps_sec:,.0f} steps/sec)")

    # Adaptive RK45
    t0 = time.perf_counter()
    res_rk45 = propagate_rk45(
        initial_state=state,
        duration=duration,
        initial_dt=30.0,
        tolerance=1e-8,
        config=PerturbationConfig(enable_j2=True),
    )
    elapsed_rk45 = time.perf_counter() - t0
    rk45_steps_sec = res_rk45.step_count / elapsed_rk45
    print(f"Adaptive RK45 (with J2):     {res_rk45.step_count:,} steps in {elapsed_rk45*1000:.2f} ms ({rk45_steps_sec:,.0f} steps/sec)")


def bench_braille_orbit_rendering():
    print("\n--- 6. Sub-Pixel Unicode Braille 3D Orbit Projections ---")
    elements = ClassicalOrbitalElements(
        a=8000.0,
        e=0.2,
        i=math.radians(35.0),
        raan=math.radians(45.0),
        arg_peri=math.radians(20.0),
        true_anomaly=0.0,
        mu=EARTH.mu,
    )
    state = orbital_elements_to_state(elements)
    # Generate 100-point trajectory
    states = [propagate_keplerian(state, dt=i * 60.0, mu=EARTH.mu) for i in range(100)]
    cam = Camera3D(azimuth_deg=40.0, elevation_deg=25.0, scale=200.0)

    t0 = time.perf_counter()
    frames = 100
    for _ in range(frames):
        render_orbit_scene(states, central_body=EARTH, camera=cam, canvas_chars_x=60, canvas_chars_y=20)
    elapsed = time.perf_counter() - t0
    fps = frames / elapsed
    print(f"Braille Orbit Projections: {frames} full frames rendered in {elapsed*1000:.2f} ms ({fps:.1f} FPS)")


def run_all_benchmarks():
    print("=" * 72)
    print("        ORBITMECH: ASTRODYNAMICS & TRAJECTORY OPTIMIZATION BENCHMARKS")
    print("=" * 72)
    bench_kepler_solver()
    bench_state_elements_conversion()
    bench_universal_propagation()
    bench_lambert_solver()
    bench_numerical_integration()
    bench_braille_orbit_rendering()
    print("\n" + "=" * 72)
    print("        ALL ORBITMECH BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    run_all_benchmarks()
