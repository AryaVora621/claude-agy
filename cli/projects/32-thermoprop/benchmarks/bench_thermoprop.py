"""
Performance Microbenchmarks for ThermoProp Aerothermodynamics & Rocket Engine.

Evaluates throughput of:
  - Isentropic compressible flow state computations
  - High-order Halley Area-Mach root-finding inversions
  - Rankine-Hugoniot normal and oblique shock wave solves
  - 2D Method of Characteristics (MOC) supersonic nozzle syntheses
  - Rocket engine thermochemistry and performance evaluations
  - Bartz convective heat transfer and coupled 1D cooling solves
  - Exhaust plume shock cell and Mach diamond field calculations
  - High-resolution sub-pixel Braille rendering
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermoprop.cooling import (
    RegenerativeCoolingEngine,
    WallMaterial,
)
from thermoprop.gas_dynamics import (
    CompressibleFlowEngine,
    GasProperties,
)
from thermoprop.moc_nozzle import MethodOfCharacteristicsNozzle
from thermoprop.plume import ExhaustPlumeEngine
from thermoprop.propulsion import (
    PropellantLibrary,
    RocketPropulsionEngine,
)
from thermoprop.visualizer import (
    BrailleCanvas,
    RocketVisualizer,
)


def benchmark_isentropic_flow(iterations: int = 50000) -> float:
    """Benchmark 1D isentropic compressible flow state computations."""
    gas = GasProperties.air()
    flow = CompressibleFlowEngine(gas)

    t0 = time.perf_counter()
    for i in range(iterations):
        mach = 0.1 + (i % 40) * 0.1
        _ = flow.isentropic_state_from_stagnation(mach, t_total=3000.0, p_total=100.0e5)
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Isentropic Flow States : {ops_per_sec:12,.0f} evals/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_area_mach_halley(iterations: int = 25000) -> float:
    """Benchmark high-order Halley root finding for Area-Mach inversion."""
    gas = GasProperties.air()
    flow = CompressibleFlowEngine(gas)

    t0 = time.perf_counter()
    for i in range(iterations):
        area_ratio = 1.1 + (i % 50) * 0.8
        _ = flow.mach_from_area_ratio(area_ratio, supersonic=True)
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Area-Mach Halley Solver: {ops_per_sec:12,.0f} solves/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_shock_waves(iterations: int = 25000) -> float:
    """Benchmark Rankine-Hugoniot normal and oblique shock wave solutions."""
    gas = GasProperties.air()
    flow = CompressibleFlowEngine(gas)

    t0 = time.perf_counter()
    for i in range(iterations):
        mach = 1.5 + (i % 30) * 0.1
        _ = flow.normal_shock(mach, p_upstream=101325.0, t_upstream=288.15)
        _ = flow.oblique_shock(mach, theta=math.radians(5.0 + (i % 10)), weak_shock=True)
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Shock Wave Solvers     : {ops_per_sec:12,.0f} pairs/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_moc_nozzle(iterations: int = 150) -> float:
    """Benchmark 2D Method of Characteristics supersonic nozzle design."""
    gas = GasProperties.methalox()
    moc = MethodOfCharacteristicsNozzle(gas)

    t0 = time.perf_counter()
    for i in range(iterations):
        mach_target = 2.0 + (i % 10) * 0.1
        _ = moc.design_minimum_length_nozzle(
            target_exit_mach=mach_target,
            throat_height=0.08,
            num_expansion_waves=8,
        )
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] 2D MOC Nozzle Designer : {ops_per_sec:12,.0f} nozzles/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_rocket_engine_eval(iterations: int = 25000) -> float:
    """Benchmark rocket propulsion thermochemistry and performance state."""
    prop = PropellantLibrary.methalox()
    engine = RocketPropulsionEngine(prop)

    t0 = time.perf_counter()
    for i in range(iterations):
        pc = 50.0e5 + (i % 50) * 2.0e5
        _ = engine.evaluate_engine(
            chamber_pressure=pc,
            throat_radius=0.10,
            expansion_ratio=35.0,
        )
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Rocket Engine State    : {ops_per_sec:12,.0f} engines/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_bartz_cooling(iterations: int = 20000) -> float:
    """Benchmark Bartz convective heat transfer and coupled 1D thermal network."""
    prop = PropellantLibrary.methalox()
    engine = RocketPropulsionEngine(prop)
    state = engine.evaluate_engine(100.0e5, 0.10, 40.0)
    cooling = RegenerativeCoolingEngine(state, prop.gas, WallMaterial.copper_cucrzr())

    t0 = time.perf_counter()
    for i in range(iterations):
        mach = 1.0 + (i % 25) * 0.1
        radius = 0.10 * (1.0 + (i % 25) * 0.08)
        _ = cooling.solve_station_thermal_equilibrium(0.1, radius, mach, 0.10, coolant_temp=120.0)
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Bartz Cooling Engine   : {ops_per_sec:12,.0f} stations/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def benchmark_plume_and_braille(iterations: int = 500) -> float:
    """Benchmark exhaust plume generation and sub-pixel Braille rendering."""
    gas = GasProperties.methalox()
    plume_engine = ExhaustPlumeEngine(gas)
    canvas = BrailleCanvas(char_width=76, char_height=30)
    viz = RocketVisualizer(canvas)

    t0 = time.perf_counter()
    for i in range(iterations):
        plume = plume_engine.simulate_plume(
            x_exit=2.0,
            exit_radius=0.5,
            exit_mach=2.8,
            p_exit=80000.0,
            chamber_pressure=80.0e5,
            p_ambient=101325.0,
            plume_length=6.0,
            num_cells=4,
        )
        canvas.clear()
        viz.draw_exhaust_plume(plume)
        _ = canvas.render_to_string()
    t1 = time.perf_counter()

    ops_per_sec = iterations / (t1 - t0)
    print(f" [BENCH] Plume + Braille Render : {ops_per_sec:12,.0f} frames/sec ({iterations} iterations in {t1 - t0:.4f}s)")
    return ops_per_sec


def main() -> None:
    print("=" * 70)
    print("      THERMOPROP : COMPRESSIBLE GAS DYNAMICS & PROPULSION BENCHMARKS")
    print("=" * 70)
    benchmark_isentropic_flow()
    benchmark_area_mach_halley()
    benchmark_shock_waves()
    benchmark_moc_nozzle()
    benchmark_rocket_engine_eval()
    benchmark_bartz_cooling()
    benchmark_plume_and_braille()
    print("=" * 70)
    print("All microbenchmarks completed successfully.")


if __name__ == "__main__":
    main()
