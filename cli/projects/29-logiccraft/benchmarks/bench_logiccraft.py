"""Comprehensive Performance Microbenchmarks for LogicCraft EDA Engine.

Measures throughput and execution latency across:
1. ROBDD ITE Operations & Shannon Decomposition (Ops/sec)
2. AIG Structural Hashing (Strashing) & Logic Simulation (Nodes/sec)
3. Liberty NLDM 2D Bilinear Interpolation (Lookups/sec)
4. DAGON Dynamic Programming Technology Mapping (Gates/sec)
5. Static Timing Analysis Forward/Backward Propagation (Pins/sec)
6. Sub-Pixel Braille Canvas Graphics Rendering (FPS)
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure logiccraft is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logiccraft.bdd import BDDManager
from logiccraft.aig import AIGGraph
from logiccraft.liberty import get_default_library
from logiccraft.techmap import TechMapper
from logiccraft.sta import StaticTimingAnalyzer
from logiccraft.visualizer import BrailleCanvas, CircuitVisualizer


def run_benchmark(name: str, fn, iterations: int = 1000) -> Tuple[float, float]:
    """Execute benchmark measuring total elapsed time and operations/second."""
    # Warmup
    fn()

    start_time = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start_time

    ops_per_sec = iterations / elapsed if elapsed > 0 else 0.0
    return elapsed, ops_per_sec


def bench_bdd() -> Tuple[float, float]:
    """Benchmark ROBDD ITE and Boolean operations."""
    mgr = BDDManager([f"v{i}" for i in range(16)])
    v = [mgr.var(f"v{i}") for i in range(16)]

    def bdd_workload() -> None:
        # Build 16-variable parity / majority tree
        res = v[0]
        for i in range(1, 8):
            res = mgr.xor_(res, v[i])
        _ = mgr.sat_count(res, total_vars=16)

    return run_benchmark("ROBDD Synthesis & SAT Counting", bdd_workload, iterations=500)


def bench_aig_strashing() -> Tuple[float, float]:
    """Benchmark AIG construction with structural hashing."""
    def aig_workload() -> None:
        aig = AIGGraph()
        pis = [aig.create_pi(f"in_{i}") for i in range(8)]
        curr = pis[0]
        for i in range(1, 8):
            curr = aig.and_(curr, pis[i])
            # Induce redundant lookups for strashing hit testing
            _ = aig.and_(curr, pis[i])
        aig.set_output("out", curr)

    return run_benchmark("AIG Construction & Strashing", aig_workload, iterations=2000)


def bench_nldm_interpolation() -> Tuple[float, float]:
    """Benchmark Liberty NLDM 2D bilinear interpolation."""
    lib = get_default_library()
    inv_arc = lib.get_cell("INV_X1").timing_arcs[0]

    def nldm_workload() -> None:
        # Perform 20 lookups across varying slew and load
        for s in [15.0, 25.0, 45.0, 80.0]:
            for c in [2.0, 5.0, 12.0, 25.0, 40.0]:
                _ = inv_arc.lookup_delay(s, c)
                _ = inv_arc.lookup_slew(s, c)

    return run_benchmark("NLDM 2D Bilinear Interpolation (20 lookups)", nldm_workload, iterations=2500)


def bench_techmap() -> Tuple[float, float]:
    """Benchmark DAGON technology mapping on multi-level AIG."""
    lib = get_default_library()
    mapper = TechMapper(lib, target_metric="area")

    # Construct synthetic 16-input logic tree
    aig = AIGGraph()
    pis = [aig.create_pi(f"in_{i}") for i in range(16)]
    stage1 = [aig.and_(pis[2 * i], pis[2 * i + 1]) for i in range(8)]
    stage2 = [aig.or_(stage1[2 * i], stage1[2 * i + 1]) for i in range(4)]
    stage3 = [aig.xor_(stage2[2 * i], stage2[2 * i + 1]) for i in range(2)]
    out = aig.and_(stage3[0], stage3[1])
    aig.set_output("Y", out)

    def map_workload() -> None:
        _ = mapper.map_aig(aig, module_name="bench_block")

    return run_benchmark("Technology Mapping (16-PI Network)", map_workload, iterations=300)


def bench_sta() -> Tuple[float, float]:
    """Benchmark Static Timing Analysis graph traversal."""
    lib = get_default_library()
    mapper = TechMapper(lib, target_metric="delay")

    aig = AIGGraph()
    pis = [aig.create_pi(f"in_{i}") for i in range(16)]
    stage1 = [aig.and_(pis[2 * i], pis[2 * i + 1]) for i in range(8)]
    stage2 = [aig.or_(stage1[2 * i], stage1[2 * i + 1]) for i in range(4)]
    stage3 = [aig.xor_(stage2[2 * i], stage2[2 * i + 1]) for i in range(2)]
    out = aig.and_(stage3[0], stage3[1])
    aig.set_output("Y", out)

    netlist = mapper.map_aig(aig, module_name="sta_bench")
    sta = StaticTimingAnalyzer(netlist)

    def sta_workload() -> None:
        _ = sta.run_sta(clock_period=1000.0)

    return run_benchmark("Static Timing Analysis (Full Pass)", sta_workload, iterations=800)


def bench_braille_rendering() -> Tuple[float, float]:
    """Benchmark Sub-Pixel Braille Canvas rasterization."""
    def braille_workload() -> None:
        canvas = BrailleCanvas(char_width=70, char_height=15)
        for i in range(canvas.pixel_width):
            y = int((canvas.pixel_height - 1) * (0.5 + 0.4 * (i % 7) / 7.0))
            canvas.set_pixel(i, y)
        _ = canvas.render()

    return run_benchmark("Sub-Pixel Braille Canvas Render", braille_workload, iterations=1500)


def main() -> None:
    print("=" * 76)
    print("LOGICCRAFT: EDA LOGIC SYNTHESIS & STA PERFORMANCE BENCHMARK")
    print("=" * 76)

    benchmarks = [
        ("ROBDD Synthesis & SAT Count (16-var)", bench_bdd),
        ("AIG Construction & Strashing", bench_aig_strashing),
        ("Liberty NLDM 2D Bilinear Interpolation", bench_nldm_interpolation),
        ("DAGON Dynamic Programming Tech Mapping", bench_techmap),
        ("Static Timing Analysis (AT/RAT/Slack)", bench_sta),
        ("Sub-Pixel Braille Canvas Rasterizer", bench_braille_rendering),
    ]

    for label, bfn in benchmarks:
        elapsed, ops_sec = bfn()
        print(f"  {label:<46}: {ops_sec:10.1f} ops/sec ({elapsed * 1000:.2f} ms total)")

    print("=" * 76)
    print("BENCHMARK SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 76)


if __name__ == "__main__":
    main()
