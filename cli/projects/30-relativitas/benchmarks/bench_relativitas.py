"""Comprehensive Performance Microbenchmarks for Relativitas.

Measures throughput and latency for:
1. Spacetime metric tensors and Christoffel connection evaluation
2. 8-state first-order geodesic RK4 integrator step throughput
3. Accretion disk kinematics, Keplerian velocity, and relativistic Doppler shift calculation
4. Curved spacetime backward ray tracing throughput (rays/sec)
5. 2x4 sub-pixel Unicode Braille canvas rasterization FPS
"""

from __future__ import annotations
import math
import time
from typing import Callable, Tuple

from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.geodesic import GeodesicIntegrator, GeodesicState
from relativitas.accretion import AccretionDisk
from relativitas.raytracer import Camera, RayTracer
from relativitas.visualizer import BrailleCanvas, BlackHoleVisualizer


def run_benchmark(name: str, fn: Callable[[], int], target_duration: float = 1.0) -> Tuple[int, float, float]:
    """Execute a benchmark function repeatedly and report total ops, duration, and ops/sec."""
    # Warmup
    fn()

    start_time = time.perf_counter()
    total_ops = 0
    iterations = 0

    while True:
        ops = fn()
        total_ops += ops
        iterations += 1
        elapsed = time.perf_counter() - start_time
        if elapsed >= target_duration and iterations >= 3:
            break

    rate = total_ops / elapsed
    return (total_ops, elapsed, rate)


def bench_schwarzschild_christoffel() -> int:
    metric = SchwarzschildMetric(mass=1.0)
    batch = 5000
    for i in range(batch):
        r = 3.0 + (i % 100) * 0.1
        gamma = metric.christoffel_symbols((0.0, r, 1.2, 0.5))
    return batch


def bench_kerr_christoffel() -> int:
    metric = KerrMetric(mass=1.0, spin=0.85)
    batch = 500
    for i in range(batch):
        r = 3.0 + (i % 50) * 0.1
        gamma = metric.christoffel_symbols((0.0, r, 1.3, 0.4))
    return batch


def bench_rk4_geodesic_steps() -> int:
    metric = SchwarzschildMetric(mass=1.0)
    integrator = GeodesicIntegrator(metric, is_null=True)
    state = GeodesicState(0.0, 15.0, 1.4, 0.0, 1.0, -0.3, 0.02, 0.05)
    batch = 2000
    h = 0.05
    for _ in range(batch):
        state = integrator.step_rk4(state, h)
    return batch


def bench_accretion_radiative_transfer() -> int:
    metric = KerrMetric(mass=1.0, spin=0.9)
    disk = AccretionDisk(metric)
    batch = 5000
    for i in range(batch):
        r = 4.0 + (i % 80) * 0.1
        p_contra = (1.0, 0.0, 0.0, 0.03 * ((i % 5) - 2))
        res = disk.evaluate_radiative_transfer(r, phi=0.0, p_contra=p_contra)
    return batch


def bench_raytracer_throughput() -> int:
    metric = SchwarzschildMetric(mass=1.0)
    rt = RayTracer(metric, max_steps=100)
    cam = Camera(r=20.0, theta=1.396, fov_degrees=45.0)
    # Render small grid: 12x8 = 96 rays
    fb = rt.render_frame(cam, width=12, height=8)
    return fb.total_rays


def bench_braille_rasterization() -> int:
    width, height = 80, 40
    canvas = BrailleCanvas(width, height)
    batch = 200
    for _ in range(batch):
        for y in range(height):
            for x in range(width):
                canvas.set_pixel(x, y, (200, 120, 50), active=True)
        _ = canvas.render_ansi(true_color=True)
    return batch


def main() -> None:
    print("=" * 80)
    print("RELATIVITAS: GENERAL RELATIVITY ENGINE PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print(f"{'Benchmark Target':<42} | {'Ops / Units':<12} | {'Time (s)':<10} | {'Throughput':<15}")
    print("-" * 80)

    benchmarks = [
        ("Schwarzschild Christoffel Symbols", bench_schwarzschild_christoffel),
        ("Kerr Numerical Christoffel Symbols", bench_kerr_christoffel),
        ("Geodesic 8-State RK4 Steps", bench_rk4_geodesic_steps),
        ("Accretion Radiative Transfer & Redshift", bench_accretion_radiative_transfer),
        ("Curved Spacetime Ray Tracing (Rays)", bench_raytracer_throughput),
        ("Sub-Pixel Braille Frame Rasterization", bench_braille_rasterization),
    ]

    for label, fn in benchmarks:
        ops, duration, rate = run_benchmark(label, fn, target_duration=0.8)
        unit = "frames/s" if "Rasterization" in label else "ops/s" if "Symbols" in label or "Transfer" in label or "Steps" in label else "rays/s"
        print(f"{label:<42} | {ops:<12} | {duration:<10.4f} | {rate:<12.1f} {unit}")

    print("=" * 80)
    print("All performance microbenchmarks completed successfully.")


if __name__ == "__main__":
    main()
