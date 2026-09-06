"""Performance microbenchmark suite for ChromaSplat 3D Gaussian Splatting engine.

Measures raw computational throughput and execution latencies across all core
graphics kernels: 3D covariance synthesis, spherical harmonics radiance evaluation,
perspective projective Jacobian and 2D EWA covariance, tiled volume rasterization,
and sub-pixel Unicode Braille terminal rendering.
"""

from __future__ import annotations

import math
import time
from typing import List

from chromasplat.gaussian import Gaussian3D, Quaternion
from chromasplat.projection import Camera, ProjectionEngine
from chromasplat.rasterizer import VolumeRasterizer
from chromasplat.renderer import GaussianRenderer
from chromasplat.scene import SceneFactory
from chromasplat.spherical_harmonics import create_specular_sh, eval_sh
from chromasplat.visualizer import BrailleCanvas


def benchmark_3d_covariance(num_iterations: int = 200_000) -> None:
    """Benchmark 3D covariance matrix computation: Sigma = R * S * S^T * R^T."""
    q = Quaternion.from_euler(0.4, 0.7, 0.2)
    g = Gaussian3D(position=(0.0, 0.0, 0.0), scale=(0.2, 0.4, 0.6), rotation=q)

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = g.compute_covariance_3d()
    elapsed = time.perf_counter() - t0

    throughput = num_iterations / max(1e-9, elapsed)
    latency_us = (elapsed / num_iterations) * 1e6
    print(f"3D Covariance Matrix:     {throughput:>12,.1f} evals/sec  ({latency_us:6.2f} us/eval)")


def benchmark_spherical_harmonics(num_iterations: int = 200_000) -> None:
    """Benchmark real spherical harmonics degree-2 directional color evaluation."""
    sh = create_specular_sh((0.2, 0.4, 0.8), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0), degree=2)
    ray_dir = (0.577, 0.577, 0.577)

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = eval_sh(2, sh, ray_dir)
    elapsed = time.perf_counter() - t0

    throughput = num_iterations / max(1e-9, elapsed)
    latency_us = (elapsed / num_iterations) * 1e6
    print(f"Spherical Harmonics (Deg 2):{throughput:>10,.1f} evals/sec  ({latency_us:6.2f} us/eval)")


def benchmark_projection_engine(num_iterations: int = 100_000) -> None:
    """Benchmark perspective projection, Jacobian, and 2D EWA screen covariance."""
    cam = Camera.orbit(width=160, height=80, azimuth_deg=45.0, elevation_deg=20.0, distance=3.5)
    basis = cam.get_view_basis()
    engine = ProjectionEngine(anti_aliasing_filter=0.3)
    g = Gaussian3D.isotropic(position=(0.2, 0.4, -0.1), radius=0.3)

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = engine.project_single(g, cam, basis, gaussian_idx=0, sh_degree=1)
    elapsed = time.perf_counter() - t0

    throughput = num_iterations / max(1e-9, elapsed)
    latency_us = (elapsed / num_iterations) * 1e6
    print(f"2D EWA Projection & Cov:  {throughput:>12,.1f} splats/sec ({latency_us:6.2f} us/splat)")


def benchmark_volume_rasterizer(num_frames: int = 200) -> None:
    """Benchmark tiled volume alpha compositing rasterizer."""
    scene = SceneFactory.orbiting_rings(num_planet_splats=60, num_ring_splats=140)
    cam = Camera.orbit(width=128, height=64, azimuth_deg=30.0, elevation_deg=15.0, distance=3.2)
    engine = ProjectionEngine()
    rasterizer = VolumeRasterizer(tile_size=16)

    projected = engine.project_scene(scene.gaussians, cam, sh_degree=1)

    t0 = time.perf_counter()
    for _ in range(num_frames):
        _ = rasterizer.rasterize(cam.width, cam.height, projected, use_tiling=True)
    elapsed = time.perf_counter() - t0

    fps = num_frames / max(1e-9, elapsed)
    frame_ms = (elapsed / num_frames) * 1000.0
    print(f"Tiled Volume Rasterizer:  {fps:>12,.1f} FPS        ({frame_ms:6.2f} ms/frame)")


def benchmark_braille_canvas(num_frames: int = 300) -> None:
    """Benchmark sub-pixel Unicode Braille terminal canvas rasterization."""
    canvas = BrailleCanvas(char_width=64, char_height=25)
    scene = SceneFactory.orbiting_rings(num_planet_splats=30, num_ring_splats=60)
    cam = Camera.orbit(width=canvas.pixel_width, height=canvas.pixel_height, azimuth_deg=25.0, elevation_deg=15.0, distance=3.0)
    renderer = GaussianRenderer(sh_degree=0)
    res = renderer.render(scene, cam)

    t0 = time.perf_counter()
    for _ in range(num_frames):
        _ = canvas.rasterize_render_result(res)
    elapsed = time.perf_counter() - t0

    fps = num_frames / max(1e-9, elapsed)
    frame_ms = (elapsed / num_frames) * 1000.0
    print(f"Sub-Pixel Braille Canvas: {fps:>12,.1f} FPS        ({frame_ms:6.2f} ms/frame)")


def benchmark_end_to_end_pipeline(num_frames: int = 150) -> None:
    """Benchmark complete end-to-end rendering pipeline with dynamic camera orbit."""
    scene = SceneFactory.orbiting_rings(num_planet_splats=50, num_ring_splats=100)
    canvas = BrailleCanvas(char_width=60, char_height=24)
    renderer = GaussianRenderer(sh_degree=1)

    t0 = time.perf_counter()
    for i in range(num_frames):
        azimuth = (360.0 / num_frames) * i
        cam = Camera.orbit(
            width=canvas.pixel_width,
            height=canvas.pixel_height,
            azimuth_deg=azimuth,
            elevation_deg=20.0,
            distance=3.5,
        )
        res = renderer.render(scene, cam)
        _ = canvas.rasterize_render_result(res)
    elapsed = time.perf_counter() - t0

    fps = num_frames / max(1e-9, elapsed)
    frame_ms = (elapsed / num_frames) * 1000.0
    print(f"Full End-to-End Pipeline:{fps:>13,.1f} FPS        ({frame_ms:6.2f} ms/frame)")


def run_all_benchmarks() -> None:
    """Execute full benchmark suite."""
    print("=" * 68)
    print(" CHROMASPLAT 3D GAUSSIAN SPLATTING PERFORMANCE BENCHMARK SUITE")
    print("=" * 68)
    benchmark_3d_covariance()
    benchmark_spherical_harmonics()
    benchmark_projection_engine()
    benchmark_volume_rasterizer()
    benchmark_braille_canvas()
    benchmark_end_to_end_pipeline()
    print("=" * 68)


if __name__ == "__main__":
    run_all_benchmarks()
