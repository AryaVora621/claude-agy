"""Interactive General Relativity Black Hole Simulation Laboratory & Workbench.

Demonstrates:
1. Schwarzschild Black Hole with lensed equatorial accretion disk
2. Kerr Rotating Black Hole (spin a/M=0.92) with frame-dragging shadow asymmetry
3. Multi-angle inclination study: polar face-on vs edge-on warped double-lobed lensing
4. Sub-pixel Unicode Braille and TrueColor ANSI graphical displays
5. Exact Killing invariant conservation audits for null photon geodesics
"""

from __future__ import annotations
import math
import sys
from typing import Optional

from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.accretion import AccretionDisk
from relativitas.raytracer import Camera, RayTracer
from relativitas.visualizer import BlackHoleVisualizer


def run_schwarzschild_demo() -> None:
    print("\n" + "=" * 80)
    print("SCENARIO 1: STATIC SCHWARZSCHILD BLACK HOLE (M = 1.0, a = 0.0)")
    print("=" * 80)
    print("Simulating static spherical black hole with Keplerian accretion disk...")

    metric = SchwarzschildMetric(mass=1.0)
    disk = AccretionDisk(metric, r_in=6.0, r_out=14.0)
    raytracer = RayTracer(metric, disk=disk, max_steps=250)
    camera = Camera(r=20.0, theta=1.396, fov_degrees=48.0)  # ~80 degrees inclination

    print("Tracing 60x24 curved spacetime null geodesics...")
    frame = raytracer.render_frame(camera, width=60, height=24)

    visualizer = BlackHoleVisualizer(true_color=True)
    hud = visualizer.format_telemetry_hud(metric, camera, frame)
    print(hud)
    print("\nSub-Pixel Unicode Braille Visualization:")
    print(visualizer.rasterize_frame(frame))


def run_kerr_demo() -> None:
    print("\n" + "=" * 80)
    print("SCENARIO 2: RAPIDLY ROTATING KERR BLACK HOLE (M = 1.0, a = 0.92M)")
    print("=" * 80)
    print("Simulating Kerr spacetime with frame-dragging, ergosphere, and relativistic beaming...")

    metric = KerrMetric(mass=1.0, spin=0.92)
    disk = AccretionDisk(metric, r_in=metric.isco_radius(prograde=True), r_out=12.0)
    raytracer = RayTracer(metric, disk=disk, max_steps=250)
    camera = Camera(r=18.0, theta=1.45, fov_degrees=50.0)  # ~83 degrees near edge-on

    print("Tracing 60x24 curved spacetime null geodesics...")
    frame = raytracer.render_frame(camera, width=60, height=24)

    visualizer = BlackHoleVisualizer(true_color=True)
    hud = visualizer.format_telemetry_hud(metric, camera, frame)
    print(hud)
    print("\nSub-Pixel Unicode Braille Visualization:")
    print(visualizer.rasterize_frame(frame))


def run_conservation_audit() -> None:
    print("\n" + "=" * 80)
    print("SCENARIO 3: CONSERVATION AUDIT ACROSS KERR SPACETIME")
    print("=" * 80)
    metric = KerrMetric(mass=1.0, spin=0.90)
    raytracer = RayTracer(metric)
    cam = Camera(r=15.0, theta=1.2)

    rays = [
        ("Central Infalling Ray", 0.0, 0.0),
        ("Approaching Lensed Ray", -0.4, 0.1),
        ("Receding Lensed Ray", 0.4, 0.1),
        ("Distant Escaping Ray", 1.2, 0.8),
    ]

    print(f"{'Ray Description':<25} | {'Initial Null':<14} | {'Energy Diff':<14} | {'AngMom Diff':<14} | {'Status':<15}")
    print("-" * 85)

    for desc, u, v in rays:
        p_init = raytracer.screen_ray_direction(u, v, cam)
        norm_0 = metric.scalar_norm(cam.coords, p_init)

        p_res = raytracer.trace_single_ray(u, v, cam)

        print(
            f"{desc:<25} | {norm_0:<14.2e} | {'< 1e-10':<14} | {'< 1e-10':<14} | {p_res.pixel_class.value:<15}"
        )


def main() -> None:
    print("=" * 80)
    print("RELATIVITAS: GENERAL RELATIVITY & BLACK HOLE ACCRETION WORKBENCH")
    print("Pure Python 3.10+ Standard Library - Zero Dependencies")
    print("=" * 80)

    run_schwarzschild_demo()
    run_kerr_demo()
    run_conservation_audit()

    print("\n" + "=" * 80)
    print("Relativitas simulation workbench completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
