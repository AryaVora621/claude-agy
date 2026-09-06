#!/usr/bin/env python3
"""
PhotonPBR: Cornell Box 3D Scene Terminal Renderer.
Renders the classic Cornell Box with:
- Red left wall, Green right wall, White floor/ceiling/back wall
- Ceiling rectangular emissive area light source
- Polished dielectric glass sphere (refraction & caustics)
- Brushed metallic sphere (reflection & roughness)
- Accelerated via Surface Area Heuristic (SAH) BVH Tree
Outputs directly to terminal in 24-bit TrueColor and ASCII art!
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from photon.vec3 import Vec3, Color
from photon.geometry import Sphere, Triangle, HittableList
from photon.bvh import BVHNode
from photon.material import Lambertian, Metal, Dielectric, DiffuseLight
from photon.camera import Camera
from photon.tracer import PathTracer
from photon.framebuffer import Framebuffer


def build_cornell_box() -> HittableList:
    """Construct 3D Cornell Box geometry."""
    world = HittableList()

    # Materials
    red = Lambertian(Color(0.65, 0.05, 0.05))
    white = Lambertian(Color(0.73, 0.73, 0.73))
    green = Lambertian(Color(0.12, 0.45, 0.15))
    light = DiffuseLight(Color(15.0, 15.0, 15.0))
    glass = Dielectric(1.5)
    gold = Metal(Color(0.85, 0.75, 0.35), fuzz=0.05)

    # Floor (y = 0)
    world.add(Triangle(Vec3(0, 0, 0), Vec3(555, 0, 0), Vec3(555, 0, 555), white))
    world.add(Triangle(Vec3(0, 0, 0), Vec3(555, 0, 555), Vec3(0, 0, 555), white))

    # Ceiling (y = 555)
    world.add(Triangle(Vec3(0, 555, 0), Vec3(555, 555, 555), Vec3(555, 555, 0), white))
    world.add(Triangle(Vec3(0, 555, 0), Vec3(0, 555, 555), Vec3(555, 555, 555), white))

    # Back wall (z = 555)
    world.add(Triangle(Vec3(0, 0, 555), Vec3(555, 0, 555), Vec3(555, 555, 555), white))
    world.add(Triangle(Vec3(0, 0, 555), Vec3(555, 555, 555), Vec3(0, 555, 555), white))

    # Left wall - Red (x = 555)
    world.add(Triangle(Vec3(555, 0, 0), Vec3(555, 555, 555), Vec3(555, 0, 555), red))
    world.add(Triangle(Vec3(555, 0, 0), Vec3(555, 555, 0), Vec3(555, 555, 555), red))

    # Right wall - Green (x = 0)
    world.add(Triangle(Vec3(0, 0, 0), Vec3(0, 0, 555), Vec3(0, 555, 555), green))
    world.add(Triangle(Vec3(0, 0, 0), Vec3(0, 555, 555), Vec3(0, 555, 0), green))

    # Ceiling Area Light
    world.add(Triangle(Vec3(213, 554, 227), Vec3(343, 554, 227), Vec3(343, 554, 332), light))
    world.add(Triangle(Vec3(213, 554, 227), Vec3(343, 554, 332), Vec3(213, 554, 332), light))

    # Sphere 1: Dielectric Glass Sphere
    world.add(Sphere(Vec3(160, 100, 180), 100, glass))

    # Sphere 2: Polished Gold Metal Sphere
    world.add(Sphere(Vec3(390, 110, 320), 110, gold))

    return world


def render_scene(width: int = 64, height: int = 48, samples: int = 16):
    print("=" * 70)
    print("        PHOTON-PBR: MONTE CARLO PATH TRACER & CORNELL BOX")
    print("=" * 70)

    print(f"\nBuilding Scene Geometry & SAH BVH Acceleration Tree...")
    raw_world = build_cornell_box()
    t0_bvh = time.perf_counter()
    bvh_world = BVHNode(list(raw_world.objects))
    bvh_time = time.perf_counter() - t0_bvh
    print(f"BVH Built successfully: {len(raw_world.objects)} primitives in {bvh_time * 1000:.2f} ms")

    # Camera setup (looking into box from front)
    lookfrom = Vec3(278, 278, -800)
    lookat = Vec3(278, 278, 0)
    vup = Vec3(0, 1, 0)
    vfov = 40.0
    aspect_ratio = width / height

    cam = Camera(
        lookfrom=lookfrom,
        lookat=lookat,
        vup=vup,
        vfov_degrees=vfov,
        aspect_ratio=aspect_ratio,
        aperture=0.0,
        focus_dist=10.0
    )

    tracer = PathTracer(max_depth=6)

    print(f"Rendering {width}x{height} image at {samples} samples/pixel...")
    t0 = time.perf_counter()

    pixels = tracer.render(
        world=bvh_world,
        camera=cam,
        width=width,
        height=height,
        samples_per_pixel=samples,
        background=Color(0.0, 0.0, 0.0)
    )

    render_time = time.perf_counter() - t0
    total_rays = tracer.rays_cast
    rps = total_rays / render_time

    fb = Framebuffer(width, height, pixels)

    print(f"\nRender Complete in {render_time:.2f}s ({total_rays:,} rays, {rps:,.0f} rays/sec)\n")
    print("--- 24-BIT TRUECOLOR TERMINAL RENDER ---")
    print(fb.to_ansi_truecolor(samples_per_pixel=samples))

    print("\n--- HIGH-CONTRAST ASCII LUMINANCE RENDER ---")
    print(fb.to_ascii(samples_per_pixel=samples))
    print("=" * 70)


if __name__ == "__main__":
    # Standard terminal preview size
    w = 60
    h = 36
    s = 8
    if len(sys.argv) > 1:
        w = int(sys.argv[1])
    if len(sys.argv) > 2:
        h = int(sys.argv[2])
    if len(sys.argv) > 3:
        s = int(sys.argv[3])
    render_scene(width=w, height=h, samples=s)
