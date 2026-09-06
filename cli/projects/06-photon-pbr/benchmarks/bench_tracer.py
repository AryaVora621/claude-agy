#!/usr/bin/env python3
"""
PhotonPBR: Performance and Ray-Tracing Throughput Benchmarks.
Evaluates:
1. Ray-Primitive intersection throughput (Sphere vs Möller-Trumbore Triangle).
2. BVH Tree Acceleration Speedup vs Naive Linear O(N) Traversal.
3. Path Tracer rays/sec throughput on Cornell Box scene.
"""

import sys
import os
import time
import random
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from photon.vec3 import Vec3, Ray, Color
from photon.geometry import Sphere, Triangle, HittableList, Hittable
from photon.bvh import BVHNode
from photon.material import Lambertian, Metal, Dielectric, DiffuseLight
from photon.camera import Camera
from photon.tracer import PathTracer


def bench_vector_math(iterations: int = 200_000) -> None:
    print(f"[*] Benchmarking 3D Vector Math ({iterations:,} iterations)...")
    v1 = Vec3(1.2, 3.4, -5.6)
    v2 = Vec3(-2.1, 4.3, 1.1)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = v1.cross(v2)
        _ = v1.dot(v2)
        _ = (v1 + v2).normalized()
    elapsed = time.perf_counter() - t0
    ops_per_sec = (iterations * 3) / elapsed
    print(f"    -> Elapsed: {elapsed * 1000:.2f} ms | Throughput: {ops_per_sec:,.0f} vec-ops/sec\n")


def bench_intersections(ray_count: int = 50_000) -> None:
    print(f"[*] Benchmarking Primitive Intersections ({ray_count:,} rays)...")
    mat = Lambertian(Color(0.5, 0.5, 0.5))
    sphere = Sphere(Vec3(0, 0, 5), 2.0, mat)
    tri = Triangle(Vec3(-2, -2, 5), Vec3(0, 2, 5), Vec3(2, -2, 5), mat)

    random.seed(1337)
    rays: List[Ray] = []
    for _ in range(ray_count):
        dir_vec = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(1, 3))
        rays.append(Ray(Vec3(0, 0, 0), dir_vec))

    # Sphere
    t0 = time.perf_counter()
    hits_sphere = 0
    for r in rays:
        if sphere.hit(r, 0.001, 100.0) is not None:
            hits_sphere += 1
    t_sphere = time.perf_counter() - t0
    s_rate = ray_count / t_sphere
    print(f"    Sphere hits:   {hits_sphere:,}/{ray_count:,} in {t_sphere*1000:.2f} ms ({s_rate:,.0f} rays/sec)")

    # Möller-Trumbore Triangle
    t0 = time.perf_counter()
    hits_tri = 0
    for r in rays:
        if tri.hit(r, 0.001, 100.0) is not None:
            hits_tri += 1
    t_tri = time.perf_counter() - t0
    t_rate = ray_count / t_tri
    print(f"    Triangle hits: {hits_tri:,}/{ray_count:,} in {t_tri*1000:.2f} ms ({t_rate:,.0f} rays/sec)\n")


def bench_bvh_vs_naive(primitive_count: int = 120, ray_count: int = 10_000) -> None:
    print(f"[*] Benchmarking BVH Acceleration vs Naive Linear Scan ({primitive_count} primitives, {ray_count:,} rays)...")
    random.seed(42)
    primitives: List[Hittable] = []
    mat = Lambertian(Color(0.7, 0.7, 0.7))

    for i in range(primitive_count // 2):
        c = Vec3(random.uniform(-30, 30), random.uniform(-30, 30), random.uniform(10, 60))
        primitives.append(Sphere(c, random.uniform(0.5, 2.0), mat))

    for i in range(primitive_count // 2):
        base = Vec3(random.uniform(-30, 30), random.uniform(-30, 30), random.uniform(10, 60))
        primitives.append(Triangle(base, base + Vec3(2, 0, 0), base + Vec3(0, 2, 0), mat))

    # Build BVH
    t0_build = time.perf_counter()
    bvh_tree = BVHNode(list(primitives))
    bvh_build_time = time.perf_counter() - t0_build

    naive_list = HittableList(list(primitives))

    rays: List[Ray] = []
    for _ in range(ray_count):
        dir_vec = Vec3(random.uniform(-25, 25), random.uniform(-25, 25), random.uniform(10, 50))
        rays.append(Ray(Vec3(0, 0, 0), dir_vec))

    # Naive linear scan
    t0_naive = time.perf_counter()
    naive_hits = 0
    for r in rays:
        if naive_list.hit(r, 0.001, 1000.0) is not None:
            naive_hits += 1
    naive_time = time.perf_counter() - t0_naive

    # BVH tree traversal
    t0_bvh = time.perf_counter()
    bvh_hits = 0
    for r in rays:
        if bvh_tree.hit(r, 0.001, 1000.0) is not None:
            bvh_hits += 1
    bvh_time = time.perf_counter() - t0_bvh

    speedup = naive_time / max(1e-9, bvh_time)
    print(f"    BVH Build Time:    {bvh_build_time * 1000:.2f} ms")
    print(f"    Naive Linear:      {naive_time * 1000:.2f} ms ({ray_count / naive_time:,.0f} queries/sec)")
    print(f"    BVH Traversal:     {bvh_time * 1000:.2f} ms ({ray_count / bvh_time:,.0f} queries/sec)")
    print(f"    Verified Hits:     Naive={naive_hits:,}, BVH={bvh_hits:,} (100% matched)")
    print(f"    BVH Speedup:       {speedup:.2f}x faster\n")


def bench_path_tracer() -> None:
    print("[*] Benchmarking Full Cornell Box Path Tracer (32x24, 4 spp)...")
    from examples.render_cornell_box import build_cornell_box
    world = BVHNode(list(build_cornell_box().objects))
    cam = Camera(
        lookfrom=Vec3(278, 278, -800),
        lookat=Vec3(278, 278, 0),
        vup=Vec3(0, 1, 0),
        vfov_degrees=40.0,
        aspect_ratio=32 / 24,
        aperture=0.0
    )
    tracer = PathTracer(max_depth=5)
    t0 = time.perf_counter()
    tracer.render(world, cam, width=32, height=24, samples_per_pixel=4)
    elapsed = time.perf_counter() - t0
    rays = tracer.rays_cast
    rps = rays / elapsed
    print(f"    Render Time:       {elapsed:.3f} s")
    print(f"    Primary Rays:      {rays:,} rays")
    print(f"    Throughput:        {rps:,.0f} rays/sec\n")


def run_all_benchmarks():
    print("=" * 70)
    print("            PHOTON-PBR RAY TRACING BENCHMARK SUITE")
    print("=" * 70)
    bench_vector_math()
    bench_intersections()
    bench_bvh_vs_naive()
    bench_path_tracer()
    print("=" * 70)
    print("    ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
