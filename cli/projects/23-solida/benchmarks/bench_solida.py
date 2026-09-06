"""
Solida: High-Performance Solid Modeling & NURBS CAD Kernel Microbenchmarks.
Benchmarks 3D differential geometry, NURBS Cox-de Boor recursion, B-Rep topology,
CSG BSP-tree Booleans, mesh tessellation, binary STL encoding, and Braille rasterization.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import sys
import os
import math
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Callable
from solida.geometry import Vector3D, Matrix4x4, Quaternion
from solida.nurbs import NURBSCurve, NURBSSurface
from solida.brep import build_solid_from_polygons
from solida.primitives import make_box, make_cylinder, make_sphere, make_torus
from solida.features import Sketch2D, extrude, revolve, loft
from solida.csg import csg_difference, csg_union
from solida.tessellation import tessellate_solid, export_stl_binary
from solida.visualizer import render_solid_cad, render_cad_hud
from solida.kernel import Part


def run_benchmark(name: str, fn: Callable[[], int], target_duration: float = 0.5) -> Dict[str, float]:
    """Runs benchmark function repeatedly for target duration and calculates ops/sec."""
    # Warmup
    fn()

    iterations = 0
    total_elements = 0
    start = time.perf_counter()
    while True:
        elems = fn()
        iterations += 1
        total_elements += elems
        now = time.perf_counter()
        if now - start >= target_duration and iterations >= 5:
            break

    elapsed = time.perf_counter() - start
    ops_per_sec = total_elements / elapsed
    us_per_op = (elapsed / total_elements) * 1e6

    return {
        "name": name,
        "iterations": iterations,
        "total_elements": total_elements,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "us_per_op": us_per_op,
    }


def main():
    print("=" * 80)
    print("  SOLIDA 3D CAD & NURBS GEOMETRIC MODELING KERNEL: PERFORMANCE BENCHMARK  ")
    print("=" * 80)

    benchmarks: List[Dict[str, float]] = []

    # 1. Vector3D Math: Dot, Cross, Normalize
    v1 = Vector3D(1.23, 4.56, 7.89)
    v2 = Vector3D(9.87, 6.54, 3.21)

    def bench_vector():
        count = 5000
        for _ in range(count):
            c = v1.cross(v2)
            d = v1.dot(v2)
            n = c.normalized()
        return count

    res = run_benchmark("Vector3D: Dot, Cross & Normalize", bench_vector)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} ops/sec ({res['us_per_op']:6.2f} us/op)")

    # 2. Matrix4x4 Affine Transformations
    mat_rot = Matrix4x4.rotation_z(0.785)
    mat_trans = Matrix4x4.translation(10, 20, 30)
    p = Vector3D(5, 10, 15)

    def bench_matrix():
        count = 4000
        for _ in range(count):
            m = mat_trans @ mat_rot
            pt = m.transform_point(p)
        return count

    res = run_benchmark("Matrix4x4: Multiply & Transform", bench_matrix)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} ops/sec ({res['us_per_op']:6.2f} us/op)")

    # 3. NURBS Cox-de Boor Basis & Curve Evaluation
    ctrl_pts = [
        Vector3D(0, 0, 0),
        Vector3D(5, 10, 0),
        Vector3D(10, 10, 0),
        Vector3D(15, 0, 0),
    ]
    curve = NURBSCurve(degree=3, control_points=ctrl_pts)

    def bench_nurbs():
        count = 1500
        for i in range(count):
            u = (i % 100) / 100.0
            pt = curve.point_at(u)
            t = curve.tangent_at(u)
        return count

    res = run_benchmark("NURBS: Cox-de Boor Curve & Tangent", bench_nurbs)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} evals/sec ({res['us_per_op']:6.2f} us/eval)")

    # 4. B-Rep Solid Creation & Divergence Theorem Volume
    def bench_brep():
        count = 400
        for _ in range(count):
            cyl = make_cylinder(5.0, 20.0, segments=24)
            v = cyl.volume()
            a = cyl.surface_area()
            cm = cyl.center_of_mass()
        return count

    res = run_benchmark("B-Rep: Cylinder & Divergence Volume", bench_brep)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} solids/sec ({res['us_per_op']:6.2f} us/solid)")

    # 5. CAD Feature Sweeps: Extrude & Revolve
    profile_rect = Sketch2D.rectangle(10, 10, center=True)
    profile_circ = Sketch2D.circle(3.0, segments=16).translate(10, 0, 0).to_plane("XZ")

    def bench_features():
        count = 200
        for _ in range(count):
            ext = extrude(profile_rect, height=15.0, draft_angle_deg=3.0, twist_angle_deg=30.0, steps=4)
            rev = revolve(profile_circ, angle_deg=360.0, segments=16)
        return count * 2

    res = run_benchmark("Features: Sweep Extrude & Revolve", bench_features)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} sweeps/sec ({res['us_per_op']:6.2f} us/sweep)")

    # 6. CSG 3D Boolean Engine: Difference & Union
    b_box = make_box(10, 10, 10, center=True)
    b_cyl = make_cylinder(3.0, 15.0, segments=16, center=True)

    def bench_csg():
        count = 50
        for _ in range(count):
            diff = csg_difference(b_box, b_cyl)
        return count

    res = run_benchmark("CSG: BSP-Tree Boolean Difference", bench_csg)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} booleans/sec ({res['us_per_op']:6.2f} us/boolean)")

    # 7. Tessellation & Binary STL Serialization
    torus = make_torus(10.0, 3.0, major_segs=24, minor_segs=16)

    def bench_tess_stl():
        count = 200
        for _ in range(count):
            facets = tessellate_solid(torus)
            data = export_stl_binary(torus)
        return count

    res = run_benchmark("Tessellation: STL Binary Export", bench_tess_stl)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} exports/sec ({res['us_per_op']:6.2f} us/export)")

    # 8. Unicode Braille 3D CAD Rasterization
    box_part = make_box(12, 12, 12, center=True)

    def bench_braille():
        count = 150
        for i in range(count):
            frame = render_solid_cad(
                box_part,
                azimuth_deg=30.0 + i,
                elevation_deg=25.0,
                char_width=60,
                char_height=14,
                mode="shaded",
                use_ansi_color=False,
            )
        return count

    res = run_benchmark("Visualizer: Shaded Braille Rasterizer", bench_braille)
    benchmarks.append(res)
    print(f"[{res['name']:<38}] {res['ops_per_sec']:12,.1f} frames/sec ({res['us_per_op']:6.2f} us/frame)")

    print("-" * 80)
    print("  ALL 8 PERFORMANCE BENCHMARKS COMPLETE - ZERO DEPENDENCIES")
    print("=" * 80)


if __name__ == "__main__":
    main()
