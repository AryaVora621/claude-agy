"""
GeoPrism: High-Performance Spatial Indexing & Computational Geometry Benchmarks.
Evaluates R*-Tree insertion and query throughput, k-NN search acceleration,
H3 DGGS coordinate codecs, Bowyer-Watson triangulation, and dual-tree spatial join.
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geoprism.aabb import AABB
from geoprism.rtree import RStarTree
from geoprism.h3 import geo_to_h3, h3_to_geo, encode_h3, decode_h3
from geoprism.geometry import delaunay_triangulation, convex_hull
from geoprism.engine import spatial_join


def bench_rtree_insertion(n: int = 10000) -> float:
    print(f"[*] Benchmarking R*-Tree Insertion (N={n:,} entries)...")
    random.seed(42)
    tree = RStarTree(max_entries=16, dimensions=2)
    boxes = [
        AABB.from_point((random.uniform(0, 10000), random.uniform(0, 10000)), radius=random.uniform(1, 10))
        for _ in range(n)
    ]

    t0 = time.perf_counter()
    for i, box in enumerate(boxes):
        tree.insert(f"item_{i}", box)
    t_total = time.perf_counter() - t0

    throughput = n / t_total
    print(f"    Total Time: {t_total * 1e3:.2f} ms")
    print(f"    Throughput: {throughput:,.0f} insertions/sec\n")
    return throughput


def bench_rtree_queries(n_index: int = 10000, n_queries: int = 2000) -> None:
    print(f"[*] Benchmarking R*-Tree Range & k-NN Queries (Index N={n_index:,}, Queries={n_queries:,})...")
    random.seed(42)
    tree = RStarTree(max_entries=16, dimensions=2)
    raw_points = []
    for i in range(n_index):
        x = random.uniform(0, 10000)
        y = random.uniform(0, 10000)
        raw_points.append((x, y))
        tree.insert(i, AABB.from_point((x, y), radius=5.0))

    # Range Queries
    query_boxes = [
        AABB.from_point((random.uniform(100, 9900), random.uniform(100, 9900)), radius=50.0)
        for _ in range(n_queries)
    ]

    t0 = time.perf_counter()
    total_matches = 0
    for qb in query_boxes:
        res = tree.search(qb)
        total_matches += len(res)
    t_range = time.perf_counter() - t0

    range_throughput = n_queries / t_range
    latency_range_us = (t_range / n_queries) * 1e6
    print(f"    Range Search Throughput : {range_throughput:,.0f} queries/sec ({latency_range_us:.2f} us/query)")
    print(f"    Average Matches / Query : {total_matches / n_queries:.1f}")

    # k-NN Queries (k=5) vs Brute Force
    k = 5
    query_pts = [(random.uniform(100, 9900), random.uniform(100, 9900)) for _ in range(n_queries)]

    # R*-Tree k-NN
    t0 = time.perf_counter()
    for pt in query_pts:
        _ = tree.nearest_neighbors(pt, k=k)
    t_tree_knn = time.perf_counter() - t0

    # Brute force k-NN on subset of 200 queries
    n_sub = 200
    t0 = time.perf_counter()
    for pt in query_pts[:n_sub]:
        qx, qy = pt
        _ = sorted(raw_points, key=lambda p: (p[0] - qx) ** 2 + (p[1] - qy) ** 2)[:k]
    t_brute_sub = time.perf_counter() - t0
    t_brute_extrapolated = t_brute_sub * (n_queries / n_sub)

    knn_throughput = n_queries / t_tree_knn
    knn_latency_us = (t_tree_knn / n_queries) * 1e6
    speedup = t_brute_extrapolated / t_tree_knn

    print(f"    k-NN Search Throughput  : {knn_throughput:,.0f} queries/sec ({knn_latency_us:.2f} us/query)")
    print(f"    k-NN Pruning Speedup    : {speedup:.1f}x vs Brute-Force Scan\n")


def bench_h3_codecs(n: int = 25000) -> None:
    print(f"[*] Benchmarking H3 DGGS Hexagonal Codecs (N={n:,} operations)...")
    random.seed(42)

    # 1. 64-bit Bit Packing / Unpacking
    t0 = time.perf_counter()
    indices = []
    for i in range(n):
        idx = encode_h3(8, (i * 7) % 122, [1, 2, 0, 4, 3, 6, 5, 2])
        indices.append(idx)
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    for idx in indices:
        _ = decode_h3(idx)
    t_unpack = time.perf_counter() - t0

    print(f"    Bit-Packing Throughput   : {n / t_pack:,.0f} encodes/sec")
    print(f"    Unpacking Throughput     : {n / t_unpack:,.0f} decodes/sec")

    # 2. Geographic Lat/Lon to H3 Cell Conversion
    coords = [(random.uniform(-80.0, 80.0), random.uniform(-170.0, 170.0)) for _ in range(n)]
    t0 = time.perf_counter()
    for lat, lon in coords:
        _ = geo_to_h3(lat, lon, resolution=8)
    t_geo = time.perf_counter() - t0

    print(f"    Geo-to-H3 Projection     : {n / t_geo:,.0f} conversions/sec\n")


def bench_computational_geometry() -> None:
    print("[*] Benchmarking Computational Geometry (Convex Hull & Delaunay)...")
    random.seed(42)

    # Convex Hull scaling
    for n in [1000, 5000, 20000]:
        pts = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]
        t0 = time.perf_counter()
        hull = convex_hull(pts)
        t_hull = time.perf_counter() - t0
        print(f"    Convex Hull N={n:5d}    : {t_hull * 1e3:6.2f} ms ({n / t_hull:,.0f} pts/sec, hull={len(hull)})")

    # Delaunay Triangulation scaling
    for n in [50, 100, 200, 400]:
        pts = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]
        t0 = time.perf_counter()
        triangles = delaunay_triangulation(pts)
        t_del = time.perf_counter() - t0
        print(f"    Delaunay Tri N={n:3d}   : {t_del * 1e3:6.2f} ms ({len(triangles)} triangles formed)")
    print()


def bench_dual_tree_spatial_join() -> None:
    print("[*] Benchmarking Dual R*-Tree Synchronous Spatial Join vs Brute Force...")
    random.seed(42)
    n_a = 1500
    n_b = 1500

    tree_a = RStarTree(max_entries=16, dimensions=2)
    tree_b = RStarTree(max_entries=16, dimensions=2)
    boxes_a = []
    boxes_b = []

    for i in range(n_a):
        b = AABB.from_point((random.uniform(0, 1000), random.uniform(0, 1000)), radius=10.0)
        tree_a.insert(i, b)
        boxes_a.append((i, b))

    for j in range(n_b):
        b = AABB.from_point((random.uniform(0, 1000), random.uniform(0, 1000)), radius=10.0)
        tree_b.insert(j, b)
        boxes_b.append((j, b))

    # Dual R*-Tree Join
    t0 = time.perf_counter()
    matches = spatial_join(tree_a, tree_b)
    t_join = time.perf_counter() - t0

    # Brute force pairwise subset (first 300 items)
    sub_a = boxes_a[:300]
    t0 = time.perf_counter()
    brute_count = 0
    for ia, ba in sub_a:
        for ib, bb in boxes_b:
            if ba.intersects(bb):
                brute_count += 1
    t_brute_sub = time.perf_counter() - t0
    t_brute_est = t_brute_sub * (n_a / 300)

    speedup = t_brute_est / t_join
    print(f"    Dual-Tree Spatial Join   : {t_join * 1e3:.2f} ms ({len(matches):,} intersection pairs)")
    print(f"    Pairwise Scan (Est)      : {t_brute_est * 1e3:.2f} ms ({n_a * n_b:,} box comparisons)")
    print(f"    Spatial Join Speedup     : {speedup:.1f}x speedup over brute-force comparison\n")


def main() -> None:
    print("=" * 80)
    print("  GEOPRISM PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    print()

    bench_rtree_insertion(10000)
    bench_rtree_queries(10000, 2000)
    bench_h3_codecs(25000)
    bench_computational_geometry()
    bench_dual_tree_spatial_join()

    print("All GeoPrism benchmarks completed successfully.")


if __name__ == "__main__":
    main()
