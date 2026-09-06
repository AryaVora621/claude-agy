"""
GeoPrism: Interactive Spatial Intelligence & DGGS Laboratory.
Demonstrates R*-Tree indexing, H3 hexagonal global grid, Delaunay triangulation,
Voronoi diagrams, dual-tree spatial join, DBSCAN clustering, and Braille visualization.
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geoprism.aabb import AABB
from geoprism.rtree import RStarTree
from geoprism.h3 import (
    geo_to_h3, h3_to_geo, h3_to_string, h3_to_geo_boundary, k_ring, h3_distance
)
from geoprism.geometry import (
    convex_hull, delaunay_triangulation, voronoi_diagram, polygon_area
)
from geoprism.engine import (
    spatial_join, dbscan, h3_hexbin, spatial_kde_raster
)
from geoprism.visualizer import BrailleCanvas, render_heatmap


def print_banner() -> None:
    banner = r"""
================================================================================
  GEOPRISM : High-Performance Spatial Indexing & Computational Geometry
  Pure Python 3 Standard Library - Zero External Dependencies
================================================================================
"""
    print(banner)


def demo_rtree_and_knn() -> None:
    print("--- 1. R*-Tree Spatial Indexing & Branch-and-Bound k-NN ---")
    tree = RStarTree(max_entries=16, dimensions=2)

    # Insert 1,000 spatial entities across a 1000x1000 field
    random.seed(1337)
    cities = [
        ("Metropolis Alpha", (150.0, 200.0)),
        ("Neo Tokyo Hub", (750.0, 820.0)),
        ("Avalon Citadel", (500.0, 500.0)),
        ("Zephyr Outpost", (180.0, 220.0)),
        ("Solaris Station", (520.0, 480.0)),
        ("Borealis Port", (120.0, 890.0)),
    ]
    for name, pt in cities:
        tree.insert(name, AABB.from_point(pt, radius=5.0))

    # Insert background points
    for i in range(500):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        tree.insert(f"Node_{i:03d}", AABB.from_point((x, y), radius=2.0))

    print(f"Total Indexed Entities: {tree.size} objects in R*-Tree")

    # Range query around Avalon Citadel [450, 450] to [550, 550]
    query_box = AABB((450.0, 450.0), (550.0, 550.0))
    t0 = time.perf_counter()
    matches = tree.search(query_box)
    t_search = (time.perf_counter() - t0) * 1e6

    print(f"\nSpatial Range Search in {query_box}:")
    print(f"  Found {len(matches)} intersecting objects in {t_search:.2f} us")
    for item, box in matches[:5]:
        print(f"    - {item} at center {box.center()}")
    if len(matches) > 5:
        print(f"    ... and {len(matches) - 5} more")

    # k-NN search from origin (160, 210) near Metropolis Alpha
    query_pt = (160.0, 210.0)
    t0 = time.perf_counter()
    nearest = tree.nearest_neighbors(query_pt, k=4)
    t_knn = (time.perf_counter() - t0) * 1e6

    print(f"\nBranch-and-Bound k-NN Query from {query_pt} (k=4):")
    print(f"  Completed in {t_knn:.2f} us using MinDist priority pruning:")
    for dist, item, box in nearest:
        print(f"    - [{dist:6.2f} units] {item} at {box.center()}")
    print()


def demo_h3_hexagonal_dggs() -> None:
    print("--- 2. H3 Hierarchical Hexagonal Discrete Global Grid System ---")
    landmarks = [
        ("San Francisco (Ferry Bldg)", 37.7955, -122.3937),
        ("London (Tower Bridge)", 51.5055, -0.0754),
        ("Tokyo (Shibuya Crossing)", 35.6595, 139.7004),
    ]

    for name, lat, lon in landmarks:
        res = 9
        h3_idx = geo_to_h3(lat, lon, res)
        h3_hex = h3_to_string(h3_idx)
        cent_lat, cent_lon = h3_to_geo(h3_idx)
        print(f"Landmark: {name}")
        print(f"  Coordinates   : ({lat:.4f}, {lon:.4f})")
        print(f"  H3 Index (R{res}): 0x{h3_hex} (64-bit int: {h3_idx})")
        print(f"  Cell Centroid : ({cent_lat:.4f}, {cent_lon:.4f})")

        # Boundary vertices
        boundary = h3_to_geo_boundary(h3_idx)
        print(f"  Hexagon Vertices ({len(boundary)} sides):")
        for idx, (v_lat, v_lon) in enumerate(boundary[:3], 1):
            print(f"    v{idx}: ({v_lat:.4f}, {v_lon:.4f})")
        print("    ...")

    # k-ring neighborhood expansion
    sf_h3 = geo_to_h3(37.7749, -122.4194, resolution=7)
    ring_1 = k_ring(sf_h3, 1)
    ring_2 = k_ring(sf_h3, 2)
    print(f"\nTopological k-Ring Expansion around SF (Res 7):")
    print(f"  k=0 (Self)   : 1 cell")
    print(f"  k=1 (Adjacency): {len(ring_1)} cells")
    print(f"  k=2 (Radius 2) : {len(ring_2)} cells")
    print()


def demo_computational_geometry() -> None:
    print("--- 3. Computational Geometry in Braille (Hull, Delaunay, Voronoi) ---")
    random.seed(42)

    # Generate 18 points on a 100x60 plane
    points = [
        (15.0, 10.0), (85.0, 12.0), (90.0, 50.0), (10.0, 48.0),
        (50.0, 30.0), (35.0, 25.0), (65.0, 25.0), (45.0, 40.0),
        (25.0, 35.0), (75.0, 38.0), (50.0, 15.0), (50.0, 52.0),
        (30.0, 18.0), (70.0, 18.0), (20.0, 45.0), (80.0, 45.0),
        (40.0, 32.0), (60.0, 32.0),
    ]

    # Convex Hull via Andrew's Monotone Chain
    hull = convex_hull(points)
    hull_area = abs(polygon_area(hull))
    print(f"Convex Hull: {len(hull)} vertices enclosing area of {hull_area:.1f} sq units")

    # Delaunay Triangulation via Bowyer-Watson
    triangles = delaunay_triangulation(points)
    print(f"Bowyer-Watson Delaunay Triangulation: {len(triangles)} triangles formed")

    # Voronoi Diagram Dual Graph
    vor = voronoi_diagram(points)
    print(f"Voronoi Diagram: {len(vor.vertices)} dual vertices, {len(vor.edges)} dual edges")

    # Render into Unicode Braille Canvas (width=60 chars, height=18 chars -> 120x72 sub-pixels)
    bbox = AABB((0.0, 0.0), (100.0, 60.0))
    canvas = BrailleCanvas(char_width=60, char_height=18, bbox=bbox)

    # Draw Delaunay Triangles
    for t in triangles:
        canvas.draw_line(t.a[0], t.a[1], t.b[0], t.b[1])
        canvas.draw_line(t.b[0], t.b[1], t.c[0], t.c[1])
        canvas.draw_line(t.c[0], t.c[1], t.a[0], t.a[1])

    # Draw Convex Hull (thicker or highlighted perimeter)
    canvas.draw_polygon(hull)

    # Draw Points
    for px, py in points:
        canvas.draw_point(px, py, radius=1)

    print("\nTerminal Braille Delaunay Mesh & Convex Hull Perimeter:")
    print(canvas.render(border=True))
    print()


def demo_spatial_join_and_dbscan() -> None:
    print("--- 4. Dual R*-Tree Spatial Join & Accelerated DBSCAN Clustering ---")
    random.seed(99)

    # Create two spatial layers: Sensors and Hazardous Zones
    tree_sensors = RStarTree(max_entries=8, dimensions=2)
    tree_zones = RStarTree(max_entries=8, dimensions=2)

    # 100 sensors
    for i in range(100):
        sx = random.uniform(0, 500)
        sy = random.uniform(0, 500)
        tree_sensors.insert(f"Sensor_{i:02d}", AABB.from_point((sx, sy), radius=4.0))

    # 15 hazard zones
    for j in range(15):
        zx = random.uniform(50, 450)
        zy = random.uniform(50, 450)
        radius = random.uniform(25.0, 50.0)
        tree_zones.insert(f"HazardZone_{j:02d}", AABB.from_point((zx, zy), radius=radius))

    t0 = time.perf_counter()
    join_results = spatial_join(tree_sensors, tree_zones)
    t_join = (time.perf_counter() - t0) * 1e3

    print(f"Dual-Tree Spatial Join (Sensors x Zones):")
    print(f"  Found {len(join_results)} intersecting sensor-hazard pairs in {t_join:.2f} ms")
    for sensor, zone in join_results[:4]:
        print(f"    - Alert: {sensor} located inside {zone}")
    if len(join_results) > 4:
        print(f"    ... and {len(join_results) - 4} more overlapping intersections")

    # DBSCAN Clustering
    cluster_a = [(random.gauss(50, 8), random.gauss(50, 8)) for _ in range(30)]
    cluster_b = [(random.gauss(180, 12), random.gauss(150, 12)) for _ in range(40)]
    cluster_c = [(random.gauss(300, 10), random.gauss(220, 10)) for _ in range(25)]
    noise = [(random.uniform(0, 400), random.uniform(0, 400)) for _ in range(15)]
    dbscan_pts = cluster_a + cluster_b + cluster_c + noise

    t0 = time.perf_counter()
    clusters = dbscan(dbscan_pts, eps=25.0, min_pts=5)
    t_db = (time.perf_counter() - t0) * 1e3

    print(f"\nR*-Tree Accelerated Spatial DBSCAN (N={len(dbscan_pts)}, eps=25, min_pts=5):")
    print(f"  Clustering converged in {t_db:.2f} ms:")
    for cid, members in sorted(clusters.items()):
        if cid == -1:
            print(f"    Noise Outliers : {len(members)} isolated points")
        else:
            print(f"    Cluster #{cid:02d}    : {len(members)} core members")
    print()


def demo_kde_heatmap() -> None:
    print("--- 5. 2D Continuous Kernel Density Estimation (KDE) Thermal Heatmap ---")
    random.seed(777)
    # Generate multimodal density peaks
    hotspot_1 = [(random.gauss(25, 4), random.gauss(15, 3)) for _ in range(60)]
    hotspot_2 = [(random.gauss(50, 6), random.gauss(20, 5)) for _ in range(90)]
    hotspot_3 = [(random.gauss(40, 3), random.gauss(8, 2)) for _ in range(40)]
    all_points = hotspot_1 + hotspot_2 + hotspot_3

    grid, bbox = spatial_kde_raster(all_points, grid_w=60, grid_h=15, bandwidth=3.5)
    print(f"Gaussian KDE Rasterization ({len(grid[0])}x{len(grid)} cells, Bandwidth=3.5):")
    print(render_heatmap(grid, use_color=True, border=True))
    print()


def main() -> None:
    print_banner()
    demo_rtree_and_knn()
    demo_h3_hexagonal_dggs()
    demo_computational_geometry()
    demo_spatial_join_and_dbscan()
    demo_kde_heatmap()
    print("GeoPrism spatial laboratory demonstration completed successfully.")


if __name__ == "__main__":
    main()
