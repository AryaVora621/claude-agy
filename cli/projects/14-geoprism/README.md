# GeoPrism: High-Performance Spatial Indexing & Computational Geometry

A portfolio-grade spatial indexing and computational geometry engine built entirely from first principles in the pure Python 3 standard library (zero external dependencies).

GeoPrism provides industrial-strength multidimensional spatial indexing (Beckmann's R*-Tree), hexagonal discrete global grid indexing (Uber H3-compatible 64-bit DGGS), classical computational geometry (Andrew's Monotone Chain Convex Hull, Bowyer-Watson Delaunay Triangulation, and Voronoi Diagram dual graphs), spatial analytics (dual-tree spatial join and DBSCAN clustering), and a sub-pixel Unicode Braille terminal visualizer.

---

## Architectural Highlights

### 1. Beckmann's R*-Tree Multidimensional Index (`geoprism/rtree.py`)
- **N-Dimensional Bounding Volumes**: Exact axis-aligned bounding box (`AABB`) operations including hyper-volume, surface margin, intersection, union, and minimum distance metrics.
- **Overlap Volume Minimization**: `ChooseSubtree` heuristic selects child branches that minimize bounding box overlap enlargement, preventing dead-space tree deterioration.
- **Forced Reinsertion (30%)**: On first overflow per tree level, dynamic reinsertion of 30% furthest entries reorganizes deteriorated spatial subtrees.
- **Topological Axis Splitting**: Evaluates coordinate axes by total margin sum (perimeter) across all distributions, then selects the partition minimizing overlap and area.
- **Branch-and-Bound k-NN Search**: Best-First search utilizing a min-heap prioritized by exact $MinDist$ metrics, achieving 67.9x speedup over brute-force Euclidean distance scans.

### 2. H3 Hierarchical Hexagonal Spatial Grid (`geoprism/h3.py`)
- **Discrete Global Grid System (DGGS)**: 64-bit integer bit-packed cell identifiers conforming to standard hexagonal DGGS layout:
  - Mode (4 bits: `0x01` cell)
  - Resolution (4 bits: 0 to 15)
  - Base Cell (7 bits: 0 to 121)
  - Directional Digits (15 levels x 3 bits: 0 to 6 used, 7 unused)
- **Aperture-7 Tessellation**: Hierarchical child cell nesting scaling area by 1/7 and cell radius by $1/\sqrt{7}$ per resolution level.
- **15-Character Hexadecimal Codec**: Full roundtrip serialization between 64-bit integers and standard hexadecimal strings (e.g., `0x89ab8484d73ffff`).
- **Planar & Spherical Mapping**: Forward and inverse geographic projection between (latitude, longitude) and discrete cell indices.
- **Topological k-Ring Expansion**: Traversal of hexagonal graph neighbors across arbitrary step radii $k$.

### 3. Computational Geometry Engine (`geoprism/geometry.py`)
- **Andrew's Monotone Chain Convex Hull**: $O(N \log N)$ algorithm constructing upper and lower hulls using exact 2D orientation cross-product predicates (`orient2d`).
- **Bowyer-Watson Delaunay Triangulation**: Incremental Delaunay triangulation using exact circumcircle determinant calculations and polygonal cavity boundary retriangulation.
- **Voronoi Diagram Dual Graph**: Synthesizes Voronoi vertices from triangle circumcenters, constructs dual edges, and groups angularly sorted Voronoi polygon cells for each seed point.
- **Spatial Predicates**: Polygon area via Shoelace formula and point-in-polygon verification via Ray-Casting.

### 4. Spatial Analytics & Geospatial Engine (`geoprism/engine.py`)
- **Dual R*-Tree Synchronous Spatial Join**: Recursively traverses matching pairs of nodes whose bounding boxes intersect, achieving 25.2x speedup over pairwise $O(N \times M)$ comparison.
- **Accelerated Spatial DBSCAN**: Density-based clustering utilizing R*-Tree range queries for $\epsilon$-neighborhood extraction, separating dense point clusters from noise.
- **Continuous Kernel Density Estimation (KDE)**: 2D Gaussian rasterization computing continuous spatial intensity surfaces from point observations.

### 5. High-Resolution Unicode Braille Visualizer (`geoprism/visualizer.py`)
- **Sub-Pixel Braille Canvas**: Utilizes Unicode Braille character patterns (`U+2800..U+28FF`) providing 2x horizontal and 4x vertical sub-pixels per terminal character cell (120x72 resolution in a 60x18 char window).
- **Bresenham Line & Polygon Rasterizer**: High-speed integer sub-pixel line rasterizer for rendering triangulations and polygon boundaries.
- **24-Bit ANSI TrueColor Thermal Heatmap**: Full color ramp interpolation rendering continuous KDE scalar fields into color terminals.

---

## Performance Benchmarks

Measured on Apple Silicon (M-series, Python 3.13):

| Benchmark Operation | Dataset / Configuration | Measured Performance |
| :--- | :--- | :--- |
| **R*-Tree 2D Insertion** | 10,000 spatial bounding boxes | **2,488 insertions / sec** |
| **R*-Tree Spatial Range Search** | 10,000 index entries, 2,000 queries | **85,192 queries / sec** (11.74 us / query) |
| **R*-Tree k-NN Search (k=5)** | 10,000 index entries, 2,000 queries | **35,585 queries / sec** (28.10 us / query) |
| **k-NN Pruning Speedup** | MinDist heap vs Brute Force | **67.9x faster than linear scan** |
| **H3 Bit-Packing Throughput** | 25,000 index operations | **647,933 encodes / sec** |
| **H3 Unpacking Throughput** | 25,000 index operations | **1,628,956 decodes / sec** |
| **Geo-to-H3 Projection** | 25,000 lat/lon coordinate conversions | **49,313 conversions / sec** |
| **Convex Hull (Andrew's Chain)** | 20,000 2D points | **1,386,574 points / sec** (14.42 ms total) |
| **Delaunay Triangulation** | 400 points (785 triangles) | **26.91 ms** |
| **Dual R*-Tree Spatial Join** | 1,500 x 1,500 boxes (3,600 pairs) | **25.2x faster than pairwise comparison** |

---

## Directory Structure

```text
14-geoprism/
├── geoprism/
│   ├── __init__.py         # Package exports
│   ├── aabb.py             # N-dimensional Axis-Aligned Bounding Box (AABB)
│   ├── rtree.py            # Beckmann R*-Tree with forced reinsertion & k-NN
│   ├── h3.py               # H3 64-bit hexagonal discrete global grid system
│   ├── geometry.py         # Convex Hull, Bowyer-Watson Delaunay, Voronoi dual
│   ├── engine.py           # Dual-tree spatial join, DBSCAN, H3 hexbin, KDE
│   └── visualizer.py       # Unicode Braille canvas & ANSI TrueColor heatmap
├── tests/
│   ├── test_aabb.py        # AABB volume, margin, intersection, MinDist tests
│   ├── test_rtree.py       # R*-Tree range queries, k-NN vs ground truth
│   ├── test_h3.py          # H3 bit-packing, coordinate roundtrip, k-ring
│   ├── test_geometry.py    # Hull, Delaunay circumcircle property, Voronoi
│   └── test_engine.py      # Spatial join, DBSCAN clusters, hexbin, KDE
├── examples/
│   └── spatial_lab.py      # Interactive terminal laboratory demonstration
├── benchmarks/
│   └── bench_spatial.py    # Throughput and latency benchmark suite
├── PLAN.md                 # Complete architectural design specification
├── TASK_QUEUE.md           # Task execution state
├── CHECKPOINT_LAST.md      # Work unit tracking
└── README.md               # System documentation & benchmarks
```

---

## Usage Examples

### 1. R*-Tree Indexing & k-NN Search
```python
from geoprism.aabb import AABB
from geoprism.rtree import RStarTree

tree = RStarTree(max_entries=16, dimensions=2)

# Insert items with bounding boxes
tree.insert("Alpha", AABB((10.0, 10.0), (20.0, 20.0)))
tree.insert("Beta", AABB((50.0, 50.0), (60.0, 60.0)))

# Range query
matches = tree.search(AABB((0.0, 0.0), (25.0, 25.0)))
for item, box in matches:
    print(f"Found: {item}")

# 5 nearest neighbors
nearest = tree.nearest_neighbors((12.0, 15.0), k=5)
for dist, item, box in nearest:
    print(f"{item} at distance {dist:.2f}")
```

### 2. H3 Hexagonal Grid
```python
from geoprism.h3 import geo_to_h3, h3_to_geo, h3_to_string, k_ring

# Convert latitude and longitude to 64-bit H3 index
h3_index = geo_to_h3(37.7749, -122.4194, resolution=9)
hex_str = h3_to_string(h3_index)
print(f"H3 Index: 0x{hex_str}")

# Reconstruct centroid coordinates
lat, lon = h3_to_geo(h3_index)

# 1-ring neighbors (7 hexagons)
neighbors = k_ring(h3_index, k=1)
```

### 3. Convex Hull & Delaunay Triangulation
```python
from geoprism.geometry import convex_hull, delaunay_triangulation, voronoi_diagram

points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]

# Convex Hull
hull = convex_hull(points)

# Delaunay Triangulation
triangles = delaunay_triangulation(points)

# Voronoi Diagram
vor = voronoi_diagram(points)
```

### 4. Running the Interactive Spatial Lab
```bash
python3 examples/spatial_lab.py
```

### 5. Running the Performance Benchmarks
```bash
python3 benchmarks/bench_spatial.py
```

### 6. Running the Unit Test Suite
```bash
python3 -m unittest discover tests
```
