# GeoPrism: Spatial Indexing, R*-Tree, H3 Hexagonal Grid & Computational Geometry Engine
## Technical Specification & Architecture Plan

---

## 1. Executive Overview

GeoPrism is an industrial-grade, zero-dependency geospatial indexing engine, computational geometry suite, and spatial database kernel implemented entirely in pure Python standard library.

Modern spatial computing underpins GIS, autonomous navigation, logistics, ride-hailing, astrophysics, and computer-aided design (CAD). GeoPrism brings together five cornerstone pillars of spatial computer science:

1. **R*-Tree Spatial Indexing**:
   - Advanced variant of the R-Tree (Beckmann et al., 1990) combining $N$-dimensional bounding boxes (`AABB`), overlap minimization in `ChooseSubtree`, forced reinsertion on first overflow to prevent tree deterioration, and topological axis splitting minimizing perimeter/margin.
   - Branch-and-bound $k$-Nearest Neighbors ($k$-NN) search utilizing MinDist metrics in a priority min-heap.
2. **H3 Hierarchical Hexagonal Spatial Grid**:
   - Discrete Global Grid System (DGGS) modeling the Earth as an icosahedron partitioned into aperture-7 hexagonal tessellations across 16 hierarchical resolution levels (0 to 15).
   - 64-bit integer bit-packed cell IDs encoding mode, resolution, base cell (0 to 121), and up to 15 3-bit direction digits.
   - Topological $k$-ring neighbor traversal, geographic coordinate forward/inverse projections (lat/lon to H3 cell and H3 cell to polygon boundary vertices).
3. **Computational Geometry Suite**:
   - **Convex Hull**: Andrew's Monotone Chain algorithm computing the exact convex hull in $O(N \log N)$ time.
   - **Delaunay Triangulation**: Bowyer-Watson incremental algorithm maintaining empty circumcircles with super-triangle containment and cavity boundary edge extraction.
   - **Voronoi Diagram**: Dual graph computation extracting Voronoi polygon cells from Delaunay triangle circumcenters.
   - **Polygon Algorithms**: Ray-casting point-in-polygon (Jordan curve theorem), polygon area via Gauss's shoelace formula, centroid calculation, and bounding box computation.
4. **Spatial Analytics & Accelerated Clustering**:
   - Dual-tree R-Tree spatial join ($R \bowtie_S S$) for $O(N \log M)$ spatial intersection.
   - R*-Tree accelerated DBSCAN (Density-Based Spatial Clustering of Applications with Noise), reducing neighborhood range query time from $O(N^2)$ to $O(N \log N)$.
   - Haversine great-circle metric for ellipsoidal geodesic distance and Euclidean $L_2$ metric for planar coordinates.
5. **Unicode Sub-Pixel Braille Terminal Visualizer**:
   - High-density $2 \times 4$ sub-pixel terminal canvas rasterizer mapping geospatial coordinates, Voronoi cell boundaries, Delaunay triangulation meshes, and H3 hexagonal grids directly onto terminal screens.

---

## 2. Technical Specifications & Mathematical Foundations

### 2.1 R*-Tree Spatial Index

#### 2.1.1 Axis-Aligned Bounding Box (AABB)
In $d$-dimensional space, a bounding box $B = \prod_{i=1}^d [l_i, u_i]$ has:
- **Volume / Area**: $V(B) = \prod_{i=1}^d (u_i - l_i)$
- **Margin / Perimeter**: $M(B) = 2 \sum_{i=1}^d (u_i - l_i)$
- **Overlap**: For boxes $A$ and $B$, $Overlap(A, B) = V(A \cap B)$ where $A \cap B = \prod_{i=1}^d [\max(l_i^A, l_i^B), \min(u_i^A, u_i^B)]$ if $\max(l_i^A, l_i^B) \le \min(u_i^A, u_i^B)$ for all $i$, else 0.

#### 2.1.2 ChooseSubtree Heuristic
- For leaf level (inserting a data entry):
  - In an ordinary R-Tree, choose the child that needs minimum area enlargement.
  - In an R*-Tree, if the child nodes are leaves, select the child that minimizes the **overlap enlargement**:
    $$\Delta Overlap(E_k) = \sum_{j \ne k} Overlap(E_k \cup E_{new}, E_j) - \sum_{j \ne k} Overlap(E_k, E_j)$$
    Tie-broken by minimum volume enlargement, then minimum volume.
  - If the child nodes are non-leaves, select the child that minimizes volume enlargement.

#### 2.1.3 Forced Reinsertion (Overflow Treatment)
- When a node overflows (entries $> M$):
  - If this is the first overflow on the current level during the insertion, do not split immediately.
  - Sort all $M+1$ entries by distance from the node's center to the bounding box center.
  - Remove the top $p = 30\%$ furthest entries from the node and re-insert them from the top of the tree!
  - This dynamically reorganizes clustered leaves, reduces dead space, and prevents tree skew.

#### 2.1.4 Topological Axis Split
- For each dimension $i \in \{1, \dots, d\}$:
  - Sort entries by lower bounds, then by upper bounds.
  - For each sorting, generate all valid partitions $(S_1, S_2)$ such that $|S_1|, |S_2| \ge m$ (minimum fill factor, e.g. $m = \lceil 0.4 M \rceil$).
  - Compute the sum of margins $S = \sum Margin(B(S_1)) + Margin(B(S_2))$.
- Choose the split dimension with the minimal sum of margins.
- Along that chosen dimension, choose the partition that minimizes $Overlap(B(S_1), B(S_2))$, tie-broken by minimal total area $V(B(S_1)) + V(B(S_2))$.

#### 2.1.5 Branch-and-Bound k-NN with MinDist
- Computes $k$ nearest neighbors to a query point $q$:
  - Distance metric:
    $$MinDist(q, B) = \sqrt{\sum_{i=1}^d r_i^2} \quad \text{where } r_i = \begin{cases} l_i - q_i & \text{if } q_i < l_i \\ 0 & \text{if } l_i \le q_i \le u_i \\ q_i - u_i & \text{if } q_i > u_i \end{cases}$$
  - Maintains a min-heap priority queue of $(MinDist, Node/Entry)$.
  - Prunes subtrees where $MinDist(q, B) \ge \text{current } k\text{-th best distance}$.

---

### 2.2 H3 Hierarchical Hexagonal Spatial Grid

- **Icosahedral Geometry**:
  - The sphere is projected onto the 20 faces of a regular icosahedron.
  - 122 base cells (110 hexagons, 12 pentagons) at resolution 0.
- **Aperture 7 Hierarchy**:
  - Each hexagonal cell at resolution $r$ decomposes into 7 sub-hexagons at resolution $r+1$, rotated by $\approx 19.106^\circ$.
  - Area ratio between successive resolutions is exactly 7.
- **64-bit Bit-Packed Cell Identifier**:
  - Bits 63: Reserved (0)
  - Bits 62-59: Mode (4 bits: 1 for H3 Cell)
  - Bits 58-52: Resolution (4 bits: 0 to 15)
  - Bits 51-45: Base Cell ID (7 bits: 0 to 121)
  - Bits 44-0: 15 directional digits (3 bits each: values 0 to 6, where 7 indicates invalid/unused).
- **Coordinate Conversion & Topological Search**:
  - Forward: Latitude / Longitude $(\phi, \lambda) \to$ Icosahedron Face $\to$ Hexagonal Coordinate $(i, j) \to$ 64-bit Index.
  - Inverse: 64-bit Index $\to$ Hexagonal Center $(\phi_c, \lambda_c)$ and 6 vertex coordinates.
  - $k$-Ring search: Computes concentric hexagonal rings of radius $k$ around an origin cell.

---

### 2.3 Computational Geometry Suite

1. **Monotone Chain Convex Hull (Andrew's Algorithm)**:
   - Sorts points lexicographically by $(x, y)$ in $O(N \log N)$.
   - Computes lower and upper hulls by evaluating the sign of the 2D cross product:
     $$Cross(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)$$
     A positive value indicates a counter-clockwise turn.
2. **Bowyer-Watson Delaunay Triangulation**:
   - Initializes a super-triangle enclosing all input points.
   - For each point $P$:
     - Identifies all existing triangles whose circumcircle contains $P$ ($dist(P, circumcenter) < radius$).
     - The union of these triangles forms a polygonal star-shaped cavity.
     - Removes internal triangles and connects $P$ to every boundary edge of the cavity.
   - Removes super-triangle vertices and incident triangles.
3. **Voronoi Diagram Extraction**:
   - The Voronoi diagram is the geometric dual of the Delaunay triangulation.
   - Each Voronoi vertex is the circumcenter of a Delaunay triangle.
   - For each original point $P$, its Voronoi polygon is formed by connecting the circumcenters of all Delaunay triangles sharing $P$ in cyclic angular order.
4. **Jordan Curve Point-in-Polygon & Gauss Shoelace Area**:
   - Ray-casting test: counts intersections of a horizontal ray $[x, \infty) \times \{y\}$ with polygon edges. Odd count $\implies$ inside.
   - Shoelace formula:
     $$Area = \frac{1}{2} \left| \sum_{i=1}^n (x_i y_{i+1} - x_{i+1} y_i) \right|$$

---

### 2.4 Spatial Analytics & DBSCAN

1. **Dual R-Tree Spatial Join ($R \bowtie_S S$)**:
   - Traverses two R-Tree roots simultaneously.
   - If bounding boxes of nodes $N_R$ and $N_S$ do not overlap, the entire pair of subtrees is pruned.
   - Reaches leaf pairs and reports exact intersecting geometries.
2. **R*-Tree Accelerated DBSCAN**:
   - Clusters spatial points without prior specification of cluster counts $k$.
   - Parameters: $\varepsilon$ (neighborhood radius) and $MinPts$ (minimum core points).
   - Standard DBSCAN has $O(N^2)$ complexity due to brute-force distance scans.
   - Using GeoPrism's $R^*$-Tree spatial range queries, neighborhood lookups run in $O(\log N)$ time, yielding an overall clustering complexity of $O(N \log N)$.

---

## 3. Directory Layout & Module Structure

```
projects/14-geoprism/
|-- geoprism/
|   |-- __init__.py          # Package exports
|   |-- aabb.py              # N-dimensional AABB, overlap, margin, volume, MinDist
|   |-- rtree.py             # R*-Tree index, forced reinsertion, topological split, k-NN
|   |-- h3.py                # H3 DGGS, 64-bit bit-packed cell IDs, lat/lon, k-ring
|   |-- geometry.py          # Monotone Chain Hull, Bowyer-Watson Delaunay, Voronoi, Point-in-Poly
|   |-- engine.py            # Dual R-tree spatial join, Haversine, R*-tree accelerated DBSCAN
|   `-- visualizer.py        # Sub-pixel Unicode Braille spatial rasterizer
|-- tests/
|   |-- test_aabb.py         # Bounding box arithmetic, intersection, MinDist
|   |-- test_rtree.py        # R*-Tree insertion, reinsertion, range query, k-NN
|   |-- test_h3.py           # 64-bit H3 bit-packing, coordinate roundtrip, k-ring
|   |-- test_geometry.py     # Convex hull, Delaunay triangulation, Voronoi cells
|   |-- test_engine.py       # Spatial join, Haversine distance, accelerated DBSCAN
|   `-- test_visualizer.py   # Braille canvas rasterization
|-- benchmarks/
|   `-- bench_spatial.py     # R*-tree insertions, k-NN queries, H3 lookups, DBSCAN
|-- examples/
|   `-- spatial_lab.py       # Interactive spatial laboratory showcase
|-- PLAN.md                  # Architectural specification
`-- README.md                # Comprehensive documentation
```

---

## 4. Implementation Phasing

1. **Phase 1 (Task #72)**: Bounding box primitives (`aabb.py`), $R^*$-Tree spatial index with overflow reinsertion and topological axis split (`rtree.py`), branch-and-bound $k$-NN search, and unit tests (`test_aabb.py`, `test_rtree.py`).
2. **Phase 2 (Task #73)**: H3 hierarchical hexagonal grid (`h3.py`), 64-bit integer bit-packing, forward/inverse geographic projection, $k$-ring neighbor search, and unit tests (`test_h3.py`).
3. **Phase 3 (Task #74)**: Computational geometry suite (`geometry.py`): Convex Hull, Bowyer-Watson Delaunay Triangulation, Voronoi diagram polygon generation, and unit tests (`test_geometry.py`).
4. **Phase 4 (Task #75)**: Spatial analytics engine (`engine.py`): Dual R-tree spatial join, Haversine geodesics, $R^*$-tree accelerated DBSCAN clustering, and unit tests (`test_engine.py`).
5. **Phase 5 (Task #76)**: Sub-pixel Braille map visualizer (`visualizer.py`) and interactive spatial lab (`examples/spatial_lab.py`).
6. **Phase 6 (Task #77)**: Comprehensive test suite and performance benchmark suite (`benchmarks/bench_spatial.py`).
7. **Phase 7 (Task #78)**: Master showcase integration (`showcase.py`), project registry (`projects.md`), tracker update, and documentation (`README.md`).
