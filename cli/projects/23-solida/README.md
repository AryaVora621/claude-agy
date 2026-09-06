# Solida: 3D Boundary Representation (B-Rep) Solid Modeling CAD & NURBS Geometric Modeling Kernel

A professional-grade 3D solid modeling CAD engine and Non-Uniform Rational B-Splines (NURBS) geometric kernel built entirely from first principles in the pure Python standard library. Solida features manifold half-edge boundary representation topology, Cox-de Boor recursive basis evaluation, exact Euler-Poincaré topological invariants, a Constructive Solid Geometry (CSG) Binary Space Partitioning (BSP) Boolean engine, multi-format STL and Wavefront OBJ codecs, an orbital sub-pixel Unicode Braille visualizer with 24-bit TrueColor Lambertian diffuse shading, and full assembly interference analysis.

Zero external dependencies. Pure Python 3.10+ standard library.

---

## Key Engineering Features

1. **3D Differential Geometry & Affine Algebra (`solida/geometry.py`)**:
   - `Vector3D`: Immutable Cartesian 3-space operations (dot, cross, norm, normalization, orthogonal projection, rejection, reflection, lerp, angular separation).
   - `Matrix4x4`: 4x4 homogeneous transformation matrices (translation, scaling, Euler axis rotations, Rodrigues axis-angle rotation, view look-at, Laplace cofactor determinant and analytical inversion).
   - `Quaternion`: Spatial unit quaternions, conjugate, vector rotation, matrix conversion, and spherical linear interpolation (slerp).
   - `Plane` & `Ray3D`: Implicit plane equations ($n \cdot x + d = 0$), signed distances, polygon half-space clipping, Möller-Trumbore ray-triangle intersection, and Kay-Kajiya slab ray-AABB intersection.

2. **NURBS Curves & Surfaces (`solida/nurbs.py`)**:
   - Arbitrary degree $p$ Cox-de Boor recursive basis formulation with division-by-zero prevention ($0/0 \equiv 0$).
   - Open clamped knot vectors with endpoint multiplicity $p + 1$, guaranteeing curve and surface endpoint interpolation.
   - Rational NURBS curves with analytical first and second derivatives via quotient rules, unit tangents, and scalar curvature $\kappa(u) = \frac{\|\mathbf{C}'(u) \times \mathbf{C}''(u)\|}{\|\mathbf{C}'(u)\|^3}$.
   - Exact conic representations: circular arcs using rational weights $w_1 = \cos(\theta/2)$.
   - Tensor-product NURBS surfaces $\mathbf{S}(u, v)$, partial derivatives $\mathbf{S}_u, \mathbf{S}_v$, unit surface normal $\mathbf{n}(u, v)$, first fundamental form ($E, F, G$), second fundamental form ($L, M, N$), and Gaussian curvature $K = \frac{LN - M^2}{EG - F^2}$.

3. **2-Manifold Half-Edge B-Rep Topology (`solida/brep.py`)**:
   - Complete topological hierarchy: `Vertex`, `HalfEdge`, `Edge`, `Loop`, `Face`, `Shell`, `Solid`.
   - Topological verification: edge twin mating, cycle traversal, and Poincaré-Euler characteristic $\chi = V - E + F = 2(S - G) + H$ ($\chi = 2$ for genus-0 solids, $\chi = 0$ for genus-1 tori).
   - Exact volumetric calculation via the Divergence Theorem: $V = \frac{1}{6} \sum \mathbf{v}_0 \cdot (\mathbf{v}_1 \times \mathbf{v}_2)$ over all surface triangles.
   - Centroid and center of mass evaluation via tetrahedral decomposition against the origin.
   - Newell unit normal evaluation for non-convex planar polygonal faces.

4. **3D Primitives & CAD Feature Sweeps (`solida/primitives.py`, `solida/features.py`)**:
   - Exact solid primitives: Box, Cylinder, Sphere, Cone, and Torus.
   - 2D Profile Sketches: `Sketch2D` (rectangles, circles, regular polygons, stars, and 4-digit NACA airfoils with analytical camber and thickness distributions).
   - Linear Extrusions: Sweeps with draft angles (tapering) and axial twist.
   - Rotational Sweeps: Arbitrary 3D axis-angle revolve ($360^\circ$ seamless solids and partial sector angles with end caps).
   - Multi-Section Lofts: Surface skinning through two or more planar profile sketches.

5. **CSG 3D Boolean Engine (`solida/csg.py`)**:
   - Binary Space Partitioning (BSP) Tree polygon partitioner.
   - Sutherland-Hodgman polygon clipping into front, back, coplanar-front, and coplanar-back sub-polygons.
   - Exact Boolean operations: Union ($A \cup B$), Difference ($A \setminus B$), and Intersection ($A \cap B$) with coplanar duplicate face elimination.

6. **Tessellation & Industry-Standard Codecs (`solida/tessellation.py`)**:
   - 2D Ear-Clipping triangulation with shoelace signed area winding verification.
   - 3D polygon projection onto dominant coordinate planes for non-convex planar faces.
   - ASCII STL export and parser (`solid ... facet normal ... endfacet ... endsolid`).
   - Binary STL export and parser (80-byte header, uint32 triangle count, 50-byte IEEE 754 little-endian records).
   - Wavefront OBJ export with vertex deduplication (`v`, `f`).

7. **Sub-Pixel Unicode Braille 3D Visualizer (`solida/visualizer.py`)**:
   - 2x4 sub-pixel Unicode Braille graphics canvas (`U+2800..U+28FF`) with floating-point depth buffering.
   - 3D orbital camera projection (azimuth, elevation, distance).
   - Wireframe boundary edge rasterization and 24-bit TrueColor Lambertian diffuse shaded rendering.
   - Live CAD engineering telemetry HUD (volume, surface area, centroid, bounding box extents, Euler characteristic $\chi$).

8. **High-Level Kernel & Assembly Orchestrator (`solida/kernel.py`)**:
   - Fluent `Part` API with operator overloading (`+` for union, `-` for difference, `&` for intersection).
   - Parametric `Workplane` for sketching on arbitrary datum planes in 3D space.
   - Multi-part `Assembly` management with mass budget, center of mass, spatial transforms, broadphase AABB and narrowphase CSG interference / clash detection, and composite multi-group OBJ export.

---

## Architecture Overview

```
                        +----------------------------+
                        |      High-Level API        |
                        |   Part, Workplane, Assembly|
                        +--------------+-------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
+---------v----------+       +---------v----------+       +---------v----------+
|  CAD Feature Sweeps|       |   CSG 3D Booleans  |       |   Mesh Exporters   |
| Extrude/Revolve/Loft|      | BSP-Tree Partition |       | Binary STL / OBJ   |
+---------+----------+       +---------+----------+       +---------+----------+
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                        +--------------v-------------+
                        |   2-Manifold B-Rep Shell   |
                        |   Half-Edge, Loop, Face    |
                        |   Divergence Theorem Vol   |
                        +--------------+-------------+
                                       |
          +----------------------------+----------------------------+
          |                                                         |
+---------v----------+                                    +---------v----------+
|  NURBS Math Engine |                                    |   3D Vector Math   |
| Cox-de Boor Basis  |                                    | Vector3D, Matrix4x4|
| Curves & Surfaces  |                                    | Quaternion, Plane  |
+--------------------+                                    +--------------------+
```

---

## Quickstart & Code Examples

### 1. Parametric Part Modeling with CSG Booleans

```python
from solida import Part, Workplane, Vector3D

# Create an aluminum mounting plate
plate = Part.box(100.0, 60.0, 12.0, center=True, name="BasePlate", density_g_cm3=2.7)

# Drill a central bore hole (diameter 30 mm)
bore = Part.cylinder(radius=15.0, height=20.0, segments=24, center=True)
drilled = plate - bore

# Add 4 corner mounting holes (radius 3.5 mm)
hole = Part.cylinder(radius=3.5, height=20.0, segments=16, center=True)
for dx in (-40.0, 40.0):
    for dy in (-20.0, 20.0):
        drilled = drilled - hole.translate(dx, dy, 0.0)

print(f"Drilled Plate Volume: {drilled.volume:.2f} mm3")
print(f"Total Mass:           {drilled.mass_grams:.2f} g")

# Export to binary STL for 3D printing
drilled.export_stl("plate.stl", binary=True)
```

### 2. Aerodynamic Wing Lofting from NACA Airfoils

```python
from solida import Sketch2D, loft, Part

# Generate NACA 2412 profiles across 3 spanwise stations
root = Sketch2D.naca_airfoil(code="2412", chord=40.0, num_points=24)
mid  = Sketch2D.naca_airfoil(code="2412", chord=25.0, num_points=24).translate(10.0, 2.0, 50.0)
tip  = Sketch2D.naca_airfoil(code="2412", chord=15.0, num_points=24).translate(22.0, 6.0, 100.0)

# Skin multi-section loft
wing = Part(loft([root, mid, tip], name="SweptWing"), density_g_cm3=1.6)
print(wing.show(azimuth_deg=45.0, elevation_deg=30.0, mode="shaded"))
```

### 3. Multi-Part Assembly & Interference Collision Check

```python
from solida import Assembly, Part

asm = Assembly("Gearbox")
case = Part.box(60, 50, 40, center=True, name="Case") - Part.box(50, 40, 32, center=True)
shaft = Part.cylinder(radius=6.0, height=70.0, center=True, name="Shaft")

asm.add_part(case, "GearboxHousing")
asm.add_part(shaft, "DriveShaft")

# Check for mechanical clashes between parts
clashes = asm.check_interferences(volume_tolerance=0.01)
print(f"Detected {len(clashes)} collision(s)")
```

---

## Interactive Terminal CAD Workbench

Run the included CAD workbench demo to view pre-modeled engineering components in Unicode Braille with live telemetry HUDs:

```bash
# View Flanged Bearing Housing and export STL/OBJ files
python3 examples/cad_workbench.py --model bearing --export

# View Swept Aircraft Wing
python3 examples/cad_workbench.py --model wing

# Run multi-part mechanical assembly collision analysis
python3 examples/cad_workbench.py --model assembly --export
```

---

## Performance Microbenchmarks

Run the benchmark suite:

```bash
python3 benchmarks/bench_solida.py
```

Measured on Apple M-series hardware:

| Benchmark Operation | Throughput | Latency |
|:---|:---:|:---:|
| Vector3D: Dot, Cross & Normalize | **1,370,443 ops/sec** | 0.73 μs/op |
| Matrix4x4: Multiply & Transform | **338,005 ops/sec** | 2.96 μs/op |
| NURBS: Cox-de Boor Curve & Tangent | **37,232 evals/sec** | 26.86 μs/eval |
| B-Rep: Cylinder & Divergence Volume | **1,528 solids/sec** | 654.34 μs/solid |
| Features: Sweep Extrude & Revolve | **238.6 sweeps/sec** | 4,191.61 μs/sweep |
| CSG: BSP-Tree Boolean Difference | **307.8 booleans/sec** | 3,248.71 μs/boolean |
| Tessellation: STL Binary Export | **284.1 exports/sec** | 3,519.99 μs/export |
| Visualizer: Shaded Braille Rasterizer | **1,619.4 frames/sec** | 617.50 μs/frame |

---

## Verification & Test Suite

Solida includes 30 unit tests covering geometry, NURBS, B-Rep topology, CSG Booleans, STL/OBJ codecs, and assembly orchestration:

```bash
python3 -m unittest discover -s tests -v
```

All 30 tests pass with zero warnings in under 0.08 seconds.
