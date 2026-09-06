# Architectural Specification: Solida - 3D B-Rep CAD & NURBS Geometric Modeling Kernel

## 1. System Philosophy & Objectives
Solida is a high-performance, first-principles 3D Boundary Representation (B-Rep) solid modeling Computer-Aided Design (CAD) and Non-Uniform Rational B-Spline (NURBS) geometric modeling kernel implemented entirely in the pure Python standard library with zero external dependencies.

Solida provides the mathematical bedrock used by industry-standard engineering kernels (such as Parasolid, ACIS, and OpenCASCADE):
1. Exact differential geometry with homogeneous transformation matrices and quaternions.
2. Tensor-product NURBS curves and surfaces with Cox-de Boor recursive basis evaluation, analytical gradients, and exact conic weights.
3. Rigorous Half-Edge manifold boundary representation (B-Rep) data structures governed by Euler-Poincaré invariants ($V - E + F = 2(S - G)$).
4. Parametric solid primitive generators (Box, Cylinder, Sphere, Cone, Torus).
5. CAD feature operations: Linear Extrusion and Rotational Revolve from 2D profile sketches.
6. Robust Constructive Solid Geometry (CSG) 3D Boolean modeling (Union, Difference, Intersection) via Binary Space Partitioning (BSP) trees and polygon clipping.
7. Mesh tessellation and industry-standard export: ASCII STL, Binary STL, and Wavefront OBJ.
8. Sub-pixel 3D Unicode Braille (`U+2800..U+28FF`) terminal visualizer with depth-buffering, Lambertian diffuse lighting, and CAD telemetry HUD.

---

## 2. Mathematical Formulations

### A. Non-Uniform Rational B-Splines (NURBS)
Let $U = \{u_0, u_1, \dots, u_{m}\}$ be a non-decreasing sequence of real numbers representing the knot vector, where $m = n + p + 1$ with $n+1$ control points and degree $p$.

#### Cox-de Boor Recursive Basis Functions
For $p = 0$:
$$N_{i,0}(u) = \begin{cases} 1 & \text{if } u_i \le u < u_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

For $p \ge 1$:
$$N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)$$
where division by zero is handled as $0/0 \equiv 0$.

#### NURBS Curve Formulation
With 3D control points $\mathbf{P}_i$ and scalar weights $w_i > 0$:
$$\mathbf{C}(u) = \frac{\sum_{i=0}^n N_{i,p}(u) w_i \mathbf{P}_i}{\sum_{i=0}^n N_{i,p}(u) w_i} = \frac{\mathbf{A}(u)}{w(u)}$$

First analytical derivative:
$$\mathbf{C}'(u) = \frac{\mathbf{A}'(u) - w'(u) \mathbf{C}(u)}{w(u)}$$

Unit tangent and curvature:
$$\mathbf{T}(u) = \frac{\mathbf{C}'(u)}{\|\mathbf{C}'(u)\|}, \quad \kappa(u) = \frac{\|\mathbf{C}'(u) \times \mathbf{C}''(u)\|}{\|\mathbf{C}'(u)\|^3}$$

#### Tensor-Product NURBS Surface Formulation
For knot vectors $U$ of degree $p$ and $V$ of degree $q$:
$$\mathbf{S}(u, v) = \frac{\sum_{i=0}^n \sum_{j=0}^m N_{i,p}(u) N_{j,q}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m N_{i,p}(u) N_{j,q}(v) w_{i,j}} = \frac{\mathbf{A}(u, v)}{w(u, v)}$$

Partial derivatives:
$$\mathbf{S}_u(u, v) = \frac{\mathbf{A}_u - w_u \mathbf{S}}{w}, \quad \mathbf{S}_v(u, v) = \frac{\mathbf{A}_v - w_v \mathbf{S}}{w}$$

Surface unit normal:
$$\mathbf{n}(u, v) = \frac{\mathbf{S}_u \times \mathbf{S}_v}{\|\mathbf{S}_u \times \mathbf{S}_v\|}$$

Fundamental forms & Curvatures:
$$E = \mathbf{S}_u \cdot \mathbf{S}_u, \quad F = \mathbf{S}_u \cdot \mathbf{S}_v, \quad G = \mathbf{S}_v \cdot \mathbf{S}_v$$
$$L = \mathbf{S}_{uu} \cdot \mathbf{n}, \quad M = \mathbf{S}_{uv} \cdot \mathbf{n}, \quad N = \mathbf{S}_{vv} \cdot \mathbf{n}$$
Gaussian curvature: $K = \frac{LN - M^2}{EG - F^2}$
Mean curvature: $H = \frac{EN + GL - 2FM}{2(EG - F^2)}$

---

### B. Half-Edge Manifold B-Rep Topology
A 2-manifold boundary representation is decomposed into:
- **Solid**: Closed physical volume.
- **Shell**: Closed 2-manifold surface.
- **Face**: 2D patch bounded by one outer loop and zero or more inner loops.
- **Loop**: Directed cycle of half-edges.
- **Edge**: Unoriented 1D boundary with exactly two twin half-edges.
- **HalfEdge**: Directed segment with `origin`, `twin`, `next`, `prev`, `face`.
- **Vertex**: 3D geometric point.

Euler-Poincaré Formula:
$$V - E + F = 2(S - G) + H$$
For standard simply connected solids ($S=1, G=0, H=0$):
$$V - E + F = 2$$

---

### C. Constructive Solid Geometry (CSG) via BSP Trees
A Binary Space Partitioning tree partitions $\mathbb{R}^3$ solid space:
- Each node stores a plane $\mathcal{P}: \mathbf{n} \cdot \mathbf{x} + d = 0$ and coplanar polygons.
- Polygons are recursively partitioned:
  - `FRONT`: $\mathbf{n} \cdot \mathbf{v} + d > \epsilon$
  - `BACK`: $\mathbf{n} \cdot \mathbf{v} + d < -\epsilon$
  - `COPLANAR`: $|\mathbf{n} \cdot \mathbf{v} + d| \le \epsilon$
  - `SPANNING`: Polygon crosses the plane, split into front and back sub-polygons by plane intersection segments.

Boolean Set Operations:
$$\text{Union}(A, B) = \operatorname{clip}(A, B_{\text{out}}) \cup \operatorname{clip}(B, A_{\text{out}})$$
$$\text{Difference}(A, B) = \operatorname{clip}(A, B_{\text{out}}) \cup \operatorname{invert}(\operatorname{clip}(B, A_{\text{in}}))$$
$$\text{Intersection}(A, B) = \operatorname{clip}(A, B_{\text{in}}) \cup \operatorname{clip}(B, A_{\text{in}})$$

---

## 3. Package Architecture

```
projects/23-solida/
├── solida/
│   ├── __init__.py           # Unified public API export
│   ├── geometry.py          # Vector3D, Matrix4x4, Quaternion, Plane, Ray3D, BoundingBox3D
│   ├── nurbs.py             # Cox-de Boor basis, NURBS curves & surfaces, curvature
│   ├── brep.py              # Half-edge topological structures, Euler operators
│   ├── primitives.py        # Box, Cylinder, Sphere, Cone, Torus exact solids
│   ├── features.py          # Linear Extrude and Rotational Revolve operations
│   ├── csg.py               # BSP tree polygon clipping and CSG Boolean engine
│   ├── tessellation.py      # Ear-clipping triangulation, STL & OBJ exporters
│   ├── visualizer.py        # 3D sub-pixel Braille renderer, camera, telemetry HUD
│   └── kernel.py            # High-level fluent CAD design API
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py     # Vectors, matrices, quaternions, planes, rays, AABB
│   ├── test_nurbs.py        # Basis recursion, curve derivatives, surface normals, curvature
│   ├── test_brep.py         # Half-edge topology, twin pairing, Euler characteristic
│   ├── test_primitives.py   # Primitive solid dimensions, topological validity
│   ├── test_features.py     # Linear extrusions, sweeps, rotational revolution
│   ├── test_csg.py          # CSG Union, Difference, Intersection operations
│   ├── test_tessellation.py # Triangulation, STL and OBJ file exports
│   └── test_kernel.py       # Fluent API, volume/mass properties, transforms
├── benchmarks/
│   └── bench_solida.py      # Microbenchmarks: vector/matrix, NURBS, CSG, Braille FPS
├── examples/
│   └── cad_workbench.py     # Interactive terminal CAD workbench with mechanical parts
├── TASK_QUEUE.md            # Local task tracker
├── CHECKPOINT_LAST.md       # Local progress checkpoint
└── README.md                # Mathematical documentation, architecture, API guide
```
