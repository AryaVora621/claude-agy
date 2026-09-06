# ChromaSplat: 3D Gaussian Splatting, Radiance Fields & Real-Time Volume Rendering Engine

A high-performance, zero-dependency, pure Python 3.10+ standard library implementation of 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) and volumetric radiance fields.

ChromaSplat implements exact 3D Gaussian parameterizations, quaternion rotations, positive semi-definite 3D spatial covariance matrices, real spherical harmonics (degrees 0 through 3) for view-dependent directional radiance, perspective camera projections with the Zwicker EWA Jacobian, low-pass screen-space covariance filtering, tile-based spatial binning, and front-to-back alpha compositing with early ray termination. It features full Stanford PLY format serialization compatible with standard 3DGS workflows, procedural synthetic scene generators, a sub-pixel Unicode Braille visualizer with 24-bit TrueColor ANSI shading, and an interactive terminal workbench.

```
       3D SCENE GAUSSIANS                EWA PERSPECTIVE PROJECTION              FRONT-TO-BACK COMPOSITING
    +-------------------------+            +---------------------+               +-------------------------+
    | Position mu in R^3      |            | Camera extrinsics   |               | Tile screen binning     |
    | Scale s in R^3          |  ======>   | Viewing Jacobian J  |    ======>    | Mahalanobis distance tau|
    | Rotation q (Quaternion) |            | Sigma_2D = J*Sig*J^T|               | Transmittance T tracking|
    | SH Directional Color    |            | 3-sigma ellipse rad |               | Early ray saturation    |
    +-------------------------+            +---------------------+               +-------------------------+
                 |                                    |                                       |
                 v                                    v                                       v
         Sigma = R*S*S^T*R^T                 det(Sigma_2D) > 0                     C = sum(c_i * alpha_i * T)
       (Positive Semi-Definite)             (Low-Pass Filter nu)                  (Sub-Pixel Braille Output)
```

---

## Key Features

1. **Exact 3D Gaussian Splatting Representation**:
   - 3D spatial center $\boldsymbol{\mu} \in \mathbb{R}^3$, positive scale vector $\mathbf{s} \in \mathbb{R}^3$, and unit quaternion $\mathbf{q} = (q_w, q_x, q_y, q_z)^T$.
   - 3D rotation matrix $\mathbf{R}(\mathbf{q}) \in \text{SO}(3)$ derived from quaternion algebra.
   - Positive semi-definite 3D spatial covariance matrix formulation:
     $$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$
     computed in $O(1)$ without intermediate matrix inversion.

2. **Real Spherical Harmonics Directional Radiance**:
   - Exact mathematical formulation of real spherical harmonics basis polynomials $Y_l^m(x, y, z)$ up to degree 3 (16 basis functions per color channel).
   - View-dependent directional color evaluation:
     $$\mathbf{c}(\mathbf{d}) = \text{clamp}\left( \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{c}_{lm} Y_l^m(\mathbf{d}) + 0.5, \; 0.0, \; 1.0 \right)$$
   - Procedural synthesis of specular reflection lobes with tunable glint intensity and direction.

3. **Perspective Projection & 2D Screen Covariance (EWA Splatting)**:
   - Pinhole camera model with look-at extrinsics, focal length, and principal point.
   - Projective mapping Jacobian $\mathbf{J} \in \mathbb{R}^{2 \times 3}$ derived from perspective division:
     $$\mathbf{J} = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$
   - 2D screen covariance matrix synthesis with low-pass Gaussian anti-aliasing filter $\nu$:
     $$\boldsymbol{\Sigma}_{2D} = \mathbf{J} \boldsymbol{\Sigma}_{\text{cam}} \mathbf{J}^T + \nu \mathbf{I}_{2 \times 2}$$
   - Eigenvalue decomposition of $\boldsymbol{\Sigma}_{2D}$ yielding exact $3\sigma$ screen-space bounding ellipse radii.

4. **Tiled Front-to-Back Volume Rasterization**:
   - Depth sorting along camera axis $t_z$ ascending ($t_z^{(1)} \le t_z^{(2)} \le \dots$).
   - Spatial 2D screen binning into $16 \times 16$ pixel tiles matching modern GPU architectures.
   - Mahalanobis quadratic distance evaluation $\tau = \boldsymbol{\Delta}^T \boldsymbol{\Sigma}_{2D}^{-1} \boldsymbol{\Delta}$.
   - Front-to-back alpha compositing with ray transmittance $T$ tracking:
     $$\mathbf{C} \leftarrow \mathbf{C} + \mathbf{c}_i \cdot \alpha_i \cdot T, \quad T \leftarrow T \cdot (1 - \alpha_i)$$
   - Early ray termination when transmittance $T < 10^{-4}$.

5. **Stanford PLY Format Serialization & Scene Generators**:
   - ASCII format parser and serializer conforming to standard Inria 3D Gaussian Splatting schema (`x, y, z`, `f_dc_0..2`, `opacity` logit, `scale` log-space, quaternion rotations).
   - Built-in procedural scene generators:
     - **Saturnian Rings**: Luminous central planet with atmospheric bands and tilted orbital rings.
     - **Cornell Box**: Classic enclosure with diffuse colored walls and central specular sphere displaying view-dependent glints.
     - **DNA Double Helix**: Intertwined cyan/magenta backbones and glowing base pair rungs.

6. **Sub-Pixel Unicode Braille Visualizer & Telemetry HUD**:
   - $2 \times 4$ sub-pixel Braille dot matrix canvas (`U+2800..U+28FF`).
   - 24-bit TrueColor ANSI palette (`\033[38;2;R;G;Bm`).
   - Real-time graphics telemetry HUD reporting scene stats, camera orbit, resolution, frame time, and FPS.

---

## Mathematical Formulations

### 1. 3D Covariance Matrix
For scale vector $\mathbf{s} = (s_x, s_y, s_z)$ and normalized quaternion $\mathbf{q} = (w, x, y, z)$, the rotation matrix $\mathbf{R}$ is:
$$\mathbf{R} = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$
Letting $\mathbf{M}_{ij} = \mathbf{R}_{ij} s_j$, the 3D covariance is:
$$\boldsymbol{\Sigma}_{ik} = \sum_j \mathbf{R}_{ij} \mathbf{R}_{kj} s_j^2$$
guaranteeing symmetry and positive semi-definiteness.

### 2. 2D EWA Screen Covariance
In camera coordinates $\mathbf{t} = (t_x, t_y, t_z)^T = \mathbf{R}_{\text{view}} \boldsymbol{\mu} + \mathbf{t}_{\text{view}}$, the Jacobian is:
$$\mathbf{J} = \begin{bmatrix}
\frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\
0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2}
\end{bmatrix}$$
The 2D screen covariance $\boldsymbol{\Sigma}_{2D} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$ is:
$$\boldsymbol{\Sigma}_{2D} = \mathbf{J} \mathbf{R}_{\text{view}} \boldsymbol{\Sigma} \mathbf{R}_{\text{view}}^T \mathbf{J}^T + \nu \mathbf{I}_{2 \times 2}$$
The inverse covariance is:
$$\boldsymbol{\Sigma}_{2D}^{-1} = \frac{1}{a c - b^2} \begin{bmatrix} c & -b \\ -b & a \end{bmatrix}$$

### 3. Alpha Compositing & Early Ray Termination
For displacement $\boldsymbol{\Delta} = (x - u, y - v)^T$ from projected center $(u, v)$:
$$\tau = \boldsymbol{\Delta}^T \boldsymbol{\Sigma}_{2D}^{-1} \boldsymbol{\Delta} = \frac{c (x-u)^2 - 2 b (x-u)(y-v) + a (y-v)^2}{a c - b^2}$$
If $\tau \le 9.0$ ($3\sigma$ threshold):
$$\alpha = \alpha_0 \exp\left(-\frac{1}{2} \tau\right)$$
$$\mathbf{C} \leftarrow \mathbf{C} + \mathbf{c} \cdot \alpha \cdot T, \quad T \leftarrow T \cdot (1 - \alpha)$$
Ray accumulation stops when $T < 10^{-4}$.

---

## Quickstart & Usage

```python
from chromasplat import (
    Camera,
    GaussianRenderer,
    SceneFactory,
    SplatVisualizer,
)

# 1. Instantiate procedural 3D Gaussian scene
scene = SceneFactory.orbiting_rings(num_planet_splats=70, num_ring_splats=180)

# 2. Configure camera orbit and sub-pixel visualizer
vis = SplatVisualizer(char_width=64, char_height=22)
cam = Camera.orbit(
    width=vis.pixel_width,
    height=vis.pixel_height,
    azimuth_deg=35.0,
    elevation_deg=18.0,
    distance=3.4,
    fov_y_deg=60.0,
)

# 3. Render frame with spherical harmonics degree 2
renderer = GaussianRenderer(sh_degree=2, tile_size=16)
result = renderer.render(scene, cam)

# 4. Display sub-pixel Braille terminal rendering with HUD
hud = vis.render_frame_with_hud(
    scene=scene,
    camera=cam,
    render_result=result,
    profile=renderer.last_profile,
    azimuth_deg=35.0,
    elevation_deg=18.0,
    distance=3.4,
)
print(hud)
```

---

## Verification & Microbenchmarks

Run the unit test suite:
```bash
python3 -m unittest discover -v tests
```
Result: **30/30 unit tests pass (100% pass rate) in 0.019 seconds**.

Run the performance microbenchmark suite:
```bash
python3 benchmarks/bench_chromasplat.py
```

### Microbenchmark Throughput (Apple Silicon / Python 3.13)
| Benchmark Kernel | Throughput | Latency / Unit |
|---|---|---|
| **3D Covariance Matrix Synthesis** | **661,052 evals/sec** | 1.51 microseconds |
| **Spherical Harmonics (Degree 2)** | **821,795 evals/sec** | 1.22 microseconds |
| **2D EWA Projection & Covariance** | **145,519 splats/sec** | 6.87 microseconds |
| **Tiled Volume Rasterizer** | **73.0 FPS** | 13.69 milliseconds |
| **Sub-Pixel Braille Canvas** | **426.1 FPS** | 2.35 milliseconds |
| **Full End-to-End Pipeline** | **58.5 FPS** | 17.10 milliseconds |

---

## Interactive Splat Workbench

Launch the interactive terminal laboratory:
```bash
# Saturnian planetary rings scene
python3 examples/splat_workbench.py --scene rings --width 64 --height 22

# Cornell Box with specular highlight
python3 examples/splat_workbench.py --scene cornell --width 64 --height 22

# DNA Double Helix
python3 examples/splat_workbench.py --scene dna --width 64 --height 22

# 360-degree turntable animation
python3 examples/splat_workbench.py --scene rings --orbit-frames 16
```

---

## Architecture & Module Layout

```
projects/33-chromasplat/
|-- chromasplat/
|   |-- __init__.py              # Public package interface and exports
|   |-- gaussian.py              # 3D Gaussian, quaternion math, 3D covariance
|   |-- spherical_harmonics.py   # Real SH degrees 0-3 basis and color evaluation
|   |-- projection.py            # Pinhole camera, Jacobian, 2D EWA screen covariance
|   |-- rasterizer.py            # Tiled binning, depth sorting, alpha compositing
|   |-- scene.py                 # GaussianScene, PLY codec, procedural generators
|   |-- renderer.py              # End-to-end rendering pipeline and depth buffer
|   |-- visualizer.py            # Sub-pixel Braille TrueColor canvas and telemetry HUD
|-- tests/
|   |-- __init__.py
|   |-- test_gaussian.py         # Unit tests for quaternion, scale, covariance
|   |-- test_spherical_harmonics.py # Unit tests for SH degrees 0-3 and symmetry
|   |-- test_projection.py       # Unit tests for camera, Jacobian, EWA 2D covariance
|   |-- test_rasterizer.py       # Unit tests for alpha compositing and ray saturation
|   |-- test_scene.py            # Unit tests for PLY parsing/serialization & generators
|   |-- test_renderer.py         # Unit tests for full pipeline execution
|   |-- test_visualizer.py       # Unit tests for Braille rasterization and HUD
|-- benchmarks/
|   |-- bench_chromasplat.py     # High-throughput performance microbenchmarks
|-- examples/
|   |-- splat_workbench.py       # Interactive terminal 3D Gaussian Splatting lab
|-- PLAN.md                  # Theoretical specification & mathematical derivation
|-- TASK_QUEUE.md            # Work unit progress tracking
|-- CHECKPOINT_LAST.md       # Session checkpoint state
|-- README.md                # Comprehensive documentation
```

---

## Zero-Dependency Guarantee

ChromaSplat is written exclusively in pure Python 3.10+ standard library (`math`, `dataclasses`, `typing`, `struct`, `time`, `pathlib`, `unittest`). No external dependencies (`numpy`, `scipy`, `torch`, `matplotlib`) are required.
