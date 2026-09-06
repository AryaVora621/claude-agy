# PLAN: Project 33 - ChromaSplat
## 3D Gaussian Splatting, Radiance Fields & Real-Time Volume Rendering Engine

### 1. Executive Summary & Goals
ChromaSplat is an ultra-high-performance, zero-dependency, pure Python 3.10+ standard library implementation of 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) and volumetric radiance fields.

The engine computes exact 3D Gaussian parameterizations (position, scale, quaternion rotations, positive semi-definite covariance matrices), real spherical harmonics (SH degrees 0 through 3) for view-dependent directional radiance, perspective camera projections with the Zwicker EWA Jacobian, low-pass screen-space covariance filtering, tile-based spatial binning, and front-to-back alpha compositing with early ray termination. It features full Stanford PLY format serialization compatible with standard 3DGS workflows, procedural synthetic scene generators, a sub-pixel Unicode Braille visualizer with 24-bit TrueColor ANSI shading, and an interactive terminal workbench.

---

### 2. Mathematical Formulations

#### 2.1 3D Gaussian Representation & 3D Covariance Matrix
A 3D Gaussian is defined by its spatial center $\boldsymbol{\mu} \in \mathbb{R}^3$ and 3D covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$:
$$G(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

To enforce positive semi-definiteness throughout transformations, $\boldsymbol{\Sigma}$ is factorized into a rotation matrix $\mathbf{R} \in \text{SO}(3)$ and diagonal scaling matrix $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$:
$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

The rotation matrix $\mathbf{R}$ is parameterized by a normalized unit quaternion $\mathbf{q} = (q_w, q_x, q_y, q_z)^T$ with $\|\mathbf{q}\| = 1$:
$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

Multiplying $\mathbf{M} = \mathbf{R} \mathbf{S}$, the covariance is:
$$\boldsymbol{\Sigma} = \mathbf{M} \mathbf{M}^T$$

#### 2.2 Real Spherical Harmonics Directional Radiance
View-dependent color $\mathbf{c}(\mathbf{d}) = (r, g, b)^T$ is modeled using real spherical harmonics expansions over normalized viewing direction $\mathbf{d} = (d_x, d_y, d_z)^T = \frac{\boldsymbol{\mu} - \mathbf{c}_{\text{cam}}}{\|\boldsymbol{\mu} - \mathbf{c}_{\text{cam}}\|}$.

For spherical harmonics up to degree $L \in \{0, 1, 2, 3\}$, the color is:
$$\mathbf{c}(\mathbf{d}) = \text{clamp}\left( \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{c}_{lm} Y_l^m(\mathbf{d}) + 0.5, \; 0.0, \; 1.0 \right)$$

Exact real spherical harmonics basis polynomials:
- **Degree 0 ($l=0$)**:
  $$Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}} \approx 0.28209479177387814$$
- **Degree 1 ($l=1$)**:
  $$Y_1^{-1} = -\sqrt{\frac{3}{4\pi}} d_y \approx -0.4886025119029199 d_y$$
  $$Y_1^0 = \sqrt{\frac{3}{4\pi}} d_z \approx 0.4886025119029199 d_z$$
  $$Y_1^1 = -\sqrt{\frac{3}{4\pi}} d_x \approx -0.4886025119029199 d_x$$
- **Degree 2 ($l=2$)**:
  $$Y_2^{-2} = \frac{1}{2} \sqrt{\frac{15}{\pi}} d_x d_y \approx 1.0925484305920792 d_x d_y$$
  $$Y_2^{-1} = -\frac{1}{2} \sqrt{\frac{15}{\pi}} d_y d_z \approx -1.0925484305920792 d_y d_z$$
  $$Y_2^0 = \frac{1}{4} \sqrt{\frac{5}{\pi}} (2 d_z^2 - d_x^2 - d_y^2) \approx 0.31539156525252005 (2 d_z^2 - d_x^2 - d_y^2)$$
  $$Y_2^1 = -\frac{1}{2} \sqrt{\frac{15}{\pi}} d_x d_z \approx -1.0925484305920792 d_x d_z$$
  $$Y_2^2 = \frac{1}{4} \sqrt{\frac{15}{\pi}} (d_x^2 - d_y^2) \approx 0.5462742152960396 (d_x^2 - d_y^2)$$
- **Degree 3 ($l=3$)**:
  7 additional cubic polynomial basis functions.

#### 2.3 Perspective Projection & 2D Screen Covariance (EWA Splatting)
Given camera extrinsic matrix $\mathbf{W} = [\mathbf{R}_{\text{view}} \mid \mathbf{t}_{\text{view}}]$ and intrinsics $(f_x, f_y, c_x, c_y)$ on an image of dimensions $W \times H$:

1. Center in camera space:
   $$\mathbf{t} = \mathbf{R}_{\text{view}} \boldsymbol{\mu} + \mathbf{t}_{\text{view}} = (t_x, t_y, t_z)^T$$
   Frustum culling: discard if $t_z \le z_{\text{near}}$ ($z_{\text{near}} > 0$).

2. Projected screen coordinates:
   $$u = f_x \frac{t_x}{t_z} + c_x, \quad v = f_y \frac{t_y}{t_z} + c_y$$

3. Jacobian of perspective projection $\mathbf{J} \in \mathbb{R}^{2 \times 3}$:
   $$\mathbf{J} = \begin{bmatrix}
   \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\
   0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2}
   \end{bmatrix}$$

4. 2D Screen Covariance Matrix:
   Transforming 3D covariance into camera space $\boldsymbol{\Sigma}_{\text{cam}} = \mathbf{R}_{\text{view}} \boldsymbol{\Sigma} \mathbf{R}_{\text{view}}^T$, the projected screen covariance is:
   $$\boldsymbol{\Sigma}_{2D} = \mathbf{J} \boldsymbol{\Sigma}_{\text{cam}} \mathbf{J}^T + \nu \mathbf{I}_{2 \times 2}$$
   where $\nu \approx 0.3 \text{ pixels}^2$ is a low-pass Gaussian anti-aliasing filter preventing singular or sub-pixel matrices.

5. Screen Bounding Box ($3\sigma$ Ellipse):
   For $\boldsymbol{\Sigma}_{2D} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$, the eigenvalues are:
   $$\lambda_{1,2} = \frac{a + c}{2} \pm \sqrt{\left(\frac{a - c}{2}\right)^2 + b^2}$$
   The $3\sigma$ bounding radius is:
   $$r_{\text{max}} = \lceil 3.0 \sqrt{\max(\lambda_1, \lambda_2)} \rceil$$

#### 2.4 Volume Alpha Compositing & Rasterization
Gaussians are sorted along camera depth $t_z$ ascending (front-to-back):
$$t_z^{(1)} \le t_z^{(2)} \le \dots \le t_z^{(N)}$$

For a pixel at $(x, y)$, let $\Delta = (x - u, y - v)^T$.
The 2D Gaussian Mahalanobis distance squared is:
$$\tau = \Delta^T \boldsymbol{\Sigma}_{2D}^{-1} \Delta = \frac{c (x - u)^2 - 2 b (x - u)(y - v) + a (y - v)^2}{a c - b^2}$$

If $\tau \le 9.0$ ($3\sigma$ cutoff):
$$G(\Delta) = \exp\left(-\frac{1}{2} \tau\right)$$
$$\alpha = \alpha_0 \cdot G(\Delta)$$

The pixel color is accumulated front-to-back:
$$\mathbf{C}_{\text{accum}} \leftarrow \mathbf{C}_{\text{accum}} + \mathbf{c}_i \cdot \alpha \cdot T$$
$$T \leftarrow T \cdot (1 - \alpha)$$
where $T$ is the remaining ray transmittance (initialized to $1.0$).
When $T < 10^{-4}$, the ray saturates and stops evaluating remaining Gaussians.

---

### 3. Architecture & File Structure

```
projects/33-chromasplat/
|-- chromasplat/
|   |-- __init__.py              # Public package exports
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
|   |-- bench_chromasplat.py     # Microbenchmark suite for all computational kernels
|-- examples/
|   |-- splat_workbench.py       # Interactive terminal 3D Gaussian Splatting lab
|-- PLAN.md                      # This specification
|-- TASK_QUEUE.md                # Task tracking
|-- CHECKPOINT_LAST.md           # Progress checkpoint
|-- README.md                    # Comprehensive documentation
```

---

### 4. Zero-Dependency & Clean Code Guidelines
- Pure Python 3.10+ standard library exclusively (`math`, `dataclasses`, `typing`, `struct`, `time`, `pathlib`, `unittest`).
- Zero external dependencies (`numpy`, `scipy`, `torch`, `matplotlib` forbidden).
- Strict avoidance of all em dashes across all files.
- Code comments explain why, not what.
- 100% test coverage with verifiable mathematical correctness.
