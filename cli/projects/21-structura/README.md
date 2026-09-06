# Structura: Finite Element Analysis, Continuum Mechanics & Modal Dynamics Engine

A high-performance, first-principles Finite Element Analysis (FEA), continuum structural mechanics, and structural vibration dynamics engine implemented entirely in the pure Python standard library with zero external dependencies.

---

## Architectural Overview

Structura provides a complete structural engineering simulation pipeline:
1. **Constitutive Modeling**: Generalized Hooke's law for isotropic linear elasticity under Plane Stress ($\sigma_z = 0$) and Plane Strain ($\varepsilon_z = 0$) states.
2. **Multi-Element Library**:
   - `Truss2D`: 2-node pin-jointed bar element carrying pure axial forces with coordinate rotation transformations.
   - `Beam2D`: 2-node Euler-Bernoulli frame element carrying coupled axial, shear, and bending moments ($6 \times 6$ stiffness).
   - `TriangleCST`: 3-node Constant Strain Triangle (T3) for 2D continuum elasticity with linear shape function gradients.
   - `QuadQ4`: 4-node bilinear isoparametric quadrilateral element using natural coordinates $(\xi, \eta)$, analytical Jacobian transformation, and $2 \times 2$ Gauss-Legendre numerical quadrature.
3. **Sparse Linear Algebra & Solvers**:
   - Dictionary of Keys (`DOKMatrix`) for $O(1)$ global stiffness matrix assembly.
   - Compressed Sparse Row (`CSRMatrix`) for cache-efficient $O(\text{nnz})$ sparse matrix-vector multiplication.
   - Jacobi Preconditioned Conjugate Gradient (`solve_pcg`) solver with exact Dirichlet boundary degree-of-freedom partitioning ($\mathbf{K}_{ff} \mathbf{u}_f = \mathbf{F}_f - \mathbf{K}_{fp} \mathbf{u}_p$).
4. **Stress Recovery & Failure Invariants**:
   - Cauchy stress tensor evaluation $[\sigma_x, \sigma_y, \tau_{xy}]^T = \mathbf{D} \mathbf{B} \mathbf{u}_e$.
   - Mohr's circle principal stresses $\sigma_1, \sigma_2$ and maximum in-plane shear $\tau_{\max}$.
   - Von Mises equivalent yield stress $\sigma_v = \sqrt{\sigma_x^2 - \sigma_x \sigma_y + \sigma_y^2 + 3\tau_{xy}^2}$.
   - Material yield factor of safety $\text{FoS} = \sigma_{\text{yield}} / \sigma_v$.
   - Internal compliance strain energy $U = \frac{1}{2} \mathbf{u}^T \mathbf{K} \mathbf{u}$.
5. **Modal Dynamics & Vibration Resonances**:
   - Lumped diagonal mass matrix assembly $\mathbf{M}$.
   - Shifted inverse power iteration with Gram-Schmidt mass-orthogonal deflation for extracting the lowest $m$ resonant frequencies and mode shapes: $(\mathbf{K} - \omega^2 \mathbf{M})\boldsymbol{\phi} = \mathbf{0}$.
6. **Sub-Pixel Terminal Visualization**:
   - Unicode Braille 2x4 sub-pixel canvas (`U+2800..U+28FF`) providing $144 \times 80$ effective resolution in standard terminal viewports.
   - Dual wireframe overlay (dim undeformed geometry vs magnified displaced geometry).
   - 24-bit TrueColor ANSI Von Mises stress contour heatmaps with Turbo colormap.

---

## Mathematical Formulations

### 1. Linear Elasticity Constitutive Matrices ($\mathbf{D}$)

Relating Cauchy stress $\boldsymbol{\sigma} = [\sigma_x, \sigma_y, \tau_{xy}]^T$ to engineering strain $\boldsymbol{\varepsilon} = [\varepsilon_x, \varepsilon_y, \gamma_{xy}]^T$:

$$\boldsymbol{\sigma} = \mathbf{D} \boldsymbol{\varepsilon}$$

#### Plane Stress ($\sigma_z = 0$, thin plates):
$$\mathbf{D}_{\text{stress}} = \frac{E}{1 - \nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1 - \nu}{2} \end{bmatrix}$$

#### Plane Strain ($\varepsilon_z = 0$, thick dams, tunnels):
$$\mathbf{D}_{\text{strain}} = \frac{E}{(1 + \nu)(1 - 2\nu)} \begin{bmatrix} 1 - \nu & \nu & 0 \\ \nu & 1 - \nu & 0 \\ 0 & 0 & \frac{1 - 2\nu}{2} \end{bmatrix}$$

---

### 2. Isoparametric Quadrilateral (Quad Q4) Formulation

Mapping natural coordinates $(\xi, \eta) \in [-1, 1] \times [-1, 1]$ to physical domain $(x, y)$:

$$N_i(\xi, \eta) = \frac{1}{4} (1 + \xi_i \xi)(1 + \eta_i \eta), \quad i = 1, 2, 3, 4$$

The Jacobian matrix $\mathbf{J}$ relates natural and physical gradients:

$$\mathbf{J} = \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\ \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta} \end{bmatrix} = \begin{bmatrix} \sum \frac{\partial N_i}{\partial \xi} x_i & \sum \frac{\partial N_i}{\partial \xi} y_i \\ \sum \frac{\partial N_i}{\partial \eta} x_i & \sum \frac{\partial N_i}{\partial \eta} y_i \end{bmatrix}$$

Physical shape function gradients are recovered via $\mathbf{J}^{-1}$:

$$\begin{bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{bmatrix} = \mathbf{J}^{-1} \begin{bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \end{bmatrix}$$

The $3 \times 8$ strain-displacement matrix $\mathbf{B}$ at any point $(\xi, \eta)$ is:

$$\mathbf{B}_i = \begin{bmatrix} \frac{\partial N_i}{\partial x} & 0 \\ 0 & \frac{\partial N_i}{\partial y} \\ \frac{\partial N_i}{\partial y} & \frac{\partial N_i}{\partial x} \end{bmatrix}$$

The element stiffness matrix is computed via $2 \times 2$ Gauss-Legendre quadrature points ($\xi_g, \eta_g = \pm \frac{1}{\sqrt{3}}$, weights $w_g = 1.0$):

$$\mathbf{K}_e = t \int_{-1}^1 \int_{-1}^1 \mathbf{B}^T \mathbf{D} \mathbf{B} \det(\mathbf{J}) \, d\xi \, d\eta = t \sum_{p=1}^4 w_p \mathbf{B}(\xi_p, \eta_p)^T \mathbf{D} \mathbf{B}(\xi_p, \eta_p) \det(\mathbf{J}(\xi_p, \eta_p))$$

---

### 3. Exact Dirichlet Boundary Partitioning

Rather than introducing artificial large penalty springs that degrade system condition numbers, Structura mathematically partitions global degrees of freedom into free DOFs ($f$) and prescribed boundary DOFs ($p$):

$$\begin{bmatrix} \mathbf{K}_{ff} & \mathbf{K}_{fp} \\ \mathbf{K}_{pf} & \mathbf{K}_{pp} \end{bmatrix} \begin{bmatrix} \mathbf{u}_f \\ \mathbf{u}_p \end{bmatrix} = \begin{bmatrix} \mathbf{F}_f \\ \mathbf{F}_p \end{bmatrix}$$

Yielding the symmetric positive-definite reduced system for active displacements:

$$\mathbf{K}_{ff} \mathbf{u}_f = \mathbf{F}_f - \mathbf{K}_{fp} \mathbf{u}_p$$

Reactions at support nodes are subsequently calculated without numerical cancellation:

$$\mathbf{R}_p = \mathbf{K}_{pf} \mathbf{u}_f + \mathbf{K}_{pp} \mathbf{u}_p - \mathbf{F}_p$$

---

### 4. Generalized Eigenvalue Modal Dynamics

Structural natural vibration frequencies $\omega$ and mode shapes $\boldsymbol{\phi}$ satisfy:

$$(\mathbf{K} - \omega^2 \mathbf{M}) \boldsymbol{\phi} = \mathbf{0}$$

Structura solves this via shifted inverse power iteration. For higher harmonic modes ($k > 1$), Gram-Schmidt mass-orthogonal deflation ensures numerical stability:

$$\mathbf{v} \leftarrow \mathbf{v} - \sum_{j=1}^{k-1} (\boldsymbol{\phi}_j^T \mathbf{M} \mathbf{v}) \boldsymbol{\phi}_j$$

Resonant angular frequency is evaluated from the Rayleigh quotient:

$$\omega_k^2 = \frac{\boldsymbol{\phi}_k^T \mathbf{K} \boldsymbol{\phi}_k}{\boldsymbol{\phi}_k^T \mathbf{M} \boldsymbol{\phi}_k}, \quad f_k = \frac{\omega_k}{2\pi} \text{ [Hz]}$$

---

## Performance Benchmarks

Benchmarked on Apple Silicon (single thread, pure Python 3 standard library):

| Operation | Metric | Performance |
|---|---|---|
| **Truss2D Stiffness** | Element evals / sec | **2,342,972 elem/s** |
| **TriangleCST Stiffness** | Element evals / sec | **134,011 elem/s** |
| **Beam2D Frame Stiffness** | Element evals / sec | **91,711 elem/s** |
| **QuadQ4 2x2 Gauss Quadrature**| Element evals / sec | **16,944 elem/s** |
| **Global Sparse Assembly** | 378 DOFs / 160 Quads | **12.32 ms** |
| **Jacobi PCG Solver** | Effective Throughput | **10,630 DOFs/s** (35.5 ms to $5.6 \times 10^{-8}$) |
| **Modal Vibration Extraction**| 3 modes / 44 DOFs | **22.90 ms** |
| **Braille Wireframe Deformation** | Render Frame Rate | **427.7 FPS** (2.34 ms/frame) |
| **Braille TrueColor Stress Heatmap**| Render Frame Rate | **736.4 FPS** (1.36 ms/frame) |

---

## Interactive Structural Engineering Laboratory

Structura includes an interactive terminal laboratory demonstrating three classic civil and mechanical engineering benchmarks:

```bash
python3 examples/structural_lab.py
```

### Case Studies Included:
1. **Continuum Cantilever Deep Beam**: 2D QuadQ4 continuum mesh vs Euler-Bernoulli analytical tip deflection $\delta = \frac{P L^3}{3 E I}$, verifying convergence within 3.5% including shear deformation.
2. **Pratt Truss Highway Bridge**: Live vehicle traversing 5 bays of a 30-meter steel bridge, monitoring peak tension/compression chord stresses and fundamental modal resonance.
3. **Kirsch Perforated Plate**: Uniaxial tension on a plate with circular hole, resolving the classical stress concentration factor $K_t \to 3.00$.

To run the automated performance benchmark suite:
```bash
python3 benchmarks/bench_fea.py
```

To run all unit tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## Python API Quickstart

```python
from structura.types import Material, STRUCTURAL_STEEL
from structura.mesh import generate_rectangular_mesh
from structura.solver import FEASolver
from structura.modal import ModalSolver
from structura.visualizer import render_mesh_deformation, render_stress_field

# 1. Generate 2D continuum mesh (16x4 Quads)
mesh = generate_rectangular_mesh(length=6.0, height=1.0, nx=16, ny=4, material=STRUCTURAL_STEEL)

# 2. Apply clamped boundary conditions at root (x = 0)
for nid in mesh.find_nodes_at_x(0.0):
    mesh.add_boundary_condition(nid, dof=0, prescribed_value=0.0)  # Ux = 0
    mesh.add_boundary_condition(nid, dof=1, prescribed_value=0.0)  # Uy = 0

# 3. Apply downward shear load at tip (x = 6)
tip_nodes = mesh.find_nodes_at_x(6.0)
for nid in tip_nodes:
    mesh.add_nodal_load(nid, fy=-50000.0 / len(tip_nodes))

# 4. Solve static equilibrium
solver = FEASolver(mesh)
result = solver.solve()
print(f"Max Tip Deflection: {result.max_displacement * 1000:.3f} mm")
print(f"Peak Von Mises Stress: {result.max_von_mises_stress / 1e6:.2f} MPa")

# 5. Extract natural vibration resonances
modal = ModalSolver(mesh)
modal_res = modal.solve(num_modes=3)
print(f"Fundamental Frequency: {modal_res.fundamental_frequency_hz:.2f} Hz")

# 6. Render sub-pixel Braille visualization
print(render_mesh_deformation(mesh, result.displacements))
print(render_stress_field(mesh, result.nodal_von_mises))
```
