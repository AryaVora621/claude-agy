# PLAN: Project 21 - Structura (Finite Element Analysis & Continuum Structural Mechanics Engine)

## 1. System Overview & Engineering Scope
Structura is a high-performance, zero-dependency 2D Finite Element Analysis (FEA), continuum structural mechanics, and modal vibration analysis engine authored in pure Python 3 standard library. It provides end-to-end capabilities from parametric geometry meshing to sparse global matrix assembly, Preconditioned Conjugate Gradient (PCG) linear solvers, element stress recovery, yield failure analysis, and generalized eigenvalue modal resonance extraction.

Key technical pillars:
1. **Element Formulations**:
   - 2D Bar/Truss Element (2-node axial tension/compression).
   - 2D Euler-Bernoulli Frame/Beam Element (2-node axial + shear + bending with 3 DOFs/node).
   - 2D Constant Strain Triangle (CST / T3) for continuum plane stress and plane strain.
   - 2D 4-Node Bilinear Isoparametric Quadrilateral (Q4) with $2 \times 2$ Gauss-Legendre numerical quadrature.
2. **Constitutive Laws**:
   - Linear isotropic elasticity with Young's modulus $E$ and Poisson's ratio $\nu$.
   - Plane stress ($\sigma_z = 0$, thin membranes/plates) vs Plane strain ($\varepsilon_z = 0$, thick dams/cylinders) constitutive elasticity matrices $\mathbf{D}$.
3. **High-Performance Sparse Assembly & Solvers**:
   - Dictionary of Keys (DOK) assembly converting into Compressed Sparse Row (CSR) matrix.
   - Preconditioned Conjugate Gradient (PCG) solver with Jacobi diagonal preconditioning guaranteeing rapid convergence for large DOF systems without external numerical libraries.
4. **Stress Field Recovery & Failure Criteria**:
   - Cauchy stress recovery $\boldsymbol{\sigma} = \mathbf{D} \mathbf{B} \mathbf{u} = (\sigma_x, \sigma_y, \tau_{xy})$.
   - Principal stresses $\sigma_1, \sigma_2 = \frac{\sigma_x + \sigma_y}{2} \pm \sqrt{\left(\frac{\sigma_x - \sigma_y}{2}\right)^2 + \tau_{xy}^2}$.
   - Von Mises equivalent stress $\sigma_v = \sqrt{\sigma_x^2 - \sigma_x \sigma_y + \sigma_y^2 + 3\tau_{xy}^2}$.
   - Factor of Safety (FoS) based on material yield limit $\sigma_y$.
5. **Modal Dynamics & Vibration Resonance**:
   - Global lumped mass matrix $\mathbf{M}$.
   - Generalized eigenvalue problem $(\mathbf{K} - \omega^2 \mathbf{M}) \boldsymbol{\phi} = \mathbf{0}$.
   - Shifted inverse power iteration with Gram-Schmidt deflation solving natural resonant frequencies $f_i = \omega_i / (2\pi)$ and harmonic mode shapes $\boldsymbol{\phi}_i$.
6. **Sub-Pixel Unicode Braille 2D Visualizer**:
   - $2 \times 4$ sub-pixel Braille canvas (`U+2800..U+28FF`) providing 144x84 resolution in terminal.
   - Deformed wireframe vs undeformed geometry overlay.
   - 24-bit TrueColor ANSI Von Mises stress contour heatmaps.
   - Structural telemetry HUD (DOFs, deflection, max stress, FoS, compliance, resonance frequencies).

---

## 2. Directory Structure
```
projects/21-structura/
├── structura/
│   ├── __init__.py           # Public exports
│   ├── types.py              # Node2D, Material, BoundaryCondition, NodalLoad, Stress/Strain
│   ├── elements.py           # Truss2D, Beam2D, TriangleCST, QuadQ4 isoparametric
│   ├── mesh.py               # Mesh2D, beam, truss bridge, perforated plate generators
│   ├── sparse.py             # DOKMatrix, CSRMatrix, Preconditioned Conjugate Gradient (PCG)
│   ├── solver.py             # Global assembly, boundary conditions, displacement & stress solve
│   ├── modal.py              # Generalized eigenvalue solver, natural frequencies, mode shapes
│   └── visualizer.py         # Sub-pixel Braille deformed mesh, TrueColor stress heatmap, HUD
├── tests/
│   ├── test_types.py         # Material D-matrix, StressTensor2D invariants, Node2D
│   ├── test_elements.py      # Element stiffness symmetry, rigid body modes, patch tests
│   ├── test_sparse.py        # CSR matrix-vector multiplication, PCG convergence
│   ├── test_mesh.py          # Parametric mesh generation, connectivity, DOF mapping
│   ├── test_solver.py        # Cantilever deflection vs Euler-Bernoulli theory, Kirsch problem
│   └── test_modal.py         # Cantilever natural resonance frequencies vs analytical solution
├── benchmarks/
│   └── bench_fea.py          # Assembly speed, PCG solves/sec, element evaluations, Braille FPS
├── examples/
│   └── structural_lab.py     # Interactive cantilever, truss bridge, and plate with hole
├── PLAN.md                   # This document
├── TASK_QUEUE.md             # Autonomous task management
├── CHECKPOINT_LAST.md        # State tracking
└── README.md                 # Complete technical documentation
```

---

## 3. Implementation Phasing
- Phase 1: Core data structures, material matrices, and stress/strain tensors (`types.py`).
- Phase 2: Element stiffness and mass matrices for Truss, Beam, CST (T3), and Quad (Q4) with Gauss quadrature (`elements.py`).
- Phase 3: Sparse matrix structures (DOK, CSR) and Jacobi Preconditioned Conjugate Gradient solver (`sparse.py`).
- Phase 4: Mesh representation and parametric generators (`mesh.py`).
- Phase 5: Global FEA solver: assembly, boundary conditions, displacement solve, reaction forces, and stress recovery (`solver.py`).
- Phase 6: Modal eigenvalue analysis for vibration natural frequencies and mode shapes (`modal.py`).
- Phase 7: Sub-pixel Unicode Braille visualizer with TrueColor ANSI stress contours and HUD (`visualizer.py`).
- Phase 8: Comprehensive test suite, performance benchmarks, and interactive structural lab demonstration.
- Phase 9: Integration into master `showcase.py`, `projects.md`, and `tracker/data.json`.
