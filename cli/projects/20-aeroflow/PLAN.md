# AeroFlow: Computational Fluid Dynamics, Lattice Boltzmann & Navier-Stokes Engine
## Architectural Design & Technical Specification (PLAN.md)

### 1. Executive Summary & Core Objectives
AeroFlow is a zero-dependency, high-performance Computational Fluid Dynamics (CFD) simulation engine built entirely from first principles in the pure Python 3 standard library. It implements two complementary, mathematically rigorous fluid modeling paradigms:
1. **Mesoscopic Lattice Boltzmann Method (LBM)**: D2Q9 lattice BGK collision and streaming model for compressible/quasi-incompressible flows, boundary momentum exchange, and unsteady vortex shedding (Von Kármán streets).
2. **Macroscopic Incompressible Navier-Stokes Solver**: Eulerian grid solver using Chorin's fractional step projection method, semi-Lagrangian advection, implicit diffusion, and Red-Black Gauss-Seidel pressure Poisson solver ensuring strict divergence-free velocity fields ($\nabla \cdot \mathbf{u} = 0$).

Both solvers include aerodynamic force calculation (Lift $C_L$, Drag $C_D$), obstacle geometry generators (NACA 4-digit airfoils, bluff cylinders, flat plates), vorticity computation, stream functions, and a high-resolution sub-pixel Unicode Braille visualizer with TrueColor ANSI styling.

---

### 2. Mathematical Foundations

#### 2.1 Lattice Boltzmann Method (D2Q9 BGK Model)
* **Discrete Velocities $\mathbf{e}_i$**:
  * $i=0$: $(0, 0)$ with lattice weight $w_0 = 4/9$.
  * $i=1, 2, 3, 4$: $(\pm 1, 0), (0, \pm 1)$ with lattice weight $w_{1..4} = 1/9$.
  * $i=5, 6, 7, 8$: $(\pm 1, \pm 1)$ with lattice weight $w_{5..8} = 1/36$.
* **Speed of Sound**: $c_s = 1/\sqrt{3}$, $c_s^2 = 1/3$.
* **Maxwell-Boltzmann Lattice Equilibrium**:
  $$f_i^{\text{eq}}(\rho, \mathbf{u}) = w_i \rho \left( 1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2} \right)$$
* **Bhatnagar-Gross-Krook (BGK) Relaxation**:
  $$f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left[ f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\rho, \mathbf{u}) \right]$$
  where $\tau = 3\nu + 0.5$ and $\nu$ is kinematic viscosity in lattice units.
* **Streaming Step**:
  $$f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)$$
* **Macroscopic Variable Recovery**:
  $$\rho(\mathbf{x}, t) = \sum_{i=0}^8 f_i(\mathbf{x}, t), \quad \mathbf{u}(\mathbf{x}, t) = \frac{1}{\rho} \sum_{i=0}^8 f_i(\mathbf{x}, t) \mathbf{e}_i, \quad p = c_s^2 \rho$$

#### 2.2 Boundary Conditions & Aerodynamic Forces
* **Half-Way Bounce-Back**: On solid boundary cells, particles reflect backward into the fluid:
  $$f_{\bar{i}}(\mathbf{x}_f, t+\Delta t) = f_i^*(\mathbf{x}_f, t)$$
  where $\mathbf{e}_{\bar{i}} = -\mathbf{e}_i$.
* **Zou-He Velocity Boundary (Inlet)**: Prescribes velocity $\mathbf{u} = (U_\infty, 0)$ and computes missing boundary distributions from macroscopic momentum conservation.
* **Momentum Exchange Method (Lift & Drag)**:
  $$\mathbf{F} = \sum_{\mathbf{x}_f} \sum_{i} \mathbf{e}_i \left( f_i^*(\mathbf{x}_f, t) + f_{\bar{i}}(\mathbf{x}_f, t+\Delta t) \right)$$
  Yielding total drag force $F_D = F_x$ and lift force $F_L = F_y$.
  Non-dimensional aerodynamic coefficients:
  $$C_D = \frac{2 F_D}{\rho_0 U_\infty^2 D}, \quad C_L = \frac{2 F_L}{\rho_0 U_\infty^2 D}$$

#### 2.3 Eulerian Incompressible Navier-Stokes (Chorin Projection)
* **Momentum Conservation**:
  $$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}_{\text{ext}}$$
  $$\nabla \cdot \mathbf{u} = 0$$
* **Fractional Step 1 (Advection via Semi-Lagrangian Backtracing)**:
  $$\mathbf{x}^* = \mathbf{x} - \mathbf{u}^n \Delta t, \quad \mathbf{u}^* = \operatorname{BilinearInterp}(\mathbf{u}^n, \mathbf{x}^*)$$
* **Fractional Step 2 (Implicit Viscous Diffusion)**:
  $$(I - \nu \Delta t \nabla^2)\mathbf{u}^{**} = \mathbf{u}^*$$
  Solved via Jacobi relaxation.
* **Fractional Step 3 (Pressure Poisson Equation)**:
  $$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^{**}$$
  Solved via Red-Black Gauss-Seidel iteration with Successive Over-Relaxation (SOR).
* **Fractional Step 4 (Divergence-Free Projection)**:
  $$\mathbf{u}^{n+1} = \mathbf{u}^{**} - \frac{\Delta t}{\rho} \nabla p$$
  Guarantees $\nabla \cdot \mathbf{u}^{n+1} = 0$.

---

### 3. Module Breakdown & Directory Structure

```
projects/20-aeroflow/
├── aeroflow/
│   ├── __init__.py           # Unified public API export
│   ├── types.py              # Grid2D, FlowField, FluidParams, ObstacleMask
│   ├── lbm.py                # D2Q9 Lattice Boltzmann BGK collision & streaming solver
│   ├── navier_stokes.py      # Chorin fractional step Eulerian projection solver
│   ├── obstacles.py          # NACA airfoil generator, cylinder, flat plate, and mask ops
│   ├── aerodynamics.py       # Momentum exchange, lift/drag coefficients, Strouhal number
│   ├── analysis.py           # Vorticity, stream function, streamlines, energy spectrum
│   └── visualizer.py         # Sub-pixel Unicode Braille vorticity heatmap, velocity vectors
├── tests/
│   ├── test_types.py         # Grid memory layouts, vector field operations
│   ├── test_lbm.py           # Mass conservation, Couette flow analytical match, BGK
│   ├── test_navier_stokes.py # Incompressibility divergence check, Taylor-Green vortex
│   ├── test_obstacles.py     # NACA profile coordinates, boundary mask tagging
│   ├── test_aerodynamics.py  # Drag/lift integration, momentum exchange verification
│   └── test_analysis.py      # Vorticity curl operator, streamline continuity
├── benchmarks/
│   └── bench_cfd.py          # MLUPS (Mega Lattice Updates/sec), Poisson solver throughput
├── examples/
│   └── wind_tunnel.py        # Interactive terminal wind tunnel (Karman street, NACA lift)
├── PLAN.md                   # System blueprint (this file)
├── TASK_QUEUE.md             # Autonomous task queue
├── CHECKPOINT_LAST.md        # State synchronization
└── README.md                 # System documentation & showcase
```

---

### 4. Verification & Validation Metrics
1. **LBM Mass & Momentum Conservation**: Total system mass $\sum \rho$ strictly conserved to $< 10^{-12}$ under periodic and bounce-back boundaries.
2. **Couette Flow Analytical Validation**: Planar shear flow matches linear velocity profile $u(y) = U_0 y / H$ within 0.1% relative error.
3. **Incompressible Divergence**: Navier-Stokes projection satisfies $\|\nabla \cdot \mathbf{u}\|_\infty < 10^{-6}$.
4. **Vortex Shedding & Strouhal Number**: Von Kármán vortex street behind circular cylinder demonstrates regular periodic frequency yielding Strouhal number $St \approx 0.15 - 0.21$ at $Re \approx 100$.
5. **High Computational Throughput**:
   - LBM D2Q9 throughput $> 500,000$ lattice site updates per second in pure Python.
   - Red-Black Gauss-Seidel Poisson solver $> 50,000$ cell iterations per millisecond.
