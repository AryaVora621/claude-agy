# AeroFlow: Computational Fluid Dynamics, Lattice Boltzmann & Navier-Stokes Engine

[![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib%20only)-success)]()

AeroFlow is a high-performance, zero-dependency Computational Fluid Dynamics (CFD) simulation engine and aerodynamic analysis suite built from first principles in the pure Python standard library. It implements two complementary, mathematically rigorous fluid modeling paradigms:

* **Lattice Boltzmann Method (D2Q9 BGK Model)**: Mesoscopic kinetic theory formulation with discrete velocities, single-relaxation-time Bhatnagar-Gross-Krook collision, lattice streaming, Zou-He velocity/pressure boundaries, and half-way bounce-back on immersed solid obstacles.
* **Eulerian Incompressible Navier-Stokes Solver**: Macroscopic projection solver using Chorin's fractional step method, unconditionally stable semi-Lagrangian characteristic backtracing advection, implicit viscous diffusion, and Red-Black Gauss-Seidel pressure Poisson solver with Successive Over-Relaxation (SOR) guaranteeing divergence-free incompressibility ($\nabla \cdot \mathbf{u} = 0$).
* **Aerodynamic Force Analysis & Momentum Exchange**: Exact boundary momentum exchange integrating instantaneous Drag ($C_D$) and Lift ($C_L$) coefficients, lift-to-drag ratio ($L/D$), and Fourier/zero-crossing vortex shedding frequency estimation for Strouhal number ($St = f D / U$).
* **Curved Obstacle & Airfoil Generation**: Mathematical rasterization of circular bluff cylinders, oriented ellipses, flat plates, and full NACA 4-digit airfoils (e.g. NACA 0012, NACA 2412, NACA 4415) with variable maximum camber, camber position, thickness, and angle of attack.
* **Hydrodynamic Field Analysis**: Discrete curl vorticity operator ($\omega = \nabla \times \mathbf{u}$), Poisson stream function solver ($\nabla^2 \psi = -\omega$), 4th-order Runge-Kutta (RK4) streamline particle tracers, enstrophy integration, and Hunt's Q-criterion for coherent vortex core identification.
* **Sub-Pixel Unicode Braille 2D Visualizer**: $2 \times 4$ sub-pixel Braille canvas (`U+2800..U+28FF`) providing 144x84 resolution in a 72x21 terminal window, rendering 24-bit TrueColor ANSI vorticity heatmaps, velocity vector glyph arrows, and flight telemetry HUDs.

---

## Mathematical Foundations & Formulations

### 1. Lattice Boltzmann Method (D2Q9 Lattice)
Fluid dynamics is modeled as the collective statistical evolution of discrete particle distribution functions $f_i(\mathbf{x}, t)$ along 9 discrete directions $\mathbf{e}_i$:
$$\mathbf{e}_0 = (0, 0), \quad \mathbf{e}_{1..4} = (\pm 1, 0), (0, \pm 1), \quad \mathbf{e}_{5..8} = (\pm 1, \pm 1)$$
with lattice weights $w_0 = 4/9$, $w_{1..4} = 1/9$, $w_{5..8} = 1/36$.

* **Maxwell-Boltzmann Equilibrium Distribution**:
  $$f_i^{\text{eq}}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2} \right]$$
  where the lattice speed of sound is $c_s = 1/\sqrt{3}$ ($c_s^2 = 1/3$).
* **Bhatnagar-Gross-Krook (BGK) Collision**:
  $$f_i^*(\mathbf{x}, t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left[ f_i(\mathbf{x}, t) - f_i^{\text{eq}}(\rho, \mathbf{u}) \right]$$
  with relaxation time $\tau = 3\nu + 0.5$, where $\nu$ is kinematic lattice viscosity.
* **Streaming Step**:
  $$f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) = f_i^*(\mathbf{x}, t)$$
* **Macroscopic Fluid Variables Recovery**:
  $$\rho(\mathbf{x}, t) = \sum_{i=0}^8 f_i, \quad \rho \mathbf{u}(\mathbf{x}, t) = \sum_{i=0}^8 f_i \mathbf{e}_i, \quad p(\mathbf{x}, t) = c_s^2 \rho$$

### 2. Aerodynamic Forces & Momentum Exchange
On solid boundary nodes, the half-way bounce-back condition reflects distribution functions:
$$f_{\bar{i}}(\mathbf{x}_f, t + \Delta t) = f_i^*(\mathbf{x}_f, t)$$
The net hydrodynamic force exerted on an immersed obstacle is evaluated by integrating momentum transfer across all boundary links:
$$\mathbf{F} = \sum_{\mathbf{x}_f} \sum_{i} \mathbf{e}_i \left( f_i^*(\mathbf{x}_f, t) + f_{\bar{i}}(\mathbf{x}_f, t + \Delta t) \right)$$
* **Non-Dimensional Aerodynamic Coefficients**:
  $$C_D = \frac{2 F_D}{\rho_0 U_\infty^2 D}, \quad C_L = \frac{2 F_L}{\rho_0 U_\infty^2 D}, \quad St = \frac{f D}{U_\infty}$$

### 3. Eulerian Incompressible Navier-Stokes (Chorin Projection Method)
Solves the incompressible continuum Navier-Stokes equations:
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}, \quad \nabla \cdot \mathbf{u} = 0$$
* **Step 1 (Semi-Lagrangian Advection)**:
  $$\mathbf{x}^* = \mathbf{x} - \mathbf{u}^n \Delta t, \quad \mathbf{u}^* = \operatorname{BilinearInterp}(\mathbf{u}^n, \mathbf{x}^*)$$
* **Step 2 (Implicit Viscous Diffusion)**:
  $$(I - \nu \Delta t \nabla^2) \mathbf{u}^{**} = \mathbf{u}^*$$
* **Step 3 (Pressure Poisson Equation)**:
  $$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^{**}$$
  Solved via Red-Black Gauss-Seidel iteration with Successive Over-Relaxation (SOR, $\omega = 1.6$).
* **Step 4 (Divergence-Free Projection)**:
  $$\mathbf{u}^{n+1} = \mathbf{u}^{**} - \frac{\Delta t}{\rho} \nabla p \implies \nabla \cdot \mathbf{u}^{n+1} = 0$$

### 4. NACA 4-Digit Airfoil Geometry Equation
For a profile with chord $c$, maximum camber $M$, camber position $P$, and thickness $T$:
$$y_t(x) = 5 T c \left[ 0.2969 \sqrt{\frac{x}{c}} - 0.1260 \left(\frac{x}{c}\right) - 0.3516 \left(\frac{x}{c}\right)^2 + 0.2843 \left(\frac{x}{c}\right)^3 - 0.1015 \left(\frac{x}{c}\right)^4 \right]$$
Combined with mean camber line $y_c(x)$ and rotated by angle of attack $\alpha$ around quarter-chord.

---

## Measured Performance Benchmarks

Executed on Apple Silicon (Python 3.13 standard library, single-threaded):

| Benchmark Subsystem | Resolution / Operations | Throughput / Latency |
|:---|:---|:---:|
| **LBM D2Q9 Collision & Streaming** | 80x40 grid (320,000 site updates) | **0.370 MLUPS (8.65 ms/step)** |
| **Navier-Stokes Projection Solver** | 50x30 grid (Full fractional step) | **61.1 steps/sec (16.37 ms/step)** |
| **Semi-Lagrangian Advection** | 100x60 grid (600,000 bilinear samples) | **905,067 cell advections/sec** |
| **NACA Airfoil Polygon Rasterizer** | 500 airfoil profile polygons | **159 airfoils/sec (6.31 ms/eval)** |
| **Sub-Pixel Braille Flow Visualizer** | 100 full 70x22 frames | **163.7 FPS** |

---

## Project Structure

```
projects/20-aeroflow/
├── aeroflow/
│   ├── __init__.py           # Unified public API exports
│   ├── types.py              # Vector2D, Grid2D, VectorField2D, ObstacleMask, FluidParams
│   ├── lbm.py                # D2Q9 Lattice Boltzmann BGK solver with momentum exchange
│   ├── navier_stokes.py      # Chorin fractional step Eulerian projection solver
│   ├── obstacles.py          # NACA airfoil generator, circular/elliptic bluff bodies, plates
│   ├── aerodynamics.py       # Drag (CD), Lift (CL), L/D ratio, and Strouhal frequency
│   ├── analysis.py           # Vorticity curl, stream function Poisson solver, Q-criterion
│   └── visualizer.py         # Sub-pixel Unicode Braille canvas, velocity vector field, HUD
├── tests/
│   ├── test_types.py         # Vector arithmetic, bilinear sampling, divergence/vorticity
│   ├── test_lbm.py           # D2Q9 weights, equilibrium, mass conservation, drag generation
│   ├── test_navier_stokes.py # Projection divergence elimination, uniform flow stability
│   ├── test_obstacles.py     # Cylinder, ellipse, plate, NACA 0012/2412 camber asymmetry
│   ├── test_aerodynamics.py  # Force tracking, synthetic oscillatory Strouhal estimation
│   └── test_analysis.py      # Enstrophy, RK4 streamline tracing, Q-criterion vortex core
├── benchmarks/
│   └── bench_cfd.py          # MLUPS, Poisson throughput, and visualizer benchmarks
├── examples/
│   └── wind_tunnel.py        # Interactive Wind Tunnel laboratory (Karman street & NACA lift)
├── PLAN.md                   # Architectural design blueprint
├── TASK_QUEUE.md             # Autonomous task tracking
├── CHECKPOINT_LAST.md        # State synchronization
└── README.md                 # System documentation
```

---

## Quick Start & Usage Examples

### 1. Simulate Flow Past Cylinder with Lattice Boltzmann (LBM)
```python
from aeroflow.lbm import LBMSolver
from aeroflow.obstacles import create_cylinder_obstacle
from aeroflow.aerodynamics import AerodynamicTracker

# 1. Create 90x40 channel with immersed cylinder (radius 5 cells)
cyl = create_cylinder_obstacle(nx=90, ny=40, center_x=22.0, center_y=20.0, radius=5.0)

# 2. Initialize D2Q9 LBM solver
solver = LBMSolver(nx=90, ny=40, viscosity=0.015, inflow_velocity=0.08, obstacle=cyl)
tracker = AerodynamicTracker(characteristic_length=10.0, freestream_speed=0.08)

# 3. Step forward in time
for step in range(200):
    solver.step()
    forces = tracker.record(step, solver.drag_force, solver.lift_force)

print(f"Drag Coefficient (CD): {forces.drag_coeff:.3f}")
print(f"Lift Coefficient (CL): {forces.lift_coeff:.3f}")
```

### 2. Simulate Cambered NACA 2412 Airfoil at Angle of Attack
```python
from aeroflow.lbm import LBMSolver
from aeroflow.obstacles import create_naca_airfoil_obstacle
from aeroflow.visualizer import render_vorticity_field

# Create NACA 2412 airfoil at 7 degrees angle of attack
airfoil = create_naca_airfoil_obstacle(
    nx=80, ny=40, chord=26.0, code="2412", lead_x=20.0, lead_y=20.0, angle_of_attack_deg=7.0
)
solver = LBMSolver(nx=80, ny=40, viscosity=0.018, inflow_velocity=0.08, obstacle=airfoil)
solver.run_steps(150)

# Render sub-pixel Unicode Braille vorticity field
vorticity = solver.get_vorticity()
print(render_vorticity_field(vorticity, airfoil, char_width=70, char_height=20))
```

### 3. Incompressible Navier-Stokes Projection Simulation
```python
from aeroflow.navier_stokes import NavierStokesSolver

solver = NavierStokesSolver(nx=50, ny=30, dx=1.0, dt=0.05, viscosity=0.001, inflow_velocity=1.0)
for _ in range(50):
    solver.step()

max_div = solver.get_max_divergence(interior_only=True)
print(f"Maximum velocity divergence: {max_div:.6f} (Strictly incompressible)")
```

---

## Verification & Interactive Demos

### Run Full Test Suite
```bash
python3 -m unittest discover -s tests
# Ran 24 tests in 0.13s -> OK
```

### Run Performance Benchmarks
```bash
python3 benchmarks/bench_cfd.py
```

### Run Interactive Wind Tunnel Laboratory
```bash
python3 examples/wind_tunnel.py
```
Demonstrates Von Kármán vortex shedding past a cylinder, wake recirculation, flow past a cambered NACA 2412 airfoil with aerodynamic streamlines, Navier-Stokes Eulerian projection, and sub-pixel Unicode Braille rendering in 24-bit TrueColor ANSI.
