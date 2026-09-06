# OptiFlow 2D: Computational Fluid Dynamics & Aerodynamics Studio

OptiFlow 2D is a standalone desktop computational fluid dynamics (CFD) workstation and aerodynamics laboratory built entirely with Python standard library `tkinter` and zero external dependencies.

## Key Features

- **Incompressible Navier-Stokes Solver**: First-principles numerical integration of 2D viscous fluid equations via the coupled Vorticity-Streamfunction ($\omega - \psi$) formulation.
- **Continuity Guaranteed**: Incompressibility $\nabla \cdot \mathbf{u} = 0$ is satisfied to machine precision identically through streamfunction formulation $u = \partial\psi/\partial y, v = -\partial\psi/\partial x$.
- **Poisson Pressure Recovery**: Successive over-relaxation (SOR) solve of Poisson pressure equation $\nabla^2 p = 2\rho (\frac{\partial u}{\partial x}\frac{\partial v}{\partial y} - \frac{\partial u}{\partial y}\frac{\partial v}{\partial x})$ for gauge pressure and stagnation points.
- **Aerodynamic Force Integration**: Real-time contour boundary integration of surface pressure $p$ and wall shear stress $\tau_w = \mu \omega_{\text{wall}}$ yielding dimensional lift $F_L$, drag $F_D$, pitching moment $C_M$, lift coefficient $C_L$, drag coefficient $C_D$, and efficiency $L/D$.
- **NACA 4-Digit Morphology**: Parametric generation of NACA symmetric (e.g. NACA 0012) and cambered (e.g. NACA 2412, NACA 4412) airfoil profiles with real-time Angle of Attack (AoA) adjustment from -18 deg to +22 deg.
- **Karman Vortex Street**: Unsteady vortex shedding in cylinder wake showing alternate sign vortex peeling and periodic lift fluctuations.
- **Smoke Streakline Advection**: RK2 midpoint particle advection tracing streamlines, separation bubbles, and recirculation zones.
- **Virtual Pitot Probe**: Interactive click-to-measure tool displaying velocity components, dynamic pressure $q$, static pressure $p$, and pressure coefficient $C_p = 1 - (|V|/U_\infty)^2$.

## Mathematical Formulation

### 1. Vorticity Transport Equation
$$\frac{\partial \omega}{\partial t} + u \frac{\partial \omega}{\partial x} + v \frac{\partial \omega}{\partial y} = \nu \left( \frac{\partial^2 \omega}{\partial x^2} + \frac{\partial^2 \omega}{\partial y^2} \right)$$

### 2. Streamfunction Kinematics
$$\nabla^2 \psi = -\omega, \quad u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

### 3. Woods Solid Wall Vorticity Boundary Condition
$$\omega_{\text{wall}} = -\frac{2(\psi_{\text{fluid}} - \psi_{\text{wall}})}{\Delta n^2}$$

### 4. Aerodynamic Force Coefficients
$$C_L = \frac{2 F_{\text{lift}}}{\rho U_\infty^2 c}, \quad C_D = \frac{2 F_{\text{drag}}}{\rho U_\infty^2 c}, \quad L/D = \frac{C_L}{C_D}$$

## Verification & Test Suite

Run the automated 16-test suite:
```bash
python3 programs/optiflow/test_optiflow.py
```
All 16 unit tests verify grid geometry, divergence-free continuity, numerical stability, Woods boundary condition, angle of attack lift scaling, and preset configurations with 100% pass rate.

## Launch Desktop Studio

```bash
python3 programs/optiflow/optiflow.py
```
