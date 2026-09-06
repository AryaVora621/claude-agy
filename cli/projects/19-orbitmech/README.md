# OrbitMech: Orbital Mechanics, Astrodynamics & Interplanetary Trajectory Optimization Engine

[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib%20only)-success)]()

OrbitMech is an ultra-fast, zero-dependency orbital mechanics, astrodynamics, and interplanetary mission trajectory optimization engine built entirely from first principles in the pure Python standard library. It provides high-precision algorithms for classical two-body Keplerian mechanics, boundary-value Lambert targeting, high-order numerical perturbation propagation, optimal orbital transfers, gravity-assist flybys, interplanetary Porkchop plot generation, and sub-pixel Unicode Braille 3D trajectory visualization:

* **Two-Body Problem & Classical Keplerian Dynamics**: Bidirectional state vector $(\mathbf{r}, \mathbf{v}) \in \mathbb{R}^6 \iff (a, e, i, \Omega, \omega, \nu)$ transformations, singularity handling for equatorial and circular orbits, and Danby 3rd-order Householder/Halley root solver for Kepler's equation converging in 2-3 iterations across all eccentricities $e \in [0, 0.99999]$ (1.37 Million solves/s).
* **Universal Variable Formulation & Stumpff Functions**: Evaluates Stumpff functions $c_0(z), c_1(z), c_2(z), c_3(z)$ with Taylor expansions for near-zero arguments, enabling singularity-free analytic propagation across circular, elliptic, parabolic, and hyperbolic orbits uniformly.
* **Lambert's Boundary Value Problem Solver**: Bate-Mueller-White and Battin universal variable formulation targeting orbital transfers between arbitrary 3D positions $\mathbf{r}_1, \mathbf{r}_2$ over specified time-of-flight $\Delta t$ in 24.32 µs per transfer (41,115 solves/sec).
* **Perturbation Modeling & Numerical Propagators**:
  * **Earth Oblateness ($J_2$)**: Zonal harmonic perturbation evaluating secular regression of the nodes ($\dot{\Omega}$) and rotation of the apsides ($\dot{\omega}$).
  * **Atmospheric Drag**: Multi-layer exponential scale height density model evaluating aerodynamic deceleration and orbital decay.
  * **Solar Radiation Pressure (SRP)**: Solar photon momentum flux modeling.
  * **Symplectic Störmer-Verlet Integrator**: Time-reversible geometric integrator strictly conserving phase-space symplectic 2-form with zero secular energy drift over thousands of orbits (222,786 steps/s).
  * **Adaptive Runge-Kutta-Fehlberg (RK45)**: 6-stage Cash-Karp embedded integrator with local truncation error monitoring and dynamic step-size control.
* **Orbital Maneuvers & Planetary Trajectories**:
  * **Hohmann Transfers**: Optimal 2-impulse coplanar transfers with rendezvous phase angles.
  * **Bi-Elliptic Transfers**: 3-impulse transfers evaluating fuel efficiency advantages when radius ratio $r_2 / r_1 > 11.9387$.
  * **Plane Change Maneuvers**: Combined apse-line rotation and inclination changes.
  * **Planetary Gravity Assists (Flybys)**: Unpowered hyperbolic flyby kinematics computing excess velocity $\mathbf{v}_\infty$, turning angle $\delta$, and heliocentric velocity boost $\Delta \mathbf{V}_{\text{helio}}$.
  * **Interplanetary Porkchop Plot Generator**: Date grid evaluator sweeping departure and arrival windows, computing departure characteristic energy $C_3$, arrival excess speed, and identifying global minimum $\Delta V$ launch opportunities.
* **3D Sub-Pixel Unicode Braille Visualizer**: Orthographic/perspective 3D camera projection rendering orbital tracks, equatorial reference rings, celestial bodies, and 2D Porkchop launch opportunity contour maps in 24-bit TrueColor ANSI.

---

## Mathematical Foundations & Core Formulations

### 1. Two-Body Keplerian Mechanics
Given state vector $\mathbf{r} = (x, y, z)$ and $\mathbf{v} = (v_x, v_y, v_z)$ in an inertial frame with gravitational parameter $\mu = GM$:
* **Specific Angular Momentum**:
  $$\mathbf{h} = \mathbf{r} \times \mathbf{v}, \quad h = \|\mathbf{h}\|$$
* **Eccentricity Vector**:
  $$\mathbf{e} = \frac{1}{\mu} \left( (v^2 - \frac{\mu}{r})\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v} \right), \quad e = \|\mathbf{e}\|$$
* **Specific Mechanical Energy**:
  $$\mathcal{E} = \frac{v^2}{2} - \frac{\mu}{r} = -\frac{\mu}{2a} \implies a = -\frac{\mu}{2\mathcal{E}}$$
* **Kepler's Equation & Danby's 3rd-Order Solver**:
  $$M = E - e \sin E$$
  Starting with Danby initial guess $E_0 = M + 0.85 e \operatorname{sgn}(\sin M)$, high-order updates compute:
  $$\delta_1 = -f / f', \quad \delta_2 = -f / (f' + \frac{1}{2} \delta_1 f''), \quad \delta_3 = -f / (f' + \frac{1}{2} \delta_2 f'' + \frac{1}{6} \delta_2^2 f''')$$
  achieving $< 10^{-13}$ accuracy in 2 iterations.

### 2. Universal Variable Formulation & Stumpff Functions
Eliminates conic case branchings by defining universal anomaly $\chi$:
* **Stumpff Series Expansions** ($|z| < 10^{-4}$):
  $$c_2(z) = \frac{1}{2!} - \frac{z}{4!} + \frac{z^2}{6!} - \dots, \quad c_3(z) = \frac{1}{3!} - \frac{z}{5!} + \frac{z^2}{7!} - \dots$$
* **Universal Kepler Equation**:
  $$\sqrt{\mu} \Delta t = \frac{\mathbf{r}_0 \cdot \mathbf{v}_0}{\sqrt{\mu}} \chi^2 c_2(\alpha \chi^2) + (1 - \alpha r_0) \chi^3 c_3(\alpha \chi^2) + r_0 \chi$$
  where $\alpha = 1/a = 2/r_0 - v_0^2/\mu$.
* **Lagrange $f$ and $g$ Coefficients**:
  $$\mathbf{r} = f \mathbf{r}_0 + g \mathbf{v}_0, \quad \mathbf{v} = \dot{f} \mathbf{r}_0 + \dot{g} \mathbf{v}_0$$

### 3. Lambert's Boundary Value Problem
Given $\mathbf{r}_1, \mathbf{r}_2$, and $\Delta t$, determines transfer velocities $\mathbf{v}_1, \mathbf{v}_2$:
* Geometric constant $A = \sin \Delta \theta \sqrt{\frac{r_1 r_2}{1 - \cos \Delta \theta}}$.
* Universal function $y(z) = r_1 + r_2 + A \frac{z c_3(z) - 1}{\sqrt{c_2(z)}}$.
* Root search for $t(z) = \frac{1}{\sqrt{\mu}} \left( (y/c_2)^{3/2} c_3 + A \sqrt{y} \right) = \Delta t$.
* Recovers departure and arrival velocities:
  $$\mathbf{v}_1 = \frac{\mathbf{r}_2 - f \mathbf{r}_1}{g}, \quad \mathbf{v}_2 = \frac{\dot{g} \mathbf{r}_2 - \mathbf{r}_1}{g}$$

### 4. Earth Oblateness ($J_2$) Secular Perturbation
The equatorial bulge of the Earth creates non-spherical potential harmonics:
$$\mathbf{a}_{J_2} = -\frac{3}{2} \frac{J_2 \mu R^2}{r^5} \left[ x\left(1 - 5\frac{z^2}{r^2}\right)\hat{\mathbf{i}} + y\left(1 - 5\frac{z^2}{r^2}\right)\hat{\mathbf{j}} + z\left(3 - 5\frac{z^2}{r^2}\right)\hat{\mathbf{k}} \right]$$
* **Secular Nodal Precession**:
  $$\dot{\Omega} = -\frac{3}{2} J_2 \left(\frac{R}{p}\right)^2 n \cos i$$
* **Secular Apsidal Precession**:
  $$\dot{\omega} = \frac{3}{4} J_2 \left(\frac{R}{p}\right)^2 n (5 \cos^2 i - 1)$$

### 5. Hyperbolic Gravity Assist Flybys
* **Turning Angle**:
  $$\delta = 2 \arcsin\left(\frac{1}{1 + \frac{r_p v_\infty^2}{\mu_p}}\right)$$
* **Heliocentric Velocity Boost**:
  $$\Delta \mathbf{V}_{\text{helio}} = \mathbf{v}_{\infty,+} - \mathbf{v}_{\infty,-}, \quad \|\Delta \mathbf{V}\| = 2 v_\infty \sin(\delta / 2)$$

---

## Measured Performance Benchmarks

Executed on Apple Silicon (Python 3.13 standard library, single-threaded):

| Benchmark Subsystem | Operations Evaluated | Throughput / Latency |
|:---|:---|:---:|
| **Danby Kepler Equation** | 100,000 root evaluations | **1,367,509 roots/sec (0.73 µs/root)** |
| **State <-> COE Conversion** | 50,000 coordinate conversions | **340,761 conversions/sec** |
| **Universal Conic Propagation** | 20,000 state propagations | **202,698 steps/sec (4.93 µs/step)** |
| **Lambert Boundary Value Solver** | 2,000 targeting transfers | **41,115 transfers/sec (24.32 µs/solve)** |
| **Symplectic Störmer-Verlet Integrator** | 12,161 time steps (with $J_2$) | **222,786 integration steps/sec** |
| **Adaptive RK45 Integrator** | 1,242 adaptive steps (with $J_2$) | **26,087 integration steps/sec** |
| **Sub-Pixel Braille 3D Projection** | 100 full 64x22 scenes | **3,740.8 FPS** |

---

## Project Structure

```
projects/19-orbitmech/
├── orbitmech/
│   ├── __init__.py           # Unified public API exports
│   ├── types.py              # Vector3, StateVector, ClassicalOrbitalElements, CelestialBody
│   ├── kepler.py             # State conversions, anomaly solvers, Danby Kepler root finder
│   ├── lambert.py            # Stumpff functions, Universal variables, Lambert BV solver
│   ├── propagator.py         # J2 oblateness, drag, SRP, Symplectic Verlet & RK45
│   ├── maneuvers.py          # Hohmann, Bi-elliptic, Plane change, Hyperbolic gravity assist
│   ├── porkchop.py           # Interplanetary Porkchop grid evaluator & launch optimizer
│   └── visualizer.py         # 3D camera projection, Braille canvas, Porkchop contour map
├── tests/
│   ├── test_types.py         # Vector3 math, orbital element properties
│   ├── test_kepler.py        # Kepler equation, state roundtrips, periodic motion
│   ├── test_lambert.py       # Stumpff functions, universal propagation, Lambert solutions
│   ├── test_propagator.py    # Symplectic energy conservation, J2 precession, drag decay
│   ├── test_maneuvers.py     # Hohmann delta-V, bi-elliptic threshold, gravity assist
│   └── test_visualizer.py    # Braille pixel offsets, 3D camera projection
├── benchmarks/
│   └── bench_orbit.py        # Performance benchmark suite across all subsystems
├── examples/
│   └── mission_control.py    # Interactive Mission Control demonstration laboratory
├── PLAN.md                   # Architectural design blueprint
├── TASK_QUEUE.md             # Autonomous task tracking
├── CHECKPOINT_LAST.md        # State synchronization
└── README.md                 # System documentation
```

---

## Quick Start & Usage Examples

### 1. State Vector to Classical Keplerian Elements
```python
import math
from orbitmech.types import Vector3, StateVector, EARTH
from orbitmech.kepler import state_to_orbital_elements

# Spacecraft state vector in Low Earth Orbit
r = Vector3(6778.137, 0.0, 0.0)    # 400 km altitude
v = Vector3(0.0, 7.67, 0.0)        # km/s
state = StateVector(r=r, v=v)

coe = state_to_orbital_elements(state, mu=EARTH.mu)
print(coe.summary())
# Semi-major axis, eccentricity, period, inclination
```

### 2. Solve Lambert's Problem Between Two Planetary Positions
```python
from orbitmech.types import Vector3, EARTH
from orbitmech.lambert import solve_lambert

r1 = Vector3(7000.0, 0.0, 0.0)
r2 = Vector3(0.0, 8500.0, 1200.0)
tof_seconds = 2400.0  # 40 minutes

sol = solve_lambert(r1, r2, time_of_flight=tof_seconds, mu=EARTH.mu)
print(f"Departure Velocity (v1): {sol.v1} (Speed: {sol.v1.norm():.3f} km/s)")
print(f"Arrival Velocity (v2):   {sol.v2} (Speed: {sol.v2.norm():.3f} km/s)")
print(f"Transfer Semi-Major Axis: {sol.semi_major_axis:,.1f} km")
```

### 3. High-Fidelity J2 Perturbed Orbit Propagation
```python
from orbitmech.propagator import PerturbationConfig, propagate_rk45

config = PerturbationConfig(enable_j2=True, enable_drag=False)
result = propagate_rk45(
    initial_state=state,
    duration=86400.0,     # 24 hours
    initial_dt=30.0,
    tolerance=1e-8,
    config=config,
)
print(f"Propagated {result.step_count} adaptive steps. Final state: {result.final_state.r}")
```

### 4. Interplanetary Porkchop Plot Generation
```python
from orbitmech.porkchop import generate_porkchop_grid

grid = generate_porkchop_grid(
    dep_start_day=0.0,
    dep_end_day=60.0,
    dep_steps=16,
    arr_start_day=160.0,
    arr_end_day=320.0,
    arr_steps=16,
)
print(grid.summary())
print(f"Optimal Launch C3: {grid.min_c3_point.c3_km2_s2:.2f} km^2/s^2")
```

---

## Verification & Interactive Demos

### Run Full Test Suite
```bash
python3 -m unittest discover -s tests
# Ran 26 tests in 0.22s -> OK
```

### Run Performance Benchmarks
```bash
python3 benchmarks/bench_orbit.py
```

### Run Mission Control Interactive Lab
```bash
python3 examples/mission_control.py
```
Demonstrates spacecraft orbital telemetry HUD, LEO-to-GEO Hohmann transfer, 3D sub-pixel Unicode Braille orbit rendering, J2 secular nodal drift, Jupiter gravity assist flyby, and Earth-Mars Porkchop contour maps.
