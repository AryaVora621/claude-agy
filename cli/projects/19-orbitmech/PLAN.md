# PLAN.md: OrbitMech - Orbital Mechanics, Astrodynamics & Interplanetary Trajectory Optimization Engine

## 1. System Overview & Architectural Vision
OrbitMech is an ultra-fast, zero-dependency orbital mechanics, astrodynamics, and interplanetary mission design engine built from first principles in the pure Python standard library. It provides high-precision algorithms for classical two-body Keplerian mechanics, boundary-value Lambert targeting, high-order numerical perturbation propagation, optimal orbital transfers, gravity-assist flybys, interplanetary Porkchop plot generation, and sub-pixel Unicode Braille 3D trajectory visualization.

---

## 2. Core Mathematical Formulations

### A. Two-Body Problem & Classical Orbital Elements
Given position vector $\mathbf{r} = (x, y, z)$ and velocity vector $\mathbf{v} = (v_x, v_y, v_z)$ relative to a central body with standard gravitational parameter $\mu = GM$:
1. **Specific Angular Momentum**:
   $$\mathbf{h} = \mathbf{r} \times \mathbf{v}, \quad h = \|\mathbf{h}\|$$
2. **Node Vector**:
   $$\mathbf{n} = \hat{\mathbf{k}} \times \mathbf{h} = (-h_y, h_x, 0), \quad n = \|\mathbf{n}\|$$
3. **Eccentricity Vector**:
   $$\mathbf{e} = \frac{1}{\mu} \left( (v^2 - \frac{\mu}{r})\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v} \right), \quad e = \|\mathbf{e}\|$$
4. **Specific Mechanical Energy & Semi-Major Axis**:
   $$\mathcal{E} = \frac{v^2}{2} - \frac{\mu}{r} = -\frac{\mu}{2a} \implies a = -\frac{\mu}{2\mathcal{E}}$$
5. **Inclination**:
   $$i = \arccos\left(\frac{h_z}{h}\right), \quad i \in [0, \pi]$$
6. **Longitude of the Ascending Node ($\Omega$)**:
   $$\Omega = \begin{cases} \arccos(n_x / n) & \text{if } n_y \ge 0 \\ 2\pi - \arccos(n_x / n) & \text{if } n_y < 0 \end{cases}$$
7. **Argument of Periapsis ($\omega$)**:
   $$\omega = \begin{cases} \arccos((\mathbf{n} \cdot \mathbf{e}) / (n e)) & \text{if } e_z \ge 0 \\ 2\pi - \arccos((\mathbf{n} \cdot \mathbf{e}) / (n e)) & \text{if } e_z < 0 \end{cases}$$
8. **True Anomaly ($\nu$)**:
   $$\nu = \begin{cases} \arccos((\mathbf{e} \cdot \mathbf{r}) / (e r)) & \text{if } \mathbf{r} \cdot \mathbf{v} \ge 0 \\ 2\pi - \arccos((\mathbf{e} \cdot \mathbf{r}) / (e r)) & \text{if } \mathbf{r} \cdot \mathbf{v} < 0 \end{cases}$$

### B. Kepler's Equation & Anomaly Transformations
* **Eccentric Anomaly ($E$) from True Anomaly ($\nu$)**:
  $$\tan\left(\frac{E}{2}\right) = \sqrt{\frac{1-e}{1+e}} \tan\left(\frac{\nu}{2}\right)$$
* **Mean Anomaly ($M$)**:
  $$M = E - e \sin E = n(t - t_p), \quad n = \sqrt{\frac{\mu}{a^3}}$$
* **Solving Kepler's Equation for $E$ given $M$ (Newton-Raphson with Danby Starting Guess)**:
  $$f(E) = E - e \sin E - M$$
  $$f'(E) = 1 - e \cos E, \quad f''(E) = e \sin E$$
  Danby initial guess:
  $$E_0 = M + 0.85 e \operatorname{sgn}(\sin M)$$
  Higher-order Householder/Newton update:
  $$\Delta E = -\frac{f(E)}{f'(E) - \frac{f''(E) f(E)}{2 f'(E)}}$$

### C. Universal Variable Formulation & Stumpff Functions
To handle circular ($e = 0$), elliptic ($0 < e < 1$), parabolic ($e = 1$), and hyperbolic ($e > 1$) orbits uniformly without singularities:
1. **Stumpff Functions**:
   $$c_0(z) = \cos\sqrt{z} \quad (\text{for } z > 0), \quad \cosh\sqrt{-z} \quad (\text{for } z < 0)$$
   $$c_1(z) = \frac{\sin\sqrt{z}}{\sqrt{z}}, \quad c_2(z) = \frac{1 - \cos\sqrt{z}}{z}, \quad c_3(z) = \frac{\sqrt{z} - \sin\sqrt{z}}{z^{3/2}}$$
   Evaluated via Taylor series when $|z| < 10^{-4}$ for numerical stability:
   $$c_2(z) = \frac{1}{2!} - \frac{z}{4!} + \frac{z^2}{6!} - \dots, \quad c_3(z) = \frac{1}{3!} - \frac{z}{5!} + \frac{z^2}{7!} - \dots$$
2. **Universal Kepler Equation**:
   $$\sqrt{\mu} \Delta t = \frac{\mathbf{r}_0 \cdot \mathbf{v}_0}{\sqrt{\mu}} \chi^2 c_2(\alpha \chi^2) + (1 - \alpha r_0) \chi^3 c_3(\alpha \chi^2) + r_0 \chi$$
   where $\alpha = 1/a = \frac{2}{r_0} - \frac{v_0^2}{\mu}$, and $\chi$ is the universal anomaly.
3. **Lagrange Coefficients ($f, g, \dot{f}, \dot{g}$)**:
   $$f = 1 - \frac{\chi^2}{r_0} c_2(\alpha \chi^2), \quad g = \Delta t - \frac{\chi^3}{\sqrt{\mu}} c_3(\alpha \chi^2)$$
   $$\dot{f} = \frac{\sqrt{\mu}}{r r_0} (\alpha \chi^3 c_3(\alpha \chi^2) - \chi), \quad \dot{g} = 1 - \frac{\chi^2}{r} c_2(\alpha \chi^2)$$
   $$\mathbf{r}(\Delta t) = f \mathbf{r}_0 + g \mathbf{v}_0, \quad \mathbf{v}(\Delta t) = \dot{f} \mathbf{r}_0 + \dot{g} \mathbf{v}_0$$

### D. Lambert's Boundary Value Problem
Given two position vectors $\mathbf{r}_1, \mathbf{r}_2$, gravitational parameter $\mu$, and time of flight $\Delta t$:
* Compute chord length $c = \|\mathbf{r}_2 - \mathbf{r}_1\|$, semi-perimeter $s = (r_1 + r_2 + c)/2$.
* Universal variable Lambert formulation:
  $$y(z) = r_1 + r_2 + A \frac{z c_3(z) - 1}{\sqrt{c_2(z)}}, \quad A = \pm \sqrt{r_1 r_2 (1 + \cos \Delta \theta)}$$
* Time of flight equation $t(z) = \frac{1}{\sqrt{\mu}} \left( \left(\frac{y(z)}{c_2(z)}\right)^{3/2} c_3(z) + A \sqrt{y(z)} \right)$.
* Solve for $z$ using secant/bisection iteration such that $t(z) - \Delta t = 0$.
* Recover initial and final velocities $\mathbf{v}_1, \mathbf{v}_2$:
  $$\mathbf{v}_1 = \frac{\mathbf{r}_2 - f \mathbf{r}_1}{g}, \quad \mathbf{v}_2 = \frac{\dot{g} \mathbf{r}_2 - \mathbf{r}_1}{g}$$

### E. Perturbation Modeling & Numerical Propagators
1. **$J_2$ Oblateness Perturbation (Earth Oblateness)**:
   $$\mathbf{a}_{J_2} = -\frac{3}{2} \frac{J_2 \mu R_E^2}{r^4} \left[ \left(1 - 5\frac{z^2}{r^2}\right)\frac{x}{r} \hat{\mathbf{i}} + \left(1 - 5\frac{z^2}{r^2}\right)\frac{y}{r} \hat{\mathbf{j}} + \left(3 - 5\frac{z^2}{r^2}\right)\frac{z}{r} \hat{\mathbf{k}} \right]$$
   Causes secular nodal precession $\dot{\Omega} \approx -\frac{3}{2} J_2 \left(\frac{R_E}{p}\right)^2 n \cos i$ and apsidal precession $\dot{\omega} \approx \frac{3}{4} J_2 \left(\frac{R_E}{p}\right)^2 n (5 \cos^2 i - 1)$.
2. **Atmospheric Drag**:
   $$\mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho(h) \frac{C_D A}{m} v_{\text{rel}} \mathbf{v}_{\text{rel}}$$
   using exponential barometric density model $\rho(h) = \rho_0 e^{-(h - h_0)/H}$.
3. **Solar Radiation Pressure (SRP)**:
   $$\mathbf{a}_{\text{srp}} = -P_{\text{sun}} C_R \frac{A}{m} \left(\frac{1 \text{ AU}}{\|\mathbf{r}_{\text{sun}}\|}\right)^2 \hat{\mathbf{r}}_{\text{sun}}$$
4. **Propagator Engines**:
   * **Adaptive Runge-Kutta-Fehlberg (RK45)**: 6-stage Embedded Runge-Kutta with adaptive time-step control based on local truncation error estimate.
   * **Symplectic Störmer-Verlet Integrator**: Time-reversible geometric integrator that strictly conserves the symplectic 2-form and maintains bounded Hamiltonian energy oscillation over millions of seconds.

### F. Orbital Maneuvers & Interplanetary Trajectories
1. **Hohmann & Bi-Elliptic Transfer**:
   * Hohmann semi-major axis: $a_{\text{tx}} = (r_1 + r_2)/2$.
   * $\Delta v_1 = \sqrt{\frac{\mu}{r_1}} \left(\sqrt{\frac{2 r_2}{r_1 + r_2}} - 1\right)$, $\Delta v_2 = \sqrt{\frac{\mu}{r_2}} \left(1 - \sqrt{\frac{2 r_1}{r_1 + r_2}}\right)$.
   * Bi-elliptic transfer: 3-impulse maneuver to intermediate radius $r_b > r_2$. Evaluates when $r_2 / r_1 > 11.94$ where bi-elliptic is more fuel-efficient than Hohmann.
2. **Plane Change Maneuver**:
   * Pure inclination change: $\Delta v = 2 v \sin(\Delta i / 2)$.
   * Combined apse-line rotation and plane change.
3. **Gravity Assist (Flyby) Kinematics**:
   * Excess arrival hyperbolic velocity: $\mathbf{v}_{\infty,-} = \mathbf{v}_1 - \mathbf{v}_{\text{planet}}$.
   * Turning angle $\delta = 2 \arcsin\left(\frac{1}{1 + \frac{r_p v_\infty^2}{\mu_p}}\right)$.
   * Departure velocity $\mathbf{v}_{\infty,+} = R(\delta) \mathbf{v}_{\infty,-}$.
   * Heliocentric velocity boost: $\Delta \mathbf{V} = \mathbf{v}_{\infty,+} - \mathbf{v}_{\infty,-}$.
4. **Porkchop Plot Generator**:
   * Grid evaluation of departure dates $t_{\text{dep}} \in [T_1, T_2]$ and arrival dates $t_{\text{arr}} \in [T_3, T_4]$.
   * Solves Lambert's problem for each grid point $(t_{\text{dep}}, t_{\text{arr}})$.
   * Computes departure characteristic energy $C_3 = \|\mathbf{v}_{\text{dep}} - \mathbf{v}_{\text{planet1}}\|^2$ and arrival capture $\Delta v_{\text{arr}} = \|\mathbf{v}_{\text{arr}} - \mathbf{v}_{\text{planet2}}\|$.
   * Generates contour grid to locate global minimum $\Delta V$ launch windows (e.g. Earth-Mars 2026 window).

### G. 3D Orbital Projection & Sub-Pixel Braille Visualizer
* 3D camera model with azimuth $\psi$, elevation $\theta$, and distance $d$.
* Transformation from celestial coordinate frames (ECI / Heliocentric) to camera view-plane.
* Sub-pixel Unicode Braille canvas (`U+2800..U+28FF`) providing $2 \times 4$ dot resolution per character cell.
* Renders planetary orbits, spacecraft trajectory trails, periapsis/apoapsis points, lines of nodes, and Porkchop contour maps.

---

## 3. Module Breakdown

```
projects/19-orbitmech/
├── orbitmech/
│   ├── __init__.py           # Unified exports
│   ├── types.py              # 3D Vector math, Celestial bodies, OrbitalConstants
│   ├── kepler.py             # Classical orbital elements, Kepler solver, State conversion
│   ├── lambert.py            # Stumpff functions, Universal variables, Lambert BV solver
│   ├── propagator.py         # J2, drag, SRP, RK45 adaptive, Symplectic Störmer-Verlet
│   ├── maneuvers.py          # Hohmann, Bi-elliptic, Plane change, Flyby kinematics
│   ├── porkchop.py           # Interplanetary Porkchop grid evaluator & contour finder
│   └── visualizer.py         # 3D camera projection, Braille canvas, ASCII telemetry
├── tests/
│   ├── test_types.py         # Vector operations, celestial constants
│   ├── test_kepler.py        # State vector <-> COE conversions, Kepler equation
│   ├── test_lambert.py       # Stumpff functions, Lambert boundary value solutions
│   ├── test_propagator.py    # J2 precession, energy conservation, RK45 vs Symplectic
│   ├── test_maneuvers.py     # Hohmann delta-V, bi-elliptic threshold, gravity assist
│   └── test_visualizer.py    # 3D camera projection, Braille rasterization
├── benchmarks/
│   └── bench_orbit.py        # Microbenchmarks: Kepler solver, Lambert, RK45, Symplectic
├── examples/
│   └── mission_control.py    # Interactive lab: Earth-Mars transfer, J2 decay, flyby, Braille orbit
├── PLAN.md
├── TASK_QUEUE.md
├── CHECKPOINT_LAST.md
└── README.md
```

---

## 4. Verification & Validation Strategy
1. **Conservation Laws**: Verify that two-body unperturbed propagation strictly conserves specific mechanical energy $\mathcal{E}$ and specific angular momentum vector $\mathbf{h}$ to within $10^{-12}$.
2. **Kepler Inversion Roundtrip**: Verify roundtrip $(\mathbf{r}, \mathbf{v}) \to (a, e, i, \Omega, \omega, \nu) \to (\mathbf{r}, \mathbf{v})$ matches to within machine precision ($< 10^{-10}$ relative error).
3. **Lambert Test Cases**: Verify against standard NASA/Battin benchmark problems (e.g. Earth-Mars 259-day transfer, rendezvous trajectories).
4. **$J_2$ Precession Rates**: Verify numerical nodal precession $\dot{\Omega}$ matches theoretical secular formula to within $1\%$.
5. **Zero External Dependencies**: Standard library Python 3.10+ only (`math`, `typing`, `dataclasses`, `time`, `random`).
