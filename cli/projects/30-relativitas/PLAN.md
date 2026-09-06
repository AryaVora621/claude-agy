# Architectural Specification: Project 30 (Relativitas)

## Executive Summary

**Relativitas** is a first-principles, zero-dependency General Relativity (GR) simulation, curved spacetime geodesic integrator, and black hole accretion disk ray tracer implemented in pure Python 3.10+ standard library.

The engine solves the Einstein geodesic equations of motion in curved 4-dimensional pseudo-Riemannian spacetime manifolds for both static (Schwarzschild) and rotating (Kerr) black holes. It models gravitational lensing, frame dragging (Lense-Thirring effect), ergosphere geometry, the photon sphere, gravitational redshift, relativistic Doppler beaming, and generates sub-pixel Unicode Braille terminal visualizations of black hole shadows and warped accretion disks.

---

## 1. Differential Geometry & Spacetime Metrics

Spacetime is modeled as a 4-dimensional Lorentzian manifold with coordinate signature $(-, +, +, +)$ and coordinates $x^\mu = (t, r, \theta, \phi)$ in natural geometric units ($G = c = 1$).

### 1.1 Schwarzschild Metric (Static Black Hole)

For a non-rotating black hole of mass $M$:

$$ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2$$

Key radii:
- Event Horizon: $r_s = 2M$
- Photon Sphere (unstable circular light orbits): $r_{ph} = 3M$
- Innermost Stable Circular Orbit (ISCO): $r_{isco} = 6M$

### 1.2 Kerr Metric in Boyer-Lindquist Coordinates (Rotating Black Hole)

For a rotating black hole of mass $M$ with angular momentum $J$ and dimensionless spin parameter $a = J/M \in [0, M)$:

$$ds^2 = -\left(1 - \frac{2Mr}{\Sigma}\right) dt^2 - \frac{4Mar\sin^2\theta}{\Sigma} dt \, d\phi + \frac{\Sigma}{\Delta} dr^2 + \Sigma \, d\theta^2 + \left(r^2 + a^2 + \frac{2Ma^2 r \sin^2\theta}{\Sigma}\right) \sin^2\theta \, d\phi^2$$

where:
$$\Sigma = r^2 + a^2 \cos^2\theta$$
$$\Delta = r^2 - 2Mr + a^2$$

Key physical boundaries:
- Outer Event Horizon: $r_+ = M + \sqrt{M^2 - a^2}$
- Inner (Cauchy) Horizon: $r_- = M - \sqrt{M^2 - a^2}$
- Outer Ergosphere Boundary: $r_{ergo}(\theta) = M + \sqrt{M^2 - a^2 \cos^2\theta}$
- Frame Dragging Angular Velocity:
  $$\omega(r, \theta) = -\frac{g_{t\phi}}{g_{\phi\phi}} = \frac{2Mar}{(r^2 + a^2)\Sigma + 2Mar a^2 \sin^2\theta}$$

### 1.3 Innermost Stable Circular Orbit (ISCO) in Kerr Spacetime

Following Bardeen, Press, and Teukolsky (1972):

$$r_{isco} = M \left(3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)}\right)$$

where the minus sign applies to prograde orbits and plus to retrograde orbits, with:
$$Z_1 = 1 + \left(1 - \frac{a^2}{M^2}\right)^{1/3} \left[\left(1 + \frac{a}{M}\right)^{1/3} + \left(1 - \frac{a}{M}\right)^{1/3}\right]$$
$$Z_2 = \sqrt{3 \frac{a^2}{M^2} + Z_1^2}$$

For $a = 0$ (Schwarzschild), $r_{isco} = 6M$.
For $a \to M$ (extremal Kerr), prograde $r_{isco} \to M$, retrograde $r_{isco} \to 9M$.

---

## 2. Christoffel Symbols & Geodesic Equations

### 2.1 Christoffel Symbols of the Second Kind

The affine Levi-Civita connection coefficients are given by:

$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} \left( \partial_\alpha g_{\beta\sigma} + \partial_\beta g_{\alpha\sigma} - \partial_\sigma g_{\alpha\beta} \right)$$

where $g^{\mu\sigma}$ is the inverse metric tensor ($g^{\mu\sigma} g_{\sigma\nu} = \delta^\mu_\nu$), with symmetry $\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\beta\alpha}$.

### 2.2 Geodesic Equations as a First-Order Hamiltonian System

The geodesic equation describes the path $x^\mu(\lambda)$ of a free-falling particle or photon parameterized by affine parameter $\lambda$:

$$\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\lambda} \frac{dx^\beta}{d\lambda} = 0$$

Let $p^\mu = \frac{dx^\mu}{d\lambda}$ be the 4-momentum. The system is decomposed into 8 coupled first-order ordinary differential equations:

$$\frac{dx^\mu}{d\lambda} = p^\mu$$
$$\frac{dp^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta} p^\alpha p^\beta$$

### 2.3 Constants of Motion & Invariants

Along any geodesic, the scalar norm of 4-momentum is strictly conserved:

$$g_{\mu\nu} p^\mu p^\nu = \kappa$$

- $\kappa = 0$: Null geodesics (photons / electromagnetic radiation).
- $\kappa = -1$: Timelike geodesics (massive matter / particles).

Furthermore, stationarity and axisymmetry imply two exact Killing vectors:
- Energy at infinity: $E = -\xi_{(t)}^\mu p_\mu = -p_t$
- Axial angular momentum: $L = \xi_{(\phi)}^\mu p_\mu = p_\phi$

For Kerr spacetime, Carter's constant $Q$ provides the fourth invariant of motion.

---

## 3. Accretion Disk Physics & Relativistic Radiative Transfer

### 3.1 Keplerian Accretion Disk Model

We model a geometrically thin, optically thick accretion disk residing on the equatorial plane $\theta = \pi/2$ extending from $r_{in} = r_{isco}$ to $r_{out}$.

The angular velocity of circular equatorial Keplerian orbits in Kerr spacetime is:

$$\Omega_K = \frac{d\phi}{dt} = \frac{\sqrt{M}}{r^{3/2} + a\sqrt{M}}$$

The 4-velocity of disk emitter matter $u_{em}^\mu$ is:

$$u_{em}^\mu = u_{em}^t (1, 0, 0, \Omega_K)$$

where normalization $g_{\mu\nu} u_{em}^\mu u_{em}^\nu = -1$ yields:

$$u_{em}^t = \frac{1}{\sqrt{-(g_{tt} + 2\Omega_K g_{t\phi} + \Omega_K^2 g_{\phi\phi})}}$$

### 3.2 Relativistic Doppler Boosting & Gravitational Redshift

When a photon with 4-momentum $p^\mu$ is emitted by the accretion disk and received by an observer at infinity with 4-velocity $u_{obs}^\mu = (1, 0, 0, 0)$, the relativistic frequency shift (redshift factor $g$) is:

$$g = \frac{\nu_{obs}}{\nu_{em}} = \frac{-p_\mu u_{obs}^\mu}{-p_\nu u_{em}^\nu} = \frac{-p_t}{- (p_t u_{em}^t + p_\phi u_{em}^\phi)}$$

By Liouville's theorem in curved spacetime, specific intensity scales as $I_\nu / \nu^3 = \text{constant}$, giving the observed bolometric flux:

$$I_{obs} = g^4 I_{em}$$

This produces intense **relativistic beaming** (Doppler boosting) where the approaching side of the accretion disk appears dramatically brighter and blue-shifted, while the receding side appears dimmer and red-shifted.

---

## 4. Backward Geodesic Ray Tracing Engine

To render the visual appearance of the black hole, we perform backward ray tracing:
1. An observer camera is placed at coordinate position $(r_{cam}, \theta_{cam}, \phi_{cam})$ with camera viewing orientation.
2. For each screen pixel $(u, v)$, a light ray with initial null 4-momentum $p^\mu$ ($g_{\mu\nu} p^\mu p^\nu = 0$) is directed into the past toward the black hole.
3. The geodesic equations are integrated backward in affine parameter $\lambda$ using an adaptive 4th-order Runge-Kutta (RK4) integrator.
4. During propagation, ray-matter interactions are tested:
   - **Event Horizon Capture**: If $r < r_+ + 10^{-3} M$, the ray falls into the event horizon. Pixel intensity = 0 (the black hole shadow).
   - **Accretion Disk Intersection**: If the ray crosses the equatorial plane ($\theta$ crosses $\pi/2$) within the disk boundaries $r_{isco} \le r \le r_{out}$, compute emission temperature, redshift factor $g$, Doppler boost $g^4$, and map to RGB.
   - **Escape to Infinity**: If $r > r_{escape}$ (e.g. $50M$), the ray escapes to the celestial sphere background.

---

## 5. Sub-Pixel Unicode Braille Visualizer & Telemetry HUD

- **2x4 Sub-Pixel Braille Canvas**: Encodes 2x4 pixel sub-grids into Unicode Braille characters `U+2800..U+28FF`.
- **24-bit TrueColor ANSI Output**: Colorizes the accretion disk based on the local redshift factor $g$:
  - Blue-White: Doppler boosted approaching gas ($g > 1.2$).
  - Golden-Yellow: Rest-frame emission ($g \approx 1.0$).
  - Deep Red / Crimson: Gravitationally redshifted receding gas ($g < 0.8$).
  - Pure Void: Event horizon shadow ($g = 0$).
- **Telemetry HUD**: Real-time display of black hole parameters ($M, a/M, r_+, r_{ergo}, r_{isco}$), observer inclination, ray count, and step statistics.

---

## 6. Directory Layout & Module Structure

```
projects/30-relativitas/
├── relativitas/
│   ├── __init__.py          # Public API exports
│   ├── metric.py            # Spacetime metric tensors (Schwarzschild & Kerr) & Christoffel symbols
│   ├── geodesic.py          # 8-state first-order geodesic RK4 integrator & null condition
│   ├── accretion.py         # Keplerian disk kinematics, redshift factor g, Doppler boost
│   ├── raytracer.py         # Backward curved-spacetime ray tracer, camera projection
│   └── visualizer.py        # Sub-pixel Unicode Braille black hole renderer & telemetry HUD
├── tests/
│   ├── test_metric.py       # Metric tensor invariants, Christoffel symmetry, horizons
│   ├── test_geodesic.py     # Energy/angular momentum conservation, null condition g_uv p^u p^v = 0
│   ├── test_accretion.py    # Keplerian velocity, ISCO computation, Doppler redshift g
│   ├── test_raytracer.py    # Shadow capture, disk intersection, celestial escape
│   └── test_visualizer.py   # Braille rasterization, ANSI color mapping, HUD rendering
├── benchmarks/
│   └── bench_relativitas.py # Microbenchmarks (Christoffel evals, RK4 steps, ray throughput, FPS)
├── examples/
│   └── blackhole_workbench.py # Interactive terminal black hole observation laboratory
├── PLAN.md                  # Comprehensive architectural specification
├── TASK_QUEUE.md            # Milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # Documentation and architectural guide
```
