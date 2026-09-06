# Relativitas: General Relativity Curved-Spacetime Geodesics & Black Hole Accretion Engine

A first-principles, zero-dependency General Relativity simulation engine, curved spacetime geodesic integrator, and black hole accretion disk ray tracer implemented in pure Python 3.10+ standard library.

Relativitas solves the Einstein geodesic equations of motion in curved 4-dimensional pseudo-Riemannian spacetime manifolds for both static (Schwarzschild) and rotating (Kerr) black holes. It simulates gravitational lensing, frame dragging (Lense-Thirring effect), the ergosphere, the photon sphere, gravitational redshift, relativistic Doppler beaming ($I_{obs} = g^4 I_{em}$), and renders sub-pixel Unicode Braille terminal visualizations of black hole shadows and warped accretion disks.

---

## Key Theoretical Foundations

### 1. 4-Dimensional Pseudo-Riemannian Geometry

Spacetime is modeled as a 4-dimensional Lorentzian manifold with coordinate signature $(-, +, +, +)$ and coordinates $x^\mu = (t, r, \theta, \phi)$ in geometric units ($G = c = 1$).

#### Schwarzschild Metric (Static Black Hole)
For a spherically symmetric black hole of mass $M$:

$$ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \left(1 - \frac{2M}{r}\right)^{-1} dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2$$

- Event Horizon: $r_s = 2M$
- Photon Sphere: $r_{ph} = 3M$
- Innermost Stable Circular Orbit (ISCO): $r_{isco} = 6M$

#### Kerr Metric in Boyer-Lindquist Coordinates (Rotating Black Hole)
For a rotating black hole of mass $M$ and spin parameter $a = J/M \in [0, M)$:

$$ds^2 = -\left(1 - \frac{2Mr}{\Sigma}\right) dt^2 - \frac{4Mar\sin^2\theta}{\Sigma} dt \, d\phi + \frac{\Sigma}{\Delta} dr^2 + \Sigma \, d\theta^2 + \left(r^2 + a^2 + \frac{2Ma^2 r \sin^2\theta}{\Sigma}\right) \sin^2\theta \, d\phi^2$$

where:
$$\Sigma = r^2 + a^2 \cos^2\theta$$
$$\Delta = r^2 - 2Mr + a^2$$

Key physical boundaries:
- Outer Event Horizon: $r_+ = M + \sqrt{M^2 - a^2}$
- Inner (Cauchy) Horizon: $r_- = M - \sqrt{M^2 - a^2}$
- Ergosphere Boundary: $r_{ergo}(\theta) = M + \sqrt{M^2 - a^2 \cos^2\theta}$
- Frame Dragging Angular Velocity: $\omega(r, \theta) = -g_{t\phi} / g_{\phi\phi}$
- Analytical ISCO (Bardeen, Press, Teukolsky 1972):
  $$r_{isco} = M \left(3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)}\right)$$

### 2. Christoffel Symbols & 8-State Geodesic RK4 Integrator

The affine Levi-Civita connection coefficients:
$$\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\sigma} \left( \partial_\alpha g_{\beta\sigma} + \partial_\beta g_{\alpha\sigma} - \partial_\sigma g_{\alpha\beta} \right)$$

The geodesic equations are decomposed into an 8-state first-order system with affine parameter $\lambda$:
$$\frac{dx^\mu}{d\lambda} = p^\mu$$
$$\frac{dp^\mu}{d\lambda} = -\Gamma^\mu_{\alpha\beta} p^\alpha p^\beta$$

Conserved Killing invariants:
- Energy at infinity: $E = -p_t = -g_{t\mu} p^\mu$
- Axial angular momentum: $L = p_\phi = g_{\phi\mu} p^\mu$
- 4-momentum norm: $g_{\mu\nu} p^\mu p^\nu = \kappa$ (0 for photons, -1 for massive matter)

### 3. Accretion Disk & Relativistic Radiative Transfer

- Keplerian Circular Orbital Velocity:
  $$\Omega_K = \frac{d\phi}{dt} = \frac{\sqrt{M}}{r^{3/2} + a\sqrt{M}}$$
- Emitter 4-Velocity: $u_{em}^\mu = u_{em}^t (1, 0, 0, \Omega_K)$ normalized to $g_{\mu\nu} u_{em}^\mu u_{em}^\nu = -1$.
- Relativistic Frequency Shift Factor $g$:
  $$g = \frac{\nu_{obs}}{\nu_{em}} = \frac{-p_t}{-u_{em}^t (p_t + \Omega_K p_\phi)}$$
- Relativistic Doppler Beaming (Liouville theorem):
  $$I_{obs} = g^4 I_{em}$$
  Producing intense blue-shifting and amplification on the approaching side of the disk and red-shifting/dimming on the receding side.

---

## Architectural Architecture & Structure

```
projects/30-relativitas/
├── relativitas/
│   ├── __init__.py          # Public API exports
│   ├── metric.py            # Schwarzschild and Kerr metric tensors and Christoffel connection symbols
│   ├── geodesic.py          # 8-state first-order geodesic RK4 integrator and conservation audits
│   ├── accretion.py         # Keplerian disk kinematics, Doppler boosting (I ~ g^4), and redshift
│   ├── raytracer.py         # Backward curved-spacetime ray tracer, camera tetrad projection
│   └── visualizer.py        # Sub-pixel Unicode Braille visualizer (2x4 dots) and telemetry HUD
├── tests/
│   ├── test_metric.py       # Tensors, inverses, horizons, and Christoffel symmetry
│   ├── test_geodesic.py     # Infalling/escaping paths, equatorial crossings, Killing invariants
│   ├── test_accretion.py    # Keplerian velocity, 4-velocity normalization, Doppler shift
│   ├── test_raytracer.py    # Camera tetrad null vectors, ray classification, framebuffer
│   └── test_visualizer.py   # Braille canvas bit-twiddling, ANSI true-color, HUD format
├── benchmarks/
│   └── bench_relativitas.py # Microbenchmarks (Christoffel evals, RK4 steps, ray throughput, FPS)
├── examples/
│   └── blackhole_workbench.py # Interactive CLI black hole observation laboratory
├── PLAN.md                  # Comprehensive architectural specification
├── TASK_QUEUE.md            # Milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # Documentation and architectural guide
```

---

## Performance Microbenchmarks

Executed on Apple Silicon (pure Python 3.10+ standard library, single-threaded):

| Benchmark Target | Operations / Units | Execution Time | Throughput |
| :--- | :--- | :--- | :--- |
| **Schwarzschild Christoffel Symbols** | 380,000 | 0.8043 s | **472,482.9 ops/s** |
| **Kerr Numerical Christoffel Symbols** | 26,500 | 0.8014 s | **33,066.0 ops/s** |
| **Geodesic 8-State RK4 Steps** | 36,000 | 0.8359 s | **43,069.2 ops/s** |
| **Accretion Radiative Transfer & Redshift** | 255,000 | 0.8111 s | **314,374.0 ops/s** |
| **Curved Spacetime Ray Tracing (Rays)** | 384 | 0.8869 s | **432.9 rays/s** |
| **Sub-Pixel Braille Frame Rasterization** | 800 | 0.8470 s | **944.5 frames/s** |

---

## Quickstart & Usage

### 1. Running the Interactive Simulation Workbench

```bash
cd projects/30-relativitas
python3 examples/blackhole_workbench.py
```

### 2. Python API Example

```python
from relativitas.metric import KerrMetric
from relativitas.accretion import AccretionDisk
from relativitas.raytracer import Camera, RayTracer
from relativitas.visualizer import BlackHoleVisualizer

# 1. Instantiate a rapidly rotating Kerr black hole (spin a = 0.92M)
metric = KerrMetric(mass=1.0, spin=0.92)

# 2. Attach a Keplerian accretion disk from ISCO out to 14M
disk = AccretionDisk(metric, r_in=metric.isco_radius(), r_out=14.0)

# 3. Position virtual observer camera at r = 20M with 80 degree inclination
camera = Camera(r=20.0, theta=1.396, fov_degrees=48.0)

# 4. Ray trace curved spacetime null geodesics backward
raytracer = RayTracer(metric, disk=disk, max_steps=300)
frame = raytracer.render_frame(camera, width=80, height=32)

# 5. Render sub-pixel Unicode Braille visualization with 24-bit TrueColor
visualizer = BlackHoleVisualizer(true_color=True)
print(visualizer.format_telemetry_hud(metric, camera, frame))
print(visualizer.rasterize_frame(frame))
```

### 3. Running the Test Suite

```bash
cd projects/30-relativitas
python3 -m unittest discover -s tests -p "test_*.py" -v
```

All 29 tests run in ~0.26 seconds with 100% pass rate.
