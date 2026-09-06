# Architectural Specification: Project 31 (StellarFusion)

## Executive Summary

**StellarFusion** is a first-principles, zero-dependency Magnetohydrodynamic (MHD) plasma physics simulation engine, Grad-Shafranov equilibrium solver, and Tokamak magnetic confinement reactor laboratory implemented in pure Python 3.10+ standard library.

The engine solves the non-linear 2D Grad-Shafranov partial differential equation for axisymmetric toroidal plasma equilibria, traces 3D helical magnetic field lines and Poincaré surface-of-section puncture plots, integrates relativistic charged-particle guiding-center and full Lorentz gyro-orbits via the symplectic Boris algorithm (capturing neoclassical banana orbits, passing particles, and magnetic mirror trapping), evaluates safety factor profiles $q(\psi)$ and magnetic shear $s(\psi)$, and renders sub-pixel Unicode Braille poloidal flux cross-sections with 24-bit TrueColor plasma temperature and current density telemetry.

---

## 1. Magnetohydrodynamics (MHD) & Grad-Shafranov Equilibrium

### 1.1 Ideal MHD Equilibrium Equation
In stationary ideal magnetohydrodynamics with scalar plasma pressure $p$, the momentum balance equation is:
$$\vec{J} \times \vec{B} = \nabla p$$
together with Ampere's law $\nabla \times \vec{B} = \mu_0 \vec{J}$ and Gauss's law for magnetism $\nabla \cdot \vec{B} = 0$.

### 1.2 Toroidal Axisymmetric Coordinates
In cylindrical coordinates $(R, \phi, Z)$ with toroidal symmetry ($\partial / \partial \phi = 0$):
The magnetic field is decomposed into poloidal flux function $\psi(R, Z)$ and poloidal current stream function $F(\psi) = R B_\phi$:
$$\vec{B} = \nabla \phi \times \nabla \psi + F(\psi) \nabla \phi = -\frac{1}{R} \frac{\partial \psi}{\partial Z} \hat{R} + \frac{F(\psi)}{R} \hat{\phi} + \frac{1}{R} \frac{\partial \psi}{\partial R} \hat{Z}$$

The toroidal current density is:
$$\mu_0 J_\phi = -\frac{1}{R} \Delta^* \psi$$
where the elliptic Shafranov operator $\Delta^*$ is:
$$\Delta^* \psi = R \frac{\partial}{\partial R}\left(\frac{1}{R} \frac{\partial \psi}{\partial R}\right) + \frac{\partial^2 \psi}{\partial Z^2} = \frac{\partial^2 \psi}{\partial R^2} - \frac{1}{R}\frac{\partial \psi}{\partial R} + \frac{\partial^2 \psi}{\partial Z^2}$$

### 1.3 The Grad-Shafranov Equation
Substituting $\vec{J} \times \vec{B} = \nabla p$ yields the fundamental non-linear 2D elliptic PDE:
$$\Delta^* \psi = -\mu_0 R^2 \frac{dp}{d\psi} - F \frac{dF}{d\psi}$$

where:
- $p(\psi)$ is the plasma pressure profile (constant on magnetic flux surfaces)
- $F(\psi) = R B_\phi$ is the diamagnetic toroidal field function
- The magnetic surfaces $\psi(R, Z) = \text{const}$ form nested toroidal topological tori.

### 1.4 Solovev Analytical Equilibrium Benchmark
For linear source profiles:
$$p'(\psi) = p_1 = \text{const}, \quad F F'(\psi) = F_1 = \text{const}$$
Solovev formulated exact analytical solutions with elongation $\kappa$ and triangularity $\delta$:
$$\psi_{sol}(R, Z) = \frac{\psi_0}{R_0^4} \left[ R^2 Z^2 + \frac{\kappa^2}{4} (R^2 - R_0^2)^2 \right]$$
used for rigorous verification of the numerical finite-difference solver.

---

## 2. Safety Factor $q(\psi)$, Magnetic Shear & Instability Boundaries

### 2.1 Safety Factor Profile $q(\psi)$
The safety factor $q$ measures the average number of toroidal turns a magnetic field line makes per single poloidal turn around the magnetic axis:
$$q(\psi) = \frac{1}{2\pi} \oint_{\psi} \frac{B_\phi}{R B_p} dl_p = \frac{F(\psi)}{2\pi} \oint_{\psi} \frac{dl_p}{R^2 |\nabla \psi|}$$

Key physics:
- $q_0 = q(\psi_{axis})$: central safety factor on the magnetic axis (typically $> 1$ to avoid internal sawtooth crashes).
- $q_{95} = q(\psi_{95\%})$: edge safety factor (typically $\ge 3$ for tokamak operational stability against external kink modes).
- Rational surfaces $q = m/n$ (e.g. $1/1, 2/1, 3/2$): locations where resonant magnetic perturbations trigger magnetic reconnection and magnetic island chains.

### 2.2 Magnetic Shear
$$s(\psi) = \frac{r}{q} \frac{dq}{dr}$$
Positive magnetic shear stabilizes pressure-driven interchange and ballooning modes.

---

## 3. Symplectic Boris Particle Pusher & Neoclassical Orbits

### 3.1 Lorentz Force Equation
For a particle of charge $q$ and mass $m$:
$$\frac{d\vec{x}}{dt} = \vec{v}, \quad \frac{d\vec{v}}{dt} = \frac{q}{m} (\vec{E} + \vec{v} \times \vec{B})$$

### 3.2 The Symplectic Boris Algorithm (Boris 1970)
Separates the electric field acceleration from the magnetic rotation:
1. Half-step electric acceleration:
   $$\vec{v}^- = \vec{v}_k + \frac{q \Delta t}{2m} \vec{E}(\vec{x}_k)$$
2. Rotation by magnetic field:
   $$\vec{t} = \frac{q \Delta t}{2m} \vec{B}(\vec{x}_k), \quad \vec{s} = \frac{2\vec{t}}{1 + |\vec{t}|^2}$$
   $$\vec{v}' = \vec{v}^- + \vec{v}^- \times \vec{t}$$
   $$\vec{v}^+ = \vec{v}^- + \vec{v}' \times \vec{s}$$
3. Second half-step electric acceleration:
   $$\vec{v}_{k+1} = \vec{v}^+ + \frac{q \Delta t}{2m} \vec{E}(\vec{x}_k)$$
4. Position update:
   $$\vec{x}_{k+1} = \vec{x}_k + \vec{v}_{k+1} \Delta t$$

The Boris pusher is volume-preserving in phase space, unconditionally stable with respect to cyclotron frequency $\omega_c \Delta t$, and exactly conserves kinetic energy when $\vec{E} = 0$.

### 3.3 Neoclassical Orbits: Passing vs Trapped Banana Particles
Because $B \propto 1/R$, a particle moving toward the high-field inboard side ($R < R_0$) experiences a magnetic mirror force $-\mu \nabla_\parallel B$.
- If $v_\parallel / v > \sqrt{2 \epsilon}$ (where $\epsilon = r/R_0$ is the inverse aspect ratio): particle is **passing** (circulates continuously).
- If $v_\parallel / v < \sqrt{2 \epsilon}$: particle reflects at the mirror point, executing a closed banana-shaped drift orbit with banana bounce frequency $\omega_b$ and radial banana width $\Delta r_b \approx \frac{q \rho_L}{\sqrt{\epsilon}}$.

---

## 4. Poincaré Puncture Plot & Magnetic Field Line Tracer

Traces magnetic field lines by integrating:
$$\frac{dR}{d\phi} = \frac{R B_R}{B_\phi}, \quad \frac{dZ}{d\phi} = \frac{R B_Z}{B_\phi}$$
Every time a field line completes a toroidal revolution ($\phi \equiv 0 \pmod{2\pi}$), its poloidal intersection $(R, Z)$ is recorded.
- In unperturbed equilibria: punctures trace smooth invariant KAM 1D curves (closed nested flux surfaces).
- Under resonant magnetic perturbations: punctures reveal $m/n$ magnetic islands and ergodic chaotic edge field layers.

---

## 5. Sub-Pixel Braille Visualizer & Fusion Reactor Telemetry

- **2x4 Sub-Pixel Braille Canvas**: Generates high-resolution $(R, Z)$ cross-sections using Unicode Braille glyphs `U+2800..U+28FF`.
- **TrueColor ANSI Temperature & Flux Gradients**:
  - Hot Core: Yellow-White ($T_i > 10$ keV)
  - Confinement Region: Golden-Amber / Cyan ($5-10$ keV)
  - Edge / Scrape-Off Layer: Deep Blue / Magenta ($< 1$ keV)
  - Vacuum Vessel & Limiter: Gray outline
- **Fusion Reactor HUD**:
  - Major Radius $R_0$, Minor Radius $a$, Aspect Ratio $A = R_0/a$
  - Toroidal Magnetic Field $B_0$, Plasma Current $I_p$
  - Safety Factors $q_0, q_{95}$
  - Plasma Beta: Toroidal Beta $\beta_t = \frac{2\mu_0 \langle p \rangle}{B_0^2}$, Poloidal Beta $\beta_p$
  - Lawson Criterion Triple Product $n_e T_i \tau_E \ge 3 \times 10^{21} \text{ keV}\cdot\text{s}\cdot\text{m}^{-3}$

---

## 6. Directory Layout & Module Structure

```
projects/31-stellarfusion/
├── stellarfusion/
│   ├── __init__.py          # Public API exports
│   ├── equilibrium.py       # Grad-Shafranov PDE solver & Solovev analytical benchmarks
│   ├── magnetic.py          # 3D magnetic field evaluation, flux surfaces & safety factor q(psi)
│   ├── particles.py         # Symplectic Boris pusher, gyro-orbits & neoclassical banana orbits
│   ├── poincare.py          # Field line integration & Poincaré surface-of-section puncture map
│   └── visualizer.py        # Sub-pixel Unicode Braille poloidal cross-section & telemetry HUD
├── tests/
│   ├── test_equilibrium.py   # Grad-Shafranov convergence & Solovev analytical agreement
│   ├── test_magnetic.py      # Field divergence div(B)=0, safety factor q, magnetic axis
│   ├── test_particles.py     # Boris energy conservation, gyrofrequency, banana orbit reflection
│   ├── test_poincare.py      # Field line tracing, puncture periodicity, flux conservation
│   └── test_visualizer.py    # Braille canvas rasterization, color mapping, HUD formatting
├── benchmarks/
│   └── bench_stellarfusion.py # Microbenchmarks (GS iterations/s, Boris steps/s, field evals/s)
├── examples/
│   └── tokamak_workbench.py  # Interactive CLI fusion laboratory with D-T burning telemetry
├── PLAN.md                  # Comprehensive architectural specification
├── TASK_QUEUE.md            # Milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # Documentation and architectural guide
```
