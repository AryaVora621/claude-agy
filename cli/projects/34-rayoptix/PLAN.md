# PLAN.md: RayOptix Architectural Specification

## Executive Summary

**RayOptix** is a high-performance, first-principles, zero-dependency optical engineering, geometric ray tracing, and automated lens design engine written exclusively in the pure Python 3.10+ standard library.

It provides a comprehensive, mathematically rigorous alternative to proprietary optical design software (such as Zemax OpticStudio, Synopsys CODE V, and OSLO). RayOptix enables exact 3D vector non-sequential ray tracing through spherical, conic, and high-order even-aspheric surfaces, full Sellmeier material dispersion across standard Schott/Ohara optical glass catalogs, 3rd-order Seidel aberration analysis, diffraction wavefront calculations with Modulation Transfer Function (MTF) synthesis, and automated Damped Least Squares (Levenberg-Marquardt) optical merit function optimization.

```
                  OPTICAL LENS STACK & 3D VECTOR RAY TRACE
  +-------------------------------------------------------------------------+
  |                                                                         |
  |  Ray In       Surface 1        Surface 2       Aperture Stop    Image   |
  |  =====>      (N-BK7 Crown)    (N-SF11 Flint)      (Stop)        Plane   |
  |                    |               |                |             |     |
  |             +------+-----+         |                |             |     |
  |             |  Curvature |         |                |             |     |
  |             |  c = 1/R   |         |                |             |     |
  |             |  Conic k   |         v                v             v     |
  |             |  Asphere   |    Snell Vector    Pupil Limiting   RMS Spot |
  |             +------------+    Refraction      Vignetting Check  Airy    |
  |                    |                                          Diameter  |
  +--------------------+----------------------------------------------------+
                       |
                       v
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
3RD-ORDER SEIDEL ABERRATIONS    WAVEFRONT & MTF ENGINE      DAMPED LEAST SQUARES
- Spherical (S_I)               - Optical Path Diff (OPD)   - Levenberg-Marquardt
- Coma (S_II)                   - RMS Wavefront Error       - Variable Curvatures & Thk
- Astigmatism (S_III)           - Strehl Ratio S            - Multi-Wavelength Targets
- Petzval Curvature (S_IV)      - Point Spread Function     - Constraint Penalties
- Distortion (S_V)              - MTF vs Spatial Freq (lp/mm)
- Chromatic (C_L, C_T)
```

---

## Mathematical Derivations & Formulations

### 1. Exact 3D Vector Snell's Law of Refraction and Reflection

Let a ray with unit direction vector $\mathbf{d}_1$ travel in a medium of refractive index $n_1$ and strike an optical interface with unit surface normal $\mathbf{n}$ pointing toward the incident medium. The relative refractive index is:
$$\mu = \frac{n_1}{n_2}$$

The cosine of the incident angle is:
$$\cos\theta_1 = -(\mathbf{d}_1 \cdot \mathbf{n})$$

By Snell's Law in vector form, the refracted unit direction $\mathbf{d}_2$ is:
$$\mathbf{d}_2 = \mu \mathbf{d}_1 + \left( \mu \cos\theta_1 - \sqrt{1 - \mu^2 (1 - \cos^2\theta_1)} \right) \mathbf{n}$$

#### Total Internal Reflection (TIR)
If the radicand is negative:
$$1 - \mu^2 (1 - \cos^2\theta_1) < 0$$
then total internal reflection occurs. In that case, or when the surface is designated as a mirror ($n_2 = -n_1$), the reflected ray unit direction is:
$$\mathbf{d}_r = \mathbf{d}_1 + 2 \cos\theta_1 \mathbf{n}$$

#### Fresnel Reflectance Coefficients
For unpolarized light, the amplitude reflection coefficients for perpendicular ($s$) and parallel ($p$) polarizations are:
$$r_s = \frac{n_1 \cos\theta_1 - n_2 \cos\theta_2}{n_1 \cos\theta_1 + n_2 \cos\theta_2}, \quad r_p = \frac{n_2 \cos\theta_1 - n_1 \cos\theta_2}{n_2 \cos\theta_1 + n_1 \cos\theta_2}$$
where $\cos\theta_2 = \sqrt{1 - \mu^2(1 - \cos^2\theta_1)}$. The power reflection coefficient is:
$$R = \frac{1}{2} (r_s^2 + r_p^2), \quad T = 1 - R$$

---

### 2. Optical Surface Geometry & Aspheric Sag

An optical surface centered on the $z$-axis has sagitta $z(r)$ where $r = \sqrt{x^2 + y^2}$.

#### Standard Conic Section
$$z_{\text{conic}}(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}}$$
where:
- $c = \frac{1}{R}$ is the vertex curvature ($R$ is radius of curvature).
- $k$ is the conic constant (Schwarzschild parameter):
  - $k = 0$: Sphere
  - $k = -1$: Paraboloid
  - $k < -1$: Hyperboloid
  - $-1 < k < 0$: Prolate ellipsoid
  - $k > 0$: Oblate ellipsoid

#### High-Order Even Aspheric Surface
$$z(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}} + \sum_{i=1}^M \alpha_i r^{2i}$$
where $\alpha_i$ are polynomial deformation coefficients ($\alpha_2 r^4, \alpha_3 r^6, \dots$).

#### Exact Ray-Surface Intersection
For a ray $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$:
1. **Conic / Quadric Surfaces**:
   Substituting into $(1+k) z^2 - 2 \frac{z}{c} + x^2 + y^2 = 0$ yields a standard quadratic equation:
   $$A t^2 + B t + C = 0$$
   yielding exact closed-form intersection distances $t$.
2. **High-Order Even Aspheres**:
   Solve $F(t) = z(t) - z(r(t)) = 0$ using Newton-Raphson iterations:
   $$t_{k+1} = t_k - \frac{F(t_k)}{F'(t_k)}$$
   converging to machine precision ($|F(t)| < 10^{-12}$) in 3 to 5 iterations.

#### Analytical Surface Normal Vector
The surface normal gradient vector $\nabla (z - z(x, y))$ gives:
$$\mathbf{n} = \frac{(-\partial z / \partial x, \; -\partial z / \partial y, \; 1)}{\sqrt{1 + (\partial z / \partial x)^2 + (\partial z / \partial y)^2}}$$
where $\frac{\partial z}{\partial x} = \frac{dz}{dr} \frac{x}{r}$ and $\frac{\partial z}{\partial y} = \frac{dz}{dr} \frac{y}{r}$.

---

### 3. Optical Dispersion & Sellmeier Equation

Refractive indices vary with optical wavelength $\lambda$ in micrometers according to the 3-term Sellmeier formula:
$$n^2(\lambda) - 1 = \frac{B_1 \lambda^2}{\lambda^2 - C_1} + \frac{B_2 \lambda^2}{\lambda^2 - C_2} + \frac{B_3 \lambda^2}{\lambda^2 - C_3}$$

#### Standard Fraunhofer Wavelengths
- **d-line**: $\lambda_d = 0.5875618 \;\mu\text{m}$ (Helium yellow)
- **F-line**: $\lambda_F = 0.4861327 \;\mu\text{m}$ (Hydrogen blue)
- **C-line**: $\lambda_C = 0.6562725 \;\mu\text{m}$ (Hydrogen red)

#### Abbe Number (V-number)
$$V_d = \frac{n_d - 1}{n_F - n_C}$$
- High $V_d$ ($> 50$): Crown glass (low dispersion).
- Low $V_d$ ($\le 50$): Flint glass (high dispersion).

---

### 4. Paraxial ABCD Ray Transfer Matrix & First-Order Cardinal Points

For paraxial rays defined by ray height $y$ and optical angle $u$:
- **Refraction Matrix**:
  $$\begin{bmatrix} y_2 \\ u_2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ -\frac{n_2 - n_1}{n_2 R} & \frac{n_1}{n_2} \end{bmatrix} \begin{bmatrix} y_1 \\ u_1 \end{bmatrix}$$
- **Translation Matrix**:
  $$\begin{bmatrix} y_2 \\ u_2 \end{bmatrix} = \begin{bmatrix} 1 & d \\ 0 & 1 \end{bmatrix} \begin{bmatrix} y_1 \\ u_1 \end{bmatrix}$$

The cumulative system matrix from object space to image space is:
$$\mathbf{M} = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$$
with $\det(\mathbf{M}) = \frac{n_0}{n_k}$.

#### Cardinal Properties
- **Effective Focal Length (EFL)**: $f = -\frac{1}{C}$
- **Back Focal Length (BFL)**: Distance from last surface to image focus $f_b = -\frac{A}{C}$
- **Front Focal Length (FFL)**: Distance from first surface to front focus $f_f = -\frac{D}{C}$
- **Working F-Number**: $F/\# = \frac{\text{EFL}}{D_{\text{EP}}}$
- **Lagrange Invariant**: $H = n (y \bar{u} - \bar{y} u)$

---

### 5. Third-Order Seidel Aberration Sums

Tracing paraxial marginal ray $(y, u)$ and paraxial chief ray $(\bar{y}, \bar{u})$ yields refraction invariants:
$$i = u + y c, \quad \bar{i} = \bar{u} + \bar{y} c$$
$$A = n (u + y c) = n i$$

For each surface $j$:
$$\Delta\left(\frac{u}{n}\right) = \frac{u'}{n'} - \frac{u}{n}$$

The five primary monochromatic Seidel aberration sums are:
1. **Spherical Aberration ($S_I$)**:
   $$S_I = \sum_j A_j^2 y_j \Delta\left(\frac{u}{n}\right)_j + a_{\text{asph}}$$
2. **Coma ($S_{II}$)**:
   $$S_{II} = \sum_j A_j \bar{A}_j y_j \Delta\left(\frac{u}{n}\right)_j$$
3. **Astigmatism ($S_{III}$)**:
   $$S_{III} = \sum_j \bar{A}_j^2 y_j \Delta\left(\frac{u}{n}\right)_j$$
4. **Petzval Field Curvature ($S_{IV}$)**:
   $$S_{IV} = \sum_j H^2 c_j \Delta\left(\frac{1}{n}\right)_j$$
5. **Distortion ($S_V$)**:
   $$S_V = \sum_j \bar{A}_j \left[ \bar{A}_j \bar{y}_j \Delta\left(\frac{u}{n}\right)_j + H \Delta\left(\frac{1}{n^2}\right)_j \right]$$

#### Chromatic Aberrations
1. **Longitudinal Chromatic Aberration ($C_L$)**:
   $$C_L = \sum_j y_j A_j \Delta\left(\frac{\delta n}{n}\right)_j$$
2. **Transverse Chromatic Aberration ($C_T$)**:
   $$C_T = \sum_j y_j \bar{A}_j \Delta\left(\frac{\delta n}{n}\right)_j$$

---

### 6. Wavefront Error & Modulation Transfer Function (MTF)

The optical path difference (OPD) $\Delta W(x_p, y_p)$ across the exit pupil is computed against a reference spherical wavefront centered at the Gaussian image point:
$$\text{OPD} = \text{OPL}_{\text{ray}} - \text{OPL}_{\text{chief}}$$

#### Root-Mean-Square Wavefront Error
$$\omega_{\text{RMS}} = \sqrt{\langle W^2 \rangle - \langle W \rangle^2}$$

#### Strehl Ratio
By the Maréchal approximation:
$$S \approx \exp\left( -\left(\frac{2\pi \omega_{\text{RMS}}}{\lambda}\right)^2 \right)$$

#### Optical Transfer Function (OTF) & MTF
The exit pupil complex transmission function is:
$$P(x, y) = A(x, y) \exp\left( i \frac{2\pi}{\lambda} W(x, y) \right)$$
The coherent Point Spread Function (PSF) is the Fourier transform of the pupil function:
$$\text{PSF}(u, v) = |\mathcal{F}\{ P(x, y) \}|^2$$
The incoherent Optical Transfer Function is:
$$\text{OTF}(f_x, f_y) = \frac{\mathcal{F}\{\text{PSF}\}}{\iint \text{PSF} \, dx \, dy}$$
The Modulation Transfer Function is the magnitude:
$$\text{MTF}(f) = |\text{OTF}(f)|$$
evaluated from zero frequency to the diffraction cutoff frequency $f_c = \frac{1}{\lambda F/\#}$.

---

### 7. Damped Least Squares (Levenberg-Marquardt) Lens Optimization

We optimize optical design parameters $\mathbf{x} = (c_1, \dots, c_m, d_1, \dots, d_p, k_1, \dots)^T$ to minimize a composite merit function $\Phi$:
$$\Phi(\mathbf{x}) = \sum_{i=1}^M w_i^2 f_i(\mathbf{x})^2$$
where operands $f_i$ represent:
- Transverse ray aberrations $(x_{\text{image}}, y_{\text{image}})$ across multiple field angles and wavelengths.
- Target focal length error: $f_{\text{target}} - \text{EFL}$.
- Glass edge thickness and center thickness constraint penalties.

At iteration $k$, the linear update solves the damped normal equations:
$$(\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \mathbf{D}^2) \Delta\mathbf{x} = -\mathbf{J}^T \mathbf{W} \mathbf{f}$$
where:
- $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$ is the numerical Jacobian computed via central differences.
- $\mathbf{D} = \text{diag}(\mathbf{J}^T \mathbf{W} \mathbf{J})$ is the Marquardt diagonal scaling matrix.
- $\lambda$ is the adaptive damping parameter adjusted according to the reduction ratio $\rho$.

---

## Architectural Layout

```
projects/34-rayoptix/
|-- rayoptix/
|   |-- __init__.py          # Public package exports
|   |-- ray.py               # 3D Ray, Vector Snell's Law, Fresnel, TIR
|   |-- surface.py           # Spherical, Conic, Aspheric, Normals, Intersections
|   |-- material.py          # Sellmeier models, Abbe number, Glass catalog
|   |-- system.py            # OpticalSystem, Paraxial ABCD, Cardinal points
|   |-- aberrations.py       # Seidel sums SI-SV, CL, CT, Wavefront polynomial
|   |-- wavefront.py         # Exit pupil OPD, RMS error, Strehl, PSF, MTF
|   |-- analysis.py          # Spot diagrams, RMS radius, Ray aberration fans
|   |-- optimizer.py         # Levenberg-Marquardt Damped Least Squares optimizer
|   |-- presets.py           # Cooke Triplet, Doublet, Singlet, Telescope
|   |-- visualizer.py        # Sub-pixel Braille 2D lens cross-section & HUD
|-- tests/
|   |-- __init__.py
|   |-- test_ray.py          # Unit tests for 3D ray & Snell refraction/reflection
|   |-- test_surface.py      # Unit tests for conic & aspheric intersections
|   |-- test_material.py     # Unit tests for Sellmeier formula & glass library
|   |-- test_system.py       # Unit tests for paraxial ABCD & cardinal points
|   |-- test_aberrations.py  # Unit tests for Seidel sums & symmetry
|   |-- test_wavefront.py    # Unit tests for OPD, Strehl, and MTF
|   |-- test_analysis.py     # Unit tests for spot diagrams and ray fans
|   |-- test_optimizer.py    # Unit tests for DLS lens optimization
|   |-- test_presets.py      # Unit tests for classical lens systems
|   |-- test_visualizer.py   # Unit tests for Braille canvas & telemetry HUD
|-- benchmarks/
|   |-- bench_rayoptix.py    # High-throughput performance microbenchmarks
|-- examples/
|   |-- lens_workbench.py    # Interactive terminal optical engineering lab
|-- PLAN.md                  # This file
|-- TASK_QUEUE.md            # Work unit progress tracking
|-- CHECKPOINT_LAST.md       # Session checkpoint state
|-- README.md                # Comprehensive documentation
```
