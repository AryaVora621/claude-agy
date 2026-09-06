# LuminaWave: Computational Electromagnetics & Silicon Photonics Engine
## Architectural Design & Engineering Specification (PLAN.md)

LuminaWave is a pure Python 3.10+ standard library simulation framework for classical electrodynamics, 2D finite-difference time-domain (FDTD) Maxwell solvers, and silicon photonic integrated circuits (PIC).

```
       +-------------------------------------------------------------------+
       |                       LUMINAWAVE FDTD                             |
       |                Computational Electrodynamics                      |
       +-------------------------------------------------------------------+
                                         |
     +-----------------+-----------------+-----------------+-----------------+
     |                 |                 |                 |                 |
     v                 v                 v                 v                 v
+---------+      +-----------+     +-----------+     +-----------+     +-----------+
| 2D Yee  |      | Uniaxial  |     | Optical   |     | Silicon   |     | On-the-Fly|
| Lattice |      | PML (UPML)|     | Source    |     | Photonic  |     | DFT Flux  |
| Stagger |      | Absorber  |     | Injectors |     | Circuits  |     | Monitors  |
+---------+      +-----------+     +-----------+     +-----------+     +-----------+
```

---

## 1. Mathematical Foundations

### 1.1 2D Transverse Magnetic (TM_z) Maxwell Equations
In a 2D planar dielectric geometry where materials and fields are invariant along $z$, the electromagnetic field decouples into Transverse Electric ($TE_z$) and Transverse Magnetic ($TM_z$) polarization modes. In the $TM_z$ mode, the non-zero field components are $E_z$, $H_x$, and $H_y$:

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu_r \mu_0} \frac{\partial E_z}{\partial y}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu_r \mu_0} \frac{\partial E_z}{\partial x}$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon_r \epsilon_0} \left( \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} - \sigma E_z \right)$$

where:
- $\epsilon_r(x, y)$ is the relative dielectric permittivity.
- $\mu_r(x, y)$ is the relative magnetic permeability ($\mu_r = 1$ for non-magnetic optical materials).
- $\sigma(x, y)$ is the electric conductivity.

### 1.2 Yee Staggered Spatial & Temporal Discretization
Using Kane Yee's 1966 staggered leapfrog formulation:
- $E_z$ is defined at integer grid coordinates $(i, j)$ and integer time steps $n$.
- $H_x$ is defined at $(i, j + 1/2)$ and half-integer time steps $n + 1/2$.
- $H_y$ is defined at $(i + 1/2, j)$ and half-integer time steps $n + 1/2$.

The discrete leapfrog update equations:

$$H_x^{n+1/2}(i, j+1/2) = H_x^{n-1/2}(i, j+1/2) - \frac{\Delta t}{\mu_0 \Delta y} \left[ E_z^n(i, j+1) - E_z^n(i, j) \right]$$

$$H_y^{n+1/2}(i+1/2, j) = H_y^{n-1/2}(i+1/2, j) + \frac{\Delta t}{\mu_0 \Delta x} \left[ E_z^n(i+1, j) - E_z^n(i, j) \right]$$

$$E_z^{n+1}(i, j) = C_a(i, j) E_z^n(i, j) + C_b(i, j) \left( \frac{H_y^{n+1/2}(i+1/2, j) - H_y^{n+1/2}(i-1/2, j)}{\Delta x} - \frac{H_x^{n+1/2}(i, j+1/2) - H_x^{n+1/2}(i, j-1/2)}{\Delta y} \right)$$

where:
$$C_a = \frac{2\epsilon - \sigma \Delta t}{2\epsilon + \sigma \Delta t}, \quad C_b = \frac{2 \Delta t}{2\epsilon + \sigma \Delta t}$$

### 1.3 Courant-Friedrichs-Lewy (CFL) Numerical Stability
To guarantee numerical stability, the time step $\Delta t$ must satisfy the CFL condition:

$$\Delta t \le \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}}}$$

For square grid cells $\Delta x = \Delta y = \Delta$:
$$\Delta t \le \frac{\Delta}{c \sqrt{2}} \approx 0.7071 \frac{\Delta}{c}$$

LuminaWave enforces a default Courant number $S = \frac{c \Delta t}{\Delta} = 0.50$ (or $0.70$), guaranteeing unconditional convergence.

---

## 2. Uniaxial Perfectly Matched Layer (UPML) Boundary Conditions

To truncate the infinite physical domain to a compact computational grid without non-physical boundary reflections, LuminaWave implements a Uniaxial Perfectly Matched Layer (UPML):
- Absorbing layers of thickness $d_{pml}$ (typically 8 to 16 cells) are placed around the simulation perimeter.
- Polynomial conductivity profile:
  $$\sigma_x(x) = \sigma_{max} \left( \frac{x}{d_{pml}} \right)^m, \quad \sigma_{max} = \frac{(m + 1) \ln(1/R_0)}{2 \eta d_{pml}}$$
  where $m = 3$, $R_0 = 10^{-6}$ (target normal reflection), and $\eta = \sqrt{\mu_0 / \epsilon_0}$ is the free-space wave impedance.
- Auxiliary differential equations update constitutive relation tensors $\mathbf{D} = \epsilon \mathbf{E}$ and $\mathbf{B} = \mu \mathbf{H}$ with complex frequency-dependent coordinate stretching, yielding $<-60\text{ dB}$ reflection across all incident angles.

---

## 3. Optical Excitation Sources

1. **Gaussian Pulse**: Broadband excitation ideal for computing continuous transfer functions and S-parameter spectra across wide bandwidths:
   $$s(t) = \exp\left( -\frac{(t - t_0)^2}{2 \tau^2} \right)$$
2. **Continuous Wave (CW)**: Monochromatic sinusoidal source with smooth cosine ramp-up eliminating high-frequency startup transients:
   $$s(t) = \sin(2\pi f_0 t) \cdot \min\left(1.0, \frac{t}{t_{ramp}}\right)$$
3. **Modulated Gaussian Pulse**: High-frequency optical carrier modulated by a Gaussian envelope centered at wavelength $\lambda_0 = 1.55 \ \mu\text{m}$:
   $$s(t) = \exp\left( -\frac{(t - t_0)^2}{2 \tau^2} \right) \cos(2\pi f_0 (t - t_0))$$
4. **Injection Modalities**:
   - **Soft Source**: Adds current density $J_z(t)$ to $E_z$, permitting backward propagating waves to pass through transparently.
   - **Hard Source**: Clamps $E_z$ directly to $s(t)$.
   - **Total-Field / Scattered-Field (TFSF)**: Huygens boundary injecting pure unidirectional guided modes.

---

## 4. Silicon Photonic Devices & Integrated Circuits

LuminaWave models planar Silicon-on-Insulator (SOI) devices with silicon core ($n_{Si} = 3.48$) and silicon dioxide cladding ($n_{SiO2} = 1.44$) at telecommunications wavelength $\lambda_0 = 1.55 \ \mu\text{m}$:

1. **Single-Mode Strip Waveguide**: High-index contrast dielectric slab guiding optical energy via Total Internal Reflection (TIR).
2. **90-Degree Waveguide Bend**: Circular dielectric bend with optimized bend radius minimizing radiative bending loss.
3. **2x2 Directional Coupler**: Two parallel waveguides brought into close proximity (coupling gap $g \approx 0.15 \ \mu\text{m}$). Evanescent wave tunneling transfers power between symmetric and anti-symmetric supermodes:
   $$P_{\text{cross}}(L) = \sin^2(\kappa L), \quad P_{\text{bar}}(L) = \cos^2(\kappa L)$$
4. **Optical Micro-Ring Resonator**: Bus waveguide coupled to a dielectric ring. Resonances occur when optical path length equals an integer number of wavelengths ($2\pi R n_{eff} = m \lambda_m$). Computes:
   - Free Spectral Range (FSR)
   - Quality Factor ($Q = \lambda_0 / \Delta \lambda_{FWHM}$)
   - Extinction Ratio
5. **Mach-Zehnder Interferometer (MZI)**: Dual-arm balanced interferometer demonstrating constructive and destructive optical interference modulated by refractive index variations.
6. **Photonic Crystal Defect Waveguide**: Periodic 2D dielectric rod lattice with a missing row of rods guiding light within the photonic bandgap (PBG).

---

## 5. On-the-Fly Discrete Fourier Transform (DFT) Monitors

Instead of storing full space-time volumes, LuminaWave accumulates complex frequency phasors on-the-fly at each time step:

$$\hat{E}_z(x, y, \omega_k) = \sum_{n=0}^{N_{steps}} E_z^n(x, y) e^{-i \omega_k n \Delta t} \Delta t$$

$$\hat{H}_y(x, y, \omega_k) = \sum_{n=0}^{N_{steps}} H_y^{n+1/2}(x, y) e^{-i \omega_k (n + 1/2) \Delta t} \Delta t$$

Time-averaged Poynting vector energy flux through an aperture monitor line:

$$P(\omega_k) = \frac{1}{2} \text{Re} \left[ \int \hat{E}_z(y, \omega_k) \cdot \hat{H}_y^*(y, \omega_k) \, dy \right]$$

Spectral S-Parameters:
- Transmission: $S_{21}(\omega) = P_{\text{through}}(\omega) / P_{\text{source}}(\omega)$
- Reflection: $S_{11}(\omega) = P_{\text{refl}}(\omega) / P_{\text{source}}(\omega)$
- Insertion Loss: $\text{IL}_{\text{dB}} = -10 \log_{10}(S_{21})$

---

## 6. Sub-Pixel Unicode Braille Visualizer & Telemetry HUD

1. **Sub-Pixel 2x4 Braille Electric Field Plotter**:
   - Maps 2x4 grid cells to Unicode Braille patterns (`U+2800..U+28FF`).
   - TrueColor ANSI escapes for field polarity: Blue/Cyan for negative $E_z$, Red/Yellow for positive $E_z$, Gray for dielectric waveguide core boundaries.
2. **Poynting Energy Flux Vectors**:
   - Directional vector arrows ($\rightarrow, \leftarrow, \uparrow, \downarrow, \nearrow, \searrow, \dots$) showing local electromagnetic energy flow.
3. **Spectral Resonance Plotter**:
   - Braille frequency response curve showing resonance dips, bandwidth, and peak transmission.
4. **Interactive Optical Workbench HUD**:
   - Real-time step counter, optical time ($fs$), Courant number, maximum field amplitudes, total stored EM energy ($J$), and insertion loss ($dB$).

---

## 7. Package Layout

```
projects/27-luminawave/
├── README.md                      # Comprehensive documentation and guide
├── PLAN.md                        # Formal engineering specifications
├── TASK_QUEUE.md                  # Task status and progress tracking
├── CHECKPOINT_LAST.md             # Autonomous execution checkpoint
├── luminawave/                    # Core electromagnetic engine
│   ├── __init__.py                # Package exports
│   ├── grid.py                    # 2D Yee staggered grid, materials, CFL
│   ├── pml.py                     # Uniaxial PML absorbing boundary layers
│   ├── sources.py                 # Gaussian, CW, Ricker, TFSF source injectors
│   ├── fdtd.py                    # Maxwell 2D FDTD solver time-stepping core
│   ├── photonics.py               # Waveguides, bends, directional couplers, rings
│   ├── monitors.py                # DFT monitors, Poynting flux, S-parameters
│   └── visualizer.py              # Braille sub-pixel TrueColor field visualizer
├── tests/                         # Comprehensive unit & verification tests
│   ├── test_grid_and_cfl.py       # Yee staggering, materials, stability
│   ├── test_pml_absorption.py     # Boundary absorption (>60dB reflection loss)
│   ├── test_sources.py            # Frequency spectra, pulse profiles
│   ├── test_fdtd_physics.py       # Wave propagation speed, energy conservation
│   ├── test_photonics_devices.py  # TIR waveguiding, coupler power split, ring resonance
│   ├── test_monitors_and_dft.py   # Phasor accumulation, S-parameters
│   └── test_visualizer.py         # Braille dot mapping, ANSI color escapes
├── benchmarks/
│   └── bench_luminawave.py        # FDTD cell updates/sec, PML overhead, Braille FPS
└── examples/
    └── photonics_workbench.py     # Terminal interactive optical workbench demo
```
