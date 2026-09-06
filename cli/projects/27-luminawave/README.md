# LuminaWave: 2D Maxwell FDTD Computational Nanophotonics Engine

LuminaWave is a pure Python 3.10+ nanophotonics simulation engine that implements 2D Transverse Magnetic ($TM_z$) Finite-Difference Time-Domain (FDTD) electromagnetic wave propagation with zero external dependencies.

It features Kane Yee staggered spatial lattices, Berenger split-field Perfectly Matched Layer (PML) absorbing boundaries, optical waveguide mode injectors, on-the-fly Discrete Fourier Transform (DFT) Poynting flux monitors, S-parameter spectral extraction, and real-time 2x4 sub-pixel Unicode Braille terminal visualization in 24-bit TrueColor ANSI.

---

## Key Features

1. **Pure Python Standard Library**: 100% zero-dependency architecture. Requires only standard library Python 3.10+.
2. **Mathematically Rigorous 2D $TM_z$ FDTD Engine**:
   - Electric field $E_z(x, y, t)$, magnetic fields $H_x(x, y, t)$ and $H_y(x, y, t)$.
   - Second-order accurate leapfrog time integration ($H^{n+1/2}$, $E^{n+1}$).
   - Strict Courant-Friedrichs-Lewy (CFL) stability verification ($\Delta t \le \frac{1}{c_0 \sqrt{1/\Delta x^2 + 1/\Delta y^2}}$).
3. **Berenger Split-Field PML Absorbing Boundaries**:
   - Field decomposition $E_z = E_{zx} + E_{zy}$ inside outer boundary layers.
   - Polynomial conductivity grading $\sigma(d) = \sigma_{max} (d/d_{pml})^m$ with order $m=3$.
   - Magnetic impedance matching $\sigma^*(d) = \sigma(d) \frac{\mu_0}{\epsilon_0}$ yielding theoretical zero reflection ($<-60\text{ dB}$ absorption).
4. **Broadband Optical Excitation Sources**:
   - Gaussian pulses, Continuous Wave (CW) with smooth cosine startup ramp, Modulated Gaussian wavepackets, and Ricker wavelets.
   - Soft current-density injection ($J_z$) and hard clamped injection.
   - Waveguide fundamental transverse spatial mode injection ($H_{10}$ cosine profile).
5. **Silicon Photonic Integrated Circuit (PIC) Factory**:
   - High-index-contrast Silicon-on-Insulator (SOI) platform: Silicon core ($n=3.48$) and Silicon Dioxide / Air cladding ($n=1.44 / 1.0$) at telecommunications wavelength $\lambda_0 = 1.55 \ \mu\text{m}$.
   - Strip waveguides, 90-degree low-loss circular bends, 2x2 evanescent directional couplers, micro-ring resonators, Mach-Zehnder interferometers (MZI), and 2D photonic crystal line-defect waveguides.
6. **On-the-Fly DFT Poynting Flux Monitors & S-Parameters**:
   - Continuous complex phasor accumulation $\hat{E}_z(f) = \sum E_z e^{-i 2\pi f t} \Delta t$ without memory-intensive time-domain dumps.
   - Line integral of time-averaged Poynting flux $P(f) = \frac{1}{2} \text{Re} \int \hat{E}_z \hat{H}^* dl$.
   - Automated S-parameter extraction: Transmission $S_{21}(f)$, Insertion Loss (dB), Resonant cavity Quality Factor ($Q = f_0 / \Delta f_{FWHM}$), and Extinction Ratio.
7. **Sub-Pixel Braille Terminal Visualizer & Optical HUD**:
   - 2x4 dot sub-pixel mapping using Unicode Braille (`U+2800..U+28FF`).
   - 24-bit TrueColor ANSI gradient rendering (+Ez red/yellow, -Ez blue/cyan, dielectric core gray).
   - Frequency transmission spectrum sparkline generator and real-time telemetry HUD.

---

## Theoretical Derivations

### 1. 2D Maxwell Curl Equations ($TM_z$ Mode)

In two spatial dimensions where $\partial / \partial z = 0$ and the electric field is polarized along $z$:

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu_r \mu_0} \frac{\partial E_z}{\partial y}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu_r \mu_0} \frac{\partial E_z}{\partial x}$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon_r \epsilon_0} \left( \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} - \sigma E_z \right)$$

### 2. Yee Lattice Discretization

Fields are spatially and temporally staggered according to Kane Yee (1966):
- $E_z$ is evaluated at integer space-time steps $(i, j, n)$.
- $H_x$ is evaluated at $(i, j + 1/2, n + 1/2)$.
- $H_y$ is evaluated at $(i + 1/2, j, n + 1/2)$.

Finite difference equations:

$$H_x^{n+1/2}(i, j+1/2) = H_x^{n-1/2}(i, j+1/2) - \frac{\Delta t}{\mu_0 \mu_r \Delta y} \left( E_z^n(i, j+1) - E_z^n(i, j) \right)$$

$$H_y^{n+1/2}(i+1/2, j) = H_y^{n-1/2}(i+1/2, j) + \frac{\Delta t}{\mu_0 \mu_r \Delta x} \left( E_z^n(i+1, j) - E_z^n(i, j) \right)$$

$$E_z^{n+1}(i, j) = C_a E_z^n(i, j) + C_b \left[ \frac{H_y^{n+1/2}(i+1/2, j) - H_y^{n+1/2}(i-1/2, j)}{\Delta x} - \frac{H_x^{n+1/2}(i, j+1/2) - H_x^{n+1/2}(i, j-1/2)}{\Delta y} \right]$$

where:

$$C_a = \frac{1 - \frac{\sigma \Delta t}{2 \epsilon_0 \epsilon_r}}{1 + \frac{\sigma \Delta t}{2 \epsilon_0 \epsilon_r}}, \qquad C_b = \frac{\frac{\Delta t}{\epsilon_0 \epsilon_r}}{1 + \frac{\sigma \Delta t}{2 \epsilon_0 \epsilon_r}}$$

### 3. Berenger Split-Field PML Formulation

In the absorbing boundary layer, $E_z = E_{zx} + E_{zy}$, where $E_{zx}$ couples to $\partial H_y / \partial x$ with conductivity $\sigma_x$, and $E_{zy}$ couples to $\partial H_x / \partial y$ with conductivity $\sigma_y$:

$$\frac{\partial H_{xy}}{\partial t} + \frac{\sigma_y^*}{\mu} H_{xy} = -\frac{1}{\mu} \frac{\partial (E_{zx} + E_{zy})}{\partial y}$$

$$\frac{\partial H_{yx}}{\partial t} + \frac{\sigma_x^*}{\mu} H_{yx} = \frac{1}{\mu} \frac{\partial (E_{zx} + E_{zy})}{\partial x}$$

Conductivities follow polynomial grading:

$$\sigma_x(x) = \sigma_{max} \left( \frac{x}{d_{pml}} \right)^m, \qquad \sigma_{max} = -\frac{(m+1) \ln(R_0)}{2 \eta_0 d_{pml}}$$

with $m=3$, target reflection $R_0 = 10^{-6}$, and $\eta_0 = \sqrt{\mu_0 / \epsilon_0} \approx 376.73 \ \Omega$.

---

## Performance Benchmarks

Measured on standard Python 3.13 (Apple M-series):

| Benchmark Component | Metric | Performance |
|:---|:---|:---|
| 2D Yee Leapfrog Engine (50x50 Grid) | Cell Update Throughput | **5.23 MegaCells/sec** |
| 2D Yee Leapfrog Engine (100x100 Grid) | Cell Update Throughput | **5.27 MegaCells/sec** |
| 2D Yee Leapfrog Engine (150x150 Grid) | Cell Update Throughput | **5.23 MegaCells/sec** |
| Berenger Split-Field PML (10 Cells) | Boundary Overhead | **+98.9%** (full split-field absorption) |
| On-The-Fly DFT Phasor Monitor | Update Rate | **0.95 MUpdates/sec** (40 frequency bins) |
| Silicon Micro-Ring Resonator PIC | Effective Throughput | **2.65 MegaCells/sec** |
| Unicode 2x4 Sub-Pixel Braille Canvas | Frame Rate | **547.3 FPS** (24-bit TrueColor ANSI) |

---

## Directory Layout

```
27-luminawave/
├── luminawave/
│   ├── __init__.py           # Package exports
│   ├── grid.py               # 2D Yee staggered grid, materials, CFL verification
│   ├── pml.py                # Berenger split-field PML absorbing boundary conditions
│   ├── sources.py            # Gaussian, CW, Modulated, Ricker optical sources
│   ├── fdtd.py               # Maxwell 2D FDTD leapfrog time-stepping engine
│   ├── photonics.py          # Silicon photonic circuit builder (waveguides, rings, couplers)
│   ├── monitors.py           # On-the-fly DFT flux monitors and S-parameter analyzer
│   └── visualizer.py         # Sub-pixel Unicode Braille renderer and telemetry HUD
├── tests/
│   ├── test_grid_and_cfl.py
│   ├── test_pml_absorption.py
│   ├── test_sources.py
│   ├── test_fdtd_physics.py
│   ├── test_photonics_devices.py
│   ├── test_monitors_and_dft.py
│   └── test_visualizer.py
├── benchmarks/
│   └── bench_luminawave.py   # Performance microbenchmarks
├── examples/
│   └── photonics_workbench.py# Interactive terminal photonic workbench
├── PLAN.md                   # Mathematical derivations and architectural plan
├── TASK_QUEUE.md             # Project task and milestone tracking
├── CHECKPOINT_LAST.md        # Session checkpoint state
└── README.md                 # Complete documentation
```

---

## Quickstart & Usage

### 1. Run Interactive Terminal Photonics Workbench

Simulate optical propagation with real-time TrueColor sub-pixel Braille visualization:

```bash
# Silicon Micro-Ring Resonator
python3 examples/photonics_workbench.py --device ring --steps 240 --fps 30

# 2x2 Directional Coupler
python3 examples/photonics_workbench.py --device coupler --steps 240 --fps 30

# 90-Degree Low-Loss Waveguide Bend
python3 examples/photonics_workbench.py --device bend --steps 220 --fps 30

# Photonic Bandgap Crystal Waveguide
python3 examples/photonics_workbench.py --device pbg --steps 220 --fps 30
```

### 2. Fast Headless Simulation

```bash
python3 examples/photonics_workbench.py --device ring --steps 200 --no-anim
```

### 3. Run Test Suite

Verify all 30 unit tests:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py"
```

### 4. Run Benchmarks

```bash
PYTHONPATH=. python3 benchmarks/bench_luminawave.py
```

---

## Python API Example

```python
from luminawave.grid import Grid2D
from luminawave.pml import PMLBoundary
from luminawave.sources import OpticalSource, WaveguideModeSource, SourceWaveform
from luminawave.fdtd import FDTDSimulator
from luminawave.photonics import PhotonicCircuitBuilder
from luminawave.monitors import LineDFTMonitor, SParameterAnalyzer

# 1. Initialize 2D Yee grid (dx = 50 nm)
grid = Grid2D(nx=80, ny=80, dx=50e-9, dy=50e-9, courant_factor=0.65)

# 2. Build Silicon micro-ring resonator (radius = 900 nm)
in_port, through_port = PhotonicCircuitBuilder.build_ring_resonator(
    grid, cx=40, cy=50, radius=18, ring_width=5, bus_y=25, bus_width=5
)

# 3. Create FDTD Simulator with 8-cell Berenger PML boundaries
sim = FDTDSimulator(grid, pml_thickness=8, enable_pml=True)

# 4. Inject 1550 nm optical mode
src_opt = OpticalSource(waveform=SourceWaveform.MODULATED_GAUSSIAN, wavelength=1.55e-6)
sim.add_source(WaveguideModeSource(src_opt, x=in_port.x, y_start=22, y_end=28))

# 5. Monitor transmission flux via on-the-fly DFT
frequencies = [1.934e14 * (1.0 + 0.002 * k) for k in range(-10, 11)]
through_mon = LineDFTMonitor("Through", coord=through_port.x - 4, start=20, end=30, frequencies=frequencies)
sim.add_monitor(through_mon)

# 6. Execute time steps
sim.run(200)

# 7. Compute Poynting flux
flux = through_mon.compute_flux(grid)
print("Poynting Flux (W/m):", flux)
```

---

## License

MIT License. Developed as part of the autonomous high-performance computational systems suite.
