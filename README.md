# AGY: Autonomous Creative Engineering & Simulation Laboratory

[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero_external-06B6D4.svg)](#zero-dependencies)
[![Tests Passing](https://img.shields.io/badge/tests-1003%20%2F%201003%20passing-10B981.svg)](#verification-and-test-suite)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#native-desktop-gui-applications)
[![Web Tech](https://img.shields.io/badge/web-HTML5%20%2F%20Canvas%20%2F%20WebGL-F59E0B.svg)](#standalone-interactive-web-studios)
[![Architecture](https://img.shields.io/badge/architecture-modular%20first--principles-8B5CF6.svg)](#architectural-overview)

AGY is a high-performance, zero-dependency creative engineering laboratory featuring standalone interactive visual simulations, physical modeling suites, native GUI engineering workstations, and first-principles computer science engines.

Every application in this repository is built entirely from foundational mathematics and computer science principles, with zero external npm packages, zero pip dependencies, and zero CDN assets.

---

## Architectural Overview

The repository is structured into three clean pillars:

```
agy/
├── websites/          # 14 Standalone Interactive Web Studios (HTML5 Canvas, WebGL, Web Audio)
│   ├── index.html     # Master Visual Showcase Portal with live animated canvas previews
│   ├── gravwave/      # 2D Numerical Relativity & Gravitational Wave Laser Interferometer
│   ├── waveoptics/    # 2D Physical Optics & FDTD Wavefield Electrodynamics
│   ├── plasmaflow/    # 2D Magnetohydrodynamics (MHD) & Magnetic Reconnection
│   ├── chromasplat/   # 3D Gaussian Splatting & Volume Radiance Studio
│   ├── neuralstudio/  # Autograd & Computational Graph Decision Studio
│   ├── opticalab/     # Optical Bench & Aspheric Lens Design Studio
│   ├── aerotunnel/    # 2D Lattice Boltzmann Wind Tunnel CFD Studio
│   ├── tokamak/       # Magnetic Confinement Fusion Reactor Simulator
│   ├── blackhole/     # Relativistic Spacetime Raymarcher & Accretion Disk
│   ├── biogenesis/    # Artificial Life & Particle Chemotaxis Neuro-Evolution
│   ├── voxelspace/    # Volumetric Terrain Raycaster & 6-DOF Flight Simulator
│   ├── quantum/       # Universal Quantum Circuit Simulator & 3D Bloch Sphere
│   ├── neuromorph/    # Neuromorphic SNN & DVS Silicon Retina Studio
│   └── synthwave/     # Polyphonic Synthesizer & 16-Step Drum Machine
│
├── programs/          # 9 Native Python GUI Desktop Applications (Tkinter, 0 pip dependencies)
│   ├── optiflow/      # Computational Fluid Dynamics & Aerodynamics Studio (16 tests)
│   ├── structura2d/   # Finite Element Analysis & Continuum Mechanics Studio (16 tests)
│   ├── spectrochem/   # Molecular Mechanics & Vibrational Spectroscopy Studio (16 tests)
│   ├── astroephemeris/# Astrodynamics & CR3BP Three-Body Mechanics Studio (16 tests)
│   ├── aeroacoustics/ # Computational Aeroacoustics & Sonic Boom Simulator (16 tests)
│   ├── retrocad/      # Mechanical CAD Modeler & Solid Modeling Kernel (24 tests)
│   ├── pycircuit/     # SPICE Analog Circuit Simulator & Schematic Designer (14 tests)
│   ├── signalscope/   # DSP Virtual Oscilloscope & Synthesizer Studio (14 tests)
│   └── gravitas/      # N-Body Orbital Mechanics Desktop Studio (10 tests)
│
├── cli/               # 33 First-Principles Computer Science Engines (861 tests)
│   ├── showcase.py    # Unified interactive terminal launcher & test runner
│   └── projects/      # Autograd, LSM-Tree, BPE, Raft, Microkernel, Path Tracer, TCP/IP, etc.
│
├── projects.md        # Comprehensive technical specification & mathematics reference
├── TASK_QUEUE.md      # Structured backlog & verification milestones
└── CHECKPOINT_LAST.md # State tracking & engineering logs
```

---

## Standalone Interactive Web Studios

Each web studio is a complete, self-contained application running directly in any modern browser by opening `websites/<app>/index.html` or through the master portal at `websites/index.html`. No build steps, web servers, or internet connections are required.

| Studio | Category | Mathematical & Technical Highlights |
| :--- | :--- | :--- |
| **GravWave Studio** | Numerical Relativity | Peters radiation reaction $da/dt \propto a^{-3}$, quadrupole metric strain $h_{ij}(t - r/c)$, Michelson laser interferometer optical dark port fringe shift, Web Audio chirp sonification. |
| **WaveOptics Studio** | Physical Optics & FDTD | Scalar wave equation $\partial^2\psi/\partial t^2 = c^2\nabla^2\psi - \gamma\dot{\psi}$, absorbing boundary layers, dielectric refraction, Fraunhofer double-slit diffraction, Sinc$^2$ fringe validation. |
| **PlasmaFlow Studio** | Magnetohydrodynamics | Solenoidal magnetic vector potential $A_z$, Lorentz force $\mathbf{J} \times \mathbf{B}$ fluid momentum coupling, Sweet-Parker magnetic reconnection, Orszag-Tang vortex turbulence. |
| **ChromaSplat Studio** | Neural Radiance & 3D Vision | 3D Gaussian Splatting, quaternion $SO(3)$ rotations, spherical harmonics (degrees 0 to 3), 2D elliptical weighted average (EWA) splat projection. |
| **NeuralStudio** | Deep Learning & Autograd | Reverse-mode automatic differentiation DAG, dynamic decision boundaries, MLP classification, real-time loss backpropagation. |
| **OpticaLab** | Geometrical Optics | Snell refraction, Fresnel reflection, chromatic dispersion, aspheric spherical aberration analysis, multi-element optical bench. |
| **AeroTunnel** | Fluid Dynamics | D2Q9 Lattice Boltzmann Method (LBM) with BGK collision operator, bounce-back solid boundary conditions, real-time obstacle insertion. |
| **TokamakCockpit** | Fusion Plasma Physics | Grad-Shafranov equilibrium solver, magnetic flux surfaces $\psi$, safety factor $q$, magnetohydrodynamic plasma stability margins. |
| **BlackHole Studio** | General Relativity | Null geodesic raymarching in curved spacetime, gravitational lensing, photon sphere ($r = 3GM/c^2$), relativistic Doppler beaming, Keplerian accretion disk. |
| **BioGenesis Studio** | Artificial Life | Non-reciprocal particle chemotaxis, action potential neural controllers, self-organizing morphological motility, sensor networks. |
| **VoxelSpace 3D** | Volumetric Rendering | Heightmap and colormap raycasting algorithm, 6-DOF camera kinematics, fog attenuation, real-time procedural terrain synthesis. |
| **QuantumLab Studio** | Quantum Computing | Statevector linear algebra ($2^N$ amplitudes), quantum gate array, partial trace density matrices, 1024-shot Monte Carlo projective measurement, 3D Bloch sphere. |
| **NeuroMorph Studio** | Neuromorphic Computing | Leaky Integrate-and-Fire (LIF) and Izhikevich spiking neural dynamics, STDP synaptic plasticity, 3D cortical column, silicon retina event camera. |
| **SynthWave Studio** | Audio DSP & Synthesis | Dual antialiased oscillators, 24dB resonant ladder lowpass filter, 808 drum synthesis, tape delay, spatial convolution reverb, phosphor oscilloscope. |

---

## Native Desktop GUI Applications

All desktop programs run using standard library Python `tkinter` without third-party graphical libraries or external computational dependencies.

Launch any suite directly from terminal:

```bash
# 1. OptiFlow 2D (Computational Fluid Dynamics & Aerodynamics Studio)
python3 programs/optiflow/optiflow.py

# 2. Structura 2D (Finite Element Analysis & Continuum Mechanics Studio)
python3 programs/structura2d/structura2d.py

# 3. SpectroChem 3D (Molecular Mechanics & Vibrational Spectroscopy Studio)
python3 programs/spectrochem/spectrochem.py

# 4. AstroEphemeris 3D (Astrodynamics & Three-Body Mechanics Studio)
python3 programs/astroephemeris/astroephemeris.py

# 5. AeroAcoustics Studio (Computational Aeroacoustics & Sonic Boom Simulator)
python3 programs/aeroacoustics/aeroacoustics.py

# 6. RetroCAD 3D Studio (Mechanical CAD Modeler & Solid Modeling Kernel)
python3 programs/retrocad/retrocad.py

# 7. PyCircuit (SPICE Analog Circuit Simulator & Schematic Designer)
python3 programs/pycircuit/pycircuit.py

# 8. SignalScope (DSP Virtual Oscilloscope & Synthesizer Studio)
python3 programs/signalscope/signalscope.py

# 9. Gravitas 3D (N-Body Orbital Mechanics Desktop Studio)
python3 programs/gravitas/gravitas.py
```

### Desktop Application Highlights

- **OptiFlow 2D**:
  - Incompressible Navier-Stokes finite difference solver using coupled vorticity-streamfunction ($\omega - \psi$) formulation.
  - Incompressibility $\nabla \cdot \mathbf{u} = 0$ is satisfied identically to machine precision ($< 10^{-14}$).
  - Successive over-relaxation (SOR) Poisson pressure recovery and Woods solid wall boundary conditions.
  - Parametric NACA 4-digit airfoil morphology (NACA 0012, NACA 2412, NACA 4412) with real-time Angle of Attack (-18 deg to +22 deg).
  - Dynamic surface contour integration of lift $C_L$, drag $C_D$, moment $C_M$, and efficiency $L/D$.
  - Virtual Pitot probe crosshair and RK2 smoke streakline particles.

- **Structura 2D**:
  - Multi-element finite element analysis kernel supporting 1D Pin-Jointed Truss, 3-Node Constant Strain Triangle (CST), and 4-Node Isoparametric Quadrilateral (Quad4).
  - $2 \times 2$ Gauss-Legendre numerical quadrature for Quad4 element stiffness integration.
  - Global stiffness matrix assembly, Dirichlet boundary condition reduction, and Gaussian elimination with partial pivoting.
  - Full stress recovery: $\sigma_{xx}, \sigma_{yy}, \tau_{xy}$, principal stresses $\sigma_{1, 2}$, and Von Mises equivalent failure yield stress $\sigma_{vM}$.
  - Real-time contour color heatmaps, continuous deformation amplification slider (1x to 1000x), and Rayleigh quotient modal vibration animation.

- **SpectroChem 3D**:
  - Molecular mechanics energy minimizer utilizing steepest descent and Armijo line-search conjugate gradient optimization.
  - Velocity Verlet integrator with Berendsen thermostat for NVT molecular dynamics.
  - Mass-weighted Hessian matrix assembly with Jacobi eigensolver for vibrational normal modes.
  - Simulated continuous FTIR infrared absorption spectrum with Lorentz line-broadening.

- **AstroEphemeris 3D**:
  - Halley-accelerated Kepler root solver for orbital mechanics.
  - Post-Newtonian 1PN Schwarzschild relativistic rosette precession and J2 planetary oblateness perturbations.
  - Circular Restricted Three-Body Problem (CR3BP) with Jacobi integral conservation and Lagrange libration points (L1 through L5).
  - Lambert boundary value solver for interplanetary Hohmann transfers.

- **AeroAcoustics Studio**:
  - Ffowcs Williams-Hawkings (FW-H) acoustic wave equation solver.
  - Doppler frequency shifts, supersonic Mach cone formation ($M > 1.0$), and Whitham non-linear shock wave profiles.
  - FFT spectral acoustic analyzer and virtual directional microphone arrays.

- **RetroCAD 3D Studio**:
  - Boundary Representation (B-Rep) topological solid modeling kernel with Euler-Poincare topology verification ($V - E + F = 2$).
  - Constructive Solid Geometry (CSG) Boolean operations (Union, Difference, Intersection).
  - Extrusion, revolution, sweep, fillet, and chamfer operations.
  - Parametric 3D wireframe and flat-shaded rendering with STL, OBJ, DXF, and SVG vector export.

- **PyCircuit**:
  - Modified Nodal Analysis (MNA) matrix solver with Newton-Raphson non-linear iterations.
  - DC operating point, AC frequency response, and transient simulation with backward Euler integration.
  - Support for resistors, capacitors, inductors, diodes, BJTs, MOSFETs, and operational amplifiers.

- **SignalScope**:
  - Dual-channel digital storage oscilloscope with real-time hardware triggering (rising/falling edge).
  - Cooley-Tukey Radix-2 Fast Fourier Transform (FFT) with Hanning, Hamming, and Blackman window functions.
  - Robert Bristow-Johnson (RBJ) biquad audio filter DSP engine.

- **Gravitas 3D**:
  - Symplectic 4th-order Yoshida numerical integrator preserving energy and angular momentum.
  - Collision detection and inelastic merging dynamics for multi-body planetary systems.
  - Real-time 3D orbit trajectory ribbons and energy conservation telemetry.

---

## First-Principles Computer Science Engines

The `cli/` directory houses 33 computer science engines covering fundamental algorithms and systems, completely self-contained in Python standard library:

1. **NanoTensor**: Autograd engine with RoPE, RMSNorm, SwiGLU, and KV-Cache.
2. **BytePairTokenizer**: BPE tokenizer with subword segmentation.
3. **ChronoDB**: LSM-tree key-value store with WAL, SSTables, Bloom filters, and HNSW vector index.
4. **AetherVM**: Register-based bytecode VM with SSA optimization pipeline.
5. **SwarmRaft**: Distributed consensus state machine with network partition simulator.
6. **NexusOS**: Capability-based microkernel with 2-level paging, COW, and Unix pipes.
7. **PhotonPBR**: Monte Carlo path tracer with SAH BVH and Moller-Trumbore intersection.
8. **HydraNet**: User-space TCP/IP network stack with RFC 793 state machine and Reno congestion control.
9. **SynapseDB**: Columnar analytics engine with vectorized execution, bit-packing, and SQL planner.
10. **QuantaLab**: Universal quantum circuit simulator with $O(2^N)$ statevector transforms.
11. **ZetaProof**: Zero-knowledge SNARK proof engine with R1CS compiler and Groth16 protocol.
12. **WasmCore**: WebAssembly MVP virtual machine with 64KB paged linear memory and LEB128 codec.
13. **NovaPhysics**: 2D rigid body physics engine with GJK/EPA collision solver and PBD cloth.
14. **HelixGit**: Git-compatible VCS with DIRC v2 binary index, packfile deltas, and Myers diff.
15. **GeoPrism**: Spatial index with Beckmann R*-Tree, H3 DGGS, Bowyer-Watson Delaunay, and Voronoi.
16. **AuraDSP**: Audio DSP engine with Cooley-Tukey FFT, RBJ biquad filters, and polyphonic synth.
17. **VeloSLAM**: Autonomous robotics with EKF-SLAM, occupancy grids, Dubins curves, and Hybrid A*.
18. **ApexMatch**: L3 limit order book with FIFO matching, Iceberg orders, and ITCH/OUCH codec.
19. **NucleoCore**: Computational genomics with FM-Index, Gotoh affine gap, and De Bruijn graph.
20. **OrbitMech**: Astrodynamics engine with Danby Kepler solver, Lambert targeting, and J2 perturbations.
21. **AeroFlow**: CFD engine with D2Q9 LBM and Chorin Navier-Stokes projection.
22. **Structura**: FEA engine with Truss/Beam/CST/Quad4 elements and sparse PCG solver.
23. **Atomix**: Molecular dynamics with LJ 12-6, Coulomb, Velocity Verlet, and SHAKE constraints.
24. **Solida**: 3D B-Rep CAD kernel with Cox-de Boor NURBS and CSG Booleans.
25. **NeuroSynapse**: Spiking neural network with LIF/Izhikevich/Hodgkin-Huxley and STDP plasticity.
26. **Avionix**: 6-DOF flight dynamics with differential flatness and SE(3) geometric control.
27. **SiliconRISC**: Cycle-accurate RV64GC processor emulator with 5-stage pipeline and SV39 MMU.
28. **LuminaWave**: 2D Maxwell FDTD computational nanophotonics engine with Yee lattice and PML.
29. **LatticeGuard**: NIST FIPS 203 ML-KEM post-quantum cryptography engine.
30. **LogicCraft**: EDA logic synthesis and static timing analysis (STA) engine with ROBDD and AIG.
31. **Relativitas**: General relativity geodesic integrator with Kerr spacetime and accretion disk.
32. **StellarFusion**: Magnetohydrodynamic tokamak plasma equilibrium with Grad-Shafranov solver.
33. **ThermoProp**: Compressible gas dynamics and supersonic De Laval nozzle solver.

Launch the unified CLI engine showcase:
```bash
python3 cli/showcase.py
```

---

## Verification and Test Suite

All 1,003 unit tests pass deterministically across all environments with 100% pass rate:

```bash
# Run all native desktop GUI program test suites (142 tests)
python3 -m unittest discover -s programs -p "test_*.py"

# Run individual program test suites
python3 programs/optiflow/test_optiflow.py           # 16 tests pass
python3 programs/structura2d/test_structura2d.py     # 16 tests pass
python3 programs/spectrochem/test_spectrochem.py     # 16 tests pass
python3 programs/astroephemeris/test_astroephemeris.py # 16 tests pass
python3 programs/aeroacoustics/test_aeroacoustics.py # 16 tests pass
python3 programs/retrocad/test_retrocad.py           # 24 tests pass
python3 programs/pycircuit/test_pycircuit.py         # 14 tests pass
python3 programs/signalscope/test_signalscope.py     # 14 tests pass
python3 programs/gravitas/test_gravitas.py           # 10 tests pass

# Run all CLI engines test suite (861 tests)
python3 cli/showcase.py --test-all
```

Test Results Breakdown:
- **CLI Systems**: 861 / 861 tests passing
- **Desktop Programs**: 142 / 142 tests passing
- **Total Suite**: 1,003 / 1,003 tests passing (100%)

---

## Zero Dependencies Policy

AGY enforces a strict zero-dependency philosophy:
- **Web Applications**: Pure HTML5 Canvas, WebGL, SVG, and Web Audio API. Zero external JavaScript frameworks, zero CSS libraries, zero remote CDN scripts. Works completely offline.
- **Desktop Applications**: Built exclusively using standard Python 3 `tkinter`, `math`, `cmath`, and `struct`. Zero pip dependencies.
- **CLI Systems**: Pure Python standard library implementation of all data structures, linear algebra, numerical integrators, and binary protocols.

---

## License

MIT License. Open for educational, scientific, and engineering exploration.
