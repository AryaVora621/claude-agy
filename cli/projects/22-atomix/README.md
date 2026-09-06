# Atomix: Molecular Dynamics, Statistical Mechanics & Biomolecular Simulation Engine

A high-performance, first-principles classical Molecular Dynamics (MD), statistical mechanics, and computational biophysics simulation engine implemented entirely in the pure Python standard library with zero external dependencies.

---

## Architectural Overview

Atomix provides a complete molecular modeling and statistical physics simulation pipeline:
1. **Force Fields & Interatomic Potentials**:
   - `LennardJonesPotential`: Truncated and shifted Lennard-Jones 12-6 potential with Lorentz-Berthelot mixing rules ($\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}$) and pairwise virial evaluation.
   - `CoulombPotential`: Shifted Coulombic electrostatics ($V(r) = f \frac{q_i q_j}{r}$) for partial atomic charges.
   - `HarmonicBondPotential`: Bond stretching ($V(r) = \frac{1}{2} k_b (r - r_0)^2$) with exact momentum conservation ($\mathbf{F}_i + \mathbf{F}_j = \mathbf{0}$).
   - `HarmonicAnglePotential`: 3-body valence angle bending ($V(\theta) = \frac{1}{2} k_\theta (\theta - \theta_0)^2$) with analytical gradient chain rules.
   - `PeriodicDihedralPotential`: 4-body torsional dihedrals ($V(\phi) = k_\phi [1 + \cos(n \phi - \delta)]$) with Blondel-Karplus torque projection.
2. **Spatial Partitioning & Pair Acceleration**:
   - 3D Periodic Boundary Conditions (PBC) and Minimum Image Convention.
   - `LinkedCellList`: 3D spatial cell partitioning with 13-direction forward half-neighborhood stencils, achieving strictly $O(N)$ pair evaluation complexity.
   - `VerletNeighborList`: Buffered neighbor skin list ($r_{\text{list}} = r_c + r_{\text{skin}}$) with displacement-triggered automatic rebuilds.
   - Automatic 1-2 (bonded) and 1-3 (angled) topological pair exclusion.
3. **Symplectic Integration & Holonomic Constraints**:
   - `VelocityVerletIntegrator`: Time-reversible, area-preserving symplectic phase-space integrator with $O(\Delta t^2)$ global energy conservation.
   - `SHAKEConstraintSolver`: Iterative constraint projection for rigid bonds ($\|\mathbf{r}_i - \mathbf{r}_j\|^2 - d_0^2 = 0$) with RATTLE velocity orthogonality ($\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} = 0$).
4. **Statistical Mechanics Ensembles & Thermostats**:
   - Microcanonical (NVE): Hamiltonian energy conservation.
   - Maxwell-Boltzmann sampling: Zero net center-of-mass momentum drift and exact kinetic temperature calibration.
   - Canonical (NVT): Berendsen weak-coupling thermostat, Andersen stochastic collision thermostat, and Nosé-Hoover extended phase space dynamical friction thermostat.
   - Isothermal-Isobaric (NPT): Berendsen isotropic barostat with volume rescaling.
5. **Thermodynamic & Structural Observables**:
   - Clausius Virial Equation of State for instantaneous pressure $P = \frac{2 E_k + W}{3 V}$.
   - Radial Distribution Function $g(r)$ with spherical shell volume normalization.
   - Mean Squared Displacement (MSD) with unwrapped periodic trajectories and Einstein self-diffusion coefficient $D = \lim_{t \to \infty} \frac{\text{MSD}(t)}{6t}$.
   - Radius of Gyration $R_g$ and Root Mean Square Deviation (RMSD) for biomolecules.
6. **Sub-Pixel 3D Terminal Visualization**:
   - 3D camera projection with orbital azimuth and elevation rotation.
   - Unicode Braille 2x4 sub-pixel canvas (`U+2800..U+28FF`) providing $140 \times 88$ effective screen resolution.
   - Depth-tested 24-bit TrueColor ANSI IUPAC CPK element color rendering (Carbon charcoal, Oxygen red, Nitrogen blue, Hydrogen white, Argon cyan).
   - Live telemetry dashboard HUD with thermodynamic state telemetry and 8-level Unicode sparkline energy trends.

---

## Mathematical Formulations

### 1. Interatomic Potentials & Force Derivations

#### Lennard-Jones (12-6) Potential
For pairwise separation vector $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ and scalar distance $r = \|\mathbf{r}_{ij}\|$:

$$V_{\text{LJ}}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

The pairwise interatomic force $\mathbf{F}_i$ acting on atom $i$ is:

$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} V_{\text{LJ}}(r) = -\frac{24\epsilon}{r^2} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right] \mathbf{r}_{ij}$$

Shifted potential to guarantee continuous energy at cutoff radius $r_c$:

$$V_{\text{shifted}}(r) = V_{\text{LJ}}(r) - V_{\text{LJ}}(r_c)$$

Pairwise scalar virial contribution for equation of state:

$$W_{ij} = \mathbf{r}_{ij} \cdot \mathbf{F}_j = 24\epsilon \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

---

### 2. Symplectic Velocity Verlet & SHAKE Constraints

The two-stage Velocity Verlet integrator advances positions and velocities:

$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2m} \mathbf{F}(t) \Delta t^2$$

$$\mathbf{v}\left(t + \frac{1}{2}\Delta t\right) = \mathbf{v}(t) + \frac{1}{2m} \mathbf{F}(t) \Delta t$$

Forces $\mathbf{F}(t + \Delta t)$ are evaluated at the updated coordinates $\mathbf{r}(t + \Delta t)$, followed by velocity completion:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}\left(t + \frac{1}{2}\Delta t\right) + \frac{1}{2m} \mathbf{F}(t + \Delta t) \Delta t$$

#### SHAKE Distance Constraint
For rigid bonds with target distance $d_0$, unconstrained coordinates $\mathbf{r}'$ are projected iteratively via Lagrange multipliers:

$$\Delta = \|\mathbf{r}_{ij}'\|^2 - d_0^2$$

$$g = \frac{\Delta}{2 \left(\frac{1}{m_i} + \frac{1}{m_j}\right) (\mathbf{r}_{ij}(t) \cdot \mathbf{r}_{ij}')}$$

$$\mathbf{r}_i \leftarrow \mathbf{r}_i + \frac{g}{m_i} \mathbf{r}_{ij}(t), \quad \mathbf{r}_j \leftarrow \mathbf{r}_j - \frac{g}{m_j} \mathbf{r}_{ij}(t)$$

---

### 3. Statistical Thermodynamics & Observables

#### Instantaneous Temperature
$$T = \frac{2 E_k}{N_{\text{df}} k_B}, \quad N_{\text{df}} = 3N - N_{\text{constraints}} - 3$$

#### Clausius Virial Pressure
$$P = \frac{2 E_k + \sum_{i < j} W_{ij}}{3 V}$$

#### Radial Distribution Function $g(r)$
$$g(r_k) = \frac{\text{count}(r_k)}{N \cdot \rho \cdot \Delta V_k \cdot N_{\text{samples}}}, \quad \Delta V_k = \frac{4}{3}\pi \left(r_{k+\frac{1}{2}}^3 - r_{k-\frac{1}{2}}^3\right)$$

#### Einstein Self-Diffusion Relation
$$D = \lim_{t \to \infty} \frac{1}{6t N} \sum_{i=1}^N \|\mathbf{r}_{\text{unwrapped}, i}(t) - \mathbf{r}_{\text{unwrapped}, i}(0)\|^2$$

---

## Performance Benchmarks

Benchmarked on Apple Silicon (single thread, pure Python 3 standard library):

| Operation | Metric | Performance |
|---|---|---|
| **Vector3D Operations** | Throughput | **5,733,926 ops/sec** |
| **Lennard-Jones Pair Force Evals** | Pair evaluations / sec | **1,666,821 evals/sec** (0.60 us/eval) |
| **Linked Cell List Partitioning** | 256 atoms, $O(N)$ pair search | **54.5 builds/sec** |
| **MD Simulation Rate (108 atoms)** | Pure integration + forces | **83.1 steps/sec** (8,977 atom-steps/s) |
| **MD Simulation Rate (256 atoms)** | Pure integration + forces | **23.7 steps/sec** (6,071 atom-steps/s) |
| **3D Braille Render Frame Rate** | Terminal visualization | **2,373.8 FPS** (0.42 ms/frame) |

---

## Interactive Biophysics Laboratory

Atomix includes an interactive terminal laboratory with three benchmark case studies:

```bash
python3 examples/biophysics_lab.py
```

### Case Studies Included:
1. **Argon Crystal Melting & Phase Transition**: FCC noble gas crystal ($T = 0.2$) heated through melting into disordered liquid ($T = 1.35$), tracking radial distribution function $g(r)$ coordination shells and Mean Squared Displacement diffusion.
2. **Polypeptide Helix Conformational Dynamics**: Coarse-grained protein alpha-helix under Nosé-Hoover NVT dynamics, monitoring structural breathing via Radius of Gyration ($R_g$) and RMSD.
3. **Explicit TIP3P Liquid Water with SHAKE**: 24 TIP3P water molecules (72 atoms) verifying rigid O-H bond length enforcement to within $3.6 \times 10^{-7}$ Angstroms.

To run the automated performance benchmark suite:
```bash
python3 benchmarks/bench_atomix.py
```

To run all unit tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## Python API Quickstart

```python
from atomix import (
    build_fcc_lattice,
    initialize_maxwell_boltzmann_velocities,
    BerendsenThermostat,
    MolecularDynamicsSimulation,
    render_molecular_system,
    render_telemetry_hud,
)

# 1. Generate 3D Face-Centered Cubic (FCC) crystal lattice (108 atoms)
atoms, box = build_fcc_lattice(n_cells=3, lattice_constant=1.65)

# 2. Sample Maxwell-Boltzmann velocities at target temperature T = 1.0
initialize_maxwell_boltzmann_velocities(atoms, target_temperature=1.0, remove_drift=True)

# 3. Configure Berendsen weak-coupling thermostat (Canonical NVT ensemble)
thermo = BerendsenThermostat(target_temperature=1.0, tau_t=0.1)

# 4. Initialize and run simulation
sim = MolecularDynamicsSimulation(
    atoms=atoms,
    box=box,
    timestep=0.002,
    thermostat=thermo,
    cutoff=2.5,
)

trajectory = sim.run(num_steps=100, sample_interval=20)

# 5. Render 3D sub-pixel Unicode Braille CPK visualization and HUD
print(render_molecular_system(atoms, box=box, azimuth_deg=35.0, elevation_deg=25.0))
print(render_telemetry_hud(
    trajectory[-1],
    num_atoms=len(atoms),
    num_bonds=0,
    energy_history=[s.total_energy for s in trajectory],
))
```
