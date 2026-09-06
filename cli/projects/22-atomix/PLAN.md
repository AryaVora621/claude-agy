# PLAN: Project 22 - Atomix (Molecular Dynamics, Statistical Mechanics & Biomolecular Simulation Engine)

## Vision & Scope
A high-performance, first-principles classical Molecular Dynamics (MD), statistical mechanics, and computational biophysics simulation engine implemented entirely in the pure Python standard library with zero external dependencies.

Atomix provides a complete scientific simulation pipeline:
1. Microscopic physical modeling (Lennard-Jones 12-6, Coulomb electrostatics with reaction field, harmonic bonds, harmonic angles, periodic dihedrals).
2. Spatial partitioning via Linked Cell Lists ($O(N)$ evaluation) and Verlet neighbor lists with periodic boundary conditions (PBC) and minimum image convention.
3. Symplectic numerical integrators (Velocity Verlet, Leapfrog) with SHAKE holonomic bond constraints.
4. Statistical thermodynamic ensembles: Microcanonical (NVE), Canonical (NVT via Berendsen and Nosé-Hoover extended Hamiltonian), and Isothermal-Isobaric (NPT via Berendsen barostat).
5. Structural and transport observables: Radial Distribution Function $g(r)$, Mean Squared Displacement (MSD), self-diffusion coefficient $D$, Radius of Gyration $R_g$, and Virial Pressure tensor.
6. Sub-pixel 3D Unicode Braille terminal visualizer with 3D camera projection, CPK coloring, and live thermodynamic telemetry HUD.

---

## Mathematical Architecture

### 1. Non-Bonded Potentials & Interatomic Forces
For pairwise distance $r = \|\mathbf{r}_{ij}\| = \|\mathbf{r}_j - \mathbf{r}_i\|$:

#### Lennard-Jones (12-6) Potential
$$V_{\text{LJ}}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

The pairwise interatomic force $\mathbf{F}_{ij}$ acting on atom $i$ from atom $j$ is:
$$\mathbf{F}_{ij} = -\nabla_{\mathbf{r}_i} V_{\text{LJ}}(r) = \frac{24\epsilon}{r^2} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right] \mathbf{r}_{ij}$$

Shifted-force cutoff at $r_c$ ensures continuous energy and force transitions:
$$V_{\text{shifted}}(r) = V_{\text{LJ}}(r) - V_{\text{LJ}}(r_c) - \left(\frac{dV_{\text{LJ}}}{dr}\right)_{r_c} (r - r_c)$$

Lorentz-Berthelot mixing rules for heterogeneous species:
$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \quad \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}$$

#### Coulomb Electrostatics
$$V_{\text{Coulomb}}(r) = \frac{q_i q_j}{4\pi \varepsilon_0 r}, \quad \mathbf{F}_{\text{Coulomb}}(\mathbf{r}_{ij}) = \frac{q_i q_j}{4\pi \varepsilon_0 r^3} \mathbf{r}_{ij}$$

---

### 2. Bonded Intramolecular Potentials

#### Harmonic Bond Stretching
$$V_{\text{bond}}(r_{ij}) = \frac{1}{2} k_b (r_{ij} - r_0)^2, \quad \mathbf{F}_i = k_b (r_{ij} - r_0) \frac{\mathbf{r}_{ij}}{r_{ij}}, \quad \mathbf{F}_j = -\mathbf{F}_i$$

#### Harmonic Angle Bending
For atoms $i - j - k$ with angle $\theta$ centered at vertex $j$:
$$\cos\theta = \frac{\mathbf{r}_{ji} \cdot \mathbf{r}_{jk}}{\|\mathbf{r}_{ji}\| \|\mathbf{r}_{jk}\|}$$
$$V_{\text{angle}}(\theta) = \frac{1}{2} k_\theta (\theta - \theta_0)^2$$
Atomic forces are computed via the analytical chain rule $\nabla_{\mathbf{r}} \theta = \frac{-1}{\sin\theta} \nabla_{\mathbf{r}} \cos\theta$.

#### Periodic Dihedral Torsions
For 4 consecutive bonded atoms $i - j - k - l$ with dihedral angle $\phi$:
$$V_{\text{dihedral}}(\phi) = \sum_{n=1}^4 k_n [1 + \cos(n \phi - \delta_n)]$$

---

### 3. Spatial Acceleration: Linked Cell Lists ($O(N)$)
A naive all-pairs loop scales as $O(N^2)$. Atomix implements 3D Linked Cell Lists:
- Simulation box of dimensions $(L_x, L_y, L_z)$ is discretized into cubic cells with edge length $d \ge r_c$.
- Each particle is mapped to cell $(c_x, c_y, c_z)$ in $O(1)$ time.
- Pair interactions are restricted to the particle's own cell and the 13 neighboring forward cells (accounting for Newton's third law $\mathbf{F}_{ji} = -\mathbf{F}_{ij}$), reducing complexity to $O(N)$.
- Minimum image convention handles periodic boundary conditions across all faces.

---

### 4. Symplectic Integrators & Constraint Dynamics

#### Velocity Verlet Integrator
1. Update coordinates:
   $$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t) \Delta t + \frac{1}{2m} \mathbf{F}(t) \Delta t^2$$
2. Half-step velocity update:
   $$\mathbf{v}(t + \frac{1}{2}\Delta t) = \mathbf{v}(t) + \frac{1}{2m} \mathbf{F}(t) \Delta t$$
3. Recompute forces $\mathbf{F}(t + \Delta t)$ at new positions.
4. Complete velocity step:
   $$\mathbf{v}(t + \Delta t) = \mathbf{v}(t + \frac{1}{2}\Delta t) + \frac{1}{2m} \mathbf{F}(t + \Delta t) \Delta t$$

#### SHAKE Holonomic Constraints
Constrains rigid bond lengths $\|\mathbf{r}_i - \mathbf{r}_j\|^2 - d_{ij}^2 = 0$ iteratively through Lagrange multipliers, allowing large simulation timesteps (1-2 fs) for rigid water models (e.g. TIP3P/SPC).

---

### 5. Statistical Thermodynamic Ensembles & Thermostats

#### Instantaneous Temperature
$$T = \frac{2 E_k}{N_{\text{df}} k_B}, \quad N_{\text{df}} = 3N - N_{\text{constraints}} - 3$$

#### Berendsen Weak-Coupling Thermostat
$$\lambda = \sqrt{1 + \frac{\Delta t}{\tau_T} \left( \frac{T_0}{T} - 1 \right)}, \quad \mathbf{v}_i \leftarrow \lambda \mathbf{v}_i$$

#### Nosé-Hoover Extended Hamiltonian Thermostat
Introduces dynamic thermodynamic heat bath friction $\xi$:
$$\dot{\mathbf{r}}_i = \mathbf{v}_i$$
$$\dot{\mathbf{v}}_i = \frac{\mathbf{F}_i}{m_i} - \xi \mathbf{v}_i$$
$$\dot{\xi} = \frac{1}{Q} \left( \sum_{i=1}^N m_i v_i^2 - N_{\text{df}} k_B T_0 \right)$$
Conserves the extended Hamiltonian invariant $H' = E_k + V + \frac{1}{2} Q \xi^2 + N_{\text{df}} k_B T_0 s$.

---

### 6. Observables & Structural Characterization

#### Virial Equation of State & Pressure
$$P = \frac{N k_B T}{V} + \frac{1}{3V} \sum_{i < j} \mathbf{r}_{ij} \cdot \mathbf{F}_{ij}$$

#### Radial Distribution Function $g(r)$
Measures local structural order and coordination shells:
$$g(r) = \frac{V}{N^2 4\pi r^2 \Delta r} \sum_{i} \sum_{j \ne i} \delta(r - r_{ij})$$

#### Mean Squared Displacement (MSD) & Diffusion
$$\text{MSD}(t) = \frac{1}{N} \sum_{i=1}^N \|\mathbf{r}_i(t) - \mathbf{r}_i(0)\|^2$$
Self-diffusion coefficient via Einstein relation: $D = \lim_{t \to \infty} \frac{\text{MSD}(t)}{6t}$.

---

## File Layout

```
projects/22-atomix/
├── atomix/
│   ├── __init__.py
│   ├── types.py          # Vector3D, Atom, Bond, Angle, Dihedral, SimulationBox, constants
│   ├── potentials.py     # Non-bonded LJ 12-6, Coulomb, harmonic bond/angle/dihedral
│   ├── spatial.py        # Periodic boundary, minimum image, Linked Cell Lists, Verlet skin
│   ├── integrators.py    # Velocity Verlet, Leapfrog, SHAKE bond constraint solver
│   ├── thermostats.py    # NVE, Velocity Rescaling, Berendsen, Nosé-Hoover, Barostat
│   ├── observables.py    # Temperature, Virial Pressure, g(r), MSD, Rg, RMSD
│   ├── builder.py        # FCC crystal, solvent box, TIP3P water, peptide backbone
│   ├── visualizer.py     # 3D sub-pixel Braille viewer, CPK coloring, telemetry HUD
│   └── simulation.py     # High-level MolecularDynamicsSimulation engine orchestrator
├── tests/
│   ├── test_potentials.py
│   ├── test_spatial.py
│   ├── test_integrators.py
│   ├── test_thermostats.py
│   ├── test_observables.py
│   └── test_visualizer.py
├── benchmarks/
│   └── bench_atomix.py
├── examples/
│   └── biophysics_lab.py
├── README.md
├── TASK_QUEUE.md
└── CHECKPOINT_LAST.md
```
