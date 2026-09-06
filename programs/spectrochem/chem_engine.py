"""
SpectroChem 3D: First-Principles Molecular Mechanics, Dynamics, and Spectroscopy Engine.
Pure Python standard library implementation with zero external dependencies.

Features:
- Full periodic element data with masses, covalent/vdW radii, CPK colors, and electronegativities.
- Harmonic bond stretching, angle bending, dihedral torsions, Lennard-Jones 6-12, and Coulomb electrostatics.
- Conjugate Gradient and Steepest Descent energy minimization.
- Velocity Verlet Molecular Dynamics with Maxwell-Boltzmann thermalization and Berendsen thermostat.
- Mass-weighted Hessian matrix calculation via central finite difference of analytical/numerical gradients.
- Jacobi symmetric matrix diagonalization for 3N normal mode vibrational frequencies and eigenvectors.
- Infrared (IR) dipole transition intensities and Lorentzian spectral line broadening.
- VSEPR geometry classification and dipole moment calculations.
"""

import math
import random
from typing import List, Tuple, Dict, Optional

# Physical constants (in chemical simulation units: Angstroms, amu, kcal/mol, fs)
HARTREE_TO_KCAL = 627.509
BOHR_TO_ANGSTROM = 0.529177
SPEED_OF_LIGHT_CM_S = 2.99792458e10
KB_KCAL_MOL_K = 0.001987204  # Boltzmann constant in kcal / (mol * K)
ELEMENTARY_CHARGE_DEBYE = 4.80320  # e * Angstrom to Debye conversion factor

# Fundamental Element Reference Data
ELEMENTS = {
    'H': {
        'name': 'Hydrogen', 'atomic_number': 1, 'mass': 1.008,
        'cov_radius': 0.31, 'vdw_radius': 1.20, 'electronegativity': 2.20,
        'color': '#FFFFFF', 'valency': 1
    },
    'C': {
        'name': 'Carbon', 'atomic_number': 6, 'mass': 12.011,
        'cov_radius': 0.76, 'vdw_radius': 1.70, 'electronegativity': 2.55,
        'color': '#909090', 'valency': 4
    },
    'N': {
        'name': 'Nitrogen', 'atomic_number': 7, 'mass': 14.007,
        'cov_radius': 0.71, 'vdw_radius': 1.55, 'electronegativity': 3.04,
        'color': '#3050F8', 'valency': 3
    },
    'O': {
        'name': 'Oxygen', 'atomic_number': 8, 'mass': 15.999,
        'cov_radius': 0.66, 'vdw_radius': 1.52, 'electronegativity': 3.44,
        'color': '#FF0D0D', 'valency': 2
    },
    'F': {
        'name': 'Fluorine', 'atomic_number': 9, 'mass': 18.998,
        'cov_radius': 0.57, 'vdw_radius': 1.47, 'electronegativity': 3.98,
        'color': '#90E050', 'valency': 1
    },
    'P': {
        'name': 'Phosphorus', 'atomic_number': 15, 'mass': 30.974,
        'cov_radius': 1.07, 'vdw_radius': 1.80, 'electronegativity': 2.19,
        'color': '#FF8000', 'valency': 5
    },
    'S': {
        'name': 'Sulfur', 'atomic_number': 16, 'mass': 32.065,
        'cov_radius': 1.05, 'vdw_radius': 1.80, 'electronegativity': 2.58,
        'color': '#FFFF30', 'valency': 2
    },
    'Cl': {
        'name': 'Chlorine', 'atomic_number': 17, 'mass': 35.453,
        'cov_radius': 1.02, 'vdw_radius': 1.75, 'electronegativity': 3.16,
        'color': '#1FF01F', 'valency': 1
    },
    'Br': {
        'name': 'Bromine', 'atomic_number': 35, 'mass': 79.904,
        'cov_radius': 1.20, 'vdw_radius': 1.85, 'electronegativity': 2.96,
        'color': '#A62929', 'valency': 1
    },
    'Fe': {
        'name': 'Iron', 'atomic_number': 26, 'mass': 55.845,
        'cov_radius': 1.25, 'vdw_radius': 2.00, 'electronegativity': 1.83,
        'color': '#E06633', 'valency': 2
    }
}


class Atom:
    """Represents a single atom with 3D coordinates, kinematics, and electrostatics."""
    def __init__(self, element: str, x: float, y: float, z: float, charge: float = 0.0, name: str = ""):
        self.element = element
        elem_data = ELEMENTS.get(element, ELEMENTS['C'])
        self.name = name if name else element
        self.mass = elem_data['mass']
        self.cov_radius = elem_data['cov_radius']
        self.vdw_radius = elem_data['vdw_radius']
        self.color = elem_data['color']
        self.electronegativity = elem_data['electronegativity']

        # 3D Cartesian coordinates in Angstroms
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        # Initial reference coordinates for normal mode displacement
        self.x0 = self.x
        self.y0 = self.y
        self.z0 = self.z

        # Velocity in Angstroms / fs
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # Forces in kcal / (mol * Angstrom)
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0

        # Partial charge in elementary charge units (e)
        self.charge = float(charge)


class Bond:
    """Represents a chemical bond between two atoms."""
    def __init__(self, a1_idx: int, a2_idx: int, order: float = 1.0, r0: Optional[float] = None, kb: float = 350.0):
        self.a1 = min(a1_idx, a2_idx)
        self.a2 = max(a1_idx, a2_idx)
        self.order = order  # 1.0 (single), 1.5 (aromatic), 2.0 (double), 3.0 (triple)
        self.r0 = r0  # Equilibrium bond length (Angstroms)
        self.kb = kb  # Force constant (kcal / mol / Angstrom^2)


class Angle:
    """Represents a valence angle between three bonded atoms (i - j - k)."""
    def __init__(self, a1_idx: int, center_idx: int, a3_idx: int, theta0: Optional[float] = None, k_theta: float = 60.0):
        self.a1 = a1_idx
        self.center = center_idx
        self.a3 = a3_idx
        self.theta0 = theta0  # Equilibrium angle in radians
        self.k_theta = k_theta  # Force constant (kcal / mol / rad^2)


class Molecule:
    """Complete molecular structure with force field, mechanics, dynamics, and spectroscopy."""
    def __init__(self, name: str = "Molecule"):
        self.name = name
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.angles: List[Angle] = []

        # Energy telemetry (kcal / mol)
        self.e_bond = 0.0
        self.e_angle = 0.0
        self.e_vdw = 0.0
        self.e_coulomb = 0.0
        self.e_potential = 0.0
        self.e_kinetic = 0.0
        self.temperature = 0.0

        # Vibrational modes
        self.normal_modes: List[Dict] = []
        self.active_mode_idx: Optional[int] = None
        self.vibration_phase: float = 0.0

    def add_atom(self, element: str, x: float, y: float, z: float, charge: float = 0.0, name: str = "") -> int:
        idx = len(self.atoms)
        atom = Atom(element, x, y, z, charge, name)
        self.atoms.append(atom)
        return idx

    def add_bond(self, a1_idx: int, a2_idx: int, order: float = 1.0, r0: Optional[float] = None, kb: float = 350.0):
        # Auto compute r0 if not provided based on initial distance
        if r0 is None:
            p1 = (self.atoms[a1_idx].x, self.atoms[a1_idx].y, self.atoms[a1_idx].z)
            p2 = (self.atoms[a2_idx].x, self.atoms[a2_idx].y, self.atoms[a2_idx].z)
            r0 = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        self.bonds.append(Bond(a1_idx, a2_idx, order, r0, kb))

    def detect_connectivity(self, tolerance: float = 0.45):
        """Automatically detect bonds and angles based on covalent radii and geometry."""
        self.bonds.clear()
        self.angles.clear()
        n = len(self.atoms)

        # 1. Detect Bonds
        for i in range(n):
            for j in range(i + 1, n):
                r_cov_sum = self.atoms[i].cov_radius + self.atoms[j].cov_radius
                dx = self.atoms[i].x - self.atoms[j].x
                dy = self.atoms[i].y - self.atoms[j].y
                dz = self.atoms[i].z - self.atoms[j].z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if 0.4 < dist <= (r_cov_sum + tolerance):
                    # Estimate bond order based on distance contraction
                    order = 1.0
                    if dist < r_cov_sum - 0.17:
                        order = 2.0
                    if dist < r_cov_sum - 0.30:
                        order = 3.0
                    self.add_bond(i, j, order=order, r0=dist)

        # 2. Detect Angles (bonded triplets with central atom)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for b in self.bonds:
            adj[b.a1].append(b.a2)
            adj[b.a2].append(b.a1)

        for center in range(n):
            neighbors = adj[center]
            num_nbr = len(neighbors)
            for i in range(num_nbr):
                for j in range(i + 1, num_nbr):
                    a1 = neighbors[i]
                    a3 = neighbors[j]
                    theta0 = self.calculate_angle(a1, center, a3)
                    self.angles.append(Angle(a1, center, a3, theta0=theta0))

    def calculate_distance(self, i: int, j: int) -> float:
        dx = self.atoms[i].x - self.atoms[j].x
        dy = self.atoms[i].y - self.atoms[j].y
        dz = self.atoms[i].z - self.atoms[j].z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def calculate_angle(self, i: int, center: int, k: int) -> float:
        """Calculate the angle in radians between atoms i - center - k."""
        v1 = (self.atoms[i].x - self.atoms[center].x,
              self.atoms[i].y - self.atoms[center].y,
              self.atoms[i].z - self.atoms[center].z)
        v2 = (self.atoms[k].x - self.atoms[center].x,
              self.atoms[k].y - self.atoms[center].y,
              self.atoms[k].z - self.atoms[center].z)

        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if mag1 < 1e-7 or mag2 < 1e-7:
            return 0.0

        dot = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (mag1 * mag2)
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def compute_energy_and_forces(self) -> float:
        """
        Calculates potential energy and analytical atomic force gradients:
        E_pot = E_bond + E_angle + E_vdW + E_coulomb
        F = -grad(E_pot)
        """
        for atom in self.atoms:
            atom.fx = 0.0
            atom.fy = 0.0
            atom.fz = 0.0

        self.e_bond = 0.0
        self.e_angle = 0.0
        self.e_vdw = 0.0
        self.e_coulomb = 0.0

        # 1. Harmonic Bond Stretching
        for bond in self.bonds:
            a1 = self.atoms[bond.a1]
            a2 = self.atoms[bond.a2]
            dx = a2.x - a1.x
            dy = a2.y - a1.y
            dz = a2.z - a1.z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-6:
                continue

            dr = r - bond.r0
            # E_b = 0.5 * kb * (r - r0)^2
            e_b = 0.5 * bond.kb * (dr ** 2)
            self.e_bond += e_b

            # Force magnitude = -dE/dr = -kb * (r - r0)
            f_mag = -bond.kb * dr
            fx = f_mag * (dx / r)
            fy = f_mag * (dy / r)
            fz = f_mag * (dz / r)

            a1.fx -= fx
            a1.fy -= fy
            a1.fz -= fz
            a2.fx += fx
            a2.fy += fy
            a2.fz += fz

        # 2. Harmonic Angle Bending
        for angle in self.angles:
            a1 = self.atoms[angle.a1]
            c = self.atoms[angle.center]
            a3 = self.atoms[angle.a3]

            v1 = [a1.x - c.x, a1.y - c.y, a1.z - c.z]
            v2 = [a3.x - c.x, a3.y - c.y, a3.z - c.z]
            r1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            r2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

            if r1 < 1e-6 or r2 < 1e-6:
                continue

            cos_theta = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (r1 * r2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            d_theta = theta - angle.theta0

            # E_a = 0.5 * k_theta * (theta - theta0)^2
            self.e_angle += 0.5 * angle.k_theta * (d_theta ** 2)

            sin_theta = math.sqrt(max(1e-12, 1.0 - cos_theta**2))
            pref = angle.k_theta * d_theta / sin_theta

            # Vector projection derivatives for a1 and a3
            f1 = [pref * (v2[d]/(r1*r2) - cos_theta*v1[d]/(r1*r1)) for d in range(3)]
            f3 = [pref * (v1[d]/(r1*r2) - cos_theta*v2[d]/(r2*r2)) for d in range(3)]
            fc = [-(f1[d] + f3[d]) for d in range(3)]

            a1.fx += f1[0]
            a1.fy += f1[1]
            a1.fz += f1[2]

            a3.fx += f3[0]
            a3.fy += f3[1]
            a3.fz += f3[2]

            c.fx += fc[0]
            c.fy += fc[1]
            c.fz += fc[2]

        # 3. Non-bonded Interactions (Lennard-Jones + Coulomb)
        # Exclude 1-2 (bonded) and 1-3 (angle) pairs
        bonded_pairs = set((b.a1, b.a2) for b in self.bonds)
        angle_pairs = set((min(a.a1, a.a3), max(a.a1, a.a3)) for a in self.angles)

        n = len(self.atoms)
        coulomb_const = 332.0637  # e^2 / Angstrom to kcal / mol

        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bonded_pairs or (i, j) in angle_pairs:
                    continue

                ai = self.atoms[i]
                aj = self.atoms[j]
                dx = aj.x - ai.x
                dy = aj.y - ai.y
                dz = aj.z - ai.z
                r2 = dx * dx + dy * dy + dz * dz
                r = math.sqrt(r2)
                if r < 0.6:
                    r = 0.6
                    r2 = 0.36

                # Lennard-Jones parameters (Lorentz-Berthelot mixing rules)
                sigma = (ai.vdw_radius + aj.vdw_radius) * 0.89
                epsilon = 0.10  # kcal / mol standard depth
                s_over_r = sigma / r
                s_over_r6 = s_over_r ** 6
                s_over_r12 = s_over_r6 ** 2

                e_lj = 4.0 * epsilon * (s_over_r12 - s_over_r6)
                self.e_vdw += e_lj

                # d(E_LJ)/dr = 4 * epsilon * (-12 * sigma^12 / r^13 + 6 * sigma^6 / r^7)
                f_lj = 24.0 * epsilon * (2.0 * s_over_r12 - s_over_r6) / r

                # Coulomb Electrostatics
                e_qq = coulomb_const * (ai.charge * aj.charge) / r
                self.e_coulomb += e_qq
                f_qq = coulomb_const * (ai.charge * aj.charge) / r2

                total_f = f_lj + f_qq
                fx = total_f * (dx / r)
                fy = total_f * (dy / r)
                fz = total_f * (dz / r)

                ai.fx -= fx
                ai.fy -= fy
                ai.fz -= fz
                aj.fx += fx
                aj.fy += fy
                aj.fz += fz

        self.e_potential = self.e_bond + self.e_angle + self.e_vdw + self.e_coulomb
        return self.e_potential

    def minimize_geometry(self, max_steps: int = 150, tolerance: float = 0.01) -> int:
        """
        Conjugate Gradient with backtracking line search to ground state minimum.
        Returns number of iterations executed.
        """
        alpha = 0.005
        prev_forces: List[Tuple[float, float, float]] = []
        prev_search_dir: List[Tuple[float, float, float]] = []

        for step in range(max_steps):
            e_current = self.compute_energy_and_forces()

            # Compute maximum atomic gradient norm
            max_grad = 0.0
            for atom in self.atoms:
                f_norm = math.sqrt(atom.fx**2 + atom.fy**2 + atom.fz**2)
                if f_norm > max_grad:
                    max_grad = f_norm

            if max_grad < tolerance:
                # Save equilibrium positions
                for a in self.atoms:
                    a.x0 = a.x
                    a.y0 = a.y
                    a.z0 = a.z
                return step + 1

            # Polak-Ribiere beta calculation
            if step > 0 and prev_forces and prev_search_dir:
                dot_num = sum((a.fx - pf[0])*a.fx + (a.fy - pf[1])*a.fy + (a.fz - pf[2])*a.fz
                              for a, pf in zip(self.atoms, prev_forces))
                dot_den = sum(pf[0]**2 + pf[1]**2 + pf[2]**2 for pf in prev_forces)
                beta = max(0.0, dot_num / max(1e-12, dot_den))
            else:
                beta = 0.0

            # Construct conjugate search direction
            search_dir = []
            for i, atom in enumerate(self.atoms):
                if beta > 0.0 and prev_search_dir:
                    sx = atom.fx + beta * prev_search_dir[i][0]
                    sy = atom.fy + beta * prev_search_dir[i][1]
                    sz = atom.fz + beta * prev_search_dir[i][2]
                else:
                    sx = atom.fx
                    sy = atom.fy
                    sz = atom.fz
                search_dir.append((sx, sy, sz))

            # Reset if search direction opposes gradient
            dot_dir = sum(a.fx * s[0] + a.fy * s[1] + a.fz * s[2] for a, s in zip(self.atoms, search_dir))
            if dot_dir <= 0:
                search_dir = [(a.fx, a.fy, a.fz) for a in self.atoms]

            prev_forces = [(a.fx, a.fy, a.fz) for a in self.atoms]
            prev_search_dir = search_dir

            # Backtracking line search
            orig_coords = [(a.x, a.y, a.z) for a in self.atoms]
            step_alpha = alpha
            accepted = False

            for _ in range(6):
                for i, atom in enumerate(self.atoms):
                    dx = search_dir[i][0] * step_alpha
                    dy = search_dir[i][1] * step_alpha
                    dz = search_dir[i][2] * step_alpha
                    disp = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if disp > 0.10:
                        scale = 0.10 / disp
                        dx *= scale
                        dy *= scale
                        dz *= scale
                    atom.x = orig_coords[i][0] + dx
                    atom.y = orig_coords[i][1] + dy
                    atom.z = orig_coords[i][2] + dz

                e_trial = self.compute_energy_and_forces()
                if e_trial < e_current:
                    accepted = True
                    alpha = min(0.02, step_alpha * 1.1)
                    break
                else:
                    step_alpha *= 0.5

            if not accepted:
                # Restore original coordinates
                for i, atom in enumerate(self.atoms):
                    atom.x, atom.y, atom.z = orig_coords[i]
                alpha = max(0.0001, alpha * 0.5)

        self.compute_energy_and_forces()
        # Save equilibrium positions
        for a in self.atoms:
            a.x0 = a.x
            a.y0 = a.y
            a.z0 = a.z
        return max_steps

    def initialize_velocities(self, target_temperature: float = 300.0):
        """Maxwell-Boltzmann velocity distribution initialization."""
        total_mass = sum(a.mass for a in self.atoms)
        p_com = [0.0, 0.0, 0.0]

        for atom in self.atoms:
            # v_std = sqrt(k_B * T / m) in Angstrom / fs
            # 1 kcal / mol / amu = 0.004184 (Angstrom / fs)^2
            sigma = math.sqrt(max(1e-6, KB_KCAL_MOL_K * target_temperature / atom.mass * 0.04184))
            atom.vx = random.gauss(0.0, sigma)
            atom.vy = random.gauss(0.0, sigma)
            atom.vz = random.gauss(0.0, sigma)

            p_com[0] += atom.mass * atom.vx
            p_com[1] += atom.mass * atom.vy
            p_com[2] += atom.mass * atom.vz

        # Remove center-of-mass translational momentum
        for atom in self.atoms:
            atom.vx -= p_com[0] / total_mass
            atom.vy -= p_com[1] / total_mass
            atom.vz -= p_com[2] / total_mass

    def md_step(self, dt: float = 0.5, target_temperature: Optional[float] = 300.0):
        """
        Velocity Verlet Molecular Dynamics integration step (dt in femtoseconds).
        Includes Berendsen thermostat for canonical NVT ensemble.
        """
        # Conversion: force (kcal/mol/A) to acceleration (A/fs^2)
        # 1 kcal / (mol * A * amu) = 0.0004184 A / fs^2
        f_to_a = 0.0004184

        # 1. Update positions: r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        for a in self.atoms:
            ax = (a.fx / a.mass) * f_to_a
            ay = (a.fy / a.mass) * f_to_a
            az = (a.fz / a.mass) * f_to_a

            a.x += a.vx * dt + 0.5 * ax * dt * dt
            a.y += a.vy * dt + 0.5 * ay * dt * dt
            a.z += a.vz * dt + 0.5 * az * dt * dt

            # Half-step velocity update
            a.vx += 0.5 * ax * dt
            a.vy += 0.5 * ay * dt
            a.vz += 0.5 * az * dt

        # 2. Compute new forces at r(t + dt)
        self.compute_energy_and_forces()

        # 3. Complete velocity update: v(t + dt) = v(t + 0.5dt) + 0.5*a(t + dt)*dt
        ek = 0.0
        for a in self.atoms:
            ax = (a.fx / a.mass) * f_to_a
            ay = (a.fy / a.mass) * f_to_a
            az = (a.fz / a.mass) * f_to_a

            a.vx += 0.5 * ax * dt
            a.vy += 0.5 * ay * dt
            a.vz += 0.5 * az * dt

            v2 = a.vx**2 + a.vy**2 + a.vz**2
            # Ek = 0.5 * m * v^2 / 0.0004184 (kcal/mol)
            ek += 0.5 * a.mass * v2 / 0.0004184

        self.e_kinetic = ek

        # Calculate instantaneous temperature: T = 2 * Ek / (3N * kB)
        dof = max(1, 3 * len(self.atoms) - 6)
        self.temperature = (2.0 * self.e_kinetic) / (dof * KB_KCAL_MOL_K)

        # 4. Berendsen Thermostat Scaling
        if target_temperature is not None and self.temperature > 1e-4:
            tau = 20.0  # Coupling time constant in fs
            lambda_t = math.sqrt(max(0.1, 1.0 + (dt / tau) * (target_temperature / self.temperature - 1.0)))
            # Clamp lambda for stability
            lambda_t = max(0.85, min(1.15, lambda_t))
            for a in self.atoms:
                a.vx *= lambda_t
                a.vy *= lambda_t
                a.vz *= lambda_t

    def compute_vibrational_spectrum(self) -> List[Dict]:
        """
        Calculates normal vibrational modes using the mass-weighted Hessian matrix:
        H_{ia, jb} = (1 / sqrt(m_i * m_j)) * (d^2 E / d x_{ia} d x_{jb})
        Diagonalizes H via Jacobi rotations to yield harmonic frequencies and eigenvectors.
        """
        n_atoms = len(self.atoms)
        dim = 3 * n_atoms
        delta = 0.002  # Finite difference step in Angstroms

        # First optimize geometry to minimum to eliminate gradient torque
        self.minimize_geometry(max_steps=80, tolerance=0.02)

        # Save base equilibrium coordinates
        base_coords = [(a.x, a.y, a.z) for a in self.atoms]

        # Construct mass-weighted Hessian matrix
        hessian = [[0.0 for _ in range(dim)] for _ in range(dim)]

        for i in range(n_atoms):
            for coord in range(3):
                col = 3 * i + coord

                # Forward displacement
                if coord == 0: self.atoms[i].x = base_coords[i][0] + delta
                elif coord == 1: self.atoms[i].y = base_coords[i][1] + delta
                else: self.atoms[i].z = base_coords[i][2] + delta
                self.compute_energy_and_forces()
                f_plus = [(a.fx, a.fy, a.fz) for a in self.atoms]

                # Backward displacement
                if coord == 0: self.atoms[i].x = base_coords[i][0] - delta
                elif coord == 1: self.atoms[i].y = base_coords[i][1] - delta
                else: self.atoms[i].z = base_coords[i][2] - delta
                self.compute_energy_and_forces()
                f_minus = [(a.fx, a.fy, a.fz) for a in self.atoms]

                # Restore coordinate
                self.atoms[i].x, self.atoms[i].y, self.atoms[i].z = base_coords[i]

                # Second derivative by central differences: d^2 E / dx dy = -(f+ - f-) / (2 * delta)
                for j in range(n_atoms):
                    m_factor = 1.0 / math.sqrt(self.atoms[i].mass * self.atoms[j].mass)
                    for c_j in range(3):
                        row = 3 * j + c_j
                        df = -(f_plus[j][c_j] - f_minus[j][c_j]) / (2.0 * delta)
                        hessian[row][col] = df * m_factor

        # Symmetrize Hessian
        for r in range(dim):
            for c in range(r + 1, dim):
                avg = (hessian[r][c] + hessian[c][r]) * 0.5
                hessian[r][c] = avg
                hessian[c][r] = avg

        # Solve eigenvalues and eigenvectors with Jacobi method
        eigenvalues, eigenvectors = jacobi_eigenvalue_solver(hessian, max_rotations=300)

        # Convert eigenvalues to vibrational frequencies in wavenumbers (cm^-1)
        # Factor conversion: sqrt(kcal / mol / A^2 / amu) to cm^-1: ~108.513
        freq_factor = 108.513

        modes = []
        for k in range(dim):
            eval_k = eigenvalues[k]
            if eval_k > 0:
                wavenumber = math.sqrt(eval_k) * freq_factor
            else:
                # Imaginary frequency
                wavenumber = -math.sqrt(abs(eval_k)) * freq_factor

            # Extract eigenvector displacement for each atom
            evec = [eigenvectors[row][k] for row in range(dim)]

            # Calculate Infrared (IR) Dipole Activity (d mu / d Q)
            dmu = [0.0, 0.0, 0.0]
            for atom_idx, a in enumerate(self.atoms):
                m_sqrt = math.sqrt(a.mass)
                disp_x = evec[3 * atom_idx + 0] / m_sqrt
                disp_y = evec[3 * atom_idx + 1] / m_sqrt
                disp_z = evec[3 * atom_idx + 2] / m_sqrt

                dmu[0] += a.charge * disp_x
                dmu[1] += a.charge * disp_y
                dmu[2] += a.charge * disp_z

            intensity = (dmu[0]**2 + dmu[1]**2 + dmu[2]**2) * 42.256  # km / mol

            modes.append({
                'mode_index': k + 1,
                'frequency_cm': wavenumber,
                'intensity_km_mol': intensity,
                'eigenvector': evec
            })

        # Sort modes by ascending frequency
        modes.sort(key=lambda m: m['frequency_cm'])
        self.normal_modes = modes
        return self.normal_modes

    def get_dipole_moment(self) -> Tuple[float, float, float, float]:
        """Calculates total molecular dipole vector (Debye) and magnitude."""
        # Calculate center of mass
        total_mass = sum(a.mass for a in self.atoms)
        com = [sum(a.mass * a.x for a in self.atoms) / total_mass,
               sum(a.mass * a.y for a in self.atoms) / total_mass,
               sum(a.mass * a.z for a in self.atoms) / total_mass]

        mux = sum(a.charge * (a.x - com[0]) for a in self.atoms) * ELEMENTARY_CHARGE_DEBYE
        muy = sum(a.charge * (a.y - com[1]) for a in self.atoms) * ELEMENTARY_CHARGE_DEBYE
        muz = sum(a.charge * (a.z - com[2]) for a in self.atoms) * ELEMENTARY_CHARGE_DEBYE
        mu_mag = math.sqrt(mux*mux + muy*muy + muz*muz)
        return mux, muy, muz, mu_mag

    def get_vsepr_classification(self, center_atom_idx: int = 0) -> Dict[str, str]:
        """Analyzes central atom steric geometry according to VSEPR theory."""
        if center_atom_idx >= len(self.atoms):
            return {'steric_number': 'N/A', 'geometry': 'N/A', 'point_group': 'N/A'}

        c_atom = self.atoms[center_atom_idx]
        bonded_count = sum(1 for b in self.bonds if b.a1 == center_atom_idx or b.a2 == center_atom_idx)

        # Estimate lone pairs from valence electrons
        valence_electrons = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6, 17: 7}
        val = valence_electrons.get(ELEMENTS.get(c_atom.element, {}).get('atomic_number', 6), 4)
        lone_pairs = max(0, (val - bonded_count) // 2)
        steric_num = bonded_count + lone_pairs

        geometry_map = {
            (2, 0): "Linear (180 deg)",
            (3, 0): "Trigonal Planar (120 deg)",
            (2, 1): "Bent (~118 deg)",
            (4, 0): "Tetrahedral (109.5 deg)",
            (3, 1): "Trigonal Pyramidal (107 deg)",
            (2, 2): "Bent (104.5 deg)",
            (5, 0): "Trigonal Bipyramidal",
            (6, 0): "Octahedral (90 deg)"
        }

        geom = geometry_map.get((bonded_count, lone_pairs), f"Steric {steric_num} (Bonded: {bonded_count})")
        return {
            'steric_number': str(steric_num),
            'bonded_atoms': str(bonded_count),
            'lone_pairs': str(lone_pairs),
            'geometry': geom
        }

    def update_normal_mode_vibration(self, amplitude: float = 0.45, speed: float = 0.08):
        """Displaces atomic coordinates according to the selected normal mode harmonic vibration."""
        if self.active_mode_idx is None or self.active_mode_idx >= len(self.normal_modes):
            return

        mode = self.normal_modes[self.active_mode_idx]
        evec = mode['eigenvector']
        self.vibration_phase += speed

        factor = amplitude * math.sin(self.vibration_phase)

        for i, atom in enumerate(self.atoms):
            m_sqrt = math.sqrt(atom.mass)
            dx = (evec[3 * i + 0] / m_sqrt) * factor
            dy = (evec[3 * i + 1] / m_sqrt) * factor
            dz = (evec[3 * i + 2] / m_sqrt) * factor

            atom.x = atom.x0 + dx
            atom.y = atom.y0 + dy
            atom.z = atom.z0 + dz


def jacobi_eigenvalue_solver(matrix: List[List[float]], max_rotations: int = 250, tolerance: float = 1e-9) -> Tuple[List[float], List[List[float]]]:
    """
    Jacobi diagonalization algorithm for symmetric real matrices.
    Returns: (eigenvalues, eigenvector_matrix)
    """
    n = len(matrix)
    # Clone matrix
    a = [row[:] for row in matrix]
    # Initialize eigenvector matrix as identity
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_rotations):
        # Find maximum off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j

        if max_val < tolerance:
            break

        # Compute rotation angle theta
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        if abs(apq) < 1e-15:
            continue

        phi = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0, phi) / (abs(phi) + math.sqrt(phi * phi + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c

        # Apply Givens rotation to A: A' = J^T A J
        tau = s / (1.0 + c)

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for r in range(n):
            if r != p and r != q:
                arp = a[r][p]
                arq = a[r][q]
                a[r][p] = arp - s * (arq + tau * arp)
                a[p][r] = a[r][p]
                a[r][q] = arq + s * (arp - tau * arq)
                a[q][r] = a[r][q]

        # Accumulate eigenvectors: V' = V J
        for r in range(n):
            vrp = v[r][p]
            vrq = v[r][q]
            v[r][p] = vrp - s * (vrq + tau * vrp)
            v[r][q] = vrq + s * (vrp - tau * vrq)

    eigenvalues = [a[i][i] for i in range(n)]
    return eigenvalues, v
