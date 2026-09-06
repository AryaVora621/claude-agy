"""
Atomix: Molecular Dynamics Simulation Engine Orchestrator.
Combines:
  - Atomic force evaluation (LJ, Coulomb, Bond, Angle, Dihedral)
  - 1-2 and 1-3 bonded pair exclusion from non-bonded loops
  - Symplectic Velocity Verlet integration with SHAKE constraints
  - Canonical (NVT) and Isothermal-Isobaric (NPT) thermodynamic ensembles
  - Trajectory collection, state snapshots, and energy conservation tracking
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Set, Callable

from atomix.types import (
    Vector3D,
    Atom,
    Bond,
    Angle,
    Dihedral,
    SimulationBox,
    KB_REDUCED,
)
from atomix.potentials import (
    LennardJonesPotential,
    CoulombPotential,
    HarmonicBondPotential,
    HarmonicAnglePotential,
    PeriodicDihedralPotential,
)
from atomix.spatial import LinkedCellList, VerletNeighborList
from atomix.integrators import VelocityVerletIntegrator
from atomix.thermostats import (
    BerendsenThermostat,
    NoseHooverThermostat,
    AndersenThermostat,
    BerendsenBarostat,
)
from atomix.observables import (
    ThermodynamicState,
    compute_kinetic_energy,
    compute_temperature,
    compute_virial_pressure,
    MSDTracker,
    RadialDistributionFunction,
)


class MolecularDynamicsSimulation:
    """
    High-level orchestrator for classical molecular dynamics simulations.
    Manages force field evaluation, time integration, thermostats, and diagnostics.
    """
    def __init__(
        self,
        atoms: List[Atom],
        box: SimulationBox,
        bonds: Optional[List[Bond]] = None,
        angles: Optional[List[Angle]] = None,
        dihedrals: Optional[List[Dihedral]] = None,
        lj_potential: Optional[LennardJonesPotential] = None,
        coulomb_potential: Optional[CoulombPotential] = None,
        timestep: float = 0.002,
        kb: float = KB_REDUCED,
        thermostat: Optional[object] = None,
        barostat: Optional[BerendsenBarostat] = None,
        use_verlet_list: bool = True,
        cutoff: float = 2.5,
        enable_shake: bool = False,
    ) -> None:
        self.atoms = atoms
        self.box = box
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.kb = float(kb)

        # Force field potentials
        self.lj = lj_potential or LennardJonesPotential(cutoff=cutoff)
        self.coulomb = coulomb_potential
        self.cutoff = cutoff

        # Filter rigid bonds for SHAKE
        self.rigid_bonds = [b for b in self.bonds if b.is_rigid]
        self.flexible_bonds = [b for b in self.bonds if not b.is_rigid]

        # Symplectic integrator
        self.integrator = VelocityVerletIntegrator(
            timestep=timestep,
            enable_shake=enable_shake or len(self.rigid_bonds) > 0,
        )

        # Statistical mechanics controllers
        self.thermostat = thermostat
        self.barostat = barostat

        # Spatial partitioner
        self.use_verlet_list = use_verlet_list
        if use_verlet_list:
            self.spatial = VerletNeighborList(box, cutoff=self.cutoff, skin=0.4)
        else:
            self.spatial = LinkedCellList(box, cutoff=self.cutoff)

        # Exclude 1-2 (bonded) and 1-3 (angled) pairs from non-bonded force loops
        self.excluded_pairs: Set[Tuple[int, int]] = set()
        for b in self.bonds:
            pair = (min(b.atom1_id, b.atom2_id), max(b.atom1_id, b.atom2_id))
            self.excluded_pairs.add(pair)
        for a in self.angles:
            pair = (min(a.atom1_id, a.atom3_id), max(a.atom1_id, a.atom3_id))
            self.excluded_pairs.add(pair)

        # Simulation clock and state history
        self.step_count = 0
        self.sim_time = 0.0
        self.energy_history: List[float] = []

        # Structural trackers
        self.msd_tracker = MSDTracker(self.atoms)
        self.rdf = RadialDistributionFunction(r_max=min(0.5 * min(box.lx, box.ly, box.lz), 6.0))

        # Initial force calculation
        self.compute_forces()

    def compute_forces(self) -> Tuple[float, float]:
        """
        Evaluate all non-bonded and bonded forces acting on every atom.
        Returns: (total_potential_energy, total_virial_sum)
        """
        # 1. Reset all atomic forces to zero
        zero = Vector3D(0.0, 0.0, 0.0)
        for a in self.atoms:
            a.force = zero

        total_pot_energy = 0.0
        total_virial = 0.0

        # 2. Non-bonded pairwise forces (Lennard-Jones + Coulomb)
        if self.use_verlet_list and isinstance(self.spatial, VerletNeighborList):
            pairs = self.spatial.get_active_pairs(self.atoms, self.excluded_pairs)
        else:
            pairs = self.spatial.find_all_pairs(self.atoms, self.excluded_pairs)

        for i, j, r12, _ in pairs:
            atom_i = self.atoms[i]
            atom_j = self.atoms[j]

            # Lennard-Jones evaluation
            sig_ij, eps_ij = LennardJonesPotential.mix_parameters(
                atom_i.sigma, atom_i.epsilon, atom_j.sigma, atom_j.epsilon
            )
            e_lj, f_i_lj, vir_lj = self.lj.evaluate_pair(r12, sig_ij, eps_ij)
            total_pot_energy += e_lj
            total_virial += vir_lj

            atom_i.force = atom_i.force + f_i_lj
            atom_j.force = atom_j.force - f_i_lj  # Newton's 3rd Law

            # Coulombic electrostatics if active
            if self.coulomb and (atom_i.charge != 0.0 or atom_j.charge != 0.0):
                e_coul, f_i_coul, vir_coul = self.coulomb.evaluate_pair(r12, atom_i.charge, atom_j.charge)
                total_pot_energy += e_coul
                total_virial += vir_coul
                atom_i.force = atom_i.force + f_i_coul
                atom_j.force = atom_j.force - f_i_coul

        # 3. Flexible harmonic bond forces
        for bond in self.flexible_bonds:
            pos_i = self.atoms[bond.atom1_id].position
            pos_j = self.atoms[bond.atom2_id].position
            e_b, f1, f2 = HarmonicBondPotential.evaluate(pos_i, pos_j, bond)
            total_pot_energy += e_b
            self.atoms[bond.atom1_id].force = self.atoms[bond.atom1_id].force + f1
            self.atoms[bond.atom2_id].force = self.atoms[bond.atom2_id].force + f2

        # 4. Valence angle bending forces
        for angle in self.angles:
            p_i = self.atoms[angle.atom1_id].position
            p_j = self.atoms[angle.vertex_id].position
            p_k = self.atoms[angle.atom3_id].position
            e_a, fi, fj, fk = HarmonicAnglePotential.evaluate(p_i, p_j, p_k, angle)
            total_pot_energy += e_a
            self.atoms[angle.atom1_id].force = self.atoms[angle.atom1_id].force + fi
            self.atoms[angle.vertex_id].force = self.atoms[angle.vertex_id].force + fj
            self.atoms[angle.atom3_id].force = self.atoms[angle.atom3_id].force + fk

        # 5. Periodic torsional dihedral forces
        for dih in self.dihedrals:
            p_i = self.atoms[dih.atom1_id].position
            p_j = self.atoms[dih.atom2_id].position
            p_k = self.atoms[dih.atom3_id].position
            p_l = self.atoms[dih.atom4_id].position
            e_d, fi, fj, fk, fl = PeriodicDihedralPotential.evaluate(p_i, p_j, p_k, p_l, dih)
            total_pot_energy += e_d
            self.atoms[dih.atom1_id].force = self.atoms[dih.atom1_id].force + fi
            self.atoms[dih.atom2_id].force = self.atoms[dih.atom2_id].force + fj
            self.atoms[dih.atom3_id].force = self.atoms[dih.atom3_id].force + fk
            self.atoms[dih.atom4_id].force = self.atoms[dih.atom4_id].force + fl

        return (total_pot_energy, total_virial)

    def step(self) -> ThermodynamicState:
        """
        Advance the simulation by one integration timestep dt.
        Returns the instantaneous thermodynamic state.
        """
        dt = self.integrator.dt

        # Stage 1: position update & half-step velocities
        self.integrator.step_stage1(self.atoms, self.box, self.rigid_bonds)

        # Recompute forces at new atomic coordinates
        pot_energy, virial_sum = self.compute_forces()

        # Stage 2: complete velocity step
        self.integrator.step_stage2(self.atoms, self.box, self.rigid_bonds)

        # Apply thermostat if present
        kin_energy = compute_kinetic_energy(self.atoms)
        curr_temp = compute_temperature(
            self.atoms,
            kb=self.kb,
            n_constraints=len(self.rigid_bonds),
        )

        if self.thermostat is not None:
            if isinstance(self.thermostat, BerendsenThermostat):
                self.thermostat.apply(self.atoms, dt, curr_temp)
            elif isinstance(self.thermostat, NoseHooverThermostat):
                self.thermostat.step(self.atoms, dt, kin_energy)
            elif isinstance(self.thermostat, AndersenThermostat):
                self.thermostat.apply(self.atoms, dt)

            # Re-evaluate kinetic energy after thermostat velocity scaling
            kin_energy = compute_kinetic_energy(self.atoms)
            curr_temp = compute_temperature(
                self.atoms,
                kb=self.kb,
                n_constraints=len(self.rigid_bonds),
            )

        # Calculate instantaneous pressure
        pressure = compute_virial_pressure(self.atoms, self.box, kin_energy, virial_sum)

        # Apply barostat if present
        if self.barostat is not None:
            self.barostat.apply(self.atoms, self.box, dt, pressure)

        # Advance simulation clock
        self.step_count += 1
        self.sim_time += dt

        total_energy = kin_energy + pot_energy
        self.energy_history.append(total_energy)

        # Density = total mass / volume
        tot_mass = sum(a.mass for a in self.atoms)
        density = tot_mass / self.box.volume

        state = ThermodynamicState(
            step=self.step_count,
            time=self.sim_time,
            kinetic_energy=kin_energy,
            potential_energy=pot_energy,
            total_energy=total_energy,
            temperature=curr_temp,
            pressure=pressure,
            virial=virial_sum,
            volume=self.box.volume,
            density=density,
        )

        return state

    def run(
        self,
        num_steps: int,
        sample_interval: int = 1,
        callback: Optional[Callable[[ThermodynamicState], None]] = None,
    ) -> List[ThermodynamicState]:
        """
        Run simulation for num_steps, recording thermodynamic states at sample_interval.
        """
        trajectory: List[ThermodynamicState] = []

        for _ in range(num_steps):
            state = self.step()
            if self.step_count % sample_interval == 0:
                trajectory.append(state)
                # Sample radial distribution function
                self.rdf.sample(self.atoms, self.box)
                # Update MSD tracker
                self.msd_tracker.update(self.atoms, self.box)
                if callback is not None:
                    callback(state)

        return trajectory
