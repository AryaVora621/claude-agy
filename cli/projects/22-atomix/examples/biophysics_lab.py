"""
Atomix: Interactive Molecular Dynamics and Computational Biophysics Laboratory.
Demonstrates 3 showcase simulations:
  1. Argon Crystal Melting & Phase Transition (FCC solid -> liquid, g(r) peak broadening, MSD diffusion)
  2. Polypeptide Helix Thermal Breathing & Conformational Dynamics (Rg, RMSD, CPK colors)
  3. Explicit TIP3P Liquid Water with SHAKE Rigid Bond Constraints (96 atoms, density, O-O coordination)
Zero external dependencies.
"""

from __future__ import annotations
import sys
import time
import math
from pathlib import Path

# Ensure atomix is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atomix import (
    Vector3D,
    Atom,
    SimulationBox,
    build_fcc_lattice,
    build_peptide_chain,
    build_tip3p_water_box,
    initialize_maxwell_boltzmann_velocities,
    BerendsenThermostat,
    NoseHooverThermostat,
    MolecularDynamicsSimulation,
    render_molecular_system,
    render_telemetry_hud,
    MSDTracker,
)


def run_argon_melting_experiment():
    print("\n" + "=" * 76)
    print("  CASE STUDY 1: ARGON CRYSTAL MELTING & PHASE TRANSITION")
    print("  FCC Solid (T=0.2) -> Liquid (T=1.4) | g(r) Structure & MSD Diffusion")
    print("=" * 76)

    # 108 atoms in FCC lattice
    atoms, box = build_fcc_lattice(n_cells=3, lattice_constant=1.65)
    print(f"[*] Initialized FCC Argon Crystal: {len(atoms)} atoms in {box.lx:.2f} A periodic box")

    # Step 1: Cold crystal
    initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.2, remove_drift=True)
    thermo = BerendsenThermostat(target_temperature=0.2, tau_t=0.05)
    sim = MolecularDynamicsSimulation(atoms, box, thermostat=thermo, timestep=0.002)

    print("[*] Equilibrating low-temperature crystal (T = 0.2)...")
    sim.run(num_steps=30)
    solid_msd = sim.msd_tracker.update(atoms, box)

    # Render solid crystal
    print("\n[Undeformed Low-Temperature FCC Solid Crystal (3D Braille CPK View)]:")
    print(render_molecular_system(atoms, box=box, azimuth_deg=30.0, elevation_deg=20.0, char_width=65, char_height=14))
    print(f"    Solid Phase MSD: {solid_msd:.4f} (bounded lattice vibrations)")

    # Step 2: Heat to liquid state (T = 1.35)
    print("\n[*] Heating system past melting point to liquid phase (T = 1.35)...")
    sim.thermostat.target_temperature = 1.35
    sim.thermostat.tau_t = 0.02
    traj = sim.run(num_steps=60, sample_interval=10)

    liquid_msd = sim.msd_tracker.update(atoms, box)
    diff_coeff = MSDTracker.calculate_diffusion_coefficient(liquid_msd, sim.sim_time)

    print("\n[Melted Disordered Liquid Phase (3D Braille CPK View)]:")
    print(render_molecular_system(atoms, box=box, azimuth_deg=45.0, elevation_deg=25.0, char_width=65, char_height=14))

    # Display HUD and radial distribution function g(r)
    last_state = traj[-1]
    print(render_telemetry_hud(
        last_state,
        len(atoms),
        num_bonds=0,
        rg=0.0,
        diffusion_coeff=diff_coeff,
        energy_history=[s.total_energy for s in traj],
        width=74,
    ))

    # Plot ASCII Radial Distribution Function g(r)
    r_centers, g_vals = sim.rdf.get_distribution(box, len(atoms))
    print("[Radial Distribution Function g(r) (Liquid Coordination Shells)]:")
    max_g = max(max(g_vals), 1.0)
    for idx in range(0, min(18, len(r_centers))):
        r = r_centers[idx]
        g = g_vals[idx]
        bar_len = int((g / max_g) * 35)
        bar = "#" * bar_len
        print(f"  r = {r:4.2f} A | g(r) = {g:5.2f} | {bar}")
    print(f"[*] Self-Diffusion Coefficient in Liquid Phase: D = {diff_coeff:.4e} (Einstein relation)")


def run_polypeptide_dynamics():
    print("\n" + "=" * 76)
    print("  CASE STUDY 2: POLYPEPTIDE HELIX CONFORMATIONAL DYNAMICS")
    print("  12-Residue Alpha-Helix | Bonded Intramolecular Force Field (Rg & RMSD)")
    print("=" * 76)

    atoms, bonds, angles, dihedrals, box = build_peptide_chain(num_residues=8, box_size=35.0)
    print(f"[*] Constructed Polypeptide: {len(atoms)} atoms, {len(bonds)} bonds, {len(angles)} angles, {len(dihedrals)} dihedrals")

    ref_positions = [a.position for a in atoms]
    initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.4)

    thermo = NoseHooverThermostat(target_temperature=0.4, tau_nh=0.1)
    sim = MolecularDynamicsSimulation(
        atoms,
        box,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        timestep=0.001,
        thermostat=thermo,
    )

    print("\n[Initial Folded Conformation (Backbone + Carbonyls in 3D Braille)]:")
    print(render_molecular_system(atoms, bonds=bonds, box=box, azimuth_deg=20.0, elevation_deg=15.0, char_width=65, char_height=14))

    print("[*] Propagating thermal dynamics (Nosé-Hoover NVT canonical ensemble)...")
    traj = sim.run(num_steps=50, sample_interval=10)

    # Compute conformational metrics
    from atomix.observables import compute_radius_of_gyration, compute_rmsd
    final_rg = compute_radius_of_gyration(atoms)
    final_rmsd = compute_rmsd(atoms, ref_positions)

    print("\n[Thermally Relaxed Conformation]:")
    print(render_molecular_system(atoms, bonds=bonds, box=box, azimuth_deg=55.0, elevation_deg=30.0, char_width=65, char_height=14))

    print(render_telemetry_hud(
        traj[-1],
        len(atoms),
        len(bonds),
        rg=final_rg,
        diffusion_coeff=0.0,
        energy_history=[s.total_energy for s in traj],
        width=74,
    ))
    print(f"[*] Structural Metrics: Final R_g = {final_rg:.2f} A | Conformation RMSD = {final_rmsd:.3f} A")


def run_tip3p_water_shake():
    print("\n" + "=" * 76)
    print("  CASE STUDY 3: EXPLICIT TIP3P LIQUID WATER WITH SHAKE CONSTRAINTS")
    print("  32 Water Molecules (96 Atoms) | Holonomic O-H Distance Constraints")
    print("=" * 76)

    num_waters = 24
    atoms, bonds, angles, box = build_tip3p_water_box(num_molecules=num_waters, box_size=18.0, is_rigid=True)
    print(f"[*] Solvated Box: {num_waters} TIP3P water molecules ({len(atoms)} atoms, {len(bonds)} rigid O-H bonds)")

    initialize_maxwell_boltzmann_velocities(atoms, target_temperature=298.15, kb=0.008314462618)
    thermo = BerendsenThermostat(target_temperature=298.15, tau_t=0.05, kb=0.008314462618)

    sim = MolecularDynamicsSimulation(
        atoms,
        box,
        bonds=bonds,
        angles=angles,
        timestep=0.001,
        kb=0.008314462618,
        thermostat=thermo,
        enable_shake=True,
    )

    print("[*] Simulating liquid water under SHAKE constraint projection...")
    traj = sim.run(num_steps=40, sample_interval=10)

    # Check rigid bond preservation
    target_oh = 0.9572
    max_dev = 0.0
    for bond in sim.rigid_bonds:
        r_ij = box.minimum_image_vector(atoms[bond.atom1_id].position, atoms[bond.atom2_id].position)
        dev = abs(r_ij.norm() - target_oh)
        if dev > max_dev:
            max_dev = dev

    print("\n[Explicit Water Molecular Box (Oxygen: Red, Hydrogen: White, Bonds: Gray)]:")
    print(render_molecular_system(atoms, bonds=bonds, box=box, azimuth_deg=40.0, elevation_deg=20.0, char_width=65, char_height=14))

    print(render_telemetry_hud(
        traj[-1],
        len(atoms),
        len(bonds),
        rg=0.0,
        diffusion_coeff=0.0,
        energy_history=[s.total_energy for s in traj],
        width=74,
    ))
    print(f"[*] SHAKE Rigidity Verification: Max O-H Bond Deviation = {max_dev:.6e} A (target = {target_oh:.4f} A)")
    print("=" * 76)


def main():
    print("\n" + "#" * 76)
    print("  ATOMIX: STATISTICAL MECHANICS & COMPUTATIONAL BIOPHYSICS LABORATORY")
    print("  First-Principles Classical Molecular Dynamics in Pure Python")
    print("#" * 76)

    run_argon_melting_experiment()
    run_polypeptide_dynamics()
    run_tip3p_water_shake()

    print("\n[+] All 3 biophysics benchmarks successfully simulated and rendered.")


if __name__ == "__main__":
    main()
