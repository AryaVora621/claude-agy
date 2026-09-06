"""
Atomix: High-Performance Molecular Dynamics Microbenchmarks.
Measures:
  - Vector3D algebraic operations throughput
  - Pairwise Lennard-Jones force evaluation rate
  - Linked Cell List spatial partitioning speed
  - Full MD simulation step rate across system sizes
  - 3D Sub-Pixel Unicode Braille visualizer frame rate
Zero external dependencies.
"""

import sys
from pathlib import Path
import time
from typing import Tuple

# Ensure atomix package is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atomix.types import Vector3D, Atom, SimulationBox
from atomix.potentials import LennardJonesPotential
from atomix.spatial import LinkedCellList, VerletNeighborList
from atomix.builder import build_fcc_lattice
from atomix.thermostats import initialize_maxwell_boltzmann_velocities, BerendsenThermostat
from atomix.simulation import MolecularDynamicsSimulation
from atomix.visualizer import render_molecular_system


def bench_vector_operations(iterations: int = 200000) -> float:
    v1 = Vector3D(1.2, 3.4, 5.6)
    v2 = Vector3D(7.8, 9.0, 2.1)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = v1.dot(v2)
        _ = v1.cross(v2)
        _ = v1 + v2
        _ = v1.norm_sq()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ops_per_sec = (iterations * 4) / elapsed
    return ops_per_sec


def bench_lj_pair_evaluation(iterations: int = 150000) -> float:
    lj = LennardJonesPotential(cutoff=2.5, shifted=True)
    r_vec = Vector3D(1.3, 0.4, 0.2)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = lj.evaluate_pair(r_vec, 1.0, 1.0)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    evals_per_sec = iterations / elapsed
    return evals_per_sec


def bench_cell_list_partitioning(num_atoms: int = 256, repetitions: int = 500) -> float:
    atoms, box = build_fcc_lattice(n_cells=4, lattice_constant=1.7)
    cell_list = LinkedCellList(box, cutoff=2.5)

    t0 = time.perf_counter()
    for _ in range(repetitions):
        _ = cell_list.find_all_pairs(atoms)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    builds_per_sec = repetitions / elapsed
    return builds_per_sec


def bench_md_step_rate(n_cells: int = 3, steps: int = 100) -> Tuple[float, int]:
    # 4 * 3^3 = 108 atoms
    atoms, box = build_fcc_lattice(n_cells=n_cells, lattice_constant=1.6)
    initialize_maxwell_boltzmann_velocities(atoms, target_temperature=1.0)
    thermo = BerendsenThermostat(target_temperature=1.0, tau_t=0.1)

    sim = MolecularDynamicsSimulation(
        atoms,
        box,
        timestep=0.002,
        thermostat=thermo,
        use_verlet_list=True,
    )

    t0 = time.perf_counter()
    sim.run(num_steps=steps)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    steps_per_sec = steps / elapsed
    return (steps_per_sec, len(atoms))


def bench_visualizer_fps(frames: int = 100) -> float:
    atoms, box = build_fcc_lattice(n_cells=2, lattice_constant=1.6)

    t0 = time.perf_counter()
    for i in range(frames):
        _ = render_molecular_system(
            atoms,
            box=box,
            azimuth_deg=float(i * 3),
            elevation_deg=25.0,
            char_width=60,
            char_height=20,
        )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = frames / elapsed
    return fps


def main():
    print("=" * 72)
    print("  ATOMIX: FIRST-PRINCIPLES MOLECULAR DYNAMICS PERFORMANCE BENCHMARKS")
    print("  Pure Python 3 Standard Library (Single Thread, Zero Dependencies)")
    print("=" * 72)

    vec_rate = bench_vector_operations()
    print(f"[*] Vector3D Operations Throughput: {vec_rate:,.0f} ops/sec")

    lj_rate = bench_lj_pair_evaluation()
    print(f"[*] Lennard-Jones Pair Force Evals: {lj_rate:,.0f} pair-evals/sec ({1e6/lj_rate:.3f} us/eval)")

    cell_rate = bench_cell_list_partitioning()
    print(f"[*] Linked Cell Neighbor Search:    {cell_rate:,.1f} builds/sec (256 atoms, O(N))")

    step_rate_108, n108 = bench_md_step_rate(n_cells=3, steps=100)
    atom_steps_108 = step_rate_108 * n108
    print(f"[*] MD Simulation Rate (108 atoms): {step_rate_108:,.1f} steps/sec ({atom_steps_108:,.0f} atom-steps/sec)")

    step_rate_256, n256 = bench_md_step_rate(n_cells=4, steps=50)
    atom_steps_256 = step_rate_256 * n256
    print(f"[*] MD Simulation Rate (256 atoms): {step_rate_256:,.1f} steps/sec ({atom_steps_256:,.0f} atom-steps/sec)")

    fps = bench_visualizer_fps()
    print(f"[*] 3D Braille Render Frame Rate:   {fps:,.1f} FPS ({1000.0/fps:.2f} ms/frame)")
    print("=" * 72)


if __name__ == "__main__":
    main()
