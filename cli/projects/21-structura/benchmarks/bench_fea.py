"""
Structura: Finite Element Analysis & Modal Vibration Performance Benchmarks.
Measures:
  - Element stiffness matrix computation throughput (QuadQ4, TriangleCST, Beam2D, Truss2D)
  - Global sparse matrix assembly & CSR conversion
  - Preconditioned Conjugate Gradient (PCG) solve latency & DOF rate
  - Modal vibration eigenvalue extraction performance
  - Unicode Braille 2x4 sub-pixel canvas rendering FPS
Zero external dependencies.
"""

import time
import math
import sys
from pathlib import Path
from typing import Dict

# Ensure structura is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structura.types import Node2D, Material, STRUCTURAL_STEEL
from structura.elements import Truss2D, Beam2D, TriangleCST, QuadQ4
from structura.sparse import DOKMatrix, solve_pcg
from structura.mesh import generate_rectangular_mesh, generate_truss_bridge_mesh
from structura.solver import FEASolver
from structura.modal import ModalSolver
from structura.visualizer import BrailleFEACanvas, render_mesh_deformation, render_stress_field


def benchmark_element_stiffness():
    print("=" * 72)
    print("  STRUCTURA FEA BENCHMARK 1: ELEMENT STIFFNESS EVALUATION")
    print("=" * 72)

    mat = STRUCTURAL_STEEL
    nodes = {
        0: Node2D(0, 0.0, 0.0),
        1: Node2D(1, 1.0, 0.0),
        2: Node2D(2, 1.0, 1.0),
        3: Node2D(3, 0.0, 1.0),
    }

    q4 = QuadQ4(0, (0, 1, 2, 3), mat)
    cst = TriangleCST(0, (0, 1, 2), mat)
    beam = Beam2D(0, (0, 1), mat, cross_section_area=0.01, moment_of_inertia=1e-4)
    truss = Truss2D(0, (0, 1), mat, cross_section_area=0.01)

    n_evals = 5000

    # 1. QuadQ4 (2x2 Gauss Quadrature)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        k_q4 = q4.compute_stiffness_matrix(nodes)
    t_q4 = time.perf_counter() - t0
    rate_q4 = n_evals / t_q4

    # 2. TriangleCST (Analytical B-matrix)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        k_cst = cst.compute_stiffness_matrix(nodes)
    t_cst = time.perf_counter() - t0
    rate_cst = n_evals / t_cst

    # 3. Beam2D (6x6 Euler-Bernoulli Frame)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        k_beam = beam.compute_stiffness_matrix(nodes)
    t_beam = time.perf_counter() - t0
    rate_beam = n_evals / t_beam

    # 4. Truss2D (4x4 Coordinate Transformed)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        k_truss = truss.compute_stiffness_matrix(nodes)
    t_truss = time.perf_counter() - t0
    rate_truss = n_evals / t_truss

    print(f"  QuadQ4 (Isoparametric 2x2 Gauss):   {rate_q4:>10,.0f} elem/s ({t_q4*1000/n_evals:.3f} ms/eval)")
    print(f"  TriangleCST (Constant Strain Tri):  {rate_cst:>10,.0f} elem/s ({t_cst*1000/n_evals:.3f} ms/eval)")
    print(f"  Beam2D (Euler-Bernoulli Frame):     {rate_beam:>10,.0f} elem/s ({t_beam*1000/n_evals:.3f} ms/eval)")
    print(f"  Truss2D (Axial Pin-Jointed Bar):    {rate_truss:>10,.0f} elem/s ({t_truss*1000/n_evals:.3f} ms/eval)")


def benchmark_solver_and_assembly():
    print("\n" + "=" * 72)
    print("  STRUCTURA FEA BENCHMARK 2: SPARSE ASSEMBLY & PCG SOLVER")
    print("=" * 72)

    # 20 x 8 Quad mesh = 160 elements, 189 nodes, 378 DOFs
    mesh = generate_rectangular_mesh(length=10.0, height=4.0, nx=20, ny=8, material=STRUCTURAL_STEEL, elem_type="quad")
    left_nodes = mesh.find_nodes_at_x(0.0)
    for nid in left_nodes:
        mesh.add_boundary_condition(nid, 0, 0.0)
        mesh.add_boundary_condition(nid, 1, 0.0)

    right_nodes = mesh.find_nodes_at_x(10.0)
    for nid in right_nodes:
        mesh.add_nodal_load(nid, fy=-10000.0 / len(right_nodes))

    fea = FEASolver(mesh)

    # Assembly Benchmark
    t0 = time.perf_counter()
    k_dok = fea.assemble_global_stiffness()
    t_assembly = time.perf_counter() - t0
    k_csr = k_dok.to_csr()

    # Solve Benchmark
    t0 = time.perf_counter()
    res = fea.solve(tolerance=1e-7)
    t_solve = time.perf_counter() - t0

    dof_rate = mesh.num_dofs / t_solve

    print(f"  Mesh Nodes: {mesh.num_nodes} | Elements: {mesh.num_elements} | DOFs: {mesh.num_dofs} | NNZ: {k_csr.nnz}")
    print(f"  Stiffness Assembly Time:            {t_assembly*1000:>8.2f} ms")
    print(f"  PCG Solve Time ({res.solver_iterations} iters, res={res.residual_norm:.1e}): {t_solve*1000:>8.2f} ms")
    print(f"  Effective PCG Throughput:           {dof_rate:>8,.0f} DOFs/s")
    print(f"  Max Tip Deflection:                 {res.max_displacement*1000:>8.3f} mm")
    print(f"  Peak Von Mises Stress:              {res.max_von_mises_stress/1e6:>8.2f} MPa")
    print(f"  Internal Strain Energy:             {res.strain_energy:>8.3f} J")


def benchmark_modal_eigenvalues():
    print("\n" + "=" * 72)
    print("  STRUCTURA FEA BENCHMARK 3: MODAL DYNAMICS EIGENVALUE SOLVER")
    print("=" * 72)

    # 10-bay Pratt truss bridge
    mesh = generate_truss_bridge_mesh(span=30.0, height=6.0, num_bays=10)
    mesh.add_boundary_condition(0, 0, 0.0)
    mesh.add_boundary_condition(0, 1, 0.0)
    mesh.add_boundary_condition(10, 1, 0.0)

    modal = ModalSolver(mesh)

    t0 = time.perf_counter()
    modal_res = modal.solve(num_modes=3)
    t_modal = time.perf_counter() - t0

    print(f"  Truss Nodes: {mesh.num_nodes} | Elements: {mesh.num_elements} | DOFs: {mesh.num_dofs}")
    print(f"  Modal Extraction Time (3 modes):    {t_modal*1000:>8.2f} ms")
    for mode in modal_res.modes:
        print(f"    Mode {mode.mode_number}: {mode.frequency_hz:>7.3f} Hz ({mode.angular_frequency_rad_s:>7.3f} rad/s)")


def benchmark_braille_visualizer():
    print("\n" + "=" * 72)
    print("  STRUCTURA FEA BENCHMARK 4: UNICODE BRAILLE 2x4 SUB-PIXEL RENDERING")
    print("=" * 72)

    mesh = generate_rectangular_mesh(length=8.0, height=2.0, nx=16, ny=4, material=STRUCTURAL_STEEL)
    for nid in mesh.find_nodes_at_x(0.0):
        mesh.add_boundary_condition(nid, 0, 0.0)
        mesh.add_boundary_condition(nid, 1, 0.0)
    for nid in mesh.find_nodes_at_x(8.0):
        mesh.add_nodal_load(nid, fy=-5000.0)

    fea = FEASolver(mesh)
    res = fea.solve()

    n_frames = 200

    # Benchmark deformation wireframe render
    t0 = time.perf_counter()
    for _ in range(n_frames):
        s = render_mesh_deformation(mesh, res.displacements, char_width=72, char_height=20)
    t_wire = time.perf_counter() - t0
    fps_wire = n_frames / t_wire

    # Benchmark TrueColor stress contour render
    t0 = time.perf_counter()
    for _ in range(n_frames):
        s = render_stress_field(mesh, res.nodal_von_mises, char_width=72, char_height=20)
    t_contour = time.perf_counter() - t0
    fps_contour = n_frames / t_contour

    print(f"  Braille Canvas Resolution: 144 x 80 sub-pixels (72x20 terminal cells)")
    print(f"  Deformed Wireframe Render:          {fps_wire:>8.1f} FPS ({t_wire*1000/n_frames:.2f} ms/frame)")
    print(f"  TrueColor ANSI Stress Contour:      {fps_contour:>8.1f} FPS ({t_contour*1000/n_frames:.2f} ms/frame)")
    print("=" * 72)


if __name__ == "__main__":
    benchmark_element_stiffness()
    benchmark_solver_and_assembly()
    benchmark_modal_eigenvalues()
    benchmark_braille_visualizer()
