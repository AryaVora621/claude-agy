"""
Structura: Interactive Structural Engineering & Continuum Mechanics Laboratory.
Showcases:
  1. Cantilever Beam Bending: 2D QuadQ4 continuum vs Euler-Bernoulli beam theory
  2. Pratt Truss Bridge: Live vehicle traversing bridge with dynamic deflection & modal resonance
  3. Kirsch Perforated Plate: Stress concentration around a circular hole (Kt -> 3.0 analytical)
Zero external dependencies.
"""

import sys
import time
import math
from pathlib import Path

# Ensure structura is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structura.types import Node2D, Material, STRUCTURAL_STEEL, ALUMINUM_6061
from structura.elements import Beam2D, QuadQ4, Truss2D
from structura.mesh import (
    Mesh2D,
    generate_rectangular_mesh,
    generate_truss_bridge_mesh,
    generate_perforated_plate_mesh,
)
from structura.solver import FEASolver
from structura.modal import ModalSolver
from structura.visualizer import (
    BrailleFEACanvas,
    render_mesh_deformation,
    render_stress_field,
    render_fea_telemetry_hud,
)


def run_cantilever_demo():
    print("\n" + "═" * 78)
    print("  CASE STUDY 1: CONTINUUM CANTILEVER BEAM UNDER END SHEAR LOAD")
    print("═" * 78)
    print("  Comparing 2D Isoparametric QuadQ4 continuum mesh against Euler-Bernoulli theory:")
    print("    Analytical Tip Deflection: delta = (P * L^3) / (3 * E * I)")

    l = 6.0    # Length = 6.0 m
    h = 1.0    # Height = 1.0 m
    b = 0.2    # Thickness = 0.2 m
    e = 200e9  # Steel 200 GPa
    nu = 0.3
    p_load = -50000.0  # 50 kN downward end shear load

    # Analytical section properties
    a = b * h
    i_z = (b * (h ** 3)) / 12.0
    exact_delta = (p_load * (l ** 3)) / (3.0 * e * i_z)  # In meters

    mat = Material(elastic_modulus=e, poissons_ratio=nu, thickness=b)
    mesh = generate_rectangular_mesh(length=l, height=h, nx=16, ny=4, material=mat, elem_type="quad")

    # Clamped boundary condition at x = 0
    for nid in mesh.find_nodes_at_x(0.0):
        mesh.add_boundary_condition(nid, 0, 0.0)
        mesh.add_boundary_condition(nid, 1, 0.0)

    # Parabolic or uniform shear load distributed along right edge x = L
    tip_nodes = mesh.find_nodes_at_x(l)
    load_per_node = p_load / len(tip_nodes)
    for nid in tip_nodes:
        mesh.add_nodal_load(nid, fy=load_per_node)

    fea = FEASolver(mesh)
    res = fea.solve()

    modal = ModalSolver(mesh)
    modal_res = modal.solve(num_modes=2)

    computed_tip_def = -res.max_displacement
    err_pct = abs(abs(computed_tip_def) - abs(exact_delta)) / abs(exact_delta) * 100.0

    print(f"\n  Dimensions: Length={l}m, Height={h}m, Width={b}m | Elements: {mesh.num_elements} QuadQ4")
    print(f"  Applied Tip Shear Load:        {p_load / 1e3:>8.1f} kN")
    print(f"  Analytical Euler-Bernoulli:    {exact_delta * 1000:>8.3f} mm")
    print(f"  2D Continuum FEA Tip Deflect:  {computed_tip_def * 1000:>8.3f} mm  (Discrepancy: {err_pct:.2f}%)")
    print(f"  Peak Von Mises Yield Stress:   {res.max_von_mises_stress / 1e6:>8.2f} MPa")
    print(f"  Safety Factor (Yield / Stress):{res.min_factor_of_safety:>8.2f}")
    print(f"  Fundamental Frequency (f1):    {modal_res.fundamental_frequency_hz:>8.2f} Hz")

    print("\n  [DEFORMATION WIREFRAME: Gray = Undeformed, Cyan/Color = Magnified Deflection]")
    print(render_mesh_deformation(mesh, res.displacements, char_width=74, char_height=14))

    print("\n  [VON MISES STRESS FIELD HEATMAP: 24-bit TrueColor ANSI]")
    print(render_stress_field(mesh, res.nodal_von_mises, char_width=74, char_height=14))
    print(render_fea_telemetry_hud(mesh, res, modal_res, title="CANTILEVER BEAM CONTINUUM TELEMETRY"))


def run_truss_bridge_demo():
    print("\n" + "═" * 78)
    print("  CASE STUDY 2: PRATT TRUSS HIGHWAY BRIDGE UNDER MOVING VEHICLE")
    print("═" * 78)
    print("  Simulating a 30-meter span Pratt truss bridge with moving live truck load (120 kN).")

    span = 30.0
    height = 5.0
    bays = 6
    truck_weight = -120000.0  # 120 kN downward

    # Bottom chord nodes are located at 0, 1, 2, ..., bays
    positions = [1, 2, 3, 4, 5]  # Lower chord interior pin joints

    print(f"  Span: {span} m | Height: {height} m | Bays: {bays} | Truck Load: {abs(truck_weight)/1e3:.0f} kN\n")

    for pos in positions:
        mesh = generate_truss_bridge_mesh(span=span, height=height, num_bays=bays, material=STRUCTURAL_STEEL)
        # Pin support at bottom-left (node 0)
        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        # Roller support at bottom-right (node bays)
        mesh.add_boundary_condition(bays, 1, 0.0)

        # Place truck load at current bottom node
        mesh.add_nodal_load(pos, fy=truck_weight)

        fea = FEASolver(mesh)
        res = fea.solve()

        # Find maximum tension (positive axial) and compression (negative axial)
        max_tension = max(s.sigma_x for s in res.element_stresses)
        max_compression = min(s.sigma_x for s in res.element_stresses)

        truck_x = mesh.nodes[pos].x
        print(f"  ── Truck at x = {truck_x:>4.1f} m (Joint #{pos}) ──────────────────────────────────────")
        print(f"     Max Deflection: {res.max_displacement * 1000:>6.2f} mm | Peak Tension: {max_tension / 1e6:>6.2f} MPa | Peak Comp: {max_compression / 1e6:>6.2f} MPa")

        if pos == 3:  # Midspan showcase
            modal = ModalSolver(mesh)
            m_res = modal.solve(num_modes=2)
            print("\n  [PRATT TRUSS AT MAXIMUM DEFLECTION (MIDSPAN)]")
            print(render_mesh_deformation(mesh, res.displacements, magnification=250.0, char_width=74, char_height=12))
            print(render_fea_telemetry_hud(mesh, res, m_res, title="PRATT TRUSS BRIDGE TELEMETRY (MIDSPAN LOAD)"))


def run_kirsch_perforated_plate_demo():
    print("\n" + "═" * 78)
    print("  CASE STUDY 3: KIRSCH PERFORATED PLATE STRESS CONCENTRATION (Kt -> 3.0)")
    print("═" * 78)
    print("  Infinite plate with circular hole of radius a under uniform far-field tension sigma_0.")
    print("  Analytical Kirsch solution predicts a theoretical stress concentration factor:")
    print("    sigma_max = 3.0 * sigma_0  =>  Kt = sigma_max / sigma_0 = 3.00")

    width = 4.0
    height = 4.0
    hole_radius = 0.5
    sigma_0 = 10.0e6  # 10 MPa far field tension
    thickness = 0.01   # 10 mm plate

    mesh = generate_perforated_plate_mesh(
        width=width,
        height=height,
        hole_radius=hole_radius,
        n_radial=5,
        n_tangential=12,
        material=STRUCTURAL_STEEL,
    )

    # Apply quarter-symmetry boundary conditions:
    # 1. Left symmetry edge (x = 0): Ux = 0
    left_nodes = mesh.find_nodes_at_x(0.0)
    for nid in left_nodes:
        mesh.add_boundary_condition(nid, 0, 0.0)

    # 2. Bottom symmetry edge (y = 0): Uy = 0
    bottom_nodes = mesh.find_nodes_at_y(0.0)
    for nid in bottom_nodes:
        mesh.add_boundary_condition(nid, 1, 0.0)

    # 3. Apply tensile traction on right edge (x = width): Fx = sigma_0 * Area
    right_nodes = mesh.find_nodes_at_x(width)
    total_tension_force = sigma_0 * (height * thickness)
    force_per_node = total_tension_force / max(1, len(right_nodes))
    for nid in right_nodes:
        mesh.add_nodal_load(nid, fx=force_per_node)

    fea = FEASolver(mesh, plane_strain=False)
    res = fea.solve()

    sigma_max = res.max_von_mises_stress
    kt_numerical = sigma_max / sigma_0

    print(f"\n  Plate Size: {width}x{height} m | Hole Radius: {hole_radius} m | Far-Field Stress: {sigma_0/1e6:.1f} MPa")
    print(f"  Kirsch Theoretical Peak Stress: {3.0 * sigma_0 / 1e6:>7.2f} MPa  (Kt = 3.00)")
    print(f"  Numerical FEA Peak Stress:      {sigma_max / 1e6:>7.2f} MPa  (Kt = {kt_numerical:.2f})")
    print(f"  Max Plate Extension:            {res.max_displacement * 1000:>7.3f} mm")

    print("\n  [KIRSCH STRESS CONCENTRATION AROUND HOLE: TrueColor ANSI Heatmap]")
    print(render_stress_field(mesh, res.nodal_von_mises, char_width=74, char_height=14))
    print(render_fea_telemetry_hud(mesh, res, title="KIRSCH PERFORATED PLATE TELEMETRY"))
    print("═" * 78)


def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                 STRUCTURA: FINITE ELEMENT ANALYSIS LABORATORY                ║")
    print("║   Continuum Mechanics, 2D Gauss Quadrature, Modal Dynamics & Braille HUD    ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    run_cantilever_demo()
    run_truss_bridge_demo()
    run_kirsch_perforated_plate_demo()


if __name__ == "__main__":
    main()
