"""
Unit tests for Structura BrailleFEACanvas, deformation wireframe, stress contours, and telemetry HUD.
"""

import unittest
from structura.types import Node2D, Material, STRUCTURAL_STEEL
from structura.elements import Beam2D, QuadQ4
from structura.mesh import Mesh2D, generate_rectangular_mesh
from structura.solver import FEASolver
from structura.modal import ModalSolver
from structura.visualizer import (
    BrailleFEACanvas,
    turbo_colormap,
    render_mesh_deformation,
    render_stress_field,
    render_fea_telemetry_hud,
)


class TestVisualizer(unittest.TestCase):
    def test_turbo_colormap(self):
        # Boundaries: 0.0 -> blue-ish, 1.0 -> red-ish
        r0, g0, b0 = turbo_colormap(0.0)
        self.assertLess(r0, 50)
        self.assertGreater(b0, 150)

        r1, g1, b1 = turbo_colormap(1.0)
        self.assertGreater(r1, 200)
        self.assertLess(b1, 50)

        # Clamping
        r_neg, _, _ = turbo_colormap(-0.5)
        self.assertEqual(r_neg, r0)
        r_pos, _, _ = turbo_colormap(1.5)
        self.assertEqual(r_pos, r1)

    def test_braille_canvas_pixels_and_lines(self):
        canvas = BrailleFEACanvas(char_width=20, char_height=10)
        self.assertEqual(canvas.pixel_width, 40)
        self.assertEqual(canvas.pixel_height, 40)

        # Draw a diagonal line
        canvas.draw_line(0, 0, 39, 39, color_rgb=(255, 100, 50))
        lines = canvas.render(use_color=False)
        self.assertEqual(len(lines), 10)
        for line in lines:
            self.assertEqual(len(line), 20)

        # Ensure non-empty braille characters exist
        total_braille_chars = sum(c != chr(0x2800) for line in lines for c in line)
        self.assertGreater(total_braille_chars, 0)

        # Out-of-bounds pixel doesn't crash
        canvas.set_pixel(-5, -5)
        canvas.set_pixel(100, 100)

    def test_render_mesh_deformation(self):
        # 2-element cantilever beam
        mesh = Mesh2D()
        mesh.add_node(0.0, 0.0)
        mesh.add_node(1.0, 0.0)
        mesh.add_node(2.0, 0.0)

        mat = Material(elastic_modulus=200e9, poissons_ratio=0.3)
        mesh.add_element(Beam2D(0, (0, 1), mat, cross_section_area=0.01, moment_of_inertia=1e-4))
        mesh.add_element(Beam2D(1, (1, 2), mat, cross_section_area=0.01, moment_of_inertia=1e-4))

        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        mesh.add_boundary_condition(0, 2, 0.0)
        mesh.add_nodal_load(2, fy=-5000.0)

        solver = FEASolver(mesh)
        res = solver.solve()

        output = render_mesh_deformation(mesh, res.displacements, char_width=40, char_height=10)
        self.assertIn("Deformation Scale", output)
        self.assertIn("┌", output)
        self.assertIn("└", output)

    def test_render_stress_field(self):
        # Rectangular quad mesh
        mesh = generate_rectangular_mesh(length=4.0, height=1.0, nx=4, ny=2, material=STRUCTURAL_STEEL)
        for nid in mesh.find_nodes_at_x(0.0):
            mesh.add_boundary_condition(nid, 0, 0.0)
            mesh.add_boundary_condition(nid, 1, 0.0)
        for nid in mesh.find_nodes_at_x(4.0):
            mesh.add_nodal_load(nid, fy=-1000.0)

        solver = FEASolver(mesh)
        res = solver.solve()

        stress_art = render_stress_field(mesh, res.nodal_von_mises, char_width=40, char_height=10)
        self.assertIn("Peak", stress_art)
        self.assertIn("Low", stress_art)

    def test_render_fea_telemetry_hud(self):
        mesh = generate_rectangular_mesh(length=2.0, height=0.5, nx=4, ny=2, material=STRUCTURAL_STEEL)
        for nid in mesh.find_nodes_at_x(0.0):
            mesh.add_boundary_condition(nid, 0, 0.0)
            mesh.add_boundary_condition(nid, 1, 0.0)
        for nid in mesh.find_nodes_at_x(2.0):
            mesh.add_nodal_load(nid, fy=-500.0)

        fea = FEASolver(mesh)
        res = fea.solve()

        modal = ModalSolver(mesh)
        m_res = modal.solve(num_modes=2)

        hud = render_fea_telemetry_hud(mesh, res, m_res, title="TEST HUD")
        self.assertIn("TEST HUD", hud)
        self.assertIn("Max Deflection", hud)
        self.assertIn("Peak Stress", hud)
        self.assertIn("Resonance (f1)", hud)
        self.assertIn("Safety Factor", hud)


if __name__ == "__main__":
    unittest.main()
