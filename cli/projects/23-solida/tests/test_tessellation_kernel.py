"""
Unit tests for Solida Tessellation, STL/OBJ Exporters, and High-Level Kernel Assembly.
"""

import os
import math
import tempfile
import unittest
from solida.geometry import Vector3D
from solida.brep import build_solid_from_polygons
from solida.primitives import make_box, make_cylinder
from solida.tessellation import (
    triangulate_polygon_2d,
    triangulate_polygon_3d,
    tessellate_solid,
    export_stl_ascii,
    export_stl_binary,
    export_obj,
    parse_stl_ascii,
    parse_stl_binary,
)
from solida.visualizer import render_solid_cad, render_cad_hud
from solida.kernel import Part, Workplane, Assembly


class TestTessellationAndKernel(unittest.TestCase):
    def test_ear_clipping_triangulation(self):
        # L-shaped 2D concave polygon
        pts_l = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (1.0, 1.0),
            (1.0, 4.0),
            (0.0, 4.0),
        ]
        tris = triangulate_polygon_2d(pts_l)
        # N-vertex simple polygon triangulates into exactly N - 2 = 4 triangles
        self.assertEqual(len(tris), 4)

    def test_stl_ascii_roundtrip(self):
        box = make_box(2, 3, 4, name="TestBox")
        stl_text = export_stl_ascii(box)
        self.assertTrue(stl_text.startswith("solid TestBox"))
        self.assertTrue(stl_text.strip().endswith("endsolid TestBox"))

        facets = parse_stl_ascii(stl_text)
        # 6 quad faces triangulated = 12 facets
        self.assertEqual(len(facets), 12)

    def test_stl_binary_roundtrip(self):
        box = make_box(2, 3, 4, name="BinBox")
        stl_bytes = export_stl_binary(box, header_text="Solida Binary STL Test")

        # Header 80 bytes + 4 bytes count + 12 * 50 bytes = 684 bytes
        self.assertEqual(len(stl_bytes), 84 + 12 * 50)

        facets = parse_stl_binary(stl_bytes)
        self.assertEqual(len(facets), 12)

    def test_obj_export(self):
        box = make_box(5, 5, 5, name="ObjBox")
        obj_text = export_obj(box)
        self.assertIn("o ObjBox", obj_text)
        self.assertIn("v ", obj_text)
        self.assertIn("f ", obj_text)

    def test_part_fluent_api_and_mass(self):
        # Aluminum 6061 block: 100 x 50 x 10 mm = 50,000 mm^3 = 50 cm^3
        # Mass = 50 * 2.7 = 135.0 g
        block = Part.box(100, 50, 10, name="Plate", density_g_cm3=2.7)
        self.assertAlmostEqual(block.volume, 50000.0, places=2)
        self.assertAlmostEqual(block.mass_grams, 135.0, places=2)

        # Drill 4 mounting holes of radius 2.5 mm
        hole = Part.cylinder(2.5, 12, segments=16, center=True)
        h1 = hole.translate(35, 15, 5)
        h2 = hole.translate(-35, 15, 5)
        drilled = block - h1 - h2
        self.assertLess(drilled.volume, block.volume)
        self.assertLess(drilled.mass_grams, block.mass_grams)

    def test_assembly_and_clash_detection(self):
        asm = Assembly("BracketAssembly")
        base = Part.box(40, 40, 10, center=True, name="Base")
        asm.add_part(base, "BaseInstance")

        # Intersecting bolt
        bolt_clash = Part.cylinder(5.0, 30.0, center=True, name="BoltClash")
        asm.add_part(bolt_clash, "ClashingBolt")

        clashes = asm.check_interferences(volume_tolerance=0.1)
        self.assertEqual(len(clashes), 1)
        self.assertEqual(clashes[0]["part_a"], "BaseInstance")
        self.assertEqual(clashes[0]["part_b"], "ClashingBolt")
        self.assertGreater(clashes[0]["clash_volume_mm3"], 0.0)

    def test_visualizer_and_hud(self):
        box = make_box(10, 10, 10, center=True, name="HudBox")
        hud = render_cad_hud(box)
        self.assertIn("SOLIDA CAD MODEL TELEMETRY", hud)
        self.assertIn("1000.0000 mm3", hud)
        self.assertIn("chi=2", hud)

        braille = render_solid_cad(box, char_width=40, char_height=10, mode="wireframe", use_ansi_color=False)
        self.assertIsInstance(braille, str)
        self.assertGreater(len(braille), 50)

        # Test shaded mode
        shaded = render_solid_cad(box, char_width=40, char_height=10, mode="shaded", use_ansi_color=True)
        self.assertIn("\033[38;2;", shaded)

    def test_workplane_parametric_extrusion(self):
        wp = Workplane.XY(z_offset=10.0)
        boss = wp.extrude_circle(radius=4.0, depth=5.0, segments=24, name="WpBoss")
        self.assertAlmostEqual(boss.volume, math.pi * 16.0 * 5.0, delta=20.0)

        wp_rect = Workplane.XZ(y_offset=-2.0)
        rect_part = wp_rect.extrude_rect(width=10.0, height=4.0, depth=6.0, center=True)
        self.assertAlmostEqual(rect_part.volume, 240.0, places=2)

    def test_assembly_composite_obj_export(self):
        asm = Assembly("TestCompound")
        b1 = Part.box(10, 10, 10, name="B1")
        b2 = Part.box(5, 5, 5, name="B2").translate(15, 0, 0)
        asm.add_part(b1, "Part1")
        asm.add_part(b2, "Part2")

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tf:
            path = tf.name
        try:
            byte_count = asm.export_composite_obj(path)
            self.assertGreater(byte_count, 0)
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("o Part1", content)
            self.assertIn("o Part2", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_stl_binary_invalid_buffer(self):
        with self.assertRaises(ValueError):
            parse_stl_binary(b"too_short")


if __name__ == "__main__":
    unittest.main()
