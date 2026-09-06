"""Comprehensive unit tests for RetroCAD 3D CAD Modeler and CSG Studio."""

import math
import unittest
import sys
import os
import tkinter as tk

# Ensure local module directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geom import (
    Vec3, Mat4, Mesh, Face, create_box, create_cylinder, create_sphere,
    create_cone, create_torus, extrude_polygon, revolve_profile, csg_boolean,
    export_stl_ascii, export_obj, export_dxf_r12, export_svg_wireframe
)
from presets import (
    PRESET_BUILDERS, build_bearing_housing, build_rocket_nozzle,
    build_spur_gear, build_hex_bolt, build_aerospace_bracket
)


class TestVectorMath(unittest.TestCase):
    """Test 3D vector arithmetic and Euclidean operations."""

    def test_vector_basic_ops(self):
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)

        # Addition and Subtraction
        v_add = v1 + v2
        self.assertEqual((v_add.x, v_add.y, v_add.z), (5.0, 7.0, 9.0))
        v_sub = v2 - v1
        self.assertEqual((v_sub.x, v_sub.y, v_sub.z), (3.0, 3.0, 3.0))

        # Scalar multiplication and division
        v_mul = v1 * 2.5
        self.assertEqual((v_mul.x, v_mul.y, v_mul.z), (2.5, 5.0, 7.5))
        v_div = v1 / 2.0
        self.assertEqual((v_div.x, v_div.y, v_div.z), (0.5, 1.0, 1.5))

    def test_dot_and_cross_product(self):
        vx = Vec3(1.0, 0.0, 0.0)
        vy = Vec3(0.0, 1.0, 0.0)

        # Dot product of orthogonal vectors is zero
        self.assertAlmostEqual(vx.dot(vy), 0.0)

        # Cross product: X x Y = Z
        vz = vx.cross(vy)
        self.assertAlmostEqual(vz.x, 0.0)
        self.assertAlmostEqual(vz.y, 0.0)
        self.assertAlmostEqual(vz.z, 1.0)

    def test_magnitude_and_normalization(self):
        v = Vec3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.magnitude(), 5.0)
        vn = v.normalize()
        self.assertAlmostEqual(vn.magnitude(), 1.0)
        self.assertAlmostEqual(vn.x, 0.6)
        self.assertAlmostEqual(vn.y, 0.8)


class TestMatrixTransforms(unittest.TestCase):
    """Test 4x4 homogeneous transformation matrix operations."""

    def test_identity_transform(self):
        m = Mat4.identity()
        p = Vec3(2.0, 3.0, 4.0)
        res = m.transform_point(p)
        self.assertAlmostEqual(res.x, 2.0)
        self.assertAlmostEqual(res.y, 3.0)
        self.assertAlmostEqual(res.z, 4.0)

    def test_translation_and_scale(self):
        t = Mat4.translation(10.0, -5.0, 20.0)
        s = Mat4.scale(2.0, 3.0, 4.0)
        # Combined s followed by t
        mat = t.mul(s)
        p = Vec3(1.0, 2.0, 3.0)
        res = mat.transform_point(p)
        # s * p = (2, 6, 12); + t = (12, 1, 32)
        self.assertAlmostEqual(res.x, 12.0)
        self.assertAlmostEqual(res.y, 1.0)
        self.assertAlmostEqual(res.z, 32.0)

    def test_rotation_z_90_deg(self):
        rot = Mat4.rotation_z(math.pi / 2.0)
        p = Vec3(1.0, 0.0, 0.0)
        res = rot.transform_point(p)
        self.assertAlmostEqual(res.x, 0.0, places=5)
        self.assertAlmostEqual(res.y, 1.0, places=5)
        self.assertAlmostEqual(res.z, 0.0, places=5)


class TestGeometricPrimitives(unittest.TestCase):
    """Test solid modeling primitives topology and volumetric metrics."""

    def test_box_primitive(self):
        w, h, d = 10.0, 20.0, 30.0
        box = create_box(w, h, d)
        self.assertEqual(len(box.vertices), 8)
        self.assertEqual(len(box.faces), 6)
        self.assertEqual(len(box.edges), 12)

        # Surface area: 2 * (10*20 + 10*30 + 20*30) = 2 * (200 + 300 + 600) = 2200
        self.assertAlmostEqual(box.surface_area(), 2200.0, places=3)
        # Volume: 10 * 20 * 30 = 6000
        self.assertAlmostEqual(box.volume(), 6000.0, places=3)

    def test_cylinder_primitive(self):
        r, h = 5.0, 10.0
        cyl = create_cylinder(radius=r, height=h, segments=32)
        self.assertEqual(len(cyl.vertices), 64)
        # 32 side quads + 2 caps = 34 faces
        self.assertEqual(len(cyl.faces), 34)

        expected_vol = math.pi * (r ** 2) * h
        self.assertAlmostEqual(cyl.volume(), expected_vol, delta=expected_vol * 0.02)

    def test_sphere_primitive(self):
        r = 10.0
        sphere = create_sphere(radius=r, rings=16, sectors=24)
        expected_vol = (4.0 / 3.0) * math.pi * (r ** 3)
        self.assertAlmostEqual(sphere.volume(), expected_vol, delta=expected_vol * 0.05)

    def test_cone_and_torus(self):
        cone = create_cone(radius=10.0, height=20.0, segments=20)
        self.assertEqual(len(cone.vertices), 21)
        expected_vol = (1.0 / 3.0) * math.pi * (10.0 ** 2) * 20.0
        self.assertAlmostEqual(cone.volume(), expected_vol, delta=expected_vol * 0.05)

        torus = create_torus(r_major=30.0, r_minor=5.0, seg_major=20, seg_minor=10)
        self.assertEqual(len(torus.vertices), 200)
        self.assertEqual(len(torus.faces), 200)


class TestParametricFeatures(unittest.TestCase):
    """Test linear extrusion and rotational revolve features."""

    def test_linear_extrusion(self):
        # 10x10 square
        contour = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        height = 15.0
        prism = extrude_polygon(contour, height=height)
        self.assertEqual(len(prism.vertices), 8)
        self.assertEqual(len(prism.faces), 6)
        expected_vol = 10.0 * 10.0 * 15.0
        self.assertAlmostEqual(prism.volume(), expected_vol, places=3)

    def test_rotational_revolve(self):
        # Revolve a vertical line at radius 10 to form a cylindrical shell
        profile = [(10.0, 0.0), (10.0, 20.0)]
        shell = revolve_profile(profile, segments=16, angle_deg=360.0)
        self.assertEqual(len(shell.vertices), 32)
        self.assertEqual(len(shell.faces), 16)


class TestCSGBooleans(unittest.TestCase):
    """Test Constructive Solid Geometry operations."""

    def test_csg_union(self):
        b1 = create_box(20.0, 20.0, 20.0, "Box1")
        b2 = create_box(20.0, 20.0, 20.0, "Box2")
        b2.transform(Mat4.translation(40.0, 0.0, 0.0))  # Disjoint
        u = csg_boolean(b1, b2, "union")
        self.assertEqual(len(u.vertices), 16)
        self.assertEqual(len(u.faces), 12)

    def test_csg_difference(self):
        b1 = create_box(20.0, 20.0, 20.0, "Box1")
        b2 = create_box(10.0, 10.0, 30.0, "HoleCutter")
        diff = csg_boolean(b1, b2, "difference")
        self.assertGreater(len(diff.faces), 6)


class TestCADFileExporters(unittest.TestCase):
    """Test STL, OBJ, DXF, and SVG serialization."""

    def setUp(self):
        self.box = create_box(10.0, 10.0, 10.0, "TestCube")

    def test_stl_ascii_export(self):
        stl_str = export_stl_ascii(self.box)
        self.assertTrue(stl_str.startswith("solid TestCube"))
        self.assertTrue(stl_str.strip().endswith("endsolid TestCube"))
        self.assertIn("facet normal", stl_str)
        self.assertIn("vertex 5.000000e+00", stl_str)

    def test_obj_export(self):
        obj_str = export_obj(self.box)
        self.assertIn("o TestCube", obj_str)
        self.assertIn("v 5.000000", obj_str)
        self.assertIn("vn 0.000000", obj_str)
        self.assertIn("f ", obj_str)

    def test_dxf_r12_export(self):
        dxf_str = export_dxf_r12(self.box)
        self.assertIn("AC1009", dxf_str)
        self.assertIn("3DFACE", dxf_str)
        self.assertIn("EOF", dxf_str)

    def test_svg_export(self):
        svg_str = export_svg_wireframe(self.box)
        self.assertIn("<svg", svg_str)
        self.assertIn("</svg>", svg_str)
        self.assertIn("<line x1=", svg_str)


class TestMechanicalPresets(unittest.TestCase):
    """Test all curated engineering mechanical presets."""

    def test_bearing_housing(self):
        part = build_bearing_housing()
        self.assertGreater(len(part.vertices), 50)
        self.assertGreater(part.surface_area(), 1000.0)

    def test_rocket_nozzle(self):
        nozzle = build_rocket_nozzle()
        self.assertGreater(len(nozzle.vertices), 100)
        self.assertGreater(nozzle.surface_area(), 5000.0)

    def test_spur_gear(self):
        gear = build_spur_gear(teeth=12)
        self.assertGreater(len(gear.faces), 20)

    def test_hex_bolt(self):
        bolt = build_hex_bolt()
        self.assertGreater(len(bolt.vertices), 30)

    def test_aerospace_bracket(self):
        bracket = build_aerospace_bracket()
        self.assertGreater(len(bracket.vertices), 20)


class TestRetroCADGUI(unittest.TestCase):
    """Test Tkinter desktop application initialization and lifecycle."""

    def test_gui_headless_lifecycle(self):
        try:
            from retrocad import RetroCADApp
            root = tk.Tk()
            root.withdraw()
            app = RetroCADApp(root)

            # Test camera rotations
            app.cam_yaw = 0.5
            app.cam_pitch = 0.3
            app.redraw()

            # Test preset loading
            app.load_preset("Rocket Engine Nozzle")
            self.assertEqual(app.active_mesh.name, "RocketNozzle")

            # Test primitive addition
            app.add_primitive("box")
            self.assertEqual(app.active_mesh.name, "Box40")

            # Test mode change
            app.mode_var.set("Phosphor CRT")
            app._on_mode_change()
            self.assertEqual(app.render_mode, "Phosphor CRT")

            app.on_close()
        except tk.TclError:
            # Gracefully handle display-less CI
            pass


if __name__ == "__main__":
    unittest.main()
