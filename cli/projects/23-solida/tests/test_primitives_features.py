"""
Unit tests for Solida 3D Primitives and CAD Feature Sweeps.
"""

import math
import unittest
from solida.geometry import Vector3D
from solida.primitives import (
    make_box,
    make_cylinder,
    make_sphere,
    make_cone,
    make_torus,
)
from solida.features import (
    Sketch2D,
    extrude,
    revolve,
    loft,
)


class TestPrimitivesAndFeatures(unittest.TestCase):
    def test_primitives_volume_and_euler(self):
        # 1. Box
        box = make_box(10, 20, 30, center=True)
        self.assertAlmostEqual(box.volume(), 6000.0, places=3)
        self.assertEqual(box.outer_shell.euler_characteristic(), 2)

        # 2. Cylinder
        r, h = 4.0, 10.0
        cyl = make_cylinder(r, h, segments=32)
        theo_cyl_vol = math.pi * (r**2) * h
        # Discretization with 32 segments is within 1% of continuous circle
        self.assertAlmostEqual(cyl.volume(), theo_cyl_vol, delta=theo_cyl_vol * 0.02)
        self.assertEqual(cyl.outer_shell.euler_characteristic(), 2)

        # 3. Sphere
        r_sph = 5.0
        sph = make_sphere(r_sph, rings=16, sectors=32)
        theo_sph_vol = (4.0 / 3.0) * math.pi * (r_sph**3)
        self.assertAlmostEqual(sph.volume(), theo_sph_vol, delta=theo_sph_vol * 0.05)
        self.assertEqual(sph.outer_shell.euler_characteristic(), 2)

        # 4. Cone
        r_cone, h_cone = 6.0, 15.0
        cone = make_cone(r_cone, h_cone, segments=32)
        theo_cone_vol = (1.0 / 3.0) * math.pi * (r_cone**2) * h_cone
        self.assertAlmostEqual(cone.volume(), theo_cone_vol, delta=theo_cone_vol * 0.02)
        self.assertEqual(cone.outer_shell.euler_characteristic(), 2)

        # 5. Torus (Genus-1: Euler chi = 0)
        R_tor, r_tor = 8.0, 2.0
        torus = make_torus(R_tor, r_tor, major_segs=24, minor_segs=16)
        theo_tor_vol = 2.0 * (math.pi**2) * R_tor * (r_tor**2)
        self.assertAlmostEqual(torus.volume(), theo_tor_vol, delta=theo_tor_vol * 0.05)
        self.assertEqual(torus.outer_shell.euler_characteristic(), 0)

    def test_sketch2d_profiles(self):
        # Rectangle
        rect = Sketch2D.rectangle(10, 5, center=True)
        self.assertEqual(len(rect), 4)

        # Circle
        circ = Sketch2D.circle(5.0, segments=24)
        self.assertEqual(len(circ), 24)

        # Regular Hexagon
        hex_sketch = Sketch2D.regular_polygon(6.0, sides=6)
        self.assertEqual(len(hex_sketch), 6)

        # Star
        star = Sketch2D.star(outer_radius=10.0, inner_radius=4.0, points=5)
        self.assertEqual(len(star), 10)

        # NACA 2412 Airfoil
        foil = Sketch2D.naca_airfoil(code="2412", chord=2.0, num_points=20)
        self.assertGreater(len(foil), 30)
        # Leading edge near (0, 0)
        self.assertAlmostEqual(foil[0].x, 0.0, delta=0.05)

    def test_extrude_with_draft_and_twist(self):
        rect = Sketch2D.rectangle(10, 10, center=True)

        # Standard linear extrusion
        solid_ext = extrude(rect, height=20.0, steps=1)
        self.assertAlmostEqual(solid_ext.volume(), 2000.0, places=3)
        self.assertEqual(solid_ext.outer_shell.euler_characteristic(), 2)

        # Extrusion with draft angle (tapered)
        solid_draft = extrude(rect, height=10.0, draft_angle_deg=5.0, steps=5)
        # Tapered volume is smaller than straight volume
        self.assertLess(solid_draft.volume(), 1000.0)

        # Extrusion with twist
        solid_twist = extrude(rect, height=15.0, twist_angle_deg=45.0, steps=10)
        self.assertGreater(solid_twist.volume(), 0.0)

    def test_revolve_and_loft(self):
        # Revolve circular profile in XZ plane around Z axis -> Torus
        circ = Sketch2D.circle(radius=2.0, segments=16).translate(6.0, 0.0, 0.0).to_plane("XZ")
        rev_solid = revolve(circ, angle_deg=360.0, segments=16)
        theo_vol = 2.0 * (math.pi**2) * 6.0 * (2.0**2)
        self.assertAlmostEqual(rev_solid.volume(), theo_vol, delta=theo_vol * 0.1)

        # Loft between two scaled rectangles
        s1 = Sketch2D.rectangle(10, 10, center=True)
        s2 = Sketch2D.rectangle(5, 5, center=True).translate(0, 0, 15)
        loft_solid = loft([s1, s2])
        # Truncated pyramid volume: h/3 * (A1 + A2 + sqrt(A1*A2)) = 15/3 * (100 + 25 + 50) = 5 * 175 = 875
        self.assertAlmostEqual(loft_solid.volume(), 875.0, places=3)


if __name__ == "__main__":
    unittest.main()
