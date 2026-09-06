"""
Unit tests for OrbitMech 3D Camera Projection & Braille Terminal Visualizer.
"""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
)
from orbitmech.kepler import orbital_elements_to_state
from orbitmech.visualizer import (
    BrailleCanvas,
    Camera3D,
    render_orbit_scene,
    render_porkchop_contour,
    render_mission_telemetry_hud,
)


class TestVisualizer(unittest.TestCase):
    def test_braille_canvas_bits(self):
        canvas = BrailleCanvas(char_width=10, char_height=5)
        # Set top-left pixel (0, 0)
        canvas.set_pixel(0, 0)
        lines = canvas.render()
        self.assertEqual(len(lines), 5)
        # First cell should have dot 1 set -> chr(0x2800 + 1) = '⠁'
        self.assertEqual(lines[0][0], "⠁")

    def test_camera_3d_projection(self):
        cam = Camera3D(azimuth_deg=0.0, elevation_deg=0.0, scale=100.0)
        pt = Vector3(0.0, 500.0, 0.0)
        # Center at (100, 100)
        px, py, zc = cam.project(pt, center_x=100, center_y=100)
        # Point along +Y with azimuth=0 maps to xc = 500 -> px = 100 + 500/100 = 105
        self.assertEqual(px, 105)
        self.assertEqual(py, 100)

    def test_render_orbit_scene(self):
        # Create a circle of state vectors
        states = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            r = Vector3(7000.0 * math.cos(rad), 7000.0 * math.sin(rad), 0.0)
            v = Vector3(-7.5 * math.sin(rad), 7.5 * math.cos(rad), 0.0)
            states.append(StateVector(r=r, v=v))

        scene = render_orbit_scene(states, central_body=EARTH, canvas_chars_x=40, canvas_chars_y=15)
        self.assertIn("┌", scene)
        self.assertIn("└", scene)
        self.assertGreater(len(scene.splitlines()), 10)

    def test_render_mission_telemetry_hud(self):
        elements = ClassicalOrbitalElements(
            a=6800.0,
            e=0.01,
            i=math.radians(28.5),
            raan=0.0,
            arg_peri=0.0,
            true_anomaly=math.radians(45.0),
            mu=EARTH.mu,
        )
        state = orbital_elements_to_state(elements)
        hud = render_mission_telemetry_hud(state, elements, central_body=EARTH)
        self.assertIn("MISSION TELEMETRY HUD", hud)
        self.assertIn("6,800.00 km", hud)
        self.assertIn("28.500°", hud)


if __name__ == "__main__":
    unittest.main()
