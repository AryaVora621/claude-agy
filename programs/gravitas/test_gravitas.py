"""Automated unit and integration test suite for Gravitas 3D Desktop Studio.

Validates symplectic Yoshida integrator conservation, collision inelastic mechanics,
presets, 3D camera projection math, and headless Tkinter UI lifecycle.
"""

import math
import unittest
import os
import sys
import tkinter as tk

# Ensure local module directory and workspace root are in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from physics import GravitasPhysics, Body
    from presets import PRESETS, setup_figure_eight, setup_sol_system
    from gravitas import Camera3D, GravitasApp
except ImportError:
    from programs.gravitas.physics import GravitasPhysics, Body
    from programs.gravitas.presets import PRESETS, setup_figure_eight, setup_sol_system
    from programs.gravitas.gravitas import Camera3D, GravitasApp


class TestGravitasPhysics(unittest.TestCase):
    """Test suite for N-body gravitational physics engine."""

    def test_single_body_energy(self):
        sim = GravitasPhysics()
        b = sim.add_body("TestBody", mass=10.0, radius=1.0, color="#FFF", pos=[0, 0, 0], vel=[3, 4, 0])
        self.assertAlmostEqual(b.speed(), 5.0)
        self.assertAlmostEqual(b.kinetic_energy(), 0.5 * 10.0 * 25.0)
        self.assertEqual(len(sim.bodies), 1)

    def test_circular_orbit_conservation(self):
        """Verify two-body circular Keplerian orbit maintains constant radius and energy."""
        sim = GravitasPhysics(g_constant=1.0, softening=0.0)
        m_star = 1000.0
        m_planet = 0.01
        r = 10.0
        v_circ = math.sqrt(sim.G * m_star / r)

        sim.add_body("Star", m_star, 1.0, "#F59E0B", [0, 0, 0], [0, 0, 0], fixed=True)
        sim.add_body("Planet", m_planet, 0.5, "#06B6D4", [r, 0, 0], [0, v_circ, 0])

        e0 = sim.total_energy()
        dt = 0.01
        # Integrate for ~100 steps
        for _ in range(100):
            sim.step_yoshida(dt)

        p = sim.bodies[1]
        dist = math.sqrt(p.pos[0]**2 + p.pos[1]**2 + p.pos[2]**2)
        # Distance should remain very close to 10.0
        self.assertAlmostEqual(dist, 10.0, delta=0.05)

        e_final = sim.total_energy()
        drift = abs((e_final - e0) / e0)
        self.assertLess(drift, 1e-4)

    def test_figure_eight_choreography_energy_drift(self):
        """Verify the 3-body Figure-Eight choreography conserves total energy."""
        sim = GravitasPhysics()
        setup_figure_eight(sim)
        self.assertEqual(len(sim.bodies), 3)

        e0 = sim.total_energy()
        dt = 0.005
        for _ in range(120):
            sim.step_yoshida(dt)

        e_final = sim.total_energy()
        drift = abs((e_final - e0) / e0)
        # Yoshida 4th-order symplectic integrator preserves Hamiltonian with negligible drift
        self.assertLess(drift, 5e-4)

    def test_inelastic_collision_momentum_conservation(self):
        """Verify colliding bodies merge into one with exact linear momentum conservation."""
        sim = GravitasPhysics(g_constant=0.0, softening=0.1)  # turn off gravity to isolate collision
        sim.enable_collisions = True

        # Body 1: moving right (+X)
        sim.add_body("A", mass=10.0, radius=1.0, color="#F00", pos=[-0.5, 0, 0], vel=[2.0, 0, 0])
        # Body 2: moving left (-X)
        sim.add_body("B", mass=5.0, radius=1.0, color="#0F0", pos=[0.5, 0, 0], vel=[-1.0, 0, 0])

        # Initial net momentum: 10*2 + 5*(-1) = 20 - 5 = 15.0
        p_init = sim.linear_momentum()
        self.assertAlmostEqual(p_init[0], 15.0)

        # Distance is 1.0, radius sum is 2.0 -> collision should trigger on step
        sim.step_verlet(0.01)

        self.assertEqual(len(sim.bodies), 1)
        merged = sim.bodies[0]
        self.assertAlmostEqual(merged.mass, 15.0)
        self.assertAlmostEqual(merged.vel[0], 15.0 / 15.0)  # v = 1.0
        self.assertEqual(len(sim.collision_events), 1)

    def test_relativistic_precession_flag(self):
        """Verify enabling relativity adds post-Newtonian correction."""
        sim = GravitasPhysics(g_constant=1.0, softening=0.01, speed_of_light=20.0)
        sim.enable_relativity = False
        sim.add_body("Central", 1000.0, 1.0, "#000", [0, 0, 0], [0, 0, 0], fixed=True)
        sim.add_body("Orbiting", 1.0, 0.5, "#FFF", [5.0, 0, 0], [0, 0, 0])

        sim.compute_accelerations()
        newton_acc = sim.bodies[1].acc[0]

        sim.enable_relativity = True
        sim.compute_accelerations()
        rel_acc = sim.bodies[1].acc[0]

        # Relativistic acceleration should be strictly larger in magnitude towards center
        self.assertLess(rel_acc, newton_acc)


class TestCamera3D(unittest.TestCase):
    """Test suite for 3D perspective projection and raycasting."""

    def test_camera_projection(self):
        cam = Camera3D(distance=50.0, azimuth=0.0, elevation=0.0)
        sx, sy, cz, sc = cam.project(0.0, 0.0, 0.0, width=800, height=600)
        self.assertIsNotNone(sx)
        self.assertIsNotNone(sy)
        self.assertAlmostEqual(sx, 400.0)
        self.assertAlmostEqual(sy, 300.0)

    def test_ground_unprojection(self):
        cam = Camera3D(distance=60.0, azimuth=0.0, elevation=45.0)
        # Center of screen should map to target origin on ground
        gx, gy = cam.screen_to_world_ground(400, 300, width=800, height=600)
        self.assertAlmostEqual(gx, 0.0, delta=0.5)
        self.assertAlmostEqual(gy, 0.0, delta=0.5)


class TestGravitasAppUI(unittest.TestCase):
    """Test suite for Gravitas Desktop GUI controls and state management."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Offscreen / headless
        self.app = GravitasApp(self.root)
        self.app._loop_running = False  # prevent continuous loop during discrete tests

    def tearDown(self):
        self.app.on_close()

    def test_initial_state_and_presets(self):
        self.assertEqual(len(self.app.sim.bodies), 3)  # Figure-8 default
        self.assertTrue(self.app.running)

        # Switch to Sol System
        self.app.load_preset("Sol System Core")
        self.assertEqual(self.app.sim.bodies[0].name, "Sol")
        self.assertGreater(len(self.app.sim.bodies), 5)

    def test_play_pause_and_step(self):
        self.assertTrue(self.app.running)
        self.app.toggle_play()
        self.assertFalse(self.app.running)

        # Single step advances time
        t0 = self.app.sim.sim_time
        self.app.single_step()
        self.assertGreater(self.app.sim.sim_time, t0)

    def test_sling_mode_toggle(self):
        self.assertFalse(self.app.sling_mode)
        self.app.toggle_sling_mode()
        self.assertTrue(self.app.sling_mode)
        self.app.toggle_sling_mode()
        self.assertFalse(self.app.sling_mode)


if __name__ == "__main__":
    unittest.main()
