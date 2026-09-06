"""
Automated Unit Tests for AstroEphemeris Astrodynamics Engine
Zero external dependencies, standard library unittest only.
Verifies Keplerian transformations, 1PN relativistic corrections,
J2 oblateness, CR3BP Lagrange libration points, Jacobi integrals,
Hohmann transfer orbits, and desktop GUI lifecycle.
"""

import math
import os
import sys
import unittest

# Ensure local directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ephemeris import (
    ASTRONOMICAL_UNIT,
    GRAVITATIONAL_CONSTANT,
    MU_SUN,
    SOLAR_MASS,
    EARTH_MASS,
    OrbitalElements,
    CartesianState,
    CelestialBody,
    Spacecraft,
    CR3BPModel,
    SolarSystemEngine,
    solve_kepler_equation,
    orbital_elements_to_cartesian,
    cartesian_to_orbital_elements,
    compute_general_relativistic_acceleration,
    compute_j2_oblateness_acceleration,
    compute_hohmann_transfer
)
from presets import PRESETS, load_preset, get_preset_list


class TestAstroEphemerisCore(unittest.TestCase):

    def test_kepler_equation_solver(self):
        """Verifies Newton-Halley solution to M = E - e*sin(E)."""
        # Circular orbit: E = M
        e_circ = solve_kepler_equation(math.pi * 0.5, 0.0)
        self.assertAlmostEqual(e_circ, math.pi * 0.5, places=8)

        # Mild eccentricity (Earth e = 0.0167)
        m_val = 1.2
        e_earth = 0.0167
        e_sol = solve_kepler_equation(m_val, e_earth)
        m_reconstructed = e_sol - e_earth * math.sin(e_sol)
        self.assertAlmostEqual(m_reconstructed, m_val, places=10)

        # High eccentricity (e = 0.85)
        e_high = 0.85
        for test_m in [0.2, 1.5, 3.14, 5.0]:
            e_res = solve_kepler_equation(test_m, e_high)
            m_rec = e_res - e_high * math.sin(e_res)
            self.assertAlmostEqual(m_rec, test_m, places=9)

    def test_orbital_elements_to_cartesian_roundtrip(self):
        """Verifies state vector conversion to orbital elements and back."""
        orig_elem = OrbitalElements(
            a=1.5 * ASTRONOMICAL_UNIT,
            e=0.15,
            i=math.radians(12.5),
            raan=math.radians(45.0),
            arg_p=math.radians(30.0),
            true_anomaly=math.radians(60.0)
        )

        state = orbital_elements_to_cartesian(orig_elem, MU_SUN)
        self.assertTrue(state.position_magnitude > 0.0)
        self.assertTrue(state.speed > 0.0)

        # Convert back
        recovered = cartesian_to_orbital_elements(state, MU_SUN)
        self.assertAlmostEqual(recovered.a / ASTRONOMICAL_UNIT, orig_elem.a / ASTRONOMICAL_UNIT, places=4)
        self.assertAlmostEqual(recovered.e, orig_elem.e, places=4)
        self.assertAlmostEqual(math.degrees(recovered.i), math.degrees(orig_elem.i), places=4)
        self.assertAlmostEqual(math.degrees(recovered.raan), math.degrees(orig_elem.raan), places=3)
        self.assertAlmostEqual(math.degrees(recovered.arg_p), math.degrees(orig_elem.arg_p), places=3)
        self.assertAlmostEqual(math.degrees(recovered.true_anomaly), math.degrees(orig_elem.true_anomaly), places=3)

    def test_vis_viva_orbital_energy(self):
        """Verifies Vis-Viva equation v^2 = mu * (2/r - 1/a) holds on state vectors."""
        elem = OrbitalElements(
            a=1.0 * ASTRONOMICAL_UNIT,
            e=0.20,
            i=math.radians(5.0),
            raan=0.0,
            arg_p=0.0,
            true_anomaly=math.radians(90.0)
        )
        state = orbital_elements_to_cartesian(elem, MU_SUN)
        r = state.position_magnitude
        v = state.speed

        vis_viva_v = math.sqrt(MU_SUN * (2.0 / r - 1.0 / elem.a))
        self.assertAlmostEqual(v, vis_viva_v, delta=1.0)

    def test_relativistic_1pn_acceleration(self):
        """Verifies 1PN Post-Newtonian acceleration vectors."""
        r = (0.387 * ASTRONOMICAL_UNIT, 0.0, 0.0)
        v = (0.0, 47000.0, 0.0) # ~47 km/s Mercury speed

        ax, ay, az = compute_general_relativistic_acceleration(r, v, MU_SUN, scale=1.0)
        # Relativistic correction is non-zero and points inward along radial axis
        self.assertNotEqual(ax, 0.0)
        self.assertAlmostEqual(az, 0.0, places=12) # In-plane

        # Verify scale multiplier works linearly
        ax_scaled, _, _ = compute_general_relativistic_acceleration(r, v, MU_SUN, scale=100.0)
        self.assertAlmostEqual(ax_scaled, ax * 100.0, places=5)

    def test_j2_oblateness_perturbation(self):
        """Verifies J2 gravitational perturbation acceleration computation."""
        r_orbit = 7000e3 # 7000 km low Earth orbit
        r_vec = (r_orbit, 0.0, 1000e3)
        earth_mu = GRAVITATIONAL_CONSTANT * EARTH_MASS
        r_eq = 6.378e6
        j2 = 1.08263e-3

        ax, ay, az = compute_j2_oblateness_acceleration(r_vec, earth_mu, r_eq, j2)
        # J2 creates a restoring force in Z towards equator
        self.assertTrue(az < 0.0)
        self.assertNotEqual(ax, 0.0)

    def test_cr3bp_lagrange_points_geometry(self):
        """Verifies analytical coordinates of the 5 CR3BP Lagrangian points."""
        mu = 0.01215 # Earth-Moon
        model = CR3BPModel(mu)
        pts = model.lagrange_points

        self.assertIn("L1", pts)
        self.assertIn("L2", pts)
        self.assertIn("L3", pts)
        self.assertIn("L4", pts)
        self.assertIn("L5", pts)

        # L4 and L5 equilateral triangle validation:
        # Distance from m1 (-mu, 0) to L4 must equal exactly 1.0
        # Distance from m2 (1-mu, 0) to L4 must equal exactly 1.0
        l4 = pts["L4"]
        dist_m1_l4 = math.hypot(l4[0] - (-mu), l4[1])
        dist_m2_l4 = math.hypot(l4[0] - (1.0 - mu), l4[1])
        self.assertAlmostEqual(dist_m1_l4, 1.0, places=7)
        self.assertAlmostEqual(dist_m2_l4, 1.0, places=7)

        # Collinear positions: L1 between masses, L2 beyond m2, L3 beyond m1
        self.assertTrue(-mu < pts["L1"][0] < (1.0 - mu))
        self.assertTrue(pts["L2"][0] > (1.0 - mu))
        self.assertTrue(pts["L3"][0] < -mu)

    def test_cr3bp_jacobi_integral_conservation(self):
        """Verifies Jacobi energy integral C_J conservation along RK4 numerical integration."""
        mu = 0.01215
        model = CR3BPModel(mu)
        # Initial state near L1
        l1_x = model.lagrange_points["L1"][0]
        state = (l1_x + 0.002, 0.0, 0.01, 0.0, 0.015, 0.0)

        c_j0 = model.jacobi_constant(*state)

        # Step forward 100 steps
        dt = 0.005
        cur_state = state
        for _ in range(100):
            cur_state = model.step_rk4(cur_state, dt)

        c_j_end = model.jacobi_constant(*cur_state)
        # Energy drift should be less than 0.01%
        rel_drift = abs(c_j_end - c_j0) / abs(c_j0)
        self.assertTrue(rel_drift < 1e-4)

    def test_hohmann_transfer_interplanetary(self):
        """Verifies Hohmann transfer delta-v and time of flight between Earth and Mars."""
        r1 = 1.0 * ASTRONOMICAL_UNIT
        r2 = 1.523679 * ASTRONOMICAL_UNIT

        dv1, dv2, total_dv, tof_sec = compute_hohmann_transfer(r1, r2, MU_SUN)

        # Standard Earth-Mars Hohmann values:
        # dv1 ~ 2945 m/s, dv2 ~ 2649 m/s, total dv ~ 5594 m/s
        # Time of flight ~ 258.9 days
        tof_days = tof_sec / 86400.0
        self.assertAlmostEqual(dv1, 2945.0, delta=25.0)
        self.assertAlmostEqual(dv2, 2649.0, delta=25.0)
        self.assertAlmostEqual(total_dv, 5594.0, delta=50.0)
        self.assertAlmostEqual(tof_days, 258.9, delta=1.5)

    def test_nbody_yoshida_integrator_energy_conservation(self):
        """Verifies 4th-order Yoshida symplectic integrator conserves mechanical energy."""
        engine = SolarSystemEngine()
        engine.initialize_default_solar_system()

        initial_energy = engine.compute_system_energy()

        # Step forward 30 days
        dt = 86400.0 * 0.5
        for _ in range(60):
            engine.step_yoshida_4th_order(dt)

        final_energy = engine.compute_system_energy()
        rel_energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        # 4th-order symplectic integrator must conserve energy within tight bound
        self.assertTrue(rel_energy_drift < 1e-4)

    def test_spacecraft_impulsive_maneuvers(self):
        """Verifies spacecraft burns accumulate delta_v and modify velocity vector."""
        sc = Spacecraft("Orbiter", CartesianState(1e8, 0, 0, 0, 30000, 0))
        self.assertEqual(sc.delta_v_spent, 0.0)

        # Apply prograde burn of 250 m/s
        sc.apply_impulse(0.0, 250.0, 0.0)
        self.assertEqual(sc.state.vy, 30250.0)
        self.assertEqual(sc.delta_v_spent, 250.0)

        # Apply normal burn of 100 m/s
        sc.apply_impulse(0.0, 0.0, 100.0)
        self.assertEqual(sc.state.vz, 100.0)
        self.assertEqual(sc.delta_v_spent, 350.0)

    def test_all_presets_validity(self):
        """Verifies all curated scenario presets load cleanly with valid parameters."""
        preset_keys = get_preset_list()
        self.assertTrue(len(preset_keys) >= 6)

        for key in preset_keys:
            p = load_preset(key)
            self.assertIn(p.view_mode, ["inertial", "cr3bp"])
            self.assertTrue(p.time_step_sec > 0.0)
            self.assertTrue(p.camera_distance > 0.0)
            self.assertTrue(len(p.description) > 0)
            if p.view_mode == "inertial":
                self.assertIsNotNone(p.bodies)
                self.assertTrue(len(p.bodies) >= 1)
            else:
                self.assertIsNotNone(p.cr3bp_mu)

    def test_escape_speed_and_parabolic_energy(self):
        """Verifies escape speed v_esc = sqrt(2*mu/r) produces near-zero parabolic specific energy."""
        r = 1.0 * ASTRONOMICAL_UNIT
        v_esc = math.sqrt(2.0 * MU_SUN / r)
        # Specific mechanical energy: E = v^2/2 - mu/r
        spec_energy = 0.5 * (v_esc ** 2) - MU_SUN / r
        self.assertAlmostEqual(spec_energy, 0.0, places=4)

    def test_cr3bp_collinear_roots_equilibrium(self):
        """Verifies that the analytical L1, L2, L3 collinear points satisfy dOmega/dx = 0."""
        mu = 0.01215
        model = CR3BPModel(mu)
        pts = model.lagrange_points

        for name in ["L1", "L2", "L3"]:
            x, y, z = pts[name]
            self.assertEqual(y, 0.0)
            self.assertEqual(z, 0.0)
            # Evaluate equations of motion with zero velocity (equilibrium test)
            _, _, _, ax, ay, az = model.equations_of_motion((x, 0.0, 0.0, 0.0, 0.0, 0.0))
            # At Lagrange points, acceleration in rotating frame must vanish
            self.assertAlmostEqual(ax, 0.0, places=5)
            self.assertAlmostEqual(ay, 0.0, places=7)
            self.assertAlmostEqual(az, 0.0, places=7)

    def test_kepler_third_law_orbital_periods(self):
        """Verifies Kepler third law T = 2*pi*sqrt(a^3 / mu) for planetary orbits."""
        a_merc = 0.387098 * ASTRONOMICAL_UNIT
        t_merc_sec = 2.0 * math.pi * math.sqrt((a_merc ** 3) / MU_SUN)
        t_merc_days = t_merc_sec / 86400.0
        # Mercury orbital period ~87.97 Earth days
        self.assertAlmostEqual(t_merc_days, 87.97, delta=0.2)

        a_earth = 1.0 * ASTRONOMICAL_UNIT
        t_earth_sec = 2.0 * math.pi * math.sqrt((a_earth ** 3) / MU_SUN)
        t_earth_days = t_earth_sec / 86400.0
        # Earth orbital period ~365.25 days
        self.assertAlmostEqual(t_earth_days, 365.25, delta=0.5)

    def test_perifocal_direction_cosine_orthonormality(self):
        """Verifies perifocal basis transformations preserve state magnitudes."""
        elem = OrbitalElements(
            a=2.0 * ASTRONOMICAL_UNIT,
            e=0.35,
            i=math.radians(28.0),
            raan=math.radians(115.0),
            arg_p=math.radians(72.0),
            true_anomaly=math.radians(140.0)
        )
        state = orbital_elements_to_cartesian(elem, MU_SUN)
        p = elem.a * (1.0 - elem.e ** 2)
        r_expected = p / (1.0 + elem.e * math.cos(elem.true_anomaly))
        self.assertAlmostEqual(state.position_magnitude, r_expected, delta=1e-3)


class TestAstroEphemerisGUI(unittest.TestCase):
    """Test Tkinter desktop application initialization and lifecycle."""

    def test_gui_headless_lifecycle(self):
        import tkinter as tk
        try:
            from astroephemeris import AstroEphemerisApp
            root = tk.Tk()
            root.withdraw()
            app = AstroEphemerisApp(root)

            # Check engine initialized
            self.assertIsNotNone(app.engine)
            self.assertTrue(len(app.engine.bodies) > 0)

            # Test camera zoom and views
            app._zoom(1.2)
            app._set_top_view()
            self.assertEqual(app.cam_pitch, 90.0)
            app._set_iso_view()
            self.assertEqual(app.cam_pitch, 35.0)

            # Test scenario loading
            app.load_preset_into_engine("earth_moon_cr3bp")
            self.assertEqual(app.view_mode, "cr3bp")
            self.assertIsNotNone(app.cr3bp_model)

            # Test simulation step and rendering in CR3BP
            app.step_simulation()
            app.render_all()
            app.update_telemetry()

            # Test thrust in CR3BP
            app._apply_thrust("prograde")

            # Switch to Hohmann scenario
            app.load_preset_into_engine("earth_mars_hohmann")
            self.assertEqual(app.view_mode, "inertial")
            self.assertIsNotNone(app.engine.spacecraft)

            # Test thrust in inertial
            app._apply_thrust("prograde")
            self.assertTrue(app.engine.spacecraft.delta_v_spent > 0.0)

            # Test play toggle
            app.toggle_play()
            self.assertFalse(app.is_running)
            app.toggle_play()
            self.assertTrue(app.is_running)

            root.destroy()
        except tk.TclError:
            # Gracefully handle display-less CI environments
            pass


if __name__ == "__main__":
    unittest.main()
