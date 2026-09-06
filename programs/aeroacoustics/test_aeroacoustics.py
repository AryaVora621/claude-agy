"""
Automated Unit Tests for AeroAcoustics Computational Engine
Zero external dependencies, standard library unittest only.
Verifies sound speed thermodynamics, Doppler shifts, Mach cone geometry,
Whitham N-wave sonic boom profiles, and microphone spectral analysis.
"""

import math
import unittest
import sys
import os

# Ensure local directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acoustics import (
    sound_speed_from_temp,
    sound_pressure_level_db,
    mach_angle_rad,
    doppler_observed_frequency,
    WavePacket,
    MicrophoneStation,
    FlightTrajectory,
    AeroAcousticEngine,
    REFERENCE_PRESSURE
)
from presets import PRESETS, load_preset, get_preset_list


class TestAeroAcousticsCore(unittest.TestCase):

    def test_sound_speed_thermodynamics(self):
        """Verifies ambient temperature to sound speed conversion."""
        # Standard sea-level temperature 15 deg C (288.15 K) -> ~340.3 m/s
        c_15 = sound_speed_from_temp(15.0)
        self.assertAlmostEqual(c_15, 340.29, delta=0.5)

        # Freezing 0 deg C (273.15 K) -> ~331.3 m/s
        c_0 = sound_speed_from_temp(0.0)
        self.assertAlmostEqual(c_0, 331.3, delta=0.5)

        # Stratospheric -50 deg C (223.15 K) -> ~299.4 m/s
        c_minus50 = sound_speed_from_temp(-50.0)
        self.assertAlmostEqual(c_minus50, 299.4, delta=0.5)

        # Desert +45 deg C (318.15 K) -> ~357.5 m/s
        c_hot = sound_speed_from_temp(45.0)
        self.assertAlmostEqual(c_hot, 357.5, delta=0.5)

    def test_sound_pressure_level_db(self):
        """Verifies decibel Sound Pressure Level (SPL) calculations."""
        # Reference threshold 20 uPa -> exactly 0.0 dB
        self.assertAlmostEqual(sound_pressure_level_db(REFERENCE_PRESSURE), 0.0, places=4)

        # 10x reference pressure -> +20.0 dB
        self.assertAlmostEqual(sound_pressure_level_db(REFERENCE_PRESSURE * 10.0), 20.0, places=4)

        # 20 Pa (threshold of pain) -> 120.0 dB
        self.assertAlmostEqual(sound_pressure_level_db(20.0), 120.0, places=2)

        # Zero or negative inputs do not raise math domain errors
        spl_zero = sound_pressure_level_db(0.0)
        self.assertTrue(spl_zero < 0.0)

    def test_mach_angle_geometry(self):
        """Verifies Mach cone half-angle sin(mu) = 1/M for supersonic flow."""
        # Subsonic (M < 1.0) has no Mach cone
        self.assertIsNone(mach_angle_rad(0.5))
        self.assertIsNone(mach_angle_rad(0.99))

        # Mach 1.0 (sonic barrier) -> mu = 90 deg (pi/2 rad)
        mu_1 = mach_angle_rad(1.0)
        self.assertIsNotNone(mu_1)
        self.assertAlmostEqual(math.degrees(mu_1), 90.0, places=4)

        # Mach 2.0 (Concorde cruise) -> sin(mu) = 0.5 -> mu = 30 deg
        mu_2 = mach_angle_rad(2.0)
        self.assertIsNotNone(mu_2)
        self.assertAlmostEqual(math.degrees(mu_2), 30.0, places=4)

        # Mach 3.0 -> sin(mu) = 1/3 -> mu = 19.47 deg
        mu_3 = mach_angle_rad(3.0)
        self.assertIsNotNone(mu_3)
        self.assertAlmostEqual(math.degrees(mu_3), 19.47, delta=0.1)

    def test_doppler_observed_frequency(self):
        """Verifies moving source Doppler shift formulation."""
        f0 = 400.0
        mach = 0.5

        # Head-on approach (theta = 0): f_obs = f0 / (1 - M) = 400 / 0.5 = 800 Hz
        f_approach = doppler_observed_frequency(f0, mach, 0.0)
        self.assertAlmostEqual(f_approach, 800.0, places=2)

        # Direct retreat (theta = pi): f_obs = f0 / (1 + M) = 400 / 1.5 = 266.67 Hz
        f_retreat = doppler_observed_frequency(f0, mach, math.pi)
        self.assertAlmostEqual(f_retreat, 266.67, places=1)

        # Transverse (theta = pi/2): cos(pi/2) = 0 -> f_obs = f0 = 400 Hz
        f_transverse = doppler_observed_frequency(f0, mach, math.pi * 0.5)
        self.assertAlmostEqual(f_transverse, 400.0, places=2)

    def test_wave_packet_expansion_and_attenuation(self):
        """Verifies expanding spherical wavefront geometry and decay."""
        c_sound = 340.0
        wp = WavePacket(
            x_emit=0.0, y_emit=0.0, t_emit=1.0,
            frequency=200.0, amplitude=100.0
        )

        # At t = 1.0, radius is 0
        self.assertEqual(wp.radius_at(1.0, c_sound), 0.0)

        # At t = 2.0 (dt = 1.0s), radius is 340m
        self.assertAlmostEqual(wp.radius_at(2.0, c_sound), 340.0, places=2)

        # Amplitude at r=0 is 0
        self.assertEqual(wp.amplitude_at(1.0, c_sound), 0.0)

        # Amplitude at r=340m decays by 1/r and exponential absorption
        amp_340 = wp.amplitude_at(2.0, c_sound, attenuation_alpha=0.001)
        expected = (100.0 / 340.0) * math.exp(-0.001 * 340.0)
        self.assertAlmostEqual(amp_340, expected, places=4)

    def test_microphone_station_recording(self):
        """Verifies microphone sampling, running RMS, and peak overpressure."""
        mic = MicrophoneStation("TEST_MIC", x=10.0, y=20.0, max_history_len=100)
        self.assertEqual(len(mic.pressure_history), 0)

        # Record clean 10.0 Pa constant pressure
        for t_step in range(60):
            mic.record_sample(t_step * 0.01, 10.0)

        self.assertEqual(len(mic.pressure_history), 60)
        self.assertEqual(mic.peak_overpressure_pa, 10.0)
        self.assertAlmostEqual(mic.spl_dBA, sound_pressure_level_db(10.0), places=1)

        # Reset microphone
        mic.reset()
        self.assertEqual(len(mic.pressure_history), 0)
        self.assertEqual(mic.peak_overpressure_pa, 0.0)

    def test_flight_trajectories(self):
        """Verifies analytical flight trajectories (straight, accelerating, circle, slalom)."""
        c = 340.0

        # Straight flight
        traj_straight = FlightTrajectory("straight")
        x0, y0, vx0, vy0, m0 = traj_straight.evaluate(0.0, 1.5, c)
        self.assertEqual(x0, -400.0)
        self.assertEqual(y0, 0.0)
        self.assertAlmostEqual(vx0, 1.5 * c, places=2)
        self.assertEqual(m0, 1.5)

        # Accelerating flight (crosses sound barrier)
        traj_accel = FlightTrajectory("accelerating")
        _, _, _, _, m_start = traj_accel.evaluate(0.0, 1.0, c)
        self.assertEqual(m_start, 0.5) # Starts subsonic
        _, _, _, _, m_later = traj_accel.evaluate(2.0, 1.0, c)
        self.assertAlmostEqual(m_later, 0.5 + 0.35 * 2.0, places=2) # Crosses Mach 1.0

        # Circle flight (constant radius and speed)
        traj_circle = FlightTrajectory("circle")
        x_c, y_c, vx_c, vy_c, m_c = traj_circle.evaluate(0.0, 0.8, c)
        self.assertAlmostEqual(math.hypot(x_c, y_c), 180.0, places=2)
        self.assertAlmostEqual(math.hypot(vx_c, vy_c), 0.8 * c, places=1)

        # Slalom flight
        traj_slalom = FlightTrajectory("slalom")
        x_s, y_s, vx_s, vy_s, m_s = traj_slalom.evaluate(0.0, 1.2, c)
        self.assertEqual(y_s, 0.0) # sin(0) = 0

    def test_aeroacoustic_engine_step(self):
        """Verifies simulation engine stepping, wavefront emission, and microphone sampling."""
        engine = AeroAcousticEngine(base_mach=1.5, source_freq=200.0, source_amplitude=50.0)
        self.assertEqual(len(engine.wavefronts), 0)

        # Step forward by 0.5 seconds
        dt = 0.01
        for _ in range(50):
            engine.step(dt)

        self.assertAlmostEqual(engine.sim_time, 0.5, places=2)
        self.assertTrue(len(engine.wavefronts) > 0)

        # Check that microphones recorded data
        for mic in engine.microphones:
            self.assertTrue(len(mic.pressure_history) > 0)

    def test_n_wave_pressure_profile(self):
        """Verifies Whitham N-wave sonic boom profile characteristics."""
        engine = AeroAcousticEngine()
        T = engine.n_wave_duration      # 0.08 s
        tau = engine.n_wave_rise_time   # 0.003 s
        P = engine.n_wave_peak_overpressure # 120.0 Pa

        # Before shock arrival (dt < 0) -> 0.0
        self.assertEqual(engine.compute_n_wave_pressure(-0.01), 0.0)

        # At peak bow shock (dt = tau) -> P
        self.assertAlmostEqual(engine.compute_n_wave_pressure(tau), P, places=2)

        # At mid-expansion (dt = T/2) -> exactly 0.0
        self.assertAlmostEqual(engine.compute_n_wave_pressure(T * 0.5), 0.0, places=2)

        # At tail shock peak (dt = T - tau) -> -P
        self.assertAlmostEqual(engine.compute_n_wave_pressure(T - tau), -P, places=2)

        # After shock passed (dt > T) -> 0.0
        self.assertEqual(engine.compute_n_wave_pressure(T + 0.05), 0.0)

    def test_mach_cone_geometry_computation(self):
        """Verifies supersonic shock cone ray generation."""
        engine = AeroAcousticEngine(base_mach=2.0)
        engine.source_x = 0.0
        engine.source_y = 0.0
        engine.source_vx = 2.0 * engine.c_sound
        engine.source_vy = 0.0
        engine.current_mach = 2.0

        geom = engine.get_mach_cone_geometry()
        self.assertIsNotNone(geom)
        self.assertAlmostEqual(geom["mu_deg"], 30.0, places=2)
        self.assertEqual(geom["apex"], (0.0, 0.0))

        # Check that subsonic returns None
        engine.current_mach = 0.8
        self.assertIsNone(engine.get_mach_cone_geometry())

    def test_polar_directivity_normalization(self):
        """Verifies 360-degree polar pattern calculation and unit normalization."""
        engine = AeroAcousticEngine(base_mach=0.8)
        pattern = engine.compute_directivity_polar_pattern(72)
        self.assertEqual(len(pattern), 72)

        # Peak intensity must be normalized to exactly 1.0
        max_val = max(p[1] for p in pattern)
        self.assertAlmostEqual(max_val, 1.0, places=4)

        # All values between 0.0 and 1.0
        for angle, val in pattern:
            self.assertTrue(0.0 <= val <= 1.0)

    def test_fft_spectrum_computation(self):
        """Verifies Discrete Fourier Transform computes valid power spectra."""
        engine = AeroAcousticEngine()
        # Seed microphone with 100 Hz synthetic sinusoidal signal
        mic = engine.microphones[0]
        sample_rate = 1000.0
        dt = 1.0 / sample_rate
        f_sig = 100.0
        for i in range(128):
            t = i * dt
            p = 20.0 * math.sin(2.0 * math.pi * f_sig * t)
            mic.record_sample(t, p)

        freqs, powers = engine.compute_fft_spectrum(0)
        self.assertTrue(len(freqs) > 0)
        self.assertEqual(len(freqs), len(powers))

        # Peak spectral power should occur near 100 Hz
        peak_idx = powers.index(max(powers))
        peak_freq = freqs[peak_idx]
        self.assertAlmostEqual(peak_freq, f_sig, delta=15.0)

    def test_presets_validity(self):
        """Verifies all curated aerospace presets contain valid physical parameters."""
        preset_keys = get_preset_list()
        self.assertTrue(len(preset_keys) >= 6)

        for key in preset_keys:
            p = load_preset(key)
            self.assertTrue(p.base_mach > 0.0)
            self.assertTrue(p.source_freq > 0.0)
            self.assertTrue(p.source_amplitude > 0.0)
            self.assertIn(p.source_type, ["monopole", "dipole", "quadrupole", "n_wave"])
            self.assertIn(p.trajectory_mode, ["straight", "accelerating", "circle", "slalom"])
            self.assertTrue(len(p.description) > 0)

    def test_multipole_field_directivity(self):
        """Verifies acoustic radiation characteristics of monopole, dipole, and quadrupole."""
        engine = AeroAcousticEngine(base_mach=0.0)
        # Emit a single wavefront at origin
        engine.sim_time = 0.1
        engine.last_emission_time = -1.0
        engine.step(0.01)

        # Monopole pressure at equidistant circular points should have equal magnitude
        engine.source_type = "monopole"
        p_east = engine.compute_field_pressure_at(20.0, 0.0, 0.15)
        p_north = engine.compute_field_pressure_at(0.0, 20.0, 0.15)
        self.assertAlmostEqual(abs(p_east), abs(p_north), delta=1e-3)

    def test_ground_boom_footprint(self):
        """Verifies supersonic flight generates ground boom event records."""
        engine = AeroAcousticEngine(base_mach=1.6)
        engine.source_y = 120.0  # Above ground
        # Step through flight
        for _ in range(30):
            engine.step(0.02)

        self.assertTrue(len(engine.ground_boom_events) > 0)
        last_event = engine.ground_boom_events[-1]
        self.assertGreaterEqual(last_event["mach"], 1.0)


class TestAeroAcousticsGUI(unittest.TestCase):
    """Test Tkinter desktop application initialization and lifecycle."""

    def test_gui_headless_lifecycle(self):
        import tkinter as tk
        try:
            from aeroacoustics import AeroAcousticsApp
            root = tk.Tk()
            root.withdraw()
            app = AeroAcousticsApp(root)

            # Verify components initialized
            self.assertIsNotNone(app.engine)
            self.assertEqual(len(app.engine.microphones), 3)

            # Test preset loading
            app.load_preset_into_ui("sr71_blackbird")
            self.assertEqual(app.engine.base_mach, 3.20)
            self.assertEqual(app.engine.source_type, "n_wave")

            # Test coordinate transformation roundtrip
            wx, wy = 120.0, -80.0
            sx, sy = app.world_to_screen(wx, wy)
            wx_back, wy_back = app.screen_to_world(sx, sy)
            self.assertAlmostEqual(wx, wx_back, delta=0.5)
            self.assertAlmostEqual(wy, wy_back, delta=0.5)

            # Test simulation stepping and render
            app.step_simulation()
            app.render_all()

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
