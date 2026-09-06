"""Unit tests for accretion disk physics, Keplerian orbits, and Doppler shifts."""

import math
import unittest
from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.accretion import AccretionDisk


class TestAccretionDisk(unittest.TestCase):
    """Test suite for accretion disk kinematics, emission profiles, and radiative transfer."""

    def setUp(self) -> None:
        self.metric = SchwarzschildMetric(mass=1.0)
        self.disk = AccretionDisk(self.metric)

    def test_disk_boundaries(self) -> None:
        self.assertEqual(self.disk.r_in, 6.0)
        self.assertEqual(self.disk.r_out, 15.0)

    def test_keplerian_frequency(self) -> None:
        # For Schwarzschild at r=8: Omega_K = sqrt(1 / 512)
        expected = math.sqrt(1.0 / (8.0**3))
        omega = self.disk.keplerian_angular_velocity(8.0)
        self.assertAlmostEqual(omega, expected, places=6)

    def test_emitter_four_velocity_normalization(self) -> None:
        # Test g_uv u^u u^v = -1
        u_em = self.disk.emitter_four_velocity(8.0)
        norm = self.metric.scalar_norm((0.0, 8.0, 0.5 * math.pi, 0.0), u_em)
        self.assertAlmostEqual(norm, -1.0, places=5)

    def test_rest_frame_profile(self) -> None:
        # Emission must be zero inside ISCO
        self.assertEqual(self.disk.rest_frame_profile(5.0), 0.0)
        # Emission must be zero outside outer boundary
        self.assertEqual(self.disk.rest_frame_profile(16.0), 0.0)
        # Emission positive within disk boundaries
        i_em = self.disk.rest_frame_profile(8.0)
        self.assertGreater(i_em, 0.0)

    def test_doppler_beaming_and_redshift(self) -> None:
        # Approaching material (photon with positive angular momentum aligned with Keplerian motion)
        res_approach = self.disk.evaluate_radiative_transfer(
            r=8.0, phi=0.0, p_contra=(1.0, 0.0, 0.0, 0.04)
        )
        # Receding material (photon moving against orbital motion)
        res_recede = self.disk.evaluate_radiative_transfer(
            r=8.0, phi=0.0, p_contra=(1.0, 0.0, 0.0, -0.04)
        )

        # Approaching side must have higher frequency shift g and higher observed intensity
        self.assertGreater(res_approach.redshift_factor_g, res_recede.redshift_factor_g)
        self.assertGreater(res_approach.observed_intensity, res_recede.observed_intensity)

    def test_kerr_accretion_disk(self) -> None:
        kerr = KerrMetric(mass=1.0, spin=0.9)
        kerr_disk = AccretionDisk(kerr)
        # ISCO for high spin is significantly smaller than 6M
        self.assertLess(kerr_disk.r_in, 3.0)
        omega = kerr_disk.keplerian_angular_velocity(4.0)
        self.assertGreater(omega, 0.0)

        u_em = kerr_disk.emitter_four_velocity(4.0)
        norm = kerr.scalar_norm((0.0, 4.0, 0.5 * math.pi, 0.0), u_em)
        self.assertAlmostEqual(norm, -1.0, places=4)


if __name__ == "__main__":
    unittest.main()
