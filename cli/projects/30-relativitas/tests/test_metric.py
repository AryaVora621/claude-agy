"""Unit tests for spacetime metrics and Christoffel connection symbols."""

import math
import unittest
from relativitas.metric import SchwarzschildMetric, KerrMetric


class TestSchwarzschildMetric(unittest.TestCase):
    """Test suite for Schwarzschild metric tensors, inverse, and connection coefficients."""

    def setUp(self) -> None:
        self.metric = SchwarzschildMetric(mass=1.0)

    def test_horizon_and_critical_radii(self) -> None:
        self.assertAlmostEqual(self.metric.horizon_radius(), 2.0, places=5)
        self.assertAlmostEqual(self.metric.photon_sphere_radius(), 3.0, places=5)
        self.assertAlmostEqual(self.metric.isco_radius(), 6.0, places=5)

    def test_metric_tensor_values(self) -> None:
        r = 4.0
        theta = 0.5 * math.pi
        g = self.metric.metric_tensor((0.0, r, theta, 0.0))

        # g_tt = -(1 - 2/4) = -0.5
        self.assertAlmostEqual(g[0][0], -0.5, places=6)
        # g_rr = 1 / (1 - 2/4) = 2.0
        self.assertAlmostEqual(g[1][1], 2.0, places=6)
        # g_th_th = r^2 = 16.0
        self.assertAlmostEqual(g[2][2], 16.0, places=6)
        # g_phi_phi = r^2 sin^2(pi/2) = 16.0
        self.assertAlmostEqual(g[3][3], 16.0, places=6)
        # Off-diagonal elements zero
        self.assertAlmostEqual(g[0][1], 0.0)
        self.assertAlmostEqual(g[0][3], 0.0)

    def test_inverse_metric(self) -> None:
        x = (0.0, 5.0, 1.2, 0.5)
        g = self.metric.metric_tensor(x)
        g_inv = self.metric.inverse_metric(x)

        # Verify g^ua * g_ab = delta^u_b
        for u in range(4):
            for b in range(4):
                val = sum(g_inv[u][a] * g[a][b] for a in range(4))
                expected = 1.0 if u == b else 0.0
                self.assertAlmostEqual(val, expected, places=5)

    def test_analytical_christoffel_symbols(self) -> None:
        x = (0.0, 5.0, 1.0, 0.0)
        gamma = self.metric.christoffel_symbols(x)

        # Check symmetry Gamma^u_ab = Gamma^u_ba
        for u in range(4):
            for a in range(4):
                for b in range(4):
                    self.assertAlmostEqual(gamma[u][a][b], gamma[u][b][a], places=7)

        # Gamma^t_tr = M / (r(r - 2M)) = 1 / (5 * 3) = 1/15
        self.assertAlmostEqual(gamma[0][0][1], 1.0 / 15.0, places=6)
        # Gamma^th_r_th = 1 / r = 0.2
        self.assertAlmostEqual(gamma[2][1][2], 0.2, places=6)
        # Gamma^ph_r_ph = 1 / r = 0.2
        self.assertAlmostEqual(gamma[3][1][3], 0.2, places=6)

    def test_null_norm(self) -> None:
        # Construct exact null vector at r=4: -(1-2/r)(pt)^2 + 1/(1-2/r)(pr)^2 = 0
        # -0.5*(pt)^2 + 2*(pr)^2 = 0 => pt = 2, pr = 1
        x = (0.0, 4.0, 0.5 * math.pi, 0.0)
        p = (2.0, 1.0, 0.0, 0.0)
        norm = self.metric.scalar_norm(x, p)
        self.assertAlmostEqual(norm, 0.0, places=6)


class TestKerrMetric(unittest.TestCase):
    """Test suite for rotating Kerr black hole metric in Boyer-Lindquist coordinates."""

    def setUp(self) -> None:
        self.metric = KerrMetric(mass=1.0, spin=0.8)

    def test_horizons_and_ergosphere(self) -> None:
        # r_+ = 1 + sqrt(1 - 0.64) = 1 + 0.6 = 1.6
        self.assertAlmostEqual(self.metric.horizon_radius(), 1.6, places=5)
        # r_- = 1 - 0.6 = 0.4
        self.assertAlmostEqual(self.metric.inner_horizon_radius(), 0.4, places=5)

        # Ergosphere at equator theta = pi/2: r_ergo = 1 + sqrt(1 - 0) = 2.0
        self.assertAlmostEqual(self.metric.ergosphere_radius(0.5 * math.pi), 2.0, places=5)
        # Ergosphere at poles theta = 0: r_ergo = 1 + sqrt(1 - 0.64) = 1.6 (touches horizon)
        self.assertAlmostEqual(self.metric.ergosphere_radius(0.0), 1.6, places=5)

    def test_isco_radius(self) -> None:
        # Prograde ISCO for spin 0.8 is strictly between M and 6M
        r_pro = self.metric.isco_radius(prograde=True)
        r_retro = self.metric.isco_radius(prograde=False)

        self.assertGreater(r_pro, 1.0)
        self.assertLess(r_pro, 6.0)
        self.assertGreater(r_retro, 6.0)

    def test_frame_dragging(self) -> None:
        # Frame dragging angular velocity omega > 0 for prograde spin
        omega = self.metric.frame_dragging_angular_velocity(r=2.5, theta=0.5 * math.pi)
        self.assertGreater(omega, 0.0)

    def test_inverse_metric(self) -> None:
        x = (0.0, 4.0, 1.2, 0.7)
        g = self.metric.metric_tensor(x)
        g_inv = self.metric.inverse_metric(x)

        # Check g^ua * g_ab = delta^u_b
        for u in range(4):
            for b in range(4):
                val = sum(g_inv[u][a] * g[a][b] for a in range(4))
                expected = 1.0 if u == b else 0.0
                self.assertAlmostEqual(val, expected, places=4)

    def test_christoffel_symmetry(self) -> None:
        x = (0.0, 5.0, 1.1, 0.0)
        gamma = self.metric.christoffel_symbols(x)
        for u in range(4):
            for a in range(4):
                for b in range(4):
                    self.assertAlmostEqual(gamma[u][a][b], gamma[u][b][a], places=4)


if __name__ == "__main__":
    unittest.main()
