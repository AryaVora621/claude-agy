"""Unit tests for curved spacetime geodesic integration."""

import math
import unittest
from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.geodesic import (
    GeodesicIntegrator,
    GeodesicState,
    GeodesicStatus,
)


class TestGeodesicIntegrator(unittest.TestCase):
    """Test suite for RK4 geodesic integrator, constants of motion, and boundary checks."""

    def setUp(self) -> None:
        self.metric = SchwarzschildMetric(mass=1.0)
        self.integrator = GeodesicIntegrator(
            metric=self.metric,
            is_null=True,
            horizon_buffer=0.05,
            escape_radius=30.0,
            max_steps=500,
        )

    def test_state_copy_and_coords(self) -> None:
        state = GeodesicState(0.0, 10.0, 1.5, 0.5, -1.0, -0.2, 0.0, 0.1)
        copied = state.copy()
        self.assertEqual(state.coords, copied.coords)
        self.assertEqual(state.momentum, copied.momentum)
        copied.r = 15.0
        self.assertEqual(state.r, 10.0)

    def test_infalling_radial_geodesic_captured(self) -> None:
        # Radial inward photon from r=8 directly towards center
        # Null condition: -(1-2/r)(pt)^2 + 1/(1-2/r)(pr)^2 = 0
        r_init = 8.0
        f = 1.0 - 2.0 / r_init
        pt = -1.0
        pr = -math.sqrt(f * f * pt * pt)  # Inward radial momentum

        state = GeodesicState(
            t=0.0,
            r=r_init,
            theta=0.5 * math.pi,
            phi=0.0,
            pt=pt,
            pr=pr,
            ptheta=0.0,
            pphi=0.0,
        )
        res = self.integrator.integrate(state, backward=False)
        self.assertEqual(res.status, GeodesicStatus.HORIZON_CAPTURED)
        self.assertLessEqual(res.final_state.r, self.metric.horizon_radius() + 0.06)

    def test_escaping_radial_geodesic(self) -> None:
        # Outward radial photon from r=10 directed away from black hole
        state = GeodesicState(
            t=0.0,
            r=10.0,
            theta=0.5 * math.pi,
            phi=0.0,
            pt=-1.0,
            pr=1.0,
            ptheta=0.0,
            pphi=0.0,
        )
        res = self.integrator.integrate(state, backward=False)
        self.assertEqual(res.status, GeodesicStatus.ESCAPED)
        self.assertGreaterEqual(res.final_state.r, 30.0)

    def test_equatorial_disk_crossing_detection(self) -> None:
        # Launch ray obliquely across equatorial plane theta = pi/2
        state = GeodesicState(
            t=0.0,
            r=10.0,
            theta=1.4,  # Above equator
            phi=0.0,
            pt=-1.0,
            pr=-0.1,
            ptheta=0.1,  # Moving towards and past equator
            pphi=0.05,
        )
        res = self.integrator.integrate(
            state,
            backward=False,
            stop_on_disk=True,
            disk_r_in=6.0,
            disk_r_out=15.0,
        )
        self.assertEqual(res.status, GeodesicStatus.DISK_INTERSECTED)
        self.assertTrue(len(res.disk_intersections) > 0)
        cross = res.disk_intersections[0]
        self.assertGreaterEqual(cross.r_cross, 6.0)
        self.assertLessEqual(cross.r_cross, 15.0)

    def test_kerr_energy_and_angular_momentum_conservation(self) -> None:
        kerr = KerrMetric(mass=1.0, spin=0.7)
        kerr_integrator = GeodesicIntegrator(kerr, is_null=True, max_steps=200)
        state = GeodesicState(
            t=0.0,
            r=12.0,
            theta=1.2,
            phi=0.0,
            pt=-1.0,
            pr=-0.3,
            ptheta=0.01,
            pphi=0.08,
        )
        res = kerr_integrator.integrate(state, backward=False)
        # Stationary axisymmetric spacetime strictly conserves pt and pphi
        self.assertTrue(res.energy_conserved)
        self.assertTrue(res.angular_momentum_conserved)


if __name__ == "__main__":
    unittest.main()
