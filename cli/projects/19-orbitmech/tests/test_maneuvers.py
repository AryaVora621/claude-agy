"""
Unit tests for OrbitMech Maneuvers, Interplanetary Transfers & Porkchop Optimization.
"""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    EARTH,
    SUN,
    JUPITER,
    AU_KM,
)
from orbitmech.maneuvers import (
    hohmann_transfer,
    bi_elliptic_transfer,
    plane_change_maneuver,
    gravity_assist_flyby,
)
from orbitmech.porkchop import generate_porkchop_grid


class TestManeuvers(unittest.TestCase):
    def test_leo_to_geo_hohmann_transfer(self):
        # 300 km LEO to GEO (42,164 km)
        r_leo = EARTH.radius + 300.0  # 6,678.137 km
        r_geo = 42164.14              # km

        res = hohmann_transfer(r_leo, r_geo, mu=EARTH.mu)

        # Standard textbook values:
        # dv1 ~ 2.42 km/s, dv2 ~ 1.47 km/s, total dv ~ 3.89 km/s
        self.assertAlmostEqual(res.delta_v1, 2.424, places=2)
        self.assertAlmostEqual(res.delta_v2, 1.468, places=2)
        self.assertAlmostEqual(res.total_delta_v, 3.892, places=2)

        # Transfer time should be ~5.26 hours (~18,930 seconds)
        self.assertAlmostEqual(res.time_of_flight / 3600.0, 5.26, delta=0.1)

    def test_bi_elliptic_transfer_threshold(self):
        # When r2 / r1 < 11.94, Hohmann is strictly superior
        r1 = 7000.0
        r2 = 14000.0  # ratio = 2.0
        rb = 30000.0
        res_bi_low = bi_elliptic_transfer(r1, r2, rb, mu=EARTH.mu)
        self.assertFalse(res_bi_low.is_more_efficient_than_hohmann)

        # When r2 / r1 > 11.94 (e.g. ratio = 20.0) and rb is very large, bi-elliptic is more fuel efficient
        r2_high = 7000.0 * 20.0  # 140,000 km
        rb_high = 7000.0 * 100.0 # 700,000 km
        res_bi_high = bi_elliptic_transfer(r1, r2_high, rb_high, mu=EARTH.mu)
        self.assertTrue(res_bi_high.is_more_efficient_than_hohmann)

    def test_plane_change_maneuver(self):
        # 28.5 degree inclination change (e.g. Cape Canaveral to equatorial) at LEO speed ~7.73 km/s
        v_leo = 7.726
        d_inc = 28.5
        res = plane_change_maneuver(d_inc, v_leo)
        # dv = 2 * 7.726 * sin(14.25 deg) ~ 3.80 km/s
        expected_dv = 2.0 * v_leo * math.sin(math.radians(d_inc * 0.5))
        self.assertAlmostEqual(res.delta_v, expected_dv, places=4)

    def test_jupiter_gravity_assist_flyby(self):
        # Spacecraft approaching Jupiter at 10 km/s excess speed
        # Jupiter orbital speed around Sun ~ 13.07 km/s along +Y axis
        v_jupiter = Vector3(0.0, 13.07, 0.0)
        # Spacecraft incoming in heliocentric frame with v_x = 10, v_y = 13.07 -> v_inf = (10, 0, 0)
        v_sc_in = Vector3(10.0, 13.07, 0.0)

        # Flyby at 500,000 km altitude
        altitude = 500000.0
        res = gravity_assist_flyby(
            v_sc_in=v_sc_in,
            v_planet=v_jupiter,
            planet=JUPITER,
            periapsis_altitude=altitude,
        )

        # Bending angle should be positive
        self.assertGreater(res.bending_angle_deg, 0.0)
        # Spacecraft receives a heliocentric delta-v boost
        self.assertGreater(res.delta_v_mag, 0.0)
        # Outgoing relative excess speed equals incoming excess speed (elastic conservation in planet frame)
        self.assertAlmostEqual(res.v_inf_in.norm(), res.v_inf_out.norm(), places=5)

    def test_porkchop_grid_earth_mars(self):
        # Generate 6x6 test grid
        grid = generate_porkchop_grid(
            dep_start_day=0.0,
            dep_end_day=60.0,
            dep_steps=6,
            arr_start_day=180.0,
            arr_end_day=300.0,
            arr_steps=6,
        )

        self.assertEqual(len(grid.departure_days), 6)
        self.assertEqual(len(grid.arrival_days), 6)
        # Minimum C3 should be reasonable (< 40 km^2/s^2)
        self.assertLess(grid.min_c3_point.c3_km2_s2, 45.0)
        self.assertGreater(grid.min_c3_point.time_of_flight_days, 100.0)
        # Matrix conversion works
        c3_matrix = grid.get_contour_matrix("c3_km2_s2")
        self.assertEqual(len(c3_matrix), 6)
        self.assertEqual(len(c3_matrix[0]), 6)


if __name__ == "__main__":
    unittest.main()
