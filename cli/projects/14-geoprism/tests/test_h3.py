"""
Unit tests for GeoPrism H3 Hierarchical Hexagonal Spatial Grid.
"""

import math
import unittest
from geoprism.h3 import (
    encode_h3, decode_h3, h3_to_string, string_to_h3,
    geo_to_h3, h3_to_geo, h3_to_geo_boundary, k_ring, h3_distance,
    h3_get_resolution, h3_get_base_cell
)


class TestH3Grid(unittest.TestCase):
    def test_bit_packing_roundtrip(self):
        resolution = 7
        base_cell = 42
        digits = [1, 0, 4, 2, 6, 3, 5]

        h3_idx = encode_h3(resolution, base_cell, digits)
        self.assertIsInstance(h3_idx, int)

        res_out, base_out, digits_out = decode_h3(h3_idx)
        self.assertEqual(res_out, resolution)
        self.assertEqual(base_out, base_cell)
        self.assertEqual(digits_out, digits)

        self.assertEqual(h3_get_resolution(h3_idx), resolution)
        self.assertEqual(h3_get_base_cell(h3_idx), base_cell)

    def test_hex_string_codec(self):
        h3_idx = encode_h3(9, 10, [1, 2, 3, 4, 5, 6, 0, 1, 2])
        hex_str = h3_to_string(h3_idx)
        self.assertEqual(len(hex_str), 15)

        reconstructed = string_to_h3(hex_str)
        self.assertEqual(reconstructed, h3_idx)

    def test_geo_coordinate_roundtrip(self):
        # Test landmarks across different hemispheres
        landmarks = [
            ("San Francisco", 37.7749, -122.4194),
            ("London", 51.5074, -0.1278),
            ("Tokyo", 35.6762, 139.6503),
            ("Sydney", -33.8688, 151.2093),
            ("Equator", 0.0, 0.0),
        ]

        for name, lat, lon in landmarks:
            h3_idx = geo_to_h3(lat, lon, resolution=8)
            cent_lat, cent_lon = h3_to_geo(h3_idx)

            # At resolution 8, cell radius is small; reconstructed center should be within tolerance
            d_lat = abs(cent_lat - lat)
            d_lon = abs(cent_lon - lon)
            self.assertLess(d_lat, 2.0, f"{name} lat error too large: {d_lat}")
            self.assertLess(d_lon, 3.0, f"{name} lon error too large: {d_lon}")

    def test_geo_boundary(self):
        h3_idx = geo_to_h3(37.7749, -122.4194, resolution=6)
        boundary = h3_to_geo_boundary(h3_idx)
        self.assertEqual(len(boundary), 6)

        center_lat, center_lon = h3_to_geo(h3_idx)
        # All boundary vertices should be within a reasonable delta of the center
        for v_lat, v_lon in boundary:
            dist = math.sqrt((v_lat - center_lat) ** 2 + (v_lon - center_lon) ** 2)
            self.assertGreater(dist, 0.0)
            self.assertLess(dist, 5.0)

    def test_k_ring_and_distance(self):
        origin = geo_to_h3(40.7128, -74.0060, resolution=5)  # NYC
        ring0 = k_ring(origin, 0)
        self.assertEqual(ring0, {origin})

        ring1 = k_ring(origin, 1)
        self.assertIn(origin, ring1)
        self.assertGreater(len(ring1), 1)

        # Distance to self is 0
        self.assertEqual(h3_distance(origin, origin), 0)


if __name__ == "__main__":
    unittest.main()
