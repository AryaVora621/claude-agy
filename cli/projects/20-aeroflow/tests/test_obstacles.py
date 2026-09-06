"""
Unit Tests for AeroFlow Obstacle Geometries & NACA Airfoils.
"""

import unittest
from aeroflow.obstacles import (
    create_cylinder_obstacle,
    create_ellipse_obstacle,
    create_naca_airfoil_obstacle,
    create_plate_obstacle,
)


class TestObstacles(unittest.TestCase):
    def test_cylinder_geometry(self):
        nx, ny = 30, 30
        cx, cy, r = 15.0, 15.0, 5.0
        cyl = create_cylinder_obstacle(nx, ny, cx, cy, r)

        # Center point must be solid
        self.assertTrue(cyl.is_solid(15, 15))
        # Points inside radius must be solid
        self.assertTrue(cyl.is_solid(15, 18))
        # Far points must be fluid
        self.assertFalse(cyl.is_solid(0, 0))
        self.assertFalse(cyl.is_solid(15, 25))

        # Check boundary node count is non-empty
        self.assertGreater(len(cyl.boundary_nodes), 0)

    def test_ellipse_geometry(self):
        nx, ny = 40, 40
        ellipse = create_ellipse_obstacle(nx, ny, 20.0, 20.0, radius_x=8.0, radius_y=4.0)
        self.assertTrue(ellipse.is_solid(20, 20))
        self.assertTrue(ellipse.is_solid(25, 20))  # along major axis
        self.assertFalse(ellipse.is_solid(20, 28)) # outside minor axis

    def test_naca_airfoil_generation(self):
        nx, ny = 60, 40
        # NACA 0012 symmetric airfoil
        airfoil = create_naca_airfoil_obstacle(
            nx=nx,
            ny=ny,
            chord=24.0,
            code="0012",
            lead_x=15.0,
            lead_y=20.0,
            angle_of_attack_deg=0.0,
        )

        # Point on chord line should be solid
        self.assertTrue(airfoil.is_solid(20, 20))
        # Point far away should be fluid
        self.assertFalse(airfoil.is_solid(5, 5))
        self.assertFalse(airfoil.is_solid(50, 35))

        # Check boundary nodes detected
        self.assertGreater(len(airfoil.boundary_nodes), 10)

    def test_cambered_airfoil_asymmetry(self):
        nx, ny = 60, 40
        # Cambered NACA 2412 vs Symmetric NACA 0012
        cambered = create_naca_airfoil_obstacle(nx, ny, chord=24.0, code="2412", lead_x=15.0, lead_y=20.0)
        symmetric = create_naca_airfoil_obstacle(nx, ny, chord=24.0, code="0012", lead_x=15.0, lead_y=20.0)

        # Count solid cells above chord line (y > 20) vs below (y < 20)
        cambered_above = sum(1 for y in range(21, ny) for x in range(nx) if cambered.is_solid(x, y))
        cambered_below = sum(1 for y in range(0, 20) for x in range(nx) if cambered.is_solid(x, y))

        symmetric_above = sum(1 for y in range(21, ny) for x in range(nx) if symmetric.is_solid(x, y))
        symmetric_below = sum(1 for y in range(0, 20) for x in range(nx) if symmetric.is_solid(x, y))

        # Symmetric airfoil must have equal cells above and below
        self.assertEqual(symmetric_above, symmetric_below)
        # Cambered airfoil must have more cells above chord line due to positive camber
        self.assertGreater(cambered_above, cambered_below)

    def test_flat_plate_geometry(self):
        nx, ny = 30, 30
        plate = create_plate_obstacle(nx, ny, start_x=10.0, start_y=15.0, length=12.0, thickness=2.0)
        self.assertTrue(plate.is_solid(15, 15))
        self.assertFalse(plate.is_solid(5, 15))


if __name__ == "__main__":
    unittest.main()
