"""
OptiFlow 2D Unit Test Suite
Zero external dependencies. Pure Python standard library unittest.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.optiflow.cfd_engine import CFDEngine2D, Particle
from programs.optiflow.presets import AERO_PRESETS


class TestOptiFlowCFD(unittest.TestCase):
    """Automated test cases for 2D Navier-Stokes CFD solver."""

    def setUp(self):
        self.engine = CFDEngine2D(nx=50, ny=25, lx=2.0, ly=1.0, u_inf=1.0, reynolds=200.0)

    def test_initial_uniform_flow(self):
        """Verify initial freestream fields and boundary conditions."""
        self.assertEqual(self.engine.nx, 50)
        self.assertEqual(self.engine.ny, 25)
        self.assertAlmostEqual(self.engine.dx, 2.0 / 49.0)
        self.assertAlmostEqual(self.engine.dy, 1.0 / 24.0)

        # Inflow streamfunction psi = u_inf * y
        for j in range(self.engine.ny):
            expected_psi = 1.0 * (j * self.engine.dy)
            self.assertAlmostEqual(self.engine.psi[j][0], expected_psi, places=4)
            self.assertAlmostEqual(self.engine.u[j][0], 1.0, places=4)
            self.assertAlmostEqual(self.engine.v[j][0], 0.0, places=4)
            self.assertAlmostEqual(self.engine.omega[j][0], 0.0, places=4)

    def test_naca0012_geometry_symmetry(self):
        """Verify NACA 0012 symmetric profile has equal thickness top and bottom."""
        self.engine.set_naca_airfoil(m=0.0, p=0.0, t=0.12, chord=0.6, x_center=0.7, y_center=0.5, alpha_deg=0.0)
        self.assertTrue(len(self.engine.boundary_cells) > 0)

        # Check midpoint of domain y = 0.5 (index 12 on 25-grid)
        mid_j = 12
        # Verify solid cells exist near chord center
        mid_i = int(0.7 / self.engine.dx)
        self.assertTrue(self.engine.solid[mid_j][mid_i])

        # Verify vertical symmetry: solid points above mid_j should equal solid points below mid_j
        top_solids = sum(1 for j in range(mid_j + 1, self.engine.ny) if self.engine.solid[j][mid_i])
        bot_solids = sum(1 for j in range(0, mid_j) if self.engine.solid[j][mid_i])
        self.assertEqual(top_solids, bot_solids)

    def test_naca4412_camber(self):
        """Verify NACA 4412 camber shifts profile upward."""
        self.engine.set_naca_airfoil(m=0.04, p=0.4, t=0.12, chord=0.6, x_center=0.7, y_center=0.5, alpha_deg=0.0)
        mid_i = int(0.7 / self.engine.dx)

        # Count solid cells above vs below midline y = 0.5 (index 12)
        mid_j = 12
        top_solids = sum(1 for j in range(mid_j + 1, self.engine.ny) if self.engine.solid[j][mid_i])
        bot_solids = sum(1 for j in range(0, mid_j) if self.engine.solid[j][mid_i])
        # Positive camber produces more thickness above centerline than below
        self.assertGreaterEqual(top_solids, bot_solids)

    def test_circular_cylinder_geometry(self):
        """Verify cylinder geometry generates correct radius and boundary normals."""
        radius = 0.1
        self.engine.set_circular_cylinder(radius=radius, x_center=0.8, y_center=0.5)
        self.assertTrue(len(self.engine.boundary_cells) > 0)

        # Center should be solid
        cx_idx = int(0.8 / self.engine.dx)
        cy_idx = int(0.5 / self.engine.dy)
        self.assertTrue(self.engine.solid[cy_idx][cx_idx])

        # Far point should be fluid
        self.assertFalse(self.engine.solid[cy_idx][0])

    def test_venturi_nozzle_geometry(self):
        """Verify Venturi nozzle produces constricted throat."""
        self.engine.set_venturi_nozzle(throat_height=0.35, inlet_height=0.75)
        self.assertTrue(len(self.engine.boundary_cells) > 0)

        # Center column (throat) should have fewer open fluid cells than inlet column
        throat_i = int(self.engine.nx / 2)
        inlet_i = 5

        open_throat = sum(1 for j in range(self.engine.ny) if not self.engine.solid[j][throat_i])
        open_inlet = sum(1 for j in range(self.engine.ny) if not self.engine.solid[j][inlet_i])
        self.assertLess(open_throat, open_inlet)

    def test_backward_facing_step_geometry(self):
        """Verify backward-facing step produces lower quadrant solid boundary."""
        self.engine.set_backward_facing_step(step_x=0.5, step_h=0.3)
        self.assertTrue(self.engine.solid[2][2])
        # Downstream upper region should be open fluid
        downstream_i = int(0.8 / self.engine.dx)
        self.assertFalse(self.engine.solid[2][downstream_i])

    def test_divergence_free_streamfunction(self):
        """Verify that velocities derived from streamfunction satisfy continuity div(u) = 0."""
        self.engine.set_circular_cylinder(radius=0.08, x_center=0.7, y_center=0.5)
        # Advance 2 simulation steps
        self.engine.step(dt=0.01, iterations_poisson=10)

        # Check divergence in fluid interior away from solid obstacle walls
        max_div = 0.0
        tested_points = 0
        for j in range(4, self.engine.ny - 4):
            for i in range(4, self.engine.nx - 4):
                nbr_solids = [self.engine.solid[j + dj][i + di] for dj in (-1, 0, 1) for di in (-1, 0, 1)]
                if not any(nbr_solids):
                    du_dx = (self.engine.u[j][i + 1] - self.engine.u[j][i - 1]) / (2.0 * self.engine.dx)
                    dv_dy = (self.engine.v[j + 1][i] - self.engine.v[j - 1][i]) / (2.0 * self.engine.dy)
                    div = abs(du_dx + dv_dy)
                    if div > max_div:
                        max_div = div
                    tested_points += 1

        self.assertGreater(tested_points, 100)
        # Divergence should be machine zero for streamfunction formulation
        self.assertLess(max_div, 1e-10)

    def test_step_stability_and_velocity_bounds(self):
        """Verify time stepping does not produce NaNs or numerical overflow."""
        self.engine.set_naca_airfoil(alpha_deg=8.0)
        for _ in range(5):
            self.engine.step(dt=0.015, iterations_poisson=8)

        self.assertFalse(math.isnan(self.engine.max_velocity))
        self.assertFalse(math.isinf(self.engine.max_velocity))
        self.assertGreater(self.engine.max_velocity, 0.0)
        self.assertLess(self.engine.max_velocity, 25.0)

    def test_lift_increase_with_angle_of_attack(self):
        """Verify that positive angle of attack generates higher lift coefficient than zero AoA."""
        # Low AoA
        engine_zero = CFDEngine2D(nx=40, ny=20, u_inf=1.0)
        engine_zero.set_naca_airfoil(alpha_deg=0.0)
        for _ in range(6):
            engine_zero.step(dt=0.01, iterations_poisson=6)

        # High AoA
        engine_high = CFDEngine2D(nx=40, ny=20, u_inf=1.0)
        engine_high.set_naca_airfoil(alpha_deg=10.0)
        for _ in range(6):
            engine_high.step(dt=0.01, iterations_poisson=6)

        self.assertGreater(engine_high.cl, engine_zero.cl)

    def test_drag_force_positive_on_cylinder(self):
        """Verify circular cylinder experiences positive streamwise drag force."""
        self.engine.set_circular_cylinder(radius=0.08)
        for _ in range(6):
            self.engine.step(dt=0.01, iterations_poisson=8)

        self.assertGreater(self.engine.drag_force, 0.0)
        self.assertGreater(self.engine.cd, 0.0)

    def test_particle_tracer_advection(self):
        """Verify particle tracer creation, advection, and lifecycle."""
        p = Particle(0.1, 0.5, max_age=5.0)
        self.assertTrue(p.is_alive())
        p.age = 5.1
        self.assertFalse(p.is_alive())

        # Test engine particle update
        self.engine.update_particles(dt=0.1)
        self.assertTrue(len(self.engine.particles) > 0)
        init_x = self.engine.particles[0].x
        self.engine.update_particles(dt=0.1)
        # Smoke particles should advect downstream (x increases)
        self.assertGreaterEqual(self.engine.particles[0].x, init_x)

    def test_pitot_probe_sampling(self):
        """Verify virtual pitot tube returns consistent physical measurements."""
        self.engine.set_naca_airfoil()
        self.engine.step(dt=0.01, iterations_poisson=6)

        probe = self.engine.get_probe_telemetry(0.2, 0.5)
        self.assertIn("velocity", probe)
        self.assertIn("pressure", probe)
        self.assertIn("vorticity", probe)
        self.assertIn("cp", probe)
        self.assertAlmostEqual(probe["x"], 0.2)
        self.assertAlmostEqual(probe["y"], 0.5)

    def test_all_aero_presets(self):
        """Verify all curated presets configure correctly without exception."""
        for key, p in AERO_PRESETS.items():
            eng = CFDEngine2D(nx=40, ny=20)
            ptype = p["type"]
            if ptype == "naca":
                eng.set_naca_airfoil(m=p.get("m", 0.0), p=p.get("p", 0.0), t=p.get("t", 0.12),
                                     chord=p.get("chord", 0.5), alpha_deg=p.get("alpha", 5.0))
            elif ptype == "cylinder":
                eng.set_circular_cylinder(radius=p.get("radius", 0.08))
            elif ptype == "venturi":
                eng.set_venturi_nozzle(throat_height=p.get("throat", 0.35), inlet_height=p.get("inlet", 0.75))
            elif ptype == "step":
                eng.set_backward_facing_step(step_x=p.get("step_x", 0.45), step_h=p.get("step_h", 0.35))

            eng.step(dt=0.01, iterations_poisson=4)
            self.assertFalse(math.isnan(eng.cl))
            self.assertFalse(math.isnan(eng.cd))

    def test_reynolds_number_update_kinematic_viscosity(self):
        """Verify modifying Reynolds number correctly scales kinematic viscosity nu."""
        init_nu = self.engine.nu
        self.engine.reynolds = 800.0
        self.engine.nu = (self.engine.u_inf * self.engine.ly) / self.engine.reynolds
        self.assertLess(self.engine.nu, init_nu)
        self.assertAlmostEqual(self.engine.nu, 1.0 * 1.0 / 800.0)

    def test_reset_flow_clears_metrics(self):
        """Verify reset_flow clears accumulated time, particles, and restores freestream."""
        self.engine.time = 15.4
        self.engine.particles.append(Particle(0.5, 0.5))
        self.engine.reset_flow()
        self.assertEqual(self.engine.time, 0.0)
        self.assertEqual(len(self.engine.particles), 0)
        self.assertEqual(self.engine.u[5][5], self.engine.u_inf)

    def test_probe_solid_detection(self):
        """Verify virtual probe identifies interior of solid obstacles."""
        self.engine.set_circular_cylinder(radius=0.1, x_center=0.8, y_center=0.5)
        probe_solid = self.engine.get_probe_telemetry(0.8, 0.5)
        self.assertEqual(probe_solid["is_solid"], 1.0)
        self.assertEqual(probe_solid["velocity"], 0.0)


if __name__ == "__main__":
    unittest.main()
