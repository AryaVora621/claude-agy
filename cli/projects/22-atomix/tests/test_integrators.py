"""
Unit tests for Atomix Symplectic Integrators and SHAKE Constraint Solver.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Atom, Bond, SimulationBox
from atomix.integrators import VelocityVerletIntegrator, SHAKEConstraintSolver


class TestIntegratorsAndConstraints(unittest.TestCase):
    def test_velocity_verlet_harmonic_oscillator(self):
        # 1D harmonic oscillator: m = 1, k = 1 -> omega = 1, period = 2*pi
        # Centered at (50, 50, 50) away from periodic boundary wraps
        box = SimulationBox(100.0, 100.0, 100.0)
        center = Vector3D(50.0, 50.0, 50.0)
        atom = Atom(
            id=0,
            name="Oscillator",
            element="Ar",
            position=Vector3D(51.0, 50.0, 50.0),
            velocity=Vector3D(0.0, 0.0, 0.0),
            mass=1.0,
        )
        disp = atom.position - center
        atom.force = -disp  # F = -k*x with k=1.0

        dt = 0.01
        integrator = VelocityVerletIntegrator(timestep=dt)

        initial_energy = 0.5 * atom.mass * atom.velocity.norm_sq() + 0.5 * disp.norm_sq()

        # Integrate for 2 complete periods (~628 steps)
        steps = int(2.0 * math.pi / dt) * 2
        for _ in range(steps):
            integrator.step_stage1([atom], box)
            disp = atom.position - center
            atom.force = -disp
            integrator.step_stage2([atom], box)

        disp = atom.position - center
        final_energy = 0.5 * atom.mass * atom.velocity.norm_sq() + 0.5 * disp.norm_sq()
        # Symplectic integrator should conserve total energy with high precision
        rel_energy_drift = abs(final_energy - initial_energy) / initial_energy
        self.assertLess(rel_energy_drift, 1e-4)

    def test_shake_rigid_bond_constraint(self):
        box = SimulationBox(20.0, 20.0, 20.0)
        target_length = 1.5

        # Two atoms connected by a rigid bond
        a1 = Atom(0, "O", "O", Vector3D(5.0, 5.0, 5.0), velocity=Vector3D(0.2, 0.1, 0.0), mass=16.0)
        a2 = Atom(1, "H", "H", Vector3D(5.0 + target_length, 5.0, 5.0), velocity=Vector3D(-0.2, 0.1, 0.0), mass=1.0)
        atoms = [a1, a2]
        rigid_bond = Bond(0, 1, length_eq=target_length, k_spring=0.0, is_rigid=True)

        integrator = VelocityVerletIntegrator(timestep=0.005, enable_shake=True, shake_tol=1e-7)

        # Run 100 integration steps with arbitrary random forces applied
        for step in range(100):
            # Apply some external perturbation force
            a1.force = Vector3D(math.sin(step), math.cos(step), 0.0)
            a2.force = Vector3D(-math.sin(step), 0.5 * math.cos(step), 0.0)

            integrator.step_stage1(atoms, box, [rigid_bond])
            integrator.step_stage2(atoms, box, [rigid_bond])

            # Measure actual bond distance
            bond_dist = box.minimum_image_vector(atoms[0].position, atoms[1].position).norm()
            self.assertAlmostEqual(bond_dist, target_length, places=5)

            # Measure relative velocity projection along bond vector (must be zero)
            r_ij = box.minimum_image_vector(atoms[0].position, atoms[1].position)
            v_ij = atoms[1].velocity - atoms[0].velocity
            v_proj = r_ij.dot(v_ij)
            self.assertAlmostEqual(v_proj, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
