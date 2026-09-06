"""Unit tests for SE(3) Geometric Tracking Control, Lie Algebra, and Motor Mixer."""

import math
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.control import (
    CascadedPIDController,
    ControlWrench,
    GeometricGains,
    MotorMixer,
    SE3GeometricController,
    skew_symmetric,
    vee_unskew,
)
from avionix.dynamics import Matrix3x3, QuadrotorParams, QuadrotorState, Quaternion, Vector3
from avionix.trajectory import FullTrajectoryState


class TestLieAlgebraOperators(unittest.TestCase):
    """Test so(3) Lie algebra hat (skew) and vee (unskew) operators."""

    def test_hat_vee_isomorphism(self) -> None:
        v = Vector3(3.2, -4.5, 1.8)
        S = skew_symmetric(v)

        # Skew-symmetry property: S + S^T = 0
        St = S.transpose()
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(S.m[r][c] + St.m[r][c], 0.0)

        # Cross product equivalence: S * u = v x u
        u = Vector3(-1.0, 2.0, 5.0)
        cross_expected = v.cross(u)
        cross_actual = S.dot_vec(u)
        self.assertAlmostEqual(cross_actual.x, cross_expected.x)
        self.assertAlmostEqual(cross_actual.y, cross_expected.y)
        self.assertAlmostEqual(cross_actual.z, cross_expected.z)

        # Vee inverse recovers original vector: vee(hat(v)) = v
        v_recovered = vee_unskew(S)
        self.assertAlmostEqual(v_recovered.x, v.x)
        self.assertAlmostEqual(v_recovered.y, v.y)
        self.assertAlmostEqual(v_recovered.z, v.z)


class TestMotorMixer(unittest.TestCase):
    """Test X-frame motor mixer and priority anti-saturation."""

    def setUp(self) -> None:
        self.params = QuadrotorParams()
        self.mixer = MotorMixer(self.params)

    def test_hover_allocation(self) -> None:
        # Hover thrust with zero moments should allocate equal speeds to all 4 motors
        hover_thrust = self.params.mass_kg * self.params.gravity_m_s2
        torques = Vector3(0.0, 0.0, 0.0)

        speeds = self.mixer.mix(hover_thrust, torques)
        self.assertEqual(len(speeds), 4)
        for i in range(4):
            self.assertAlmostEqual(speeds[i], self.params.hover_rotor_speed, places=2)

    def test_roll_torque_differential(self) -> None:
        hover_thrust = self.params.mass_kg * self.params.gravity_m_s2
        # Positive roll moment requires right motors (1, 2) to slow down and left motors (3, 4) to speed up
        roll_torque = Vector3(0.05, 0.0, 0.0)
        speeds = self.mixer.mix(hover_thrust, roll_torque)

        # Motors 3 and 4 (left) should have higher speed than 1 and 2 (right)
        self.assertGreater(speeds[2], speeds[0])
        self.assertGreater(speeds[3], speeds[1])

    def test_attitude_priority_saturation(self) -> None:
        # Extreme torque exceeding max rotor speed
        huge_torque = Vector3(5.0, 5.0, 5.0)
        speeds = self.mixer.mix(20.0, huge_torque)

        # All speeds must remain strictly within [omega_min, omega_max] bounds
        for s in speeds:
            self.assertGreaterEqual(s, self.params.omega_min_rad_s - 1e-4)
            self.assertLessEqual(s, self.params.omega_max_rad_s + 1e-4)


class TestSE3GeometricController(unittest.TestCase):
    """Test Lee-Leok-McClamroch SE(3) geometric tracking controller."""

    def setUp(self) -> None:
        self.params = QuadrotorParams(mass_kg=1.0)
        self.controller = SE3GeometricController(self.params)

    def test_hover_equilibrium(self) -> None:
        hover_pos = Vector3(0.0, 0.0, 2.0)
        state = QuadrotorState(position=hover_pos)
        ref = FullTrajectoryState(
            time_s=0.0,
            position=hover_pos,
            velocity=Vector3(0.0, 0.0, 0.0),
            acceleration=Vector3(0.0, 0.0, 0.0),
            thrust_n=self.params.mass_kg * self.params.gravity_m_s2,
            attitude=Quaternion.identity(),
            rotation_matrix=Matrix3x3.identity(),
            angular_velocity=Vector3(0.0, 0.0, 0.0),
            angular_acceleration=Vector3(0.0, 0.0, 0.0),
            body_rates=Vector3(0.0, 0.0, 0.0),
        )

        wrench = self.controller.update(state, ref, dt_s=0.01)
        # In hover, collective thrust must equal m * g
        expected_thrust = self.params.mass_kg * self.params.gravity_m_s2
        self.assertAlmostEqual(wrench.collective_thrust, expected_thrust, places=3)
        self.assertAlmostEqual(wrench.moments.norm(), 0.0, places=4)

    def test_position_error_restoring_force(self) -> None:
        # Quadrotor is 1m below reference -> Controller must command thrust > m * g
        ref = FullTrajectoryState(
            time_s=0.0,
            position=Vector3(0.0, 0.0, 3.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            acceleration=Vector3(0.0, 0.0, 0.0),
            thrust_n=self.params.mass_kg * self.params.gravity_m_s2,
            attitude=Quaternion.identity(),
            rotation_matrix=Matrix3x3.identity(),
            angular_velocity=Vector3(0.0, 0.0, 0.0),
            angular_acceleration=Vector3(0.0, 0.0, 0.0),
            body_rates=Vector3(0.0, 0.0, 0.0),
        )
        state = QuadrotorState(position=Vector3(0.0, 0.0, 2.0))

        wrench = self.controller.update(state, ref, dt_s=0.01)
        hover_thrust = self.params.mass_kg * self.params.gravity_m_s2
        self.assertGreater(wrench.collective_thrust, hover_thrust)


class TestCascadedPIDController(unittest.TestCase):
    """Test Cascaded PID controller baseline."""

    def test_pid_thrust_response(self) -> None:
        params = QuadrotorParams()
        pid = CascadedPIDController(params)

        state = QuadrotorState(position=Vector3(0.0, 0.0, 1.0))
        ref = FullTrajectoryState(
            time_s=0.0,
            position=Vector3(0.0, 0.0, 2.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            acceleration=Vector3(0.0, 0.0, 0.0),
            thrust_n=params.mass_kg * params.gravity_m_s2,
            attitude=Quaternion.identity(),
            rotation_matrix=Matrix3x3.identity(),
            angular_velocity=Vector3(0.0, 0.0, 0.0),
            angular_acceleration=Vector3(0.0, 0.0, 0.0),
            body_rates=Vector3(0.0, 0.0, 0.0),
        )

        wrench = pid.update(state, ref, dt_s=0.01)
        self.assertGreater(wrench.collective_thrust, params.mass_kg * params.gravity_m_s2)
        self.assertEqual(len(wrench.rotor_commands), 4)


if __name__ == "__main__":
    unittest.main()
