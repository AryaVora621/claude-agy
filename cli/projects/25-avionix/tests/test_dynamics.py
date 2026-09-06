"""Unit tests for 6-DOF Quadrotor Dynamics, Kinematics, and Quaternion Mathematics."""

import math
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.dynamics import (
    Matrix3x3,
    QuadrotorDynamics,
    QuadrotorParams,
    QuadrotorState,
    Quaternion,
    Vector3,
)


class TestVector3(unittest.TestCase):
    """Test 3D Vector algebra."""

    def test_arithmetic(self) -> None:
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, -5.0, 6.0)

        add = v1 + v2
        self.assertAlmostEqual(add.x, 5.0)
        self.assertAlmostEqual(add.y, -3.0)
        self.assertAlmostEqual(add.z, 9.0)

        sub = v1 - v2
        self.assertAlmostEqual(sub.x, -3.0)
        self.assertAlmostEqual(sub.y, 7.0)
        self.assertAlmostEqual(sub.z, -3.0)

        scale = v1 * 2.5
        self.assertAlmostEqual(scale.x, 2.5)
        self.assertAlmostEqual(scale.y, 5.0)
        self.assertAlmostEqual(scale.z, 7.5)

        neg = -v1
        self.assertAlmostEqual(neg.x, -1.0)
        self.assertAlmostEqual(neg.y, -2.0)
        self.assertAlmostEqual(neg.z, -3.0)

    def test_products_and_norms(self) -> None:
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)

        # Dot product
        self.assertAlmostEqual(v1.dot(v2), 0.0)
        self.assertAlmostEqual(v1.dot(v1), 1.0)

        # Cross product: i x j = k
        cross = v1.cross(v2)
        self.assertAlmostEqual(cross.x, 0.0)
        self.assertAlmostEqual(cross.y, 0.0)
        self.assertAlmostEqual(cross.z, 1.0)

        # Norms
        v3 = Vector3(3.0, 4.0, 12.0)
        self.assertAlmostEqual(v3.norm_sq(), 169.0)
        self.assertAlmostEqual(v3.norm(), 13.0)

        normed = v3.normalized()
        self.assertAlmostEqual(normed.norm(), 1.0)
        self.assertAlmostEqual(normed.x, 3.0 / 13.0)

        # Zero vector normalization protection
        zero = Vector3(0.0, 0.0, 0.0)
        self.assertEqual(zero.normalized(), zero)


class TestMatrix3x3(unittest.TestCase):
    """Test 3x3 Matrix operations."""

    def test_matrix_operations(self) -> None:
        I = Matrix3x3.identity()
        v = Vector3(2.5, -3.0, 7.2)
        v_res = I.dot_vec(v)
        self.assertAlmostEqual(v_res.x, v.x)
        self.assertAlmostEqual(v_res.y, v.y)
        self.assertAlmostEqual(v_res.z, v.z)

        # Transpose
        M = Matrix3x3([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        Mt = M.transpose()
        self.assertEqual(Mt.m[0][1], 4.0)
        self.assertEqual(Mt.m[1][0], 2.0)
        self.assertEqual(Mt.m[2][0], 3.0)

        # Matrix multiplication
        res = I.matmul(M)
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(res.m[r][c], M.m[r][c])

        # Inversion
        diag = Matrix3x3.from_diagonal(2.0, 4.0, 8.0)
        inv = diag.inverse()
        self.assertAlmostEqual(inv.m[0][0], 0.5)
        self.assertAlmostEqual(inv.m[1][1], 0.25)
        self.assertAlmostEqual(inv.m[2][2], 0.125)


class TestQuaternion(unittest.TestCase):
    """Test Unit Quaternion kinematics and SO(3) rotations."""

    def test_quaternion_rotations(self) -> None:
        # 90 degree yaw rotation around Z axis
        angle = math.pi / 2.0
        q_yaw = Quaternion(math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)).normalized()

        # Vector along X axis rotated 90 deg yaw should point along Y axis
        vx = Vector3(1.0, 0.0, 0.0)
        v_rot = q_yaw.rotate_vector(vx)
        self.assertAlmostEqual(v_rot.x, 0.0, places=5)
        self.assertAlmostEqual(v_rot.y, 1.0, places=5)
        self.assertAlmostEqual(v_rot.z, 0.0, places=5)

    def test_euler_conversions(self) -> None:
        # Test Euler roundtrip
        roll, pitch, yaw = math.radians(20.0), math.radians(-15.0), math.radians(65.0)
        q = Quaternion.from_euler(roll, pitch, yaw)
        r_out, p_out, y_out = q.to_euler()

        self.assertAlmostEqual(r_out, roll, places=4)
        self.assertAlmostEqual(p_out, pitch, places=4)
        self.assertAlmostEqual(y_out, yaw, places=4)

    def test_rotation_matrix_roundtrip(self) -> None:
        q = Quaternion.from_euler(math.radians(30.0), math.radians(45.0), math.radians(-60.0))
        R = q.to_rotation_matrix()
        # Rotation matrix columns must be orthonormal
        c0 = Vector3(R.m[0][0], R.m[1][0], R.m[2][0])
        c1 = Vector3(R.m[0][1], R.m[1][1], R.m[2][1])
        c2 = Vector3(R.m[0][2], R.m[1][2], R.m[2][2])

        self.assertAlmostEqual(c0.norm(), 1.0, places=5)
        self.assertAlmostEqual(c1.norm(), 1.0, places=5)
        self.assertAlmostEqual(c2.norm(), 1.0, places=5)
        self.assertAlmostEqual(c0.dot(c1), 0.0, places=5)
        self.assertAlmostEqual(c1.dot(c2), 0.0, places=5)


class TestQuadrotorDynamics(unittest.TestCase):
    """Test physical 6-DOF equations of motion and numerical integration."""

    def setUp(self) -> None:
        self.params = QuadrotorParams(mass_kg=1.5)
        self.dynamics = QuadrotorDynamics(self.params)

    def test_hover_equilibrium(self) -> None:
        hover_speed = self.dynamics.hover_rotor_speed_rad_s()
        state = QuadrotorState(rotor_speeds=[hover_speed] * 4)

        # In hover, total thrust must equal m * g exactly
        thrust_n = 4.0 * self.params.thrust_coeff_c_t * (hover_speed ** 2)
        weight_n = self.params.mass_kg * self.params.gravity_m_s2
        self.assertAlmostEqual(thrust_n, weight_n, places=4)

        # Derivatives at hover with rotor commands matching hover speed should have near-zero acceleration
        p_dot, v_dot, q_dot, w_dot, r_dot = self.dynamics.derivatives(
            state, [hover_speed] * 4
        )
        self.assertAlmostEqual(v_dot.x, 0.0, places=5)
        self.assertAlmostEqual(v_dot.y, 0.0, places=5)
        self.assertAlmostEqual(v_dot.z, 0.0, places=5)

    def test_free_fall_dynamics(self) -> None:
        # Zero rotor speed -> Free fall under gravity
        state = QuadrotorState(rotor_speeds=[0.0] * 4)
        dt = 0.01
        new_state = self.dynamics.step_rk4(state, dt, [0.0] * 4)

        # Velocity after dt should be -g * dt
        expected_vz = -self.params.gravity_m_s2 * dt
        self.assertAlmostEqual(new_state.velocity.z, expected_vz, places=3)
        self.assertLess(new_state.position.z, 0.0)

    def test_quaternion_normalization_preservation(self) -> None:
        state = QuadrotorState(angular_velocity=Vector3(2.0, -1.5, 3.0))
        dt = 0.01
        stepped = self.dynamics.step_rk4(state, dt, [500.0] * 4)
        self.assertAlmostEqual(stepped.attitude.norm(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
