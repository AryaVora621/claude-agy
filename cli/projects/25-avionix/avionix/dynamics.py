"""6-DOF Quadrotor Dynamics and Kinematics Engine.

Implements full rigid-body equations of motion on SE(3) with quaternion kinematics,
first-order rotor motor dynamics, aerodynamic drag, gyroscopic rotor torques,
and 4th-Order Runge-Kutta (RK4) numerical integration.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


class Vector3:
    """Immutable 3D spatial vector with linear algebra operations."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        if abs(scalar) < 1e-14:
            raise ZeroDivisionError("Division by zero in Vector3")
        inv = 1.0 / scalar
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: Vector3) -> float:
        """Standard Euclidean inner product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Vector cross product producing an orthogonal vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_sq(self) -> float:
        """Squared L2 Euclidean norm."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        """L2 Euclidean norm."""
        return math.sqrt(self.norm_sq())

    def normalized(self) -> Vector3:
        """Unit vector in the same direction, returning zero vector if degenerate."""
        n = self.norm()
        if n < 1e-12:
            return Vector3(0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: List[float]) -> Vector3:
        return cls(values[0], values[1], values[2])

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return False
        return (
            abs(self.x - other.x) < 1e-7
            and abs(self.y - other.y) < 1e-7
            and abs(self.z - other.z) < 1e-7
        )


class Matrix3x3:
    """3x3 rotation, inertia, and transformation matrix."""

    __slots__ = ("m",)

    def __init__(self, elements: Optional[List[List[float]]] = None) -> None:
        if elements is None:
            # Default to 3x3 identity matrix
            self.m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.m = [[float(elements[i][j]) for j in range(3)] for i in range(3)]

    @classmethod
    def identity(cls) -> Matrix3x3:
        return cls()

    @classmethod
    def diagonal(cls, d1: float, d2: float, d3: float) -> Matrix3x3:
        """Construct diagonal matrix, useful for principal moments of inertia."""
        return cls([[d1, 0.0, 0.0], [0.0, d2, 0.0], [0.0, 0.0, d3]])

    @classmethod
    def from_diagonal(cls, d1: float, d2: float, d3: float) -> Matrix3x3:
        return cls.diagonal(d1, d2, d3)

    @classmethod
    def from_columns(cls, col0: Vector3, col1: Vector3, col2: Vector3) -> Matrix3x3:
        """Construct matrix from 3 column vectors."""
        return cls([
            [col0.x, col1.x, col2.x],
            [col0.y, col1.y, col2.y],
            [col0.z, col1.z, col2.z],
        ])

    def dot_vec(self, v: Vector3) -> Vector3:
        """Matrix-vector product M * v."""
        return Vector3(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )

    def transpose(self) -> Matrix3x3:
        """Matrix transpose."""
        return Matrix3x3([
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        ])

    def matmul(self, other: Matrix3x3) -> Matrix3x3:
        """Matrix multiplication M * other."""
        res = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                res[i][j] = sum(self.m[i][k] * other.m[k][j] for k in range(3))
        return Matrix3x3(res)

    def determinant(self) -> float:
        """Determinant of 3x3 matrix via Laplace expansion."""
        a, b, c = self.m[0]
        d, e, f = self.m[1]
        g, h, k = self.m[2]
        return a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)

    def inverse(self) -> Matrix3x3:
        """Analytical inverse of 3x3 matrix via adjugate transpose."""
        det = self.determinant()
        if abs(det) < 1e-12:
            raise ValueError("Singular matrix cannot be inverted")
        inv_det = 1.0 / det
        a, b, c = self.m[0]
        d, e, f = self.m[1]
        g, h, k = self.m[2]

        adj = [
            [(e * k - f * h) * inv_det, (c * h - b * k) * inv_det, (b * f - c * e) * inv_det],
            [(f * g - d * k) * inv_det, (a * k - c * g) * inv_det, (c * d - a * f) * inv_det],
            [(d * h - e * g) * inv_det, (g * b - a * h) * inv_det, (a * e - b * d) * inv_det],
        ]
        return Matrix3x3(adj)

    def trace(self) -> float:
        """Sum of main diagonal elements."""
        return self.m[0][0] + self.m[1][1] + self.m[2][2]


class Quaternion:
    """Unit quaternion [w, x, y, z] parameterizing 3D rotations on SO(3).

    Avoids Euler angle kinematic singularities (gimbal lock) and allows smooth
    interpolation and stable numerical integration of angular velocity.
    """

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle_rad: float) -> Quaternion:
        """Construct quaternion from rotation axis and angle in radians."""
        u = axis.normalized()
        half = angle_rad * 0.5
        sin_half = math.sin(half)
        return cls(math.cos(half), u.x * sin_half, u.y * sin_half, u.z * sin_half).normalized()

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Construct quaternion from Z-Y-X Tait-Bryan Euler angles (yaw-pitch-roll)."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        ).normalized()

    @classmethod
    def from_rotation_matrix(cls, R: Matrix3x3) -> Quaternion:
        """Shepperd algorithm for stable extraction of quaternion from SO(3) matrix."""
        m = R.m
        tr = m[0][0] + m[1][1] + m[2][2]

        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2][1] - m[1][2]) / s
            y = (m[0][2] - m[2][0]) / s
            z = (m[1][0] - m[0][1]) / s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s

        return cls(w, x, y, z).normalized()

    def norm_sq(self) -> float:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def normalized(self) -> Quaternion:
        n = self.norm()
        if n < 1e-12:
            return Quaternion.identity()
        inv = 1.0 / n
        return Quaternion(self.w * inv, self.x * inv, self.y * inv, self.z * inv)

    def conjugate(self) -> Quaternion:
        """Quaternion conjugate reversing the spatial rotation direction."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> Quaternion:
        n2 = self.norm_sq()
        if n2 < 1e-12:
            return Quaternion.identity()
        inv = 1.0 / n2
        return Quaternion(self.w * inv, -self.x * inv, -self.y * inv, -self.z * inv)

    def multiply(self, other: Quaternion) -> Quaternion:
        """Hamilton quaternion product q1 * q2."""
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate_vec(self, v: Vector3) -> Vector3:
        """Rotate vector v from body frame to world frame via q * [0, v] * q*."""
        qv = Quaternion(0.0, v.x, v.y, v.z)
        rotated = self.multiply(qv).multiply(self.conjugate())
        return Vector3(rotated.x, rotated.y, rotated.z)

    def rotate_vector(self, v: Vector3) -> Vector3:
        return self.rotate_vec(v)

    def to_rotation_matrix(self) -> Matrix3x3:
        """Convert quaternion to 3x3 orthogonal rotation matrix in SO(3)."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return Matrix3x3([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ])

    def to_euler(self) -> Tuple[float, float, float]:
        """Extract roll, pitch, yaw (in radians) with gimbal-lock avoidance."""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi * 0.5, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def derivative(self, omega: Vector3) -> Tuple[float, float, float, float]:
        """Time derivative of attitude quaternion: q_dot = 0.5 * q * [0, omega]."""
        dw = 0.5 * (-self.x * omega.x - self.y * omega.y - self.z * omega.z)
        dx = 0.5 * (self.w * omega.x + self.y * omega.z - self.z * omega.y)
        dy = 0.5 * (self.w * omega.y - self.x * omega.z + self.z * omega.x)
        dz = 0.5 * (self.w * omega.z + self.x * omega.y - self.y * omega.x)
        return dw, dx, dy, dz

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


@dataclass
class QuadrotorParams:
    """Physical parameters of the quadrotor UAV in X-frame configuration."""

    mass_kg: float = 1.0
    arm_length_m: float = 0.225
    inertia_diagonal: Tuple[float, float, float] = (0.008, 0.008, 0.015)
    rotor_inertia_kg_m2: float = 6.0e-5
    thrust_coeff_c_t: float = 1.5e-5
    torque_coeff_c_q: float = 2.5e-7
    motor_tau_s: float = 0.025
    omega_min_rad_s: float = 100.0
    omega_max_rad_s: float = 1200.0
    drag_coeff_translational: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    gravity_m_s2: float = 9.81

    @property
    def J(self) -> Matrix3x3:
        """Diagonal inertia matrix."""
        d = self.inertia_diagonal
        return Matrix3x3.diagonal(d[0], d[1], d[2])

    @property
    def J_inv(self) -> Matrix3x3:
        """Inverse diagonal inertia matrix."""
        d = self.inertia_diagonal
        return Matrix3x3.diagonal(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])

    @property
    def hover_rotor_speed(self) -> float:
        """Rotor speed (rad/s) required for stationary hovering: 4 * c_T * Omega^2 = m * g."""
        hover_thrust_per_motor = (self.mass_kg * self.gravity_m_s2) / 4.0
        return math.sqrt(hover_thrust_per_motor / self.thrust_coeff_c_t)


@dataclass
class QuadrotorState:
    """Complete 6-DOF dynamic state vector of the quadrotor."""

    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    attitude: Quaternion = field(default_factory=Quaternion.identity)
    angular_velocity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    rotor_speeds: List[float] = field(default_factory=lambda: [404.0, 404.0, 404.0, 404.0])

    def copy(self) -> QuadrotorState:
        return QuadrotorState(
            position=Vector3(self.position.x, self.position.y, self.position.z),
            velocity=Vector3(self.velocity.x, self.velocity.y, self.velocity.z),
            attitude=Quaternion(self.attitude.w, self.attitude.x, self.attitude.y, self.attitude.z),
            angular_velocity=Vector3(self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z),
            rotor_speeds=list(self.rotor_speeds),
        )


class QuadrotorDynamics:
    """Nonlinear 6-DOF equations of motion for quadrotor UAV.

    Coordinate Frames:
      - World Frame W: East-North-Up (ENU) orthogonal right-handed frame (+Z Up).
      - Body Frame B: Forward-Left-Up (+X Forward, +Y Left, +Z Normal Upward).
      - Gravity Vector in W: [0, 0, -g]^T.
    """

    def __init__(self, params: Optional[QuadrotorParams] = None) -> None:
        self.params = params or QuadrotorParams()

    def hover_rotor_speed_rad_s(self) -> float:
        """Hover angular speed in rad/s."""
        return self.params.hover_rotor_speed

    def compute_forces_and_torques(
        self, rotor_speeds: List[float]
    ) -> Tuple[float, Vector3]:
        """Compute collective thrust f and body torques tau from rotor speeds in X-configuration.

        Motors are located in standard X-frame at angle pi/4:
          Motor 1: Front-Right (+x, -y relative to body center), CCW
          Motor 2: Rear-Right (-x, -y), CW
          Motor 3: Rear-Left (-x, +y), CCW
          Motor 4: Front-Left (+x, +y), CW
        """
        p = self.params
        c_t = p.thrust_coeff_c_t
        c_q = p.torque_coeff_c_q
        d_arm = p.arm_length_m * (1.0 / math.sqrt(2.0))

        # Thrust produced by each propeller
        f1 = c_t * (rotor_speeds[0] ** 2)
        f2 = c_t * (rotor_speeds[1] ** 2)
        f3 = c_t * (rotor_speeds[2] ** 2)
        f4 = c_t * (rotor_speeds[3] ** 2)

        # Collective thrust along body Z axis
        collective_thrust = f1 + f2 + f3 + f4

        # Roll moment (about body X axis, right/left differential)
        # Positive roll tilts right (+y side pushes up, -y side pushes down)
        tau_x = d_arm * (f3 + f4 - f1 - f2)

        # Pitch moment (about body Y axis, rear/front differential)
        # Positive pitch pitches up (nose up: rear motors push up, front motors push down)
        tau_y = d_arm * (f2 + f3 - f1 - f4)

        # Yaw moment (about body Z axis, reaction drag torques)
        # CCW rotors (1 and 3) impart CW reaction torque (-Z)
        # CW rotors (2 and 4) impart CCW reaction torque (+Z)
        tau_z = c_q * (-(rotor_speeds[0] ** 2) + (rotor_speeds[1] ** 2) - (rotor_speeds[2] ** 2) + (rotor_speeds[3] ** 2))

        return collective_thrust, Vector3(tau_x, tau_y, tau_z)

    def derivatives(
        self,
        state: QuadrotorState,
        rotor_commands: List[float],
        external_force_w: Optional[Vector3] = None,
    ) -> Tuple[Vector3, Vector3, Tuple[float, float, float, float], Vector3, List[float]]:
        """Calculate time derivatives (p_dot, v_dot, q_dot, omega_dot, rotor_dot)."""
        p = self.params
        ext_f = external_force_w or Vector3(0.0, 0.0, 0.0)

        # 1. Position derivative: p_dot = v
        p_dot = state.velocity

        # 2. Rotor dynamics: first-order motor lag
        # dOmega_i / dt = (Omega_cmd,i - Omega_i) / tau_m
        rotor_dot = []
        for i in range(4):
            cmd_clamped = max(p.omega_min_rad_s, min(p.omega_max_rad_s, rotor_commands[i]))
            rotor_dot.append((cmd_clamped - state.rotor_speeds[i]) / p.motor_tau_s)

        # 3. Collective thrust and body torques
        thrust, torques = self.compute_forces_and_torques(state.rotor_speeds)

        # 4. Translational acceleration in world frame:
        # m * v_dot = R * [0, 0, thrust]^T - m * g * e_3 - F_drag + F_ext
        thrust_body = Vector3(0.0, 0.0, thrust)
        thrust_world = state.attitude.rotate_vec(thrust_body)
        gravity_world = Vector3(0.0, 0.0, -p.mass_kg * p.gravity_m_s2)

        # Parasitic drag
        drag_world = Vector3(
            -p.drag_coeff_translational[0] * state.velocity.x,
            -p.drag_coeff_translational[1] * state.velocity.y,
            -p.drag_coeff_translational[2] * state.velocity.z,
        )

        total_force_w = thrust_world + gravity_world + drag_world + ext_f
        v_dot = total_force_w / p.mass_kg

        # 5. Quaternion attitude kinematics: q_dot = 0.5 * q * [0, omega]
        q_dot = state.attitude.derivative(state.angular_velocity)

        # 6. Rotational acceleration in body frame:
        # J * omega_dot = tau_b - omega x (J * omega) - tau_gyroscopic
        omega = state.angular_velocity
        J_omega = p.J.dot_vec(omega)
        gyroscopic_body = omega.cross(J_omega)

        # Propeller gyroscopic precession torque
        # tau_gyro = -sum(J_r * (omega x e3) * (-1)^i * Omega_i)
        e3 = Vector3(0.0, 0.0, 1.0)
        omega_cross_e3 = omega.cross(e3)
        # Signs: CCW (+), CW (-)
        signed_speed_sum = (
            state.rotor_speeds[0]
            - state.rotor_speeds[1]
            + state.rotor_speeds[2]
            - state.rotor_speeds[3]
        )
        prop_gyro_torque = omega_cross_e3 * (-p.rotor_inertia_kg_m2 * signed_speed_sum)

        total_torque = torques - gyroscopic_body - prop_gyro_torque
        omega_dot = p.J_inv.dot_vec(total_torque)

        return p_dot, v_dot, q_dot, omega_dot, rotor_dot

    def step_rk4(
        self,
        state: QuadrotorState,
        dt_s: float,
        rotor_commands: List[float],
        external_force_w: Optional[Vector3] = None,
    ) -> QuadrotorState:
        """Advance the full state by dt_s using 4th-Order Runge-Kutta integration.

        Re-normalizes quaternion after integration to maintain numerical orthogonality on SO(3).
        """
        # k1
        p_dot1, v_dot1, q_dot1, w_dot1, r_dot1 = self.derivatives(state, rotor_commands, external_force_w)

        # k2 state
        s2 = QuadrotorState(
            position=state.position + p_dot1 * (0.5 * dt_s),
            velocity=state.velocity + v_dot1 * (0.5 * dt_s),
            attitude=Quaternion(
                state.attitude.w + q_dot1[0] * (0.5 * dt_s),
                state.attitude.x + q_dot1[1] * (0.5 * dt_s),
                state.attitude.y + q_dot1[2] * (0.5 * dt_s),
                state.attitude.z + q_dot1[3] * (0.5 * dt_s),
            ).normalized(),
            angular_velocity=state.angular_velocity + w_dot1 * (0.5 * dt_s),
            rotor_speeds=[state.rotor_speeds[i] + r_dot1[i] * (0.5 * dt_s) for i in range(4)],
        )
        p_dot2, v_dot2, q_dot2, w_dot2, r_dot2 = self.derivatives(s2, rotor_commands, external_force_w)

        # k3 state
        s3 = QuadrotorState(
            position=state.position + p_dot2 * (0.5 * dt_s),
            velocity=state.velocity + v_dot2 * (0.5 * dt_s),
            attitude=Quaternion(
                state.attitude.w + q_dot2[0] * (0.5 * dt_s),
                state.attitude.x + q_dot2[1] * (0.5 * dt_s),
                state.attitude.y + q_dot2[2] * (0.5 * dt_s),
                state.attitude.z + q_dot2[3] * (0.5 * dt_s),
            ).normalized(),
            angular_velocity=state.angular_velocity + w_dot2 * (0.5 * dt_s),
            rotor_speeds=[state.rotor_speeds[i] + r_dot2[i] * (0.5 * dt_s) for i in range(4)],
        )
        p_dot3, v_dot3, q_dot3, w_dot3, r_dot3 = self.derivatives(s3, rotor_commands, external_force_w)

        # k4 state
        s4 = QuadrotorState(
            position=state.position + p_dot3 * dt_s,
            velocity=state.velocity + v_dot3 * dt_s,
            attitude=Quaternion(
                state.attitude.w + q_dot3[0] * dt_s,
                state.attitude.x + q_dot3[1] * dt_s,
                state.attitude.y + q_dot3[2] * dt_s,
                state.attitude.z + q_dot3[3] * dt_s,
            ).normalized(),
            angular_velocity=state.angular_velocity + w_dot3 * dt_s,
            rotor_speeds=[state.rotor_speeds[i] + r_dot3[i] * dt_s for i in range(4)],
        )
        p_dot4, v_dot4, q_dot4, w_dot4, r_dot4 = self.derivatives(s4, rotor_commands, external_force_w)

        # Weighted RK4 combination
        new_pos = state.position + (p_dot1 + 2.0 * p_dot2 + 2.0 * p_dot3 + p_dot4) * (dt_s / 6.0)
        new_vel = state.velocity + (v_dot1 + 2.0 * v_dot2 + 2.0 * v_dot3 + v_dot4) * (dt_s / 6.0)

        new_qw = state.attitude.w + (q_dot1[0] + 2.0 * q_dot2[0] + 2.0 * q_dot3[0] + q_dot4[0]) * (dt_s / 6.0)
        new_qx = state.attitude.x + (q_dot1[1] + 2.0 * q_dot2[1] + 2.0 * q_dot3[1] + q_dot4[1]) * (dt_s / 6.0)
        new_qy = state.attitude.y + (q_dot1[2] + 2.0 * q_dot2[2] + 2.0 * q_dot3[2] + q_dot4[2]) * (dt_s / 6.0)
        new_qz = state.attitude.z + (q_dot1[3] + 2.0 * q_dot2[3] + 2.0 * q_dot3[3] + q_dot4[3]) * (dt_s / 6.0)
        new_att = Quaternion(new_qw, new_qx, new_qy, new_qz).normalized()

        new_omega = state.angular_velocity + (w_dot1 + 2.0 * w_dot2 + 2.0 * w_dot3 + w_dot4) * (dt_s / 6.0)
        new_rotors = [
            state.rotor_speeds[i] + (r_dot1[i] + 2.0 * r_dot2[i] + 2.0 * r_dot3[i] + r_dot4[i]) * (dt_s / 6.0)
            for i in range(4)
        ]

        return QuadrotorState(
            position=new_pos,
            velocity=new_vel,
            attitude=new_att,
            angular_velocity=new_omega,
            rotor_speeds=new_rotors,
        )
