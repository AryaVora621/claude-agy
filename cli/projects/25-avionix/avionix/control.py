"""SE(3) Geometric Tracking Control and Motor Mixer for Quadrotor UAVs.

Implements:
1. Lee-Leok-McClamroch (2010) Geometric Tracking Controller directly on SE(3) and SO(3).
2. Almost-global Lyapunov attitude stability without gimbal-lock or Euler singularities.
3. Cascaded Nonlinear PID Controller as secondary benchmark architecture.
4. Quadrotor X-Configuration Motor Mixer with attitude-priority anti-saturation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .dynamics import Matrix3x3, QuadrotorParams, QuadrotorState, Quaternion, Vector3
from .trajectory import FullTrajectoryState


@dataclass
class GeometricGains:
    """Controller gain parameters for SE(3) geometric tracking."""

    # Position loop gains (world frame)
    kx: Vector3 = field(default_factory=lambda: Vector3(6.5, 6.5, 12.0))
    kv: Vector3 = field(default_factory=lambda: Vector3(4.0, 4.0, 6.5))
    ki: Vector3 = field(default_factory=lambda: Vector3(0.5, 0.5, 1.2))

    # Attitude loop gains (SO(3) Lie algebra)
    kR: Vector3 = field(default_factory=lambda: Vector3(4.5, 4.5, 2.8))
    kOmega: Vector3 = field(default_factory=lambda: Vector3(0.75, 0.75, 0.45))

    # Integrator anti-windup saturation limit in Newtons
    integral_force_max_n: float = 3.0


@dataclass
class ControlWrench:
    """Resulting control wrench commanded by controller."""

    collective_thrust: float
    moments: Vector3  # [tau_x, tau_y, tau_z] in body frame
    rotor_commands: List[float]


def skew_symmetric(v: Vector3) -> Matrix3x3:
    """Hat operator (^) mapping vector in R^3 to Lie algebra so(3)."""
    return Matrix3x3([
        [0.0, -v.z, v.y],
        [v.z, 0.0, -v.x],
        [-v.y, v.x, 0.0],
    ])


def vee_unskew(S: Matrix3x3) -> Vector3:
    """Vee operator (v) mapping so(3) skew-symmetric matrix back to R^3."""
    # (S - S^T) / 2 is strictly skew-symmetric
    m = S.m
    return Vector3(
        0.5 * (m[2][1] - m[1][2]),
        0.5 * (m[0][2] - m[2][0]),
        0.5 * (m[1][0] - m[0][1]),
    )


class MotorMixer:
    """Maps total thrust and body torques to individual rotor speeds in X-configuration."""

    def __init__(self, params: Optional[QuadrotorParams] = None) -> None:
        self.params = params or QuadrotorParams()
        p = self.params
        c_t = p.thrust_coeff_c_t
        c_q = p.torque_coeff_c_q
        d_arm = p.arm_length_m * (1.0 / math.sqrt(2.0))

        # Forward mixer matrix M mapping [Omega_1^2, ..., Omega_4^2]^T to [f, tau_x, tau_y, tau_z]^T
        # Row 0: Thrust   = c_t * (w1 + w2 + w3 + w4)
        # Row 1: Roll Tau = d * c_t * (-w1 - w2 + w3 + w4)
        # Row 2: PitchTau = d * c_t * (-w1 + w2 + w3 - w4)
        # Row 3: Yaw Tau  = c_q * (-w1 + w2 - w3 + w4)
        # We invert this 4x4 matrix analytically for numerical efficiency:
        inv_4ct = 1.0 / (4.0 * c_t)
        inv_4ct_d = 1.0 / (4.0 * c_t * d_arm)
        inv_4cq = 1.0 / (4.0 * c_q)

        self.inv_mixer = [
            [inv_4ct, -inv_4ct_d, -inv_4ct_d, -inv_4cq],
            [inv_4ct, -inv_4ct_d,  inv_4ct_d,  inv_4cq],
            [inv_4ct,  inv_4ct_d,  inv_4ct_d, -inv_4cq],
            [inv_4ct,  inv_4ct_d, -inv_4ct_d,  inv_4cq],
        ]

    def mix(self, thrust_n: float, torques: Vector3) -> List[float]:
        """Convert [f, tau_x, tau_y, tau_z] to motor speeds with priority desaturation."""
        p = self.params
        w = [thrust_n, torques.x, torques.y, torques.z]

        omega_sq = [0.0] * 4
        for i in range(4):
            omega_sq[i] = sum(self.inv_mixer[i][j] * w[j] for j in range(4))

        # Check for saturation
        min_sq = p.omega_min_rad_s ** 2
        max_sq = p.omega_max_rad_s ** 2

        # Attitude-priority desaturation: if any rotor drops below min or exceeds max,
        # shift collective thrust to maintain attitude authority
        shift = 0.0
        most_negative = min(omega_sq)
        most_positive = max(omega_sq)

        if most_negative < min_sq:
            shift = min_sq - most_negative
        elif most_positive > max_sq:
            shift = max_sq - most_positive

        rotor_speeds = []
        for i in range(4):
            val = max(min_sq, min(max_sq, omega_sq[i] + shift))
            rotor_speeds.append(math.sqrt(val))

        return rotor_speeds


class SE3GeometricController:
    """Lee-Leok-McClamroch (2010) Geometric Tracking Controller on SE(3).

    Guarantees almost-global exponential tracking on the Lie group SO(3),
    completely eliminating gimbal-lock singularities during aggressive flips.
    """

    def __init__(
        self,
        params: Optional[QuadrotorParams] = None,
        gains: Optional[GeometricGains] = None,
    ) -> None:
        self.params = params or QuadrotorParams()
        self.gains = gains or GeometricGains()
        self.mixer = MotorMixer(self.params)

        # Position error integral accumulator
        self.pos_error_integral = Vector3(0.0, 0.0, 0.0)

    def reset_integrators(self) -> None:
        """Reset internal error accumulators."""
        self.pos_error_integral = Vector3(0.0, 0.0, 0.0)

    def update(
        self,
        current_state: QuadrotorState,
        ref_state: FullTrajectoryState,
        dt_s: float,
    ) -> ControlWrench:
        """Compute control thrust, torques, and rotor commands from state feedback."""
        p = self.params
        g = self.gains
        m = p.mass_kg

        # 1. Position and velocity errors in world frame
        e_p = current_state.position - ref_state.position
        e_v = current_state.velocity - ref_state.velocity

        # Update position integral with anti-windup clamping
        self.pos_error_integral = self.pos_error_integral + e_p * dt_s
        i_max = g.integral_force_max_n
        ix = max(-i_max, min(i_max, self.pos_error_integral.x * g.ki.x))
        iy = max(-i_max, min(i_max, self.pos_error_integral.y * g.ki.y))
        iz = max(-i_max, min(i_max, self.pos_error_integral.z * g.ki.z))
        integral_force = Vector3(ix, iy, iz)

        # 2. Desired total force vector in world frame
        # A = -k_x * e_p - k_v * e_v - k_i * e_i + m * g * e3 + m * a_d
        gravity_force = Vector3(0.0, 0.0, m * p.gravity_m_s2)
        feedforward_force = ref_state.acceleration * m
        prop_force = Vector3(
            -g.kx.x * e_p.x - g.kv.x * e_v.x,
            -g.kx.y * e_p.y - g.kv.y * e_v.y,
            -g.kx.z * e_p.z - g.kv.z * e_v.z,
        )

        A = prop_force - integral_force + gravity_force + feedforward_force

        # 3. Collective thrust projected along actual body z-axis
        R = current_state.attitude.to_rotation_matrix()
        b3_actual = R.dot_vec(Vector3(0.0, 0.0, 1.0))
        collective_thrust = A.dot(b3_actual)

        # Ensure positive thrust command
        collective_thrust = max(0.05 * m * p.gravity_m_s2, collective_thrust)

        # 4. Desired rotation matrix R_d = [b1_d, b2_d, b3_d]
        b3_d = A.normalized()

        # Desired heading vector from reference yaw
        _, _, ref_yaw = ref_state.attitude.to_euler()
        b1_c = Vector3(math.cos(ref_yaw), math.sin(ref_yaw), 0.0)

        b2_d_unnorm = b3_d.cross(b1_c)
        if b2_d_unnorm.norm() < 1e-4:
            # Singularity: pitch is +-90 degrees
            b2_d = Vector3(-math.sin(ref_yaw), math.cos(ref_yaw), 0.0)
            b1_d = b2_d.cross(b3_d).normalized()
        else:
            b2_d = b2_d_unnorm.normalized()
            b1_d = b2_d.cross(b3_d)

        R_d = Matrix3x3.from_columns(b1_d, b2_d, b3_d)

        # 5. Attitude error vector on SO(3): e_R = 0.5 * (R_d^T * R - R^T * R_d)^v
        # Measures the chordal distance on the Lie group SO(3)
        R_d_T = R_d.transpose()
        R_T = R.transpose()

        error_matrix = R_d_T.matmul(R).matmul(Matrix3x3.identity())
        error_matrix_rev = R_T.matmul(R_d)
        skew_error = Matrix3x3([
            [error_matrix.m[i][j] - error_matrix_rev.m[i][j] for j in range(3)]
            for i in range(3)
        ])
        e_R = vee_unskew(skew_error)

        # 6. Angular rate error: e_omega = omega - R^T * R_d * omega_d
        omega_d_in_body = R_T.matmul(R_d).dot_vec(ref_state.angular_velocity)
        e_omega = current_state.angular_velocity - omega_d_in_body

        # 7. Control moments on body frame:
        # tau = -k_R * e_R - k_omega * e_omega + omega x (J * omega) - J * feedforward
        J = p.J
        omega = current_state.angular_velocity
        coriolis_torque = omega.cross(J.dot_vec(omega))

        prop_moment = Vector3(
            -g.kR.x * e_R.x - g.kOmega.x * e_omega.x,
            -g.kR.y * e_R.y - g.kOmega.y * e_omega.y,
            -g.kR.z * e_R.z - g.kOmega.z * e_omega.z,
        )

        feedforward_torque = J.dot_vec(R_T.matmul(R_d).dot_vec(ref_state.angular_acceleration))
        total_moments = prop_moment + coriolis_torque - feedforward_torque

        # 8. Motor speed allocation
        rotor_cmds = self.mixer.mix(collective_thrust, total_moments)

        return ControlWrench(
            collective_thrust=collective_thrust,
            moments=total_moments,
            rotor_commands=rotor_cmds,
        )


class CascadedPIDController:
    """Cascaded PID controller (Position -> Attitude -> Body Rates) as alternative architecture."""

    def __init__(self, params: Optional[QuadrotorParams] = None) -> None:
        self.params = params or QuadrotorParams()
        self.mixer = MotorMixer(self.params)

        # Position gains
        self.kp_pos = Vector3(2.0, 2.0, 3.5)
        self.kd_pos = Vector3(1.5, 1.5, 2.5)

        # Attitude gains (Euler roll, pitch, yaw)
        self.kp_att = Vector3(6.0, 6.0, 4.0)

        # Rate gains
        self.kp_rate = Vector3(0.15, 0.15, 0.08)
        self.kd_rate = Vector3(0.01, 0.01, 0.005)

        self.last_rate_error = Vector3(0.0, 0.0, 0.0)

    def update(
        self,
        current_state: QuadrotorState,
        ref_state: FullTrajectoryState,
        dt_s: float,
    ) -> ControlWrench:
        """Execute cascaded PID loops."""
        p = self.params
        m = p.mass_kg

        # 1. Position loop -> Desired acceleration and collective thrust
        e_pos = ref_state.position - current_state.position
        e_vel = ref_state.velocity - current_state.velocity

        cmd_acc = Vector3(
            self.kp_pos.x * e_pos.x + self.kd_pos.x * e_vel.x + ref_state.acceleration.x,
            self.kp_pos.y * e_pos.y + self.kd_pos.y * e_vel.y + ref_state.acceleration.y,
            self.kp_pos.z * e_pos.z + self.kd_pos.z * e_vel.z + ref_state.acceleration.z + p.gravity_m_s2,
        )

        thrust = m * max(1.0, cmd_acc.z)

        # Desired roll and pitch from horizontal acceleration commands
        # Small angle approximation: roll ~ -acc_y / g, pitch ~ acc_x / g
        cur_roll, cur_pitch, cur_yaw = current_state.attitude.to_euler()
        _, _, ref_yaw = ref_state.attitude.to_euler()

        des_roll = (-cmd_acc.x * math.sin(cur_yaw) + cmd_acc.y * math.cos(cur_yaw)) / p.gravity_m_s2
        des_pitch = (cmd_acc.x * math.cos(cur_yaw) + cmd_acc.y * math.sin(cur_yaw)) / p.gravity_m_s2
        # Clamp tilt angles to 45 degrees
        max_tilt = math.radians(45.0)
        des_roll = max(-max_tilt, min(max_tilt, des_roll))
        des_pitch = max(-max_tilt, min(max_tilt, des_pitch))

        # 2. Attitude loop -> Desired body rates
        des_p = self.kp_att.x * (des_roll - cur_roll)
        des_q = self.kp_att.y * (des_pitch - cur_pitch)
        des_r = self.kp_att.z * (ref_yaw - cur_yaw)

        # 3. Rate loop -> Body torques
        e_rate = Vector3(
            des_p - current_state.angular_velocity.x,
            des_q - current_state.angular_velocity.y,
            des_r - current_state.angular_velocity.z,
        )

        rate_deriv = (e_rate - self.last_rate_error) * (1.0 / max(1e-5, dt_s))
        self.last_rate_error = e_rate

        tau_x = self.kp_rate.x * e_rate.x + self.kd_rate.x * rate_deriv.x
        tau_y = self.kp_rate.y * e_rate.y + self.kd_rate.y * rate_deriv.y
        tau_z = self.kp_rate.z * e_rate.z + self.kd_rate.z * rate_deriv.z
        moments = Vector3(tau_x, tau_y, tau_z)

        rotor_cmds = self.mixer.mix(thrust, moments)

        return ControlWrench(
            collective_thrust=thrust,
            moments=moments,
            rotor_commands=rotor_cmds,
        )
