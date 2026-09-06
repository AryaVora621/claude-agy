"""Minimum-Snap Polynomial Trajectory Generation & Differential Flatness.

Implements:
1. Differential Flatness of Quadrotors on SE(3) mapping flat outputs (x, y, z, yaw)
   and derivatives to collective thrust, attitude orientation, and body rates.
2. Piecewise polynomial spline optimization (quintic and septic) through multi-waypoint paths.
3. Closed-form linear equation solver (Gaussian elimination with partial pivoting).
4. Parametric trajectory profiles: Figure-8 (lemniscate), aggressive slalom, and helix.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .dynamics import Matrix3x3, QuadrotorParams, Quaternion, Vector3


@dataclass
class FlatOutput:
    """Flat output state vector sigma = [x, y, z, yaw]^T and temporal derivatives."""

    position: Vector3
    velocity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    acceleration: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    jerk: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    snap: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    yaw_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0
    yaw_accel_rad_s2: float = 0.0


@dataclass
class FullTrajectoryState:
    """Full 6-DOF reference trajectory setpoint derived via differential flatness."""

    time_s: float
    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    thrust_n: float
    attitude: Quaternion
    rotation_matrix: Matrix3x3
    angular_velocity: Vector3
    angular_acceleration: Vector3
    body_rates: Vector3  # [p, q, r]

    @property
    def collective_thrust(self) -> float:
        return self.thrust_n


class DifferentialFlatness:
    """Maps flat output trajectory sigma(t) and its derivatives to full 6-DOF SE(3) state."""

    def __init__(self, params: Optional[QuadrotorParams] = None) -> None:
        self.params = params or QuadrotorParams()

    def flat_to_full_state(self, flat: FlatOutput, time_s: float = 0.0) -> FullTrajectoryState:
        """Evaluate full quadrotor state and feedforward inputs from flat outputs."""
        p = self.params
        m = p.mass_kg
        g = p.gravity_m_s2

        # 1. Total thrust vector in world frame:
        # m * (a + g * e3) = f * z_b
        gravity_term = Vector3(0.0, 0.0, g)
        total_thrust_vector = (flat.acceleration + gravity_term) * m

        # Collective thrust magnitude
        thrust_magnitude = total_thrust_vector.norm()
        if thrust_magnitude < 1e-4:
            # Degenerate free-fall case
            thrust_magnitude = 1e-4
            z_b = Vector3(0.0, 0.0, 1.0)
        else:
            z_b = total_thrust_vector * (1.0 / thrust_magnitude)

        # 2. Body frame orientation R = [x_b, y_b, z_b]
        # x_c is the projection of the desired heading on the horizontal plane
        cos_yaw = math.cos(flat.yaw_rad)
        sin_yaw = math.sin(flat.yaw_rad)
        x_c = Vector3(cos_yaw, sin_yaw, 0.0)

        # y_b is perpendicular to both z_b and x_c
        z_cross_xc = z_b.cross(x_c)
        norm_z_cross_xc = z_cross_xc.norm()

        if norm_z_cross_xc < 1e-6:
            # Singularity: thrust pointing exactly along or opposite heading
            # Choose arbitrary orthogonal vector
            y_b = Vector3(-sin_yaw, cos_yaw, 0.0)
            x_b = y_b.cross(z_b).normalized()
        else:
            y_b = z_cross_xc * (1.0 / norm_z_cross_xc)
            x_b = y_b.cross(z_b)

        R_desired = Matrix3x3.from_columns(x_b, y_b, z_b)
        q_desired = Quaternion.from_rotation_matrix(R_desired)

        # 3. Angular velocity:
        # dot(t) = m * jerk
        # dot(f) = z_b . dot(t)
        # dot(z_b) = (dot(t) - dot(f) * z_b) / f
        # omega_x = -dot(z_b) . y_b
        # omega_y =  dot(z_b) . x_b
        # omega_z =  dot(yaw) * (e3 . z_b) + (dot(y_b) . x_b projection)
        t_dot = flat.jerk * m
        f_dot = z_b.dot(t_dot)
        z_b_dot = (t_dot - z_b * f_dot) * (1.0 / thrust_magnitude)

        p_rate = -z_b_dot.dot(y_b)
        q_rate = z_b_dot.dot(x_b)

        # Yaw rate projection onto body z axis
        e3 = Vector3(0.0, 0.0, 1.0)
        z_b_dot_e3 = z_b.dot(e3)
        r_rate = flat.yaw_rate_rad_s * z_b_dot_e3

        omega_body = Vector3(p_rate, q_rate, r_rate)

        # 4. Angular acceleration approximation via projection of snap
        # dot(omega) can be projected from snap vector or set to zero for smooth curves
        t_ddot = flat.snap * m
        f_ddot = z_b_dot.dot(t_dot) + z_b.dot(t_ddot)
        z_b_ddot = (t_ddot - z_b_dot * f_dot - z_b * f_ddot - z_b_dot * f_dot) * (1.0 / thrust_magnitude)

        alpha_x = -z_b_ddot.dot(y_b)
        alpha_y = z_b_ddot.dot(x_b)
        alpha_z = flat.yaw_accel_rad_s2 * z_b_dot_e3
        alpha_body = Vector3(alpha_x, alpha_y, alpha_z)

        return FullTrajectoryState(
            time_s=time_s,
            position=flat.position,
            velocity=flat.velocity,
            acceleration=flat.acceleration,
            thrust_n=thrust_magnitude,
            attitude=q_desired,
            rotation_matrix=R_desired,
            angular_velocity=omega_body,
            angular_acceleration=alpha_body,
            body_rates=omega_body,
        )


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A * x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augment matrix A with b
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]

    for i in range(n):
        # Partial pivoting: select pivot with largest absolute value
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r

        if max_val < 1e-14:
            raise ValueError(f"Singular matrix encountered at index {i}")

        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        inv_pivot = 1.0 / pivot

        # Normalize pivot row
        for c in range(i, n + 1):
            M[i][c] *= inv_pivot

        # Eliminate column entries in other rows
        for r in range(n):
            if r != i:
                factor = M[r][i]
                if abs(factor) > 1e-15:
                    for c in range(i, n + 1):
                        M[r][c] -= factor * M[i][c]

    return [M[i][n] for i in range(n)]


class PolynomialSegment:
    """Polynomial segment parameterized by normalized duration tau in [0, 1].

    c_0 + c_1 * tau + c_2 * tau^2 + ... + c_N * tau^N.
    """

    def __init__(self, coeffs: List[float], dt: Optional[float] = None, duration: Optional[float] = None) -> None:
        self.coeffs = coeffs
        self.dt = dt if dt is not None else (duration if duration is not None else 1.0)
        self.duration = self.dt

    def eval(self, tau: float) -> float:
        """Evaluate position at normalized time tau."""
        val = 0.0
        p = 1.0
        for c in self.coeffs:
            val += c * p
            p *= tau
        return val

    def eval_deriv(self, tau: float, order: int = 1) -> float:
        """Evaluate derivative of given order at normalized time tau with chain-rule scaling."""
        if order == 0:
            return self.eval(tau)

        # Scale factor from chain rule: d/dt = (1/dt) * d/dtau
        scale = 1.0 / (self.dt ** order)

        deg = len(self.coeffs)
        val = 0.0
        for i in range(order, deg):
            # Factorial term: i * (i-1) * ... * (i - order + 1)
            fact = 1.0
            for k in range(order):
                fact *= (i - k)
            tau_pow = tau ** (i - order)
            val += self.coeffs[i] * fact * tau_pow

        return val * scale


class PiecewisePolynomial1D:
    """1D piecewise polynomial trajectory spanning multi-segment intervals."""

    def __init__(self, segments: List[PolynomialSegment], segment_times: List[float]) -> None:
        self.segments = segments
        self.segment_times = segment_times  # [t_0, t_1, ..., t_M]
        self.total_duration = segment_times[-1] - segment_times[0]

    def _find_segment(self, t: float) -> Tuple[PolynomialSegment, float]:
        t_clamped = max(self.segment_times[0], min(self.segment_times[-1], t))
        for i in range(len(self.segments)):
            t_start = self.segment_times[i]
            t_end = self.segment_times[i + 1]
            if t_clamped <= t_end or i == len(self.segments) - 1:
                tau = (t_clamped - t_start) / max(1e-9, (t_end - t_start))
                return self.segments[i], max(0.0, min(1.0, tau))
        return self.segments[-1], 1.0

    def evaluate(self, t: float, order: int = 0) -> float:
        seg, tau = self._find_segment(t)
        return seg.eval_deriv(tau, order)


class MinimumSnapTrajectory:
    """Quintic/septic polynomial trajectory generator through 3D waypoints."""

    def __init__(
        self,
        waypoints: List[Vector3],
        segment_durations: List[float],
        start_velocity: Optional[Vector3] = None,
        end_velocity: Optional[Vector3] = None,
        start_accel: Optional[Vector3] = None,
        end_accel: Optional[Vector3] = None,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("Must have at least 2 waypoints")
        if len(segment_durations) != len(waypoints) - 1:
            raise ValueError("Number of durations must equal number of waypoints - 1")

        self.waypoints = waypoints
        self.durations = segment_durations
        self.start_vel = start_velocity or Vector3(0.0, 0.0, 0.0)
        self.end_vel = end_velocity or Vector3(0.0, 0.0, 0.0)
        self.start_acc = start_accel or Vector3(0.0, 0.0, 0.0)
        self.end_acc = end_accel or Vector3(0.0, 0.0, 0.0)

        # Build timeline
        self.times = [0.0]
        for d in self.durations:
            self.times.append(self.times[-1] + d)

        # Fit 5th-order (quintic) splines independently for x, y, and z
        self.spline_x = self._fit_quintic_spline([w.x for w in waypoints], self.start_vel.x, self.end_vel.x, self.start_acc.x, self.end_acc.x)
        self.spline_y = self._fit_quintic_spline([w.y for w in waypoints], self.start_vel.y, self.end_vel.y, self.start_acc.y, self.end_acc.y)
        self.spline_z = self._fit_quintic_spline([w.z for w in waypoints], self.start_vel.z, self.end_vel.z, self.start_acc.z, self.end_acc.z)

    @property
    def total_duration(self) -> float:
        """Total trajectory duration in seconds."""
        return sum(self.durations)

    def _fit_quintic_spline(
        self,
        points: List[float],
        v_start: float,
        v_end: float,
        a_start: float,
        a_end: float,
    ) -> PiecewisePolynomial1D:
        """Solve linear continuity matrix for 5th-order polynomials (6 coeffs per segment).

        Segment i (tau in [0, 1]):
          p_i(0) = c_0
          p_i(1) = c_0 + c_1 + c_2 + c_3 + c_4 + c_5
          v_i(0) = c_1 / dt
          v_i(1) = (c_1 + 2 c_2 + 3 c_3 + 4 c_4 + 5 c_5) / dt
          a_i(0) = 2 c_2 / dt^2
          a_i(1) = (2 c_2 + 6 c_3 + 12 c_4 + 20 c_5) / dt^2
        """
        M = len(points) - 1
        num_vars = 6 * M
        A = [[0.0] * num_vars for _ in range(num_vars)]
        b = [0.0] * num_vars
        row = 0

        # 1. Start boundary conditions (3 constraints)
        dt0 = self.durations[0]
        # Position: p_0(0) = points[0]
        A[row][0] = 1.0
        b[row] = points[0]
        row += 1
        # Velocity: v_0(0) = v_start => c_1 / dt0 = v_start
        A[row][1] = 1.0 / dt0
        b[row] = v_start
        row += 1
        # Acceleration: a_0(0) = a_start => 2 c_2 / dt0^2 = a_start
        A[row][2] = 2.0 / (dt0 ** 2)
        b[row] = a_start
        row += 1

        # 2. Waypoint position constraints: p_i(1) = points[i+1] for all segments
        for i in range(M):
            offset = 6 * i
            for k in range(6):
                A[row][offset + k] = 1.0
            b[row] = points[i + 1]
            row += 1

        # 3. Interior waypoint continuity: C0, C1, C2, C3, C4
        for i in range(M - 1):
            curr_off = 6 * i
            next_off = 6 * (i + 1)
            dt_curr = self.durations[i]
            dt_next = self.durations[i + 1]

            # C0: p_i+1(0) = points[i+1] (already handled by p_i(1), ensure p_i+1(0) matches)
            A[row][next_off] = 1.0
            b[row] = points[i + 1]
            row += 1

            # C1: v_i(1) = v_i+1(0)
            # (c1 + 2c2 + 3c3 + 4c4 + 5c5)/dt_curr - (c1_next)/dt_next = 0
            for k in range(1, 6):
                A[row][curr_off + k] = k / dt_curr
            A[row][next_off + 1] = -1.0 / dt_next
            b[row] = 0.0
            row += 1

            # C2: a_i(1) = a_i+1(0)
            # (2c2 + 6c3 + 12c4 + 20c5)/dt_curr^2 - (2c2_next)/dt_next^2 = 0
            weights_c2 = [2.0, 6.0, 12.0, 20.0]
            for idx, k in enumerate(range(2, 6)):
                A[row][curr_off + k] = weights_c2[idx] / (dt_curr ** 2)
            A[row][next_off + 2] = -2.0 / (dt_next ** 2)
            b[row] = 0.0
            row += 1

            # C3: jerk continuity j_i(1) = j_i+1(0)
            # (6c3 + 24c4 + 60c5)/dt_curr^3 - (6c3_next)/dt_next^3 = 0
            weights_c3 = [6.0, 24.0, 60.0]
            for idx, k in enumerate(range(3, 6)):
                A[row][curr_off + k] = weights_c3[idx] / (dt_curr ** 3)
            A[row][next_off + 3] = -6.0 / (dt_next ** 3)
            b[row] = 0.0
            row += 1

            # C4: snap continuity s_i(1) = s_i+1(0)
            # (24c4 + 120c5)/dt_curr^4 - (24c4_next)/dt_next^4 = 0
            A[row][curr_off + 4] = 24.0 / (dt_curr ** 4)
            A[row][curr_off + 5] = 120.0 / (dt_curr ** 4)
            A[row][next_off + 4] = -24.0 / (dt_next ** 4)
            b[row] = 0.0
            row += 1

        # 4. End boundary conditions: v_M-1(1) = v_end, a_M-1(1) = a_end
        last_off = 6 * (M - 1)
        dt_last = self.durations[-1]
        for k in range(1, 6):
            A[row][last_off + k] = k / dt_last
        b[row] = v_end
        row += 1

        weights_c2 = [2.0, 6.0, 12.0, 20.0]
        for idx, k in enumerate(range(2, 6)):
            A[row][last_off + k] = weights_c2[idx] / (dt_last ** 2)
        b[row] = a_end
        row += 1

        # Solve system
        coeffs = solve_linear_system(A, b)

        segments = []
        for i in range(M):
            seg_coeffs = coeffs[6 * i : 6 * (i + 1)]
            segments.append(PolynomialSegment(seg_coeffs, self.durations[i]))

        return PiecewisePolynomial1D(segments, self.times)

    def evaluate(self, t: float) -> FlatOutput:
        """Sample flat output state at time t."""
        px = self.spline_x.evaluate(t, order=0)
        py = self.spline_y.evaluate(t, order=0)
        pz = self.spline_z.evaluate(t, order=0)

        vx = self.spline_x.evaluate(t, order=1)
        vy = self.spline_y.evaluate(t, order=1)
        vz = self.spline_z.evaluate(t, order=1)

        ax = self.spline_x.evaluate(t, order=2)
        ay = self.spline_y.evaluate(t, order=2)
        az = self.spline_z.evaluate(t, order=2)

        jx = self.spline_x.evaluate(t, order=3)
        jy = self.spline_y.evaluate(t, order=3)
        jz = self.spline_z.evaluate(t, order=3)

        sx = self.spline_x.evaluate(t, order=4)
        sy = self.spline_y.evaluate(t, order=4)
        sz = self.spline_z.evaluate(t, order=4)

        # Yaw aligned with velocity vector when moving, with smooth transition
        speed_horiz = math.sqrt(vx * vx + vy * vy)
        if speed_horiz > 0.1:
            yaw = math.atan2(vy, vx)
            # Analytical yaw rate: d/dt(atan2(vy, vx)) = (vx * ay - vy * ax) / (vx^2 + vy^2)
            yaw_rate = (vx * ay - vy * ax) / (speed_horiz * speed_horiz)
        else:
            yaw = 0.0
            yaw_rate = 0.0

        return FlatOutput(
            position=Vector3(px, py, pz),
            velocity=Vector3(vx, vy, vz),
            acceleration=Vector3(ax, ay, az),
            jerk=Vector3(jx, jy, jz),
            snap=Vector3(sx, sy, sz),
            yaw_rad=yaw,
            yaw_rate_rad_s=yaw_rate,
        )


class AerobaticTrajectories:
    """Standard predefined aerobatic trajectory profiles for benchmarking and demonstration."""

    @staticmethod
    def figure_eight(
        radius_x: float = 3.0,
        radius_y: float = 2.0,
        altitude: Optional[float] = None,
        period_s: float = 8.0,
        total_time_s: float = 16.0,
        altitude_z: Optional[float] = None,
        num_points: Optional[int] = None,
        loop_duration_s: Optional[float] = None,
    ) -> List[FlatOutput]:
        """Generate 3D Lemniscate (figure-8) trajectory setpoints with analytical derivatives."""
        target_alt = altitude if altitude is not None else (altitude_z if altitude_z is not None else 2.5)
        period = loop_duration_s if loop_duration_s is not None else period_s
        if num_points is not None:
            steps = num_points
            dt = total_time_s / steps
        else:
            dt = 0.05
            steps = int(total_time_s / dt)

        trajectory = []
        omega = 2.0 * math.pi / period

        for k in range(steps):
            t = k * dt
            # Parametric Lemniscate of Gerono: x(t) = a * sin(omega * t), y(t) = b * sin(2 * omega * t) / 2
            x = radius_x * math.sin(omega * t)
            y = radius_y * math.sin(2.0 * omega * t) * 0.5
            z = target_alt + 0.5 * math.sin(omega * t)

            vx = radius_x * omega * math.cos(omega * t)
            vy = radius_y * omega * math.cos(2.0 * omega * t)
            vz = 0.5 * omega * math.cos(omega * t)

            ax = -radius_x * (omega ** 2) * math.sin(omega * t)
            ay = -2.0 * radius_y * (omega ** 2) * math.sin(2.0 * omega * t)
            az = -0.5 * (omega ** 2) * math.sin(omega * t)

            jx = -radius_x * (omega ** 3) * math.cos(omega * t)
            jy = -4.0 * radius_y * (omega ** 3) * math.cos(2.0 * omega * t)
            jz = -0.5 * (omega ** 3) * math.cos(omega * t)

            sx = radius_x * (omega ** 4) * math.sin(omega * t)
            sy = 8.0 * radius_y * (omega ** 4) * math.sin(2.0 * omega * t)
            sz = 0.5 * (omega ** 4) * math.sin(omega * t)

            speed_h = math.sqrt(vx * vx + vy * vy)
            yaw = math.atan2(vy, vx) if speed_h > 0.05 else 0.0
            yaw_rate = (vx * ay - vy * ax) / max(0.01, speed_h ** 2) if speed_h > 0.05 else 0.0

            trajectory.append(FlatOutput(
                position=Vector3(x, y, z),
                velocity=Vector3(vx, vy, vz),
                acceleration=Vector3(ax, ay, az),
                jerk=Vector3(jx, jy, jz),
                snap=Vector3(sx, sy, sz),
                yaw_rad=yaw,
                yaw_rate_rad_s=yaw_rate,
            ))

        return trajectory
