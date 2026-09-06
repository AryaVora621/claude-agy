"""Sensor Simulation and Multi-Rate Error-State Extended Kalman Filter (ES-EKF).

Implements:
1. Realistic 6-axis IMU (accelerometer + rate gyroscope with random-walk bias drift),
   barometric altimeter, 3-axis magnetometer, and GPS receiver simulation.
2. 15-state Error-State Extended Kalman Filter (ES-EKF) avoiding quaternion rank deficiency.
3. Joseph-form numerically stabilized covariance updates.
4. Asynchronous multi-rate sensor fusion pipeline.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .dynamics import Matrix3x3, QuadrotorState, Quaternion, Vector3


@dataclass
class IMUReading:
    """High-rate inertial measurement unit sample."""

    timestamp_s: float
    accel_m_s2: Vector3  # Specific force in body frame
    gyro_rad_s: Vector3  # Angular velocity in body frame


@dataclass
class BaroReading:
    """Barometric altimeter sample."""

    timestamp_s: float
    altitude_m: float


@dataclass
class MagReading:
    """Tri-axial magnetometer sample."""

    timestamp_s: float
    field_body: Vector3


@dataclass
class GPSReading:
    """Low-rate global positioning system fix."""

    timestamp_s: float
    position_m: Vector3
    velocity_m_s: Vector3


class SensorSimulator:
    """Simulates physical sensor outputs with Gaussian noise and bias drift."""

    def __init__(
        self,
        accel_noise_std: float = 0.05,
        gyro_noise_std: float = 0.005,
        accel_bias_drift_std: float = 1e-4,
        gyro_bias_drift_std: float = 1e-5,
        baro_noise_std: float = 0.15,
        mag_noise_std: float = 0.02,
        gps_pos_noise_std: float = 0.3,
        gps_vel_noise_std: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.accel_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std
        self.accel_bias_drift = accel_bias_drift_std
        self.gyro_bias_drift = gyro_bias_drift_std
        self.baro_noise = baro_noise_std
        self.mag_noise = mag_noise_std
        self.gps_pos_noise = gps_pos_noise_std
        self.gps_vel_noise = gps_vel_noise_std

        # Simulated physical sensor biases
        self.accel_bias = Vector3(0.02, -0.015, 0.03)
        self.gyro_bias = Vector3(0.005, -0.004, 0.008)

        # Standard Earth magnetic field vector in world frame (normalized: North, East, Down/Up)
        # In ENU: North is +Y, Up is +Z, so field has positive Y and negative Z in northern hemisphere
        self.earth_mag_world = Vector3(0.0, 0.6, -0.8).normalized()

    def generate_imu(
        self,
        true_state: QuadrotorState,
        true_accel_w: Vector3,
        timestamp_s: float,
        dt_s: float,
    ) -> IMUReading:
        """Generate high-rate IMU specific force and angular rates in body frame."""
        # Accelerometer measures: R^T * (a_w + g * e3) + bias + noise
        g_world = Vector3(0.0, 0.0, 9.81)
        total_acc_w = true_accel_w + g_world
        # Rotate to body frame
        R = true_state.attitude.to_rotation_matrix()
        R_T = R.transpose()
        acc_body = R_T.dot_vec(total_acc_w)

        # Apply bias drift
        self.accel_bias = self.accel_bias + Vector3(
            self.rng.gauss(0.0, self.accel_bias_drift * math.sqrt(dt_s)),
            self.rng.gauss(0.0, self.accel_bias_drift * math.sqrt(dt_s)),
            self.rng.gauss(0.0, self.accel_bias_drift * math.sqrt(dt_s)),
        )
        self.gyro_bias = self.gyro_bias + Vector3(
            self.rng.gauss(0.0, self.gyro_bias_drift * math.sqrt(dt_s)),
            self.rng.gauss(0.0, self.gyro_bias_drift * math.sqrt(dt_s)),
            self.rng.gauss(0.0, self.gyro_bias_drift * math.sqrt(dt_s)),
        )

        measured_accel = acc_body + self.accel_bias + Vector3(
            self.rng.gauss(0.0, self.accel_noise),
            self.rng.gauss(0.0, self.accel_noise),
            self.rng.gauss(0.0, self.accel_noise),
        )

        measured_gyro = true_state.angular_velocity + self.gyro_bias + Vector3(
            self.rng.gauss(0.0, self.gyro_noise),
            self.rng.gauss(0.0, self.gyro_noise),
            self.rng.gauss(0.0, self.gyro_noise),
        )

        return IMUReading(timestamp_s, measured_accel, measured_gyro)

    def generate_barometer(self, true_state: QuadrotorState, timestamp_s: float) -> BaroReading:
        """Generate barometric altitude measurement (Z coordinate in ENU)."""
        alt = true_state.position.z + self.rng.gauss(0.0, self.baro_noise)
        return BaroReading(timestamp_s, alt)

    def generate_magnetometer(self, true_state: QuadrotorState, timestamp_s: float) -> MagReading:
        """Generate 3-axis magnetometer measurement in body frame."""
        R_T = true_state.attitude.to_rotation_matrix().transpose()
        mag_body = R_T.dot_vec(self.earth_mag_world)
        noisy_mag = mag_body + Vector3(
            self.rng.gauss(0.0, self.mag_noise),
            self.rng.gauss(0.0, self.mag_noise),
            self.rng.gauss(0.0, self.mag_noise),
        )
        return MagReading(timestamp_s, noisy_mag.normalized())

    def generate_gps(self, true_state: QuadrotorState, timestamp_s: float) -> GPSReading:
        """Generate GPS position and velocity fix in world frame."""
        pos = true_state.position + Vector3(
            self.rng.gauss(0.0, self.gps_pos_noise),
            self.rng.gauss(0.0, self.gps_pos_noise),
            self.rng.gauss(0.0, self.gps_pos_noise),
        )
        vel = true_state.velocity + Vector3(
            self.rng.gauss(0.0, self.gps_vel_noise),
            self.rng.gauss(0.0, self.gps_vel_noise),
            self.rng.gauss(0.0, self.gps_vel_noise),
        )
        return GPSReading(timestamp_s, pos, vel)


@dataclass
class EKFState:
    """Estimated state vector and covariance."""

    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    attitude: Quaternion = field(default_factory=Quaternion.identity)
    accel_bias: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    gyro_bias: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))


class MultiRateESEKF:
    """15-State Error-State Extended Kalman Filter (ES-EKF) for Quadrotor UAVs.

    State Vector:
      - Nominal State (16D): Position (3), Velocity (3), Attitude Quaternion (4), Accel Bias (3), Gyro Bias (3)
      - Error State delta_x (15D): delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)
    """

    def __init__(
        self,
        init_pos: Optional[Vector3] = None,
        init_vel: Optional[Vector3] = None,
        init_att: Optional[Quaternion] = None,
    ) -> None:
        self.state = EKFState(
            position=init_pos or Vector3(0.0, 0.0, 0.0),
            velocity=init_vel or Vector3(0.0, 0.0, 0.0),
            attitude=init_att or Quaternion.identity(),
            accel_bias=Vector3(0.0, 0.0, 0.0),
            gyro_bias=Vector3(0.0, 0.0, 0.0),
        )

        # 15x15 Error state covariance matrix P
        self.P = [[0.0] * 15 for _ in range(15)]
        # Initial standard deviations
        sig_p = 0.5
        sig_v = 0.2
        sig_att = 0.05
        sig_ba = 0.1
        sig_bg = 0.02

        for i in range(3):
            self.P[i][i] = sig_p ** 2
            self.P[3 + i][3 + i] = sig_v ** 2
            self.P[6 + i][6 + i] = sig_att ** 2
            self.P[9 + i][9 + i] = sig_ba ** 2
            self.P[12 + i][12 + i] = sig_bg ** 2

        # Process noise spectral densities
        self.var_accel = 0.05 ** 2
        self.var_gyro = 0.005 ** 2
        self.var_accel_bias = 1e-4 ** 2
        self.var_gyro_bias = 1e-5 ** 2

    def predict(self, imu: IMUReading, dt_s: float) -> None:
        """High-rate IMU kinematic prediction and error covariance propagation."""
        # 1. Unbiased inertial measurements
        acc_unbiased = imu.accel_m_s2 - self.state.accel_bias
        gyro_unbiased = imu.gyro_rad_s - self.state.gyro_bias

        # 2. Propagate nominal attitude
        # Delta angle = omega * dt
        delta_angle = gyro_unbiased * dt_s
        angle_norm = delta_angle.norm()
        if angle_norm > 1e-12:
            delta_q = Quaternion.from_axis_angle(delta_angle, angle_norm)
        else:
            delta_q = Quaternion.identity()

        new_att = self.state.attitude.multiply(delta_q).normalized()

        # 3. Propagate velocity and position in world frame
        R = self.state.attitude.to_rotation_matrix()
        acc_world = R.dot_vec(acc_unbiased)
        gravity_w = Vector3(0.0, 0.0, 9.81)
        net_acc_world = acc_world - gravity_w

        new_pos = self.state.position + self.state.velocity * dt_s + net_acc_world * (0.5 * dt_s * dt_s)
        new_vel = self.state.velocity + net_acc_world * dt_s

        self.state.position = new_pos
        self.state.velocity = new_vel
        self.state.attitude = new_att

        # 4. Discrete error-state transition matrix F (15x15)
        # F = I_15 + F_continuous * dt
        F = [[1.0 if i == j else 0.0 for j in range(15)] for i in range(15)]

        # Position block: d(delta_p)/dt = delta_v => F[0..2, 3..5] = I * dt
        for i in range(3):
            F[i][3 + i] = dt_s

        # Velocity block:
        # d(delta_v)/dt = -R * [acc_unbiased]_x * delta_theta - R * delta_ba
        # Skew-symmetric of acc_unbiased
        a_x = [
            [0.0, -acc_unbiased.z, acc_unbiased.y],
            [acc_unbiased.z, 0.0, -acc_unbiased.x],
            [-acc_unbiased.y, acc_unbiased.x, 0.0],
        ]
        # Multiply R * a_x
        R_ax = [[sum(R.m[r][k] * a_x[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                # Coupling into attitude error
                F[3 + r][6 + c] = -R_ax[r][c] * dt_s
                # Coupling into accel bias error
                F[3 + r][9 + c] = -R.m[r][c] * dt_s

        # Attitude block:
        # d(delta_theta)/dt = -[gyro_unbiased]_x * delta_theta - delta_bg
        w_x = [
            [0.0, -gyro_unbiased.z, gyro_unbiased.y],
            [gyro_unbiased.z, 0.0, -gyro_unbiased.x],
            [-gyro_unbiased.y, gyro_unbiased.x, 0.0],
        ]
        for r in range(3):
            for c in range(3):
                F[6 + r][6 + c] += -w_x[r][c] * dt_s
            F[6 + r][12 + r] = -dt_s

        # 5. Discrete process noise covariance Q (15x15)
        Q = [[0.0] * 15 for _ in range(15)]
        for i in range(3):
            Q[3 + i][3 + i] = self.var_accel * dt_s
            Q[6 + i][6 + i] = self.var_gyro * dt_s
            Q[9 + i][9 + i] = self.var_accel_bias * dt_s
            Q[12 + i][12 + i] = self.var_gyro_bias * dt_s

        # 6. Propagate covariance: P = F * P * F^T + Q
        FP = [[sum(F[i][k] * self.P[k][j] for k in range(15)) for j in range(15)] for i in range(15)]
        new_P = [[sum(FP[i][k] * F[j][k] for k in range(15)) + Q[i][j] for j in range(15)] for i in range(15)]

        self.P = new_P

    def update_barometer(self, baro: BaroReading) -> None:
        """Scalar measurement update from barometric altimeter: z = pos_z."""
        # Measurement matrix H: 1 x 15, selects pos.z (index 2)
        H = [0.0] * 15
        H[2] = 1.0

        r_baro = 0.15 ** 2
        # Innovation
        y = baro.altitude_m - self.state.position.z

        # S = H * P * H^T + R
        S = self.P[2][2] + r_baro
        inv_S = 1.0 / S

        # Kalman gain K = P * H^T * inv(S)
        K = [self.P[i][2] * inv_S for i in range(15)]

        # Apply error state
        self._apply_correction(K, [y], [[1.0 if i == 2 else 0.0 for i in range(15)]], [[r_baro]])

    def update_gps(self, gps: GPSReading) -> None:
        """6D measurement update from GPS position and velocity."""
        # Innovation vector: [p_meas - p_est, v_meas - v_est]
        y = [
            gps.position_m.x - self.state.position.x,
            gps.position_m.y - self.state.position.y,
            gps.position_m.z - self.state.position.z,
            gps.velocity_m_s.x - self.state.velocity.x,
            gps.velocity_m_s.y - self.state.velocity.y,
            gps.velocity_m_s.z - self.state.velocity.z,
        ]

        # 6x15 H matrix: top 3x3 is I for position, next 3x3 is I for velocity
        H = [[0.0] * 15 for _ in range(6)]
        for i in range(3):
            H[i][i] = 1.0
            H[3 + i][3 + i] = 1.0

        R_meas = [[0.0] * 6 for _ in range(6)]
        pos_var = 0.3 ** 2
        vel_var = 0.05 ** 2
        for i in range(3):
            R_meas[i][i] = pos_var
            R_meas[3 + i][3 + i] = vel_var

        # Compute S = H * P * H^T + R (6x6)
        HP = [[sum(H[i][k] * self.P[k][j] for k in range(15)) for j in range(15)] for i in range(6)]
        S = [[sum(HP[i][k] * H[j][k] for k in range(15)) + R_meas[i][j] for j in range(6)] for i in range(6)]

        # Invert S (6x6) via Gauss-Jordan elimination
        inv_S = self._invert_matrix(S)

        # K = P * H^T * inv_S (15x6)
        PHT = [[sum(self.P[i][k] * H[j][k] for k in range(15)) for j in range(6)] for i in range(15)]
        K = [[sum(PHT[i][k] * inv_S[k][j] for k in range(6)) for j in range(6)] for i in range(15)]

        # Correction
        delta_x = [sum(K[i][j] * y[j] for j in range(6)) for i in range(15)]

        # Inject into nominal state
        self.state.position = self.state.position + Vector3(delta_x[0], delta_x[1], delta_x[2])
        self.state.velocity = self.state.velocity + Vector3(delta_x[3], delta_x[4], delta_x[5])

        # Attitude correction: delta_q = [1, 0.5 * delta_theta]
        d_theta = Vector3(delta_x[6], delta_x[7], delta_x[8])
        th_norm = d_theta.norm()
        if th_norm > 1e-12:
            dq = Quaternion.from_axis_angle(d_theta, th_norm)
            self.state.attitude = self.state.attitude.multiply(dq).normalized()

        self.state.accel_bias = self.state.accel_bias + Vector3(delta_x[9], delta_x[10], delta_x[11])
        self.state.gyro_bias = self.state.gyro_bias + Vector3(delta_x[12], delta_x[13], delta_x[14])

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        KH = [[sum(K[i][k] * H[k][j] for k in range(6)) for j in range(15)] for i in range(15)]
        IKH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(15)] for i in range(15)]

        IKHP = [[sum(IKH[i][k] * self.P[k][j] for k in range(15)) for j in range(15)] for i in range(15)]
        P_part1 = [[sum(IKHP[i][k] * IKH[j][k] for k in range(15)) for j in range(15)] for i in range(15)]

        KR = [[sum(K[i][k] * R_meas[k][j] for k in range(6)) for j in range(6)] for i in range(15)]
        KRKT = [[sum(KR[i][k] * K[j][k] for k in range(6)) for j in range(15)] for i in range(15)]

        self.P = [[P_part1[i][j] + KRKT[i][j] for j in range(15)] for i in range(15)]

    def _apply_correction(
        self,
        K_1d: List[float],
        y_1d: List[float],
        H_2d: List[List[float]],
        R_2d: List[List[float]],
    ) -> None:
        """Apply scalar Kalman update."""
        delta_x = [K_1d[i] * y_1d[0] for i in range(15)]

        self.state.position = self.state.position + Vector3(delta_x[0], delta_x[1], delta_x[2])
        self.state.velocity = self.state.velocity + Vector3(delta_x[3], delta_x[4], delta_x[5])

        d_theta = Vector3(delta_x[6], delta_x[7], delta_x[8])
        th_norm = d_theta.norm()
        if th_norm > 1e-12:
            dq = Quaternion.from_axis_angle(d_theta, th_norm)
            self.state.attitude = self.state.attitude.multiply(dq).normalized()

        self.state.accel_bias = self.state.accel_bias + Vector3(delta_x[9], delta_x[10], delta_x[11])
        self.state.gyro_bias = self.state.gyro_bias + Vector3(delta_x[12], delta_x[13], delta_x[14])

        # Covariance update
        # P = (I - K*H) * P
        H = H_2d[0]
        KH = [[K_1d[i] * H[j] for j in range(15)] for i in range(15)]
        IKH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(15)] for i in range(15)]

        new_P = [[sum(IKH[i][k] * self.P[k][j] for k in range(15)) for j in range(15)] for i in range(15)]
        self.P = new_P

    def _invert_matrix(self, A: List[List[float]]) -> List[List[float]]:
        """Invert N x N matrix via Gauss-Jordan elimination with partial pivoting."""
        n = len(A)
        # Augment with identity
        M = [[float(A[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            max_row = i
            max_val = abs(M[i][i])
            for r in range(i + 1, n):
                if abs(M[r][i]) > max_val:
                    max_val = abs(M[r][i])
                    max_row = r

            if max_val < 1e-14:
                raise ValueError("Singular matrix in EKF")

            if max_row != i:
                M[i], M[max_row] = M[max_row], M[i]

            pivot = M[i][i]
            inv_p = 1.0 / pivot
            for c in range(i, 2 * n):
                M[i][c] *= inv_p

            for r in range(n):
                if r != i:
                    factor = M[r][i]
                    if abs(factor) > 1e-15:
                        for c in range(i, 2 * n):
                            M[r][c] -= factor * M[i][c]

        return [[M[i][n + j] for j in range(n)] for i in range(n)]
