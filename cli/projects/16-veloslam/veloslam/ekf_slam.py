"""
VeloSLAM: Extended Kalman Filter SLAM (EKF-SLAM) Engine.
Simultaneous state estimation of mobile robot pose (x, y, theta) and unknown 2D landmarks.
Features non-linear motion/measurement Jacobians, block covariance propagation,
and Mahalanobis gating data association.
"""

import math
from typing import List, Tuple, Optional, Dict
from veloslam.linalg import (
    Vector, Matrix, normalize_angle, mahalanobis_distance, covariance_ellipse_2d
)


class Landmark:
    """Tracked landmark metadata."""

    def __init__(self, landmark_id: int, x: float, y: float) -> None:
        self.id = landmark_id
        self.x = x
        self.y = y
        self.observations = 1


class EKFSLAM:
    """
    Extended Kalman Filter for Simultaneous Localization and Mapping.
    State vector: [x_r, y_r, theta_r, m_1x, m_1y, ..., m_Nx, m_Ny]^T.
    """

    def __init__(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_theta: float = 0.0,
        sigma_xy_init: float = 1e-4,
        sigma_theta_init: float = 1e-4,
        sigma_v: float = 0.05,
        sigma_omega: float = 0.02,
        sigma_range: float = 0.1,
        sigma_bearing: float = 0.03,
        gate_threshold: float = 9.21  # chi2 for 2 DOF at 99% confidence
    ) -> None:
        # State: [x, y, theta]
        self.mu = Vector([init_x, init_y, init_theta])
        # Covariance: 3x3
        self.sigma = Matrix([
            [sigma_xy_init ** 2, 0.0, 0.0],
            [0.0, sigma_xy_init ** 2, 0.0],
            [0.0, 0.0, sigma_theta_init ** 2],
        ])

        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega

        # Sensor noise covariance Q (2x2)
        self.Q = Matrix([
            [sigma_range ** 2, 0.0],
            [0.0, sigma_bearing ** 2],
        ])

        self.gate_threshold = gate_threshold
        self.landmarks: List[Landmark] = []
        self._next_landmark_id = 0

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Returns (x, y, theta) of the estimated robot pose."""
        return self.mu[0], self.mu[1], self.mu[2]

    def get_robot_covariance(self) -> Matrix:
        """Returns the 3x3 covariance matrix of the robot pose."""
        return Matrix([[self.sigma[r][c] for c in range(3)] for r in range(3)])

    def get_landmarks(self) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Returns list of tuples for each landmark:
        (id, x, y, semi_major, semi_minor, orientation_angle)
        """
        res = []
        for i, lm in enumerate(self.landmarks):
            idx = 3 + 2 * i
            cov_2x2 = Matrix([
                [self.sigma[idx][idx], self.sigma[idx][idx + 1]],
                [self.sigma[idx + 1][idx], self.sigma[idx + 1][idx + 1]],
            ])
            s_maj, s_min, ang = covariance_ellipse_2d(cov_2x2)
            res.append((lm.id, self.mu[idx], self.mu[idx + 1], s_maj, s_min, ang))
        return res

    def predict(self, v: float, omega: float, dt: float) -> None:
        """
        EKF Prediction Step based on Differential Drive / Unicycle Motion Model.
        v: linear velocity (m/s)
        omega: angular velocity (rad/s)
        dt: time delta (s)
        """
        theta = self.mu[2]
        dist = v * dt
        dtheta = omega * dt

        # Update robot pose mean
        dx = dist * math.cos(theta)
        dy = dist * math.sin(theta)
        new_x = self.mu[0] + dx
        new_y = self.mu[1] + dy
        new_theta = normalize_angle(theta + dtheta)

        self.mu[0] = new_x
        self.mu[1] = new_y
        self.mu[2] = new_theta

        # Robot motion Jacobian G_R (3x3)
        g_r = Matrix([
            [1.0, 0.0, -dist * math.sin(theta)],
            [0.0, 1.0,  dist * math.cos(theta)],
            [0.0, 0.0, 1.0],
        ])

        # Process noise covariance R (3x3)
        var_dist = (self.sigma_v * dt) ** 2
        var_angle = (self.sigma_omega * dt) ** 2
        r_cov = Matrix([
            [var_dist, 0.0, 0.0],
            [0.0, var_dist, 0.0],
            [0.0, 0.0, var_angle],
        ])

        # Efficient block-wise covariance update
        # 1. Sigma_RR <- G_R @ Sigma_RR @ G_R.T + R
        sigma_rr = Matrix([[self.sigma[r][c] for c in range(3)] for r in range(3)])
        new_sigma_rr = (g_r @ sigma_rr @ g_r.transpose()) + r_cov

        for r in range(3):
            for c in range(3):
                self.sigma[r][c] = new_sigma_rr[r][c]

        # 2. If landmarks exist, update cross-covariance: Sigma_RM <- G_R @ Sigma_RM
        m_dim = 2 * self.num_landmarks
        if m_dim > 0:
            sigma_rm = Matrix([[self.sigma[r][3 + c] for c in range(m_dim)] for r in range(3)])
            new_sigma_rm = g_r @ sigma_rm

            for r in range(3):
                for c in range(m_dim):
                    self.sigma[r][3 + c] = new_sigma_rm[r][c]
                    self.sigma[3 + c][r] = new_sigma_rm[r][c]  # symmetry

    def update(self, observations: List[Tuple[float, float]]) -> List[int]:
        """
        EKF Measurement Update Step.
        observations: list of (range_r, bearing_phi_radians)
        Returns list of matched or created landmark IDs.
        """
        matched_ids: List[int] = []

        for r_meas, phi_meas in observations:
            z = Vector([r_meas, phi_meas])
            xr, yr, thetar = self.mu[0], self.mu[1], self.mu[2]

            best_idx = -1
            best_dist = float("inf")
            best_h: Optional[Matrix] = None
            best_s_inv: Optional[Matrix] = None
            best_residual: Optional[Vector] = None

            # Compare against all known landmarks
            for j in range(self.num_landmarks):
                lm_idx = 3 + 2 * j
                lx, ly = self.mu[lm_idx], self.mu[lm_idx + 1]

                dx = lx - xr
                dy = ly - yr
                q = dx * dx + dy * dy
                r_hat = math.sqrt(q)

                if r_hat < 1e-6:
                    continue

                phi_hat = normalize_angle(math.atan2(dy, dx) - thetar)
                z_hat = Vector([r_hat, phi_hat])

                residual = z - z_hat
                residual[1] = normalize_angle(residual[1])

                # Construct measurement Jacobian H (2 x N)
                total_dim = 3 + 2 * self.num_landmarks
                h_mat = Matrix.zeros(2, total_dim)

                # Robot pose block: d(h)/d([xr, yr, thetar])
                h_mat[0][0] = -dx / r_hat
                h_mat[0][1] = -dy / r_hat
                h_mat[0][2] = 0.0

                h_mat[1][0] = dy / q
                h_mat[1][1] = -dx / q
                h_mat[1][2] = -1.0

                # Landmark position block: d(h)/d([lx, ly])
                h_mat[0][lm_idx] = dx / r_hat
                h_mat[0][lm_idx + 1] = dy / r_hat

                h_mat[1][lm_idx] = -dy / q
                h_mat[1][lm_idx + 1] = dx / q

                # Innovation covariance S = H @ Sigma @ H.T + Q (2x2)
                s = (h_mat @ self.sigma @ h_mat.transpose()) + self.Q
                try:
                    s_inv = s.inv()
                except ValueError:
                    continue

                d_m2 = mahalanobis_distance(residual, s_inv)
                if d_m2 < best_dist:
                    best_dist = d_m2
                    best_idx = j
                    best_h = h_mat
                    best_s_inv = s_inv
                    best_residual = residual

            # Data association decision
            if best_idx != -1 and best_dist < self.gate_threshold:
                # Existing landmark matched: execute EKF state update
                h_mat = best_h
                s_inv = best_s_inv
                residual = best_residual

                # Kalman Gain K = Sigma @ H.T @ S^-1 (total_dim x 2)
                k_gain = self.sigma @ h_mat.transpose() @ s_inv

                # Mean update: mu <- mu + K @ residual
                delta_mu = k_gain @ residual
                self.mu = self.mu + delta_mu
                self.mu[2] = normalize_angle(self.mu[2])

                # Covariance update: Sigma <- (I - K @ H) @ Sigma
                total_dim = 3 + 2 * self.num_landmarks
                identity_mat = Matrix.identity(total_dim)
                kh = k_gain @ h_mat
                self.sigma = (identity_mat - kh) @ self.sigma

                lm = self.landmarks[best_idx]
                lm.observations += 1
                matched_ids.append(lm.id)
            else:
                # New landmark initialization
                new_id = self._initialize_landmark(r_meas, phi_meas)
                matched_ids.append(new_id)

        return matched_ids

    def _initialize_landmark(self, r_meas: float, phi_meas: float) -> int:
        """Initializes a new landmark in the state vector and covariance matrix."""
        xr, yr, thetar = self.mu[0], self.mu[1], self.mu[2]
        angle = thetar + phi_meas

        lx = xr + r_meas * math.cos(angle)
        ly = yr + r_meas * math.sin(angle)

        new_id = self._next_landmark_id
        self._next_landmark_id += 1
        self.landmarks.append(Landmark(new_id, lx, ly))

        old_dim = 3 + 2 * (self.num_landmarks - 1)
        new_dim = old_dim + 2

        # Expand state vector
        self.mu = Vector(self.mu.to_list() + [lx, ly])

        # Jacobian w.r.t robot pose G_p (2x3)
        g_p = Matrix([
            [1.0, 0.0, -r_meas * math.sin(angle)],
            [0.0, 1.0,  r_meas * math.cos(angle)],
        ])

        # Jacobian w.r.t sensor measurement G_z (2x2)
        g_z = Matrix([
            [math.cos(angle), -r_meas * math.sin(angle)],
            [math.sin(angle),  r_meas * math.cos(angle)],
        ])

        # Landmark self-covariance Sigma_LL = G_p @ Sigma_RR @ G_p.T + G_z @ Q @ G_z.T
        sigma_rr = Matrix([[self.sigma[r][c] for c in range(3)] for r in range(3)])
        sigma_ll = (g_p @ sigma_rr @ g_p.transpose()) + (g_z @ self.Q @ g_z.transpose())

        # Cross-covariance with existing state: Sigma_L_old = G_p @ Sigma[0:3, 0:old_dim]
        sigma_r_all = Matrix([[self.sigma[r][c] for c in range(old_dim)] for r in range(3)])
        sigma_l_old = g_p @ sigma_r_all  # (2 x old_dim)

        # Build expanded new covariance matrix
        new_sigma = Matrix.zeros(new_dim, new_dim)

        # Copy existing covariance
        for r in range(old_dim):
            for c in range(old_dim):
                new_sigma[r][c] = self.sigma[r][c]

        # Insert new cross-covariance blocks
        for r in range(2):
            for c in range(old_dim):
                val = sigma_l_old[r][c]
                new_sigma[old_dim + r][c] = val
                new_sigma[c][old_dim + r] = val  # symmetry

        # Insert landmark self-covariance block
        for r in range(2):
            for c in range(2):
                new_sigma[old_dim + r][old_dim + c] = sigma_ll[r][c]

        self.sigma = new_sigma
        return new_id
