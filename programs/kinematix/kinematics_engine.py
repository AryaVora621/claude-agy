"""
KineMatix 3D: First-Principles Robotics & Multibody Inverse Kinematics Engine.
Standard library Python: zero external dependencies.
"""

import math
from typing import List, Tuple, Optional, Dict, Any


class Vector3:
    """Immutable 3D Cartesian vector with fundamental linear algebra operations."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        if abs(scalar) < 1e-12:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def normalized(self) -> "Vector3":
        n = self.norm()
        if n < 1e-12:
            return Vector3(0.0, 0.0, 1.0)
        return self / n

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Matrix4:
    """4x4 homogeneous transformation matrix stored in row-major order."""

    __slots__ = ("m",)

    def __init__(self, elements: Optional[List[float]] = None):
        if elements is None:
            # Identity matrix
            self.m = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        else:
            self.m = list(elements)

    @classmethod
    def from_dh(cls, theta: float, d: float, a: float, alpha: float) -> "Matrix4":
        """
        Construct homogeneous transform from standard Denavit-Hartenberg parameters:
        Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
        """
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return cls([
            ct, -st * ca,  st * sa, a * ct,
            st,  ct * ca, -ct * sa, a * st,
            0.0,      sa,       ca,      d,
            0.0,     0.0,      0.0,    1.0,
        ])

    @classmethod
    def from_translation(cls, x: float, y: float, z: float) -> "Matrix4":
        return cls([
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z,
            0.0, 0.0, 0.0, 1.0,
        ])

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float) -> "Matrix4":
        """Construct rotation matrix from Roll-Pitch-Yaw (Z-Y-X Euler angles)."""
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        return cls([
            r00, r01, r02, 0.0,
            r10, r11, r12, 0.0,
            r20, r21, r22, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])

    def __matmul__(self, other: "Matrix4") -> "Matrix4":
        """Matrix multiplication C = A * B."""
        a = self.m
        b = other.m
        c = [0.0] * 16
        for i in range(4):
            r = i * 4
            for j in range(4):
                c[r + j] = (
                    a[r + 0] * b[0 * 4 + j] +
                    a[r + 1] * b[1 * 4 + j] +
                    a[r + 2] * b[2 * 4 + j] +
                    a[r + 3] * b[3 * 4 + j]
                )
        return Matrix4(c)

    def transform_point(self, p: Vector3) -> Vector3:
        """Apply homogeneous transform to 3D point (w = 1)."""
        m = self.m
        return Vector3(
            m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
        )

    def transform_vector(self, v: Vector3) -> Vector3:
        """Apply rotation component to 3D direction vector (w = 0)."""
        m = self.m
        return Vector3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z,
        )

    @property
    def translation(self) -> Vector3:
        return Vector3(self.m[3], self.m[7], self.m[11])

    @property
    def z_axis(self) -> Vector3:
        """Unit Z-axis direction (3rd column of rotation submatrix)."""
        return Vector3(self.m[2], self.m[6], self.m[10]).normalized()

    @property
    def x_axis(self) -> Vector3:
        return Vector3(self.m[0], self.m[4], self.m[8]).normalized()

    @property
    def y_axis(self) -> Vector3:
        return Vector3(self.m[1], self.m[5], self.m[9]).normalized()


class DHLink:
    """Denavit-Hartenberg link definition for a single kinematic joint."""

    __slots__ = (
        "theta_offset",
        "d_offset",
        "a",
        "alpha",
        "is_revolute",
        "joint_min",
        "joint_max",
        "name",
    )

    def __init__(
        self,
        theta_offset: float = 0.0,
        d_offset: float = 0.0,
        a: float = 0.0,
        alpha: float = 0.0,
        is_revolute: bool = True,
        joint_min: float = -math.pi,
        joint_max: float = math.pi,
        name: str = "Joint",
    ):
        self.theta_offset = theta_offset
        self.d_offset = d_offset
        self.a = a
        self.alpha = alpha
        self.is_revolute = is_revolute
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.name = name

    def get_transform(self, joint_val: float) -> Matrix4:
        if self.is_revolute:
            theta = self.theta_offset + joint_val
            d = self.d_offset
        else:
            theta = self.theta_offset
            d = self.d_offset + joint_val
        return Matrix4.from_dh(theta, d, self.a, self.alpha)


class SerialRobotArm:
    """
    Serial kinematic chain manipulator supporting forward kinematics,
    geometric Jacobian derivation, and singularity-robust DLS inverse kinematics.
    """

    def __init__(self, links: List[DHLink], base_name: str = "Robot Base"):
        self.links = list(links)
        self.num_joints = len(self.links)
        self.base_name = base_name

    def forward_kinematics(self, joint_angles: List[float]) -> List[Matrix4]:
        """
        Compute cumulative transformation matrices from base frame (0) to each link frame.
        Returns list of transforms [T_0_0, T_0_1, T_0_2, ..., T_0_n].
        """
        transforms = [Matrix4()]  # Frame 0 is base identity
        t_curr = Matrix4()

        for i, link in enumerate(self.links):
            val = joint_angles[i] if i < len(joint_angles) else 0.0
            t_link = link.get_transform(val)
            t_curr = t_curr @ t_link
            transforms.append(t_curr)

        return transforms

    def get_end_effector_pose(self, joint_angles: List[float]) -> Tuple[Vector3, Matrix4]:
        transforms = self.forward_kinematics(joint_angles)
        t_end = transforms[-1]
        return t_end.translation, t_end

    def compute_jacobian(self, joint_angles: List[float]) -> List[List[float]]:
        """
        Compute 6 x N geometric Jacobian matrix J(q).
        Top 3 rows: Linear velocity Jacobian J_v
        Bottom 3 rows: Angular velocity Jacobian J_w
        """
        transforms = self.forward_kinematics(joint_angles)
        p_e = transforms[-1].translation
        n = self.num_joints

        # J is 6 rows by n columns
        j_matrix = [[0.0] * n for _ in range(6)]

        for i in range(n):
            t_prev = transforms[i]
            p_prev = t_prev.translation
            z_prev = t_prev.z_axis

            link = self.links[i]
            if link.is_revolute:
                # J_v = z_(i-1) x (p_e - p_(i-1))
                r = p_e - p_prev
                j_v = z_prev.cross(r)
                j_w = z_prev

                j_matrix[0][i] = j_v.x
                j_matrix[1][i] = j_v.y
                j_matrix[2][i] = j_v.z
                j_matrix[3][i] = j_w.x
                j_matrix[4][i] = j_w.y
                j_matrix[5][i] = j_w.z
            else:
                # Prismatic joint: J_v = z_(i-1), J_w = 0
                j_matrix[0][i] = z_prev.x
                j_matrix[1][i] = z_prev.y
                j_matrix[2][i] = z_prev.z
                j_matrix[3][i] = 0.0
                j_matrix[4][i] = 0.0
                j_matrix[5][i] = 0.0

        return j_matrix

    def compute_manipulability(self, joint_angles: List[float]) -> float:
        """
        Yoshikawa Manipulability Measure: w = sqrt(det(J * J^T))
        Computes dexterity volume of the end-effector.
        """
        j = self.compute_jacobian(joint_angles)
        # Compute positional 3x3 Gram matrix: G = J_v * J_v^T
        g = [[0.0] * 3 for _ in range(3)]
        n = self.num_joints
        for r1 in range(3):
            for r2 in range(3):
                s = sum(j[r1][k] * j[r2][k] for k in range(n))
                g[r1][r2] = s

        # Determinant of 3x3 matrix
        det_g = (
            g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) -
            g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
            g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
        )

        return math.sqrt(max(0.0, det_g))

    def solve_inverse_kinematics_dls(
        self,
        target_pos: Vector3,
        initial_angles: Optional[List[float]] = None,
        max_iter: int = 50,
        pos_tolerance: float = 0.002,
        damping: float = 0.05,
    ) -> Tuple[List[float], bool, float]:
        """
        Damped Least-Squares (DLS / Levenberg-Marquardt) Inverse Kinematics:
        Delta_q = J^T * (J * J^T + lambda^2 * I)^(-1) * e
        Unconditionally avoids singular velocity explosions.
        """
        if initial_angles is None:
            angles = [0.0] * self.num_joints
        else:
            angles = list(initial_angles)

        converged = False
        final_err = 0.0

        for _ in range(max_iter):
            curr_pos, _ = self.get_end_effector_pose(angles)
            err_vec = target_pos - curr_pos
            final_err = err_vec.norm()

            if final_err < pos_tolerance:
                converged = True
                break

            # Clip step size to prevent overshoot
            step_err = err_vec
            if final_err > 0.15:
                step_err = err_vec * (0.15 / final_err)

            # Use 3xN position Jacobian
            j_full = self.compute_jacobian(angles)
            j_pos = [j_full[0], j_full[1], j_full[2]]  # 3 rows x N cols
            n = self.num_joints

            # Compute A = J * J^T + lambda^2 * I (3x3 matrix)
            a = [[0.0] * 3 for _ in range(3)]
            for r1 in range(3):
                for r2 in range(3):
                    val = sum(j_pos[r1][k] * j_pos[r2][k] for k in range(n))
                    if r1 == r2:
                        val += damping * damping
                    a[r1][r2] = val

            # Solve A * y = step_err for y (3x1 vector) via Gaussian elimination
            b = [step_err.x, step_err.y, step_err.z]
            y = self._solve_linear_system_3x3(a, b)
            if y is None:
                # Singular breakdown: fallback to transpose
                for k in range(n):
                    dq = 0.05 * (j_pos[0][k] * b[0] + j_pos[1][k] * b[1] + j_pos[2][k] * b[2])
                    angles[k] = self._clamp_joint(k, angles[k] + dq)
                continue

            # Delta_q = J^T * y
            for k in range(n):
                dq = j_pos[0][k] * y[0] + j_pos[1][k] * y[1] + j_pos[2][k] * y[2]
                angles[k] = self._clamp_joint(k, angles[k] + dq)

        return angles, converged, final_err

    def _clamp_joint(self, index: int, val: float) -> float:
        link = self.links[index]
        return max(link.joint_min, min(link.joint_max, val))

    def _solve_linear_system_3x3(self, a_in: List[List[float]], b_in: List[float]) -> Optional[List[float]]:
        """Solve 3x3 linear system A * x = b with partial pivoting."""
        a = [list(row) for row in a_in]
        b = list(b_in)

        # Forward elimination with partial pivoting
        for i in range(3):
            # Find pivot
            max_row = i
            max_val = abs(a[i][i])
            for r in range(i + 1, 3):
                if abs(a[r][i]) > max_val:
                    max_val = abs(a[r][i])
                    max_row = r

            if max_val < 1e-12:
                return None  # Singular matrix

            if max_row != i:
                a[i], a[max_row] = a[max_row], a[i]
                b[i], b[max_row] = b[max_row], b[i]

            for r in range(i + 1, 3):
                factor = a[r][i] / a[i][i]
                for c in range(i, 3):
                    a[r][c] -= factor * a[i][c]
                b[r] -= factor * b[i]

        # Back substitution
        x = [0.0] * 3
        for i in range(2, -1, -1):
            s = sum(a[i][j] * x[j] for j in range(i + 1, 3))
            x[i] = (b[i] - s) / a[i][i]

        return x


class StewartPlatform:
    """
    6-DOF Stewart-Gough Parallel Kinematic Platform (Hexapod).
    Solves exact analytical closed-form inverse kinematics:
    Computes required actuator lengths L_1..L_6 for any 6-DOF payload pose (X, Y, Z, Roll, Pitch, Yaw).
    """

    def __init__(
        self,
        base_radius: float = 0.35,
        platform_radius: float = 0.22,
        home_height: float = 0.40,
    ):
        self.base_radius = base_radius
        self.platform_radius = platform_radius
        self.home_height = home_height

        # Hexagonal distribution in pairs for base and platform
        delta_b = 0.15
        delta_p = 0.15

        base_angles = [
            -math.pi / 6 - delta_b,
            -math.pi / 6 + delta_b,
             math.pi / 2 - delta_b,
             math.pi / 2 + delta_b,
             7 * math.pi / 6 - delta_b,
             7 * math.pi / 6 + delta_b,
        ]

        plat_angles = [
            -math.pi / 2 + delta_p,
             math.pi / 6 - delta_p,
             math.pi / 6 + delta_p,
             5 * math.pi / 6 - delta_p,
             5 * math.pi / 6 + delta_p,
            -math.pi / 2 - delta_p,
        ]

        # Base anchor coordinates B_i (fixed frame)
        self.base_anchors: List[Vector3] = [
            Vector3(base_radius * math.cos(th), base_radius * math.sin(th), 0.0)
            for th in base_angles
        ]

        # Platform attachment coordinates P_i (local platform frame)
        self.platform_joints: List[Vector3] = [
            Vector3(platform_radius * math.cos(th), platform_radius * math.sin(th), 0.0)
            for th in plat_angles
        ]

    def inverse_kinematics(
        self,
        translation: Vector3,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Tuple[List[float], List[Vector3]]:
        """
        Compute 6 leg vectors and lengths:
        l_i = t + R * p_i - b_i
        L_i = ||l_i||
        """
        rot = Matrix4.from_rpy(roll, pitch, yaw)
        pos = translation + Vector3(0.0, 0.0, self.home_height)

        leg_lengths: List[float] = []
        world_platform_pts: List[Vector3] = []

        for i in range(6):
            # Transform local platform joint to world frame
            p_local = self.platform_joints[i]
            p_world = pos + rot.transform_vector(p_local)
            world_platform_pts.append(p_world)

            # Leg vector from base anchor to platform joint
            b_world = self.base_anchors[i]
            leg_vec = p_world - b_world
            leg_lengths.append(leg_vec.norm())

        return leg_lengths, world_platform_pts
