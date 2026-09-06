"""
VeloSLAM: Dynamic Window Approach (DWA) Real-Time Local Motion Planner.
Searches velocity space (v, omega) bounded by acceleration limits,
simulates predictive trajectory rollouts, and optimizes collision clearance and heading.
"""

import math
from typing import List, Tuple, Optional
from veloslam.linalg import normalize_angle
from veloslam.occupancy import OccupancyGrid


class RobotState:
    """Dynamic state of differential-drive / unicycle robot."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 0.0,
        omega: float = 0.0
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.theta = normalize_angle(theta)
        self.v = float(v)          # Linear velocity (m/s)
        self.omega = float(omega)  # Angular velocity (rad/s)

    def pose_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta


class DWAConfig:
    """Kinodynamic constraints and objective cost weights for DWA."""

    def __init__(
        self,
        max_speed: float = 1.2,
        min_speed: float = 0.0,
        max_yaw_rate: float = math.radians(100.0),
        max_accel: float = 0.8,
        max_dyaw_rate: float = math.radians(150.0),
        v_resolution: float = 0.08,
        yaw_rate_resolution: float = math.radians(5.0),
        dt: float = 0.1,
        predict_time: float = 1.8,
        heading_weight: float = 0.15,
        clearance_weight: float = 0.8,
        velocity_weight: float = 0.3,
        robot_radius: float = 0.35
    ) -> None:
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_dyaw_rate = max_dyaw_rate
        self.v_resolution = v_resolution
        self.yaw_rate_resolution = yaw_rate_resolution
        self.dt = dt
        self.predict_time = predict_time
        self.heading_weight = heading_weight
        self.clearance_weight = clearance_weight
        self.velocity_weight = velocity_weight
        self.robot_radius = robot_radius


class DWAPlanner:
    """
    Dynamic Window Approach local reactive collision avoidance planner.
    """

    def __init__(self, config: Optional[DWAConfig] = None) -> None:
        self.cfg = config or DWAConfig()

    def calc_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """
        Computes velocity search space [v_min, v_max, omega_min, omega_max]
        taking acceleration limits into account.
        """
        # Vehicle physical capability limits
        vs = [self.cfg.min_speed, self.cfg.max_speed, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate]

        # Dynamic window based on acceleration
        vd = [
            state.v - self.cfg.max_accel * self.cfg.dt,
            state.v + self.cfg.max_accel * self.cfg.dt,
            state.omega - self.cfg.max_dyaw_rate * self.cfg.dt,
            state.omega + self.cfg.max_dyaw_rate * self.cfg.dt,
        ]

        # Intersection
        dw = (
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3]),
        )
        return dw

    def predict_trajectory(
        self,
        state: RobotState,
        v: float,
        omega: float
    ) -> List[Tuple[float, float, float]]:
        """Simulates forward trajectory over prediction horizon."""
        traj: List[Tuple[float, float, float]] = []
        cx, cy, cth = state.x, state.y, state.theta
        time_elapsed = 0.0

        while time_elapsed <= self.cfg.predict_time:
            cth = normalize_angle(cth + omega * self.cfg.dt)
            cx += v * math.cos(cth) * self.cfg.dt
            cy += v * math.sin(cth) * self.cfg.dt
            traj.append((cx, cy, cth))
            time_elapsed += self.cfg.dt

        return traj

    def evaluate_clearance(
        self,
        traj: List[Tuple[float, float, float]],
        grid: OccupancyGrid
    ) -> float:
        """
        Computes clearance distance from trajectory to nearest occupied obstacle.
        Returns 0.0 if trajectory collides with obstacle.
        """
        min_dist = float("inf")
        radius = self.cfg.robot_radius

        for x, y, _ in traj:
            # Check vehicle footprint
            if grid.is_world_occupied(x, y):
                return 0.0

            # Scan surrounding neighborhood for clearance
            gx, gy = grid.world_to_grid(x, y)
            search_cells = int(math.ceil(radius / grid.resolution))

            for dy in range(-search_cells, search_cells + 1):
                for dx in range(-search_cells, search_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if grid.is_valid_grid(nx, ny) and grid.is_occupied(nx, ny):
                        ox, oy = grid.grid_to_world(nx, ny)
                        d = math.hypot(x - ox, y - oy)
                        if d <= radius:
                            return 0.0  # Collision!
                        if d < min_dist:
                            min_dist = d

        return min(min_dist, 5.0)

    def plan(
        self,
        state: RobotState,
        goal: Tuple[float, float],
        grid: OccupancyGrid
    ) -> Tuple[float, float, List[Tuple[float, float, float]]]:
        """
        Computes optimal (v, omega) control command and selected trajectory rollout.
        Returns: (best_v, best_omega, best_trajectory)
        """
        v_min, v_max, w_min, w_max = self.calc_dynamic_window(state)

        best_score = -float("inf")
        best_v = 0.0
        best_w = 0.0
        best_traj: List[Tuple[float, float, float]] = []

        # Discretize velocity window
        v_steps = max(1, int(math.ceil((v_max - v_min) / self.cfg.v_resolution)))
        w_steps = max(1, int(math.ceil((w_max - w_min) / self.cfg.yaw_rate_resolution)))

        candidates: List[Tuple[float, float, List[Tuple[float, float, float]], float, float, float]] = []

        for iv in range(v_steps + 1):
            v = v_min + iv * self.cfg.v_resolution
            if v > v_max:
                v = v_max

            for iw in range(w_steps + 1):
                w = w_min + iw * self.cfg.yaw_rate_resolution
                if w > w_max:
                    w = w_max

                traj = self.predict_trajectory(state, v, w)
                clearance = self.evaluate_clearance(traj, grid)

                # Kinetic stopping distance constraint: v <= sqrt(2 * a * dist)
                if clearance <= 0.0:
                    continue
                stopping_dist = (v * v) / (2.0 * max(1e-3, self.cfg.max_accel))
                if clearance < stopping_dist:
                    continue

                # Heading score: alignment between end heading and direction to goal
                end_x, end_y, end_th = traj[-1]
                goal_angle = math.atan2(goal[1] - end_y, goal[0] - end_x)
                heading_err = abs(normalize_angle(goal_angle - end_th))
                heading_score = math.pi - heading_err  # Maximize alignment

                candidates.append((v, w, traj, heading_score, clearance, v))

        if not candidates:
            # Emergency brake: stop forward motion, turn towards goal
            goal_angle = math.atan2(goal[1] - state.y, goal[0] - state.x)
            turn_err = normalize_angle(goal_angle - state.theta)
            turn_w = math.copysign(min(abs(turn_err), self.cfg.max_yaw_rate), turn_err)
            return 0.0, turn_w, []

        # Normalize score components across valid candidates
        max_h = max(c[3] for c in candidates) or 1.0
        max_c = max(c[4] for c in candidates) or 1.0
        max_v = max(c[5] for c in candidates) or 1.0

        for v, w, traj, h_score, c_score, v_score in candidates:
            norm_h = h_score / max_h
            norm_c = c_score / max_c
            norm_v = v_score / max_v

            total_score = (
                self.cfg.heading_weight * norm_h +
                self.cfg.clearance_weight * norm_c +
                self.cfg.velocity_weight * norm_v
            )

            if total_score > best_score:
                best_score = total_score
                best_v = v
                best_w = w
                best_traj = traj

        return best_v, best_w, best_traj
