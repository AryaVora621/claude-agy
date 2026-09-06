"""
VeloSLAM: Kinodynamic Hybrid A* Non-Holonomic Path Planner.
Searches continuous (x, y, theta) state space using motion primitives,
3D discretized duplicate pruning, dual heuristics, and analytical Dubins shots.
"""

import heapq
import math
from typing import List, Tuple, Optional, Dict, Set
from veloslam.linalg import normalize_angle
from veloslam.occupancy import OccupancyGrid
from veloslam.dubins import dubins_shortest_path, DubinsPath


class Pose2D:
    """Continuous 2D oriented pose."""

    def __init__(self, x: float, y: float, theta: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.theta = normalize_angle(theta)

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class HybridNode:
    """Search node for Hybrid A*."""

    def __init__(
        self,
        pose: Pose2D,
        g_cost: float,
        h_cost: float,
        parent: Optional["HybridNode"] = None,
        intermediate_pts: Optional[List[Tuple[float, float, float]]] = None
    ) -> None:
        self.pose = pose
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.intermediate_pts = intermediate_pts or []

    def __lt__(self, other: "HybridNode") -> bool:
        return self.f_cost < other.f_cost


class HybridAStar:
    """
    Kinodynamic Hybrid A* Planner for vehicles with minimum turning radius.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        min_turning_radius: float = 1.0,
        xy_resolution: float = 0.5,
        theta_resolution: float = math.radians(15.0),
        step_size: float = 0.5,
        analytic_shot_freq: int = 5
    ) -> None:
        self.grid = grid
        self.rho = min_turning_radius
        self.xy_res = xy_resolution
        self.theta_res = theta_resolution
        self.step_size = step_size
        self.analytic_shot_freq = analytic_shot_freq

        # Steering angles to evaluate at each expansion: Left, Slight-Left, Straight, Slight-Right, Right
        self.steer_angles = [
            -1.0 / self.rho,
            -0.5 / self.rho,
            0.0,
            0.5 / self.rho,
            1.0 / self.rho
        ]

    def _state_to_index(self, pose: Pose2D) -> Tuple[int, int, int]:
        ix = int(math.floor(pose.x / self.xy_res))
        iy = int(math.floor(pose.y / self.xy_res))
        ith = int(math.floor((pose.theta + math.pi) / self.theta_res))
        return ix, iy, ith

    def _heuristic(self, pose: Pose2D, goal: Pose2D) -> float:
        # Dual heuristic: max(Euclidean 2D, Dubins obstacle-free)
        h_euclid = pose.distance_to(goal)
        dub = dubins_shortest_path(pose.to_tuple(), goal.to_tuple(), self.rho)
        return max(h_euclid, dub.length)

    def _is_trajectory_collision_free(self, points: List[Tuple[float, float, float]]) -> bool:
        for x, y, _ in points:
            if self.grid.is_world_occupied(x, y):
                return False
        return True

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        max_iterations: int = 2000
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plans kinodynamic collision-free path from start to goal.
        Returns ordered list of (x, y, theta) waypoints.
        """
        # Start and goal validity checks
        if self.grid.is_world_occupied(start.x, start.y):
            return None
        if self.grid.is_world_occupied(goal.x, goal.y):
            return None

        open_set: List[Tuple[float, int, HybridNode]] = []
        closed_set: Set[Tuple[int, int, int]] = set()

        start_h = self._heuristic(start, goal)
        start_node = HybridNode(start, g_cost=0.0, h_cost=start_h)
        counter = 0
        heapq.heappush(open_set, (start_node.f_cost, counter, start_node))

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            state_idx = self._state_to_index(current.pose)
            if state_idx in closed_set:
                continue
            closed_set.add(state_idx)

            # Goal check: close enough in distance and angle
            dist_to_goal = current.pose.distance_to(goal)
            ang_diff = abs(normalize_angle(current.pose.theta - goal.theta))

            if dist_to_goal < 0.5 and ang_diff < math.radians(20.0):
                return self._reconstruct_path(current)

            # Analytic Shot via Dubins shortest path
            if iterations % self.analytic_shot_freq == 0 or dist_to_goal < 4.0 * self.rho:
                dub_path = dubins_shortest_path(current.pose.to_tuple(), goal.to_tuple(), self.rho)
                samples = dub_path.sample_path(step_size=0.1)
                if self._is_trajectory_collision_free(samples):
                    # Found direct collision-free analytical trajectory!
                    base_path = self._reconstruct_path(current)
                    return base_path + samples[1:]

            # Expand successor nodes using motion primitives
            for curvature in self.steer_angles:
                succ_pose, inter_pts = self._simulate_motion(current.pose, curvature, self.step_size)

                # Collision check along arc
                if not self._is_trajectory_collision_free(inter_pts):
                    continue

                succ_idx = self._state_to_index(succ_pose)
                if succ_idx in closed_set:
                    continue

                # Steering penalty to prefer straight driving
                steer_penalty = 1.0 + 0.1 * abs(curvature * self.rho)
                new_g = current.g_cost + self.step_size * steer_penalty
                new_h = self._heuristic(succ_pose, goal)

                succ_node = HybridNode(
                    pose=succ_pose,
                    g_cost=new_g,
                    h_cost=new_h,
                    parent=current,
                    intermediate_pts=inter_pts
                )

                counter += 1
                heapq.heappush(open_set, (succ_node.f_cost, counter, succ_node))

        # Direct Dubins fallback if start and goal are directly connectable
        dub_direct = dubins_shortest_path(start.to_tuple(), goal.to_tuple(), self.rho)
        samples = dub_direct.sample_path(step_size=0.1)
        if self._is_trajectory_collision_free(samples):
            return samples

        return None

    def _simulate_motion(
        self,
        pose: Pose2D,
        curvature: float,
        length: float,
        num_substeps: int = 5
    ) -> Tuple[Pose2D, List[Tuple[float, float, float]]]:
        """Simulates vehicle forward motion along arc with specified curvature (1/R)."""
        dt_sub = length / num_substeps
        cx, cy, cth = pose.x, pose.y, pose.theta
        pts: List[Tuple[float, float, float]] = []

        for _ in range(num_substeps):
            if abs(curvature) < 1e-6:
                # Straight motion
                cx += dt_sub * math.cos(cth)
                cy += dt_sub * math.sin(cth)
            else:
                # Circular arc motion
                dphi = dt_sub * curvature
                rho = 1.0 / curvature
                cx += rho * (math.sin(cth + dphi) - math.sin(cth))
                cy += rho * (-math.cos(cth + dphi) + math.cos(cth))
                cth = normalize_angle(cth + dphi)

            pts.append((cx, cy, cth))

        return Pose2D(cx, cy, cth), pts

    def _reconstruct_path(self, node: HybridNode) -> List[Tuple[float, float, float]]:
        waypoints: List[Tuple[float, float, float]] = []
        curr: Optional[HybridNode] = node
        while curr is not None:
            if curr.intermediate_pts:
                waypoints = curr.intermediate_pts + waypoints
            else:
                waypoints.insert(0, curr.pose.to_tuple())
            curr = curr.parent
        return waypoints
