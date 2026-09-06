"""
VeloSLAM: Probabilistic Occupancy Grid Mapping & LiDAR Raycasting Engine.
Implements Bayesian log-odds mapping, inverse rangefinder sensor models,
Bresenham line traversal, and obstacle inflation layers.
"""

import math
from typing import List, Tuple, Optional


class LaserScan:
    """LiDAR / Laser Rangefinder Scan Data."""

    def __init__(
        self,
        ranges: List[float],
        angle_min: float = -math.pi * 0.75,
        angle_max: float = math.pi * 0.75,
        angle_increment: Optional[float] = None,
        range_min: float = 0.1,
        range_max: float = 12.0
    ) -> None:
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_max = angle_max
        n = len(ranges)
        if angle_increment is None:
            self.angle_increment = (angle_max - angle_min) / max(1, n - 1) if n > 1 else 0.0
        else:
            self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Classic Bresenham integer line algorithm.
    Returns list of grid points (x, y) connecting (x0, y0) to (x1, y1).
    """
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    cx, cy = x0, y0
    while True:
        points.append((cx, cy))
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy

    return points


class OccupancyGrid:
    """
    2D Probabilistic Occupancy Grid with Log-Odds Bayesian Updates.
    """

    def __init__(
        self,
        width_m: float = 20.0,
        height_m: float = 20.0,
        resolution: float = 0.2,
        origin_x: float = -10.0,
        origin_y: float = -10.0,
        p_prior: float = 0.5,
        p_free: float = 0.35,
        p_occ: float = 0.85,
        log_odds_min: float = -5.0,
        log_odds_max: float = 5.0
    ) -> None:
        self.width_m = width_m
        self.height_m = height_m
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.width_cells = int(math.ceil(width_m / resolution))
        self.height_cells = int(math.ceil(height_m / resolution))

        # Log-odds parameters
        self.l_free = math.log(p_free / (1.0 - p_free))
        self.l_occ = math.log(p_occ / (1.0 - p_occ))
        self.l_prior = math.log(p_prior / (1.0 - p_prior))
        self.l_min = log_odds_min
        self.l_max = log_odds_max

        # Grid storage initialized to prior (unknown = 0.0 log-odds)
        self.cells = [self.l_prior] * (self.width_cells * self.height_cells)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Converts world coordinates (meters) to integer grid indices (gx, gy)."""
        gx = int(math.floor((wx - self.origin_x) / self.resolution))
        gy = int(math.floor((wy - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Converts grid cell center to world coordinates (meters)."""
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    def is_valid_grid(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width_cells and 0 <= gy < self.height_cells

    def _get_index(self, gx: int, gy: int) -> int:
        return gy * self.width_cells + gx

    def get_log_odds(self, gx: int, gy: int) -> float:
        if not self.is_valid_grid(gx, gy):
            return self.l_prior
        return self.cells[self._get_index(gx, gy)]

    def get_probability(self, gx: int, gy: int) -> float:
        """Returns occupancy probability P in range [0.0, 1.0]."""
        l = self.get_log_odds(gx, gy)
        return 1.0 - (1.0 / (1.0 + math.exp(l)))

    def is_occupied(self, gx: int, gy: int, threshold: float = 0.65) -> bool:
        return self.get_probability(gx, gy) >= threshold

    def is_free(self, gx: int, gy: int, threshold: float = 0.35) -> bool:
        return self.get_probability(gx, gy) <= threshold

    def is_world_occupied(self, wx: float, wy: float, threshold: float = 0.65) -> bool:
        gx, gy = self.world_to_grid(wx, wy)
        if not self.is_valid_grid(gx, gy):
            return True  # Unknown/out of bounds treated as collision
        return self.is_occupied(gx, gy, threshold=threshold)

    def update_with_scan(
        self,
        robot_pose: Tuple[float, float, float],
        scan: LaserScan
    ) -> None:
        """
        Updates occupancy grid using inverse sensor model along each laser beam.
        robot_pose: (x, y, theta)
        """
        rx, ry, rtheta = robot_pose
        start_gx, start_gy = self.world_to_grid(rx, ry)

        for i, dist in enumerate(scan.ranges):
            beam_angle = rtheta + scan.angle_min + i * scan.angle_increment

            if math.isnan(dist) or dist <= scan.range_min:
                continue

            is_hit = (dist < scan.range_max)
            clamped_dist = min(dist, scan.range_max)

            # End point in world coordinates
            end_wx = rx + clamped_dist * math.cos(beam_angle)
            end_wy = ry + clamped_dist * math.sin(beam_angle)
            end_gx, end_gy = self.world_to_grid(end_wx, end_wy)

            # Raycast line
            line_pts = bresenham_line(start_gx, start_gy, end_gx, end_gy)

            # Free space along the ray
            free_pts = line_pts[:-1] if is_hit else line_pts
            for gx, gy in free_pts:
                if self.is_valid_grid(gx, gy):
                    idx = self._get_index(gx, gy)
                    new_l = self.cells[idx] + self.l_free - self.l_prior
                    self.cells[idx] = max(self.l_min, min(self.l_max, new_l))

            # Occupied hit at the obstacle endpoint
            if is_hit and self.is_valid_grid(end_gx, end_gy):
                idx = self._get_index(end_gx, end_gy)
                new_l = self.cells[idx] + self.l_occ - self.l_prior
                self.cells[idx] = max(self.l_min, min(self.l_max, new_l))

    def get_inflated_grid(self, inflation_radius_m: float) -> "OccupancyGrid":
        """
        Returns a new OccupancyGrid where occupied obstacles are dilated by inflation_radius_m.
        Allows path planners to safely plan trajectories considering vehicle footprint.
        """
        inflated = OccupancyGrid(
            width_m=self.width_m,
            height_m=self.height_m,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            log_odds_min=self.l_min,
            log_odds_max=self.l_max
        )
        # Copy original grid
        inflated.cells = list(self.cells)

        radius_cells = int(math.ceil(inflation_radius_m / self.resolution))
        r2 = radius_cells * radius_cells

        for gy in range(self.height_cells):
            for gx in range(self.width_cells):
                if self.is_occupied(gx, gy):
                    # Inflate surrounding circle
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            if dx * dx + dy * dy <= r2:
                                nx, ny = gx + dx, gy + dy
                                if self.is_valid_grid(nx, ny):
                                    inflated.cells[inflated._get_index(nx, ny)] = self.l_max

        return inflated
