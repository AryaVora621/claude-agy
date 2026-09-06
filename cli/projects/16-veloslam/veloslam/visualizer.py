"""
VeloSLAM: High-Resolution Unicode Braille SLAM & Trajectory Visualizer.
Renders 2D occupancy grid maps, robot pose and heading, LiDAR rays,
planned paths, and SLAM landmark covariance ellipses into terminal Braille patterns.
"""

import math
from typing import List, Tuple, Optional
from veloslam.occupancy import OccupancyGrid, LaserScan, bresenham_line
from veloslam.linalg import normalize_angle

# 2x4 sub-pixel Braille dot bitmasks
BRAILLE_DOTS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def render_braille_map(
    grid: OccupancyGrid,
    robot_pose: Optional[Tuple[float, float, float]] = None,
    path: Optional[List[Tuple[float, float, float]]] = None,
    landmarks: Optional[List[Tuple[int, float, float, float, float, float]]] = None,
    laser_scan: Optional[LaserScan] = None,
    width_chars: int = 60,
    height_chars: int = 24,
    border: bool = True
) -> str:
    """
    Renders the SLAM environment to a high-resolution sub-pixel Braille canvas.
    width_chars x height_chars maps to (width_chars*2) x (height_chars*4) sub-pixels.
    """
    sub_w = width_chars * 2
    sub_h = height_chars * 4

    # 0 = empty, 1 = obstacle, 2 = lidar ray, 3 = path, 4 = robot, 5 = landmark
    pixel_grid = [[0 for _ in range(sub_w)] for _ in range(sub_h)]

    # World bounds to sub-pixel mapping
    min_x, max_x = grid.origin_x, grid.origin_x + grid.width_m
    min_y, max_y = grid.origin_y, grid.origin_y + grid.height_m

    def world_to_subpixel(wx: float, wy: float) -> Tuple[int, int]:
        norm_x = (wx - min_x) / (max_x - min_x)
        norm_y = (wy - min_y) / (max_y - min_y)
        # Invert Y for screen display (top = max_y, bottom = min_y)
        px = int(norm_x * (sub_w - 1))
        py = int((1.0 - norm_y) * (sub_h - 1))
        return px, py

    # 1. Render Occupancy Grid obstacles
    for gy in range(grid.height_cells):
        for gx in range(grid.width_cells):
            if grid.is_occupied(gx, gy):
                wx, wy = grid.grid_to_world(gx, gy)
                px, py = world_to_subpixel(wx, wy)
                if 0 <= px < sub_w and 0 <= py < sub_h:
                    pixel_grid[py][px] = 1

    # 2. Render LiDAR beam rays
    if robot_pose and laser_scan:
        rx, ry, rth = robot_pose
        rpx, rpy = world_to_subpixel(rx, ry)
        for i, dist in enumerate(laser_scan.ranges):
            if dist <= laser_scan.range_min:
                continue
            d = min(dist, laser_scan.range_max)
            beam_th = rth + laser_scan.angle_min + i * laser_scan.angle_increment
            ex = rx + d * math.cos(beam_th)
            ey = ry + d * math.sin(beam_th)
            epx, epy = world_to_subpixel(ex, ey)

            line_pts = bresenham_line(rpx, rpy, epx, epy)
            for lx, ly in line_pts[:-1]:  # Exclude end point
                if 0 <= lx < sub_w and 0 <= ly < sub_h:
                    if pixel_grid[ly][lx] == 0:
                        pixel_grid[ly][lx] = 2

    # 3. Render Planned Path
    if path:
        for x, y, _ in path:
            px, py = world_to_subpixel(x, y)
            if 0 <= px < sub_w and 0 <= py < sub_h:
                pixel_grid[py][px] = 3

    # 4. Render Landmarks and covariance ellipses
    if landmarks:
        for lm_id, lx, ly, s_maj, s_min, ang in landmarks:
            cpx, cpy = world_to_subpixel(lx, ly)
            if 0 <= cpx < sub_w and 0 <= cpy < sub_h:
                pixel_grid[cpy][cpx] = 5
            # Draw ellipse points
            steps = 16
            for step in range(steps):
                t = (2.0 * math.pi * step) / steps
                # Parametric ellipse
                ex = lx + s_maj * math.cos(t) * math.cos(ang) - s_min * math.sin(t) * math.sin(ang)
                ey = ly + s_maj * math.cos(t) * math.sin(ang) + s_min * math.sin(t) * math.cos(ang)
                epx, epy = world_to_subpixel(ex, ey)
                if 0 <= epx < sub_w and 0 <= epy < sub_h:
                    if pixel_grid[epy][epx] == 0:
                        pixel_grid[epy][epx] = 5

    # 5. Render Robot pose and heading vector
    if robot_pose:
        rx, ry, rth = robot_pose
        rpx, rpy = world_to_subpixel(rx, ry)
        # Center marker (2x2 sub-pixels)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = rpx + dx, rpy + dy
                if 0 <= nx < sub_w and 0 <= ny < sub_h:
                    pixel_grid[ny][nx] = 4

        # Heading vector line (0.8 meters ahead)
        hx = rx + 0.8 * math.cos(rth)
        hy = ry + 0.8 * math.sin(rth)
        hpx, hpy = world_to_subpixel(hx, hy)
        for lx, ly in bresenham_line(rpx, rpy, hpx, hpy):
            if 0 <= lx < sub_w and 0 <= ly < sub_h:
                pixel_grid[ly][lx] = 4

    # Build ASCII/Braille characters
    lines: List[str] = []
    if border:
        lines.append("+" + "-" * width_chars + "+")

    for cy in range(height_chars):
        row_chars: List[str] = []
        for cx in range(width_chars):
            code = 0
            primary_layer = 0
            for by in range(4):
                py = cy * 4 + by
                for bx in range(2):
                    px = cx * 2 + bx
                    layer = pixel_grid[py][px]
                    if layer > 0:
                        code |= BRAILLE_DOTS[by][bx]
                        primary_layer = max(primary_layer, layer)

            char = chr(0x2800 + code)

            # ANSI 24-bit TrueColor styling based on layer
            if primary_layer == 4:
                # Robot pose: Bright Neon Green
                styled = f"\033[38;2;50;255;50m{char}\033[0m"
            elif primary_layer == 5:
                # Landmark & Ellipse: Vivid Coral Orange
                styled = f"\033[38;2;255;120;30m{char}\033[0m"
            elif primary_layer == 3:
                # Path: Bright Magenta
                styled = f"\033[38;2;240;60;220m{char}\033[0m"
            elif primary_layer == 1:
                # Obstacle: Slate Gray / White
                styled = f"\033[38;2;200;210;220m{char}\033[0m"
            elif primary_layer == 2:
                # LiDAR ray: Cyan Dim
                styled = f"\033[38;2;40;140;200m{char}\033[0m"
            else:
                # Empty space
                styled = " "

            row_chars.append(styled)

        content = "".join(row_chars)
        lines.append(f"|{content}|" if border else content)

    if border:
        lines.append("+" + "-" * width_chars + "+")

    return "\n".join(lines)
