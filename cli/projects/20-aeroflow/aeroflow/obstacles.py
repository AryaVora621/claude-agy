"""
AeroFlow: Solid Obstacle Geometries & Rasterization Masks.
Supports circular bluff bodies, ellipses, NACA 4-digit airfoils (with camber & angle of attack),
flat plates, and backward-facing steps.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

from aeroflow.types import ObstacleMask, Vector2D


def create_cylinder_obstacle(
    nx: int,
    ny: int,
    center_x: float,
    center_y: float,
    radius: float,
) -> ObstacleMask:
    """
    Generate circular cylinder bluff body obstacle mask.
    """
    mask = ObstacleMask(nx, ny)
    r_sq = radius * radius
    for y in range(ny):
        dy = y - center_y
        dy_sq = dy * dy
        for x in range(nx):
            dx = x - center_x
            if (dx * dx + dy_sq) <= r_sq:
                mask.set_solid(x, y, True)
    mask.update_boundary_list()
    return mask


def create_ellipse_obstacle(
    nx: int,
    ny: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle_deg: float = 0.0,
) -> ObstacleMask:
    """
    Generate oriented elliptical cylinder obstacle mask.
    """
    mask = ObstacleMask(nx, ny)
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    inv_rx_sq = 1.0 / (radius_x * radius_x) if radius_x > 0 else 1.0
    inv_ry_sq = 1.0 / (radius_y * radius_y) if radius_y > 0 else 1.0

    for y in range(ny):
        dy = y - center_y
        for x in range(nx):
            dx = x - center_x
            # Transform to ellipse principal axes
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            if (x_rot * x_rot * inv_rx_sq + y_rot * y_rot * inv_ry_sq) <= 1.0:
                mask.set_solid(x, y, True)

    mask.update_boundary_list()
    return mask


def create_naca_airfoil_obstacle(
    nx: int,
    ny: int,
    chord: float,
    code: str = "0012",
    lead_x: float = 25.0,
    lead_y: float = 40.0,
    angle_of_attack_deg: float = 0.0,
) -> ObstacleMask:
    """
    Generate NACA 4-digit series airfoil solid mask with variable camber and angle of attack.
    Code format: MPXX (e.g. '0012', '2412', '4415')
      M: Max camber (first digit / 100)
      P: Position of max camber (second digit / 10)
      XX: Max thickness (last two digits / 100)
    """
    mask = ObstacleMask(nx, ny)

    # Parse 4-digit code
    if len(code) != 4 or not code.isdigit():
        code = "0012"

    m = float(code[0]) / 100.0
    p = float(code[1]) / 10.0
    t = float(code[2:]) / 100.0

    # Rotation angle (negative for standard pitch-up angle of attack)
    alpha = math.radians(angle_of_attack_deg)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    # Quarter-chord rotation center
    qc_x = lead_x + 0.25 * chord
    qc_y = lead_y

    # Generate dense upper and lower coordinates along chord
    num_samples = max(200, int(chord * 6))
    upper_pts: List[Tuple[float, float]] = []
    lower_pts: List[Tuple[float, float]] = []

    for i in range(num_samples + 1):
        # Cosine clustering of points near leading edge
        beta = math.pi * (i / num_samples)
        xc = 0.5 * (1.0 - math.cos(beta))  # x / c in [0, 1]
        x_phys = xc * chord

        # Half-thickness yt
        yt = 5.0 * t * chord * (
            0.2969 * math.sqrt(max(0.0, xc))
            - 0.1260 * xc
            - 0.3516 * (xc ** 2)
            + 0.2843 * (xc ** 3)
            - 0.1015 * (xc ** 4)
        )

        # Camber line yc and camber slope theta
        if xc < p and p > 0.0:
            yc = (m / (p * p)) * (2.0 * p * xc - xc * xc) * chord
            dyc_dx = (2.0 * m / (p * p)) * (p - xc)
        elif p < 1.0 and p > 0.0:
            yc = (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc * xc) * chord
            dyc_dx = (2.0 * m / ((1.0 - p) ** 2)) * (p - xc)
        else:
            yc = 0.0
            dyc_dx = 0.0

        theta_c = math.atan(dyc_dx)

        # Upper and lower coordinates before pitch rotation
        xu = x_phys - yt * math.sin(theta_c)
        yu = yc + yt * math.cos(theta_c)

        xl = x_phys + yt * math.sin(theta_c)
        yl = yc - yt * math.cos(theta_c)

        # Translate relative to quarter-chord, rotate by alpha, translate to domain
        for raw_x, raw_y, pt_list in [(xu, yu, upper_pts), (xl, yl, lower_pts)]:
            dx = (lead_x + raw_x) - qc_x
            dy = (lead_y + raw_y) - qc_y
            rx = qc_x + (dx * cos_a - dy * sin_a)
            ry = qc_y + (dx * sin_a + dy * cos_a)
            pt_list.append((rx, ry))

    # Construct closed polygon: leading edge -> upper -> trailing edge -> lower -> leading edge
    polygon = upper_pts + list(reversed(lower_pts))

    # Rasterize polygon into boolean grid using ray-casting point-in-polygon
    poly_len = len(polygon)
    min_x = max(0, int(min(pt[0] for pt in polygon) - 1))
    max_x = min(nx - 1, int(max(pt[0] for pt in polygon) + 1))
    min_y = max(0, int(min(pt[1] for pt in polygon) - 1))
    max_y = min(ny - 1, int(max(pt[1] for pt in polygon) + 1))

    for y in range(min_y, max_y + 1):
        py = float(y)
        for x in range(min_x, max_x + 1):
            px = float(x)
            # Point in polygon test (Jordan curve theorem)
            inside = False
            p1x, p1y = polygon[0]
            for j in range(1, poly_len + 1):
                p2x, p2y = polygon[j % poly_len]
                if min(p1y, p2y) < py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or px <= x_inters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            if inside:
                mask.set_solid(x, y, True)

    mask.update_boundary_list()
    return mask


def create_plate_obstacle(
    nx: int,
    ny: int,
    start_x: float,
    start_y: float,
    length: float,
    thickness: float = 2.0,
    angle_deg: float = 0.0,
) -> ObstacleMask:
    """
    Generate rectangular flat plate solid obstacle.
    """
    mask = ObstacleMask(nx, ny)
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_t = thickness * 0.5

    for y in range(ny):
        dy = y - start_y
        for x in range(nx):
            dx = x - start_x
            # Project onto plate tangent and normal
            s = dx * cos_t + dy * sin_t
            n = -dx * sin_t + dy * cos_t
            if 0.0 <= s <= length and -half_t <= n <= half_t:
                mask.set_solid(x, y, True)

    mask.update_boundary_list()
    return mask
