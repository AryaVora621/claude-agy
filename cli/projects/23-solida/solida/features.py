"""
Solida: CAD Feature Operations - Linear Extrusion, Rotational Revolve & Multi-Section Loft.
Converts 2D profile sketches into watertight 3D B-Rep manifold solids.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Optional, Tuple
from solida.geometry import Vector3D, Matrix4x4, Plane
from solida.brep import Solid, build_solid_from_polygons


class Sketch2D:
    """Planar 2D closed profile for CAD feature sweeps."""
    __slots__ = ("points",)

    def __init__(self, points: Sequence[Vector3D]) -> None:
        if len(points) < 3:
            raise ValueError("Sketch2D profile requires at least 3 points")
        self.points = list(points)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Vector3D:
        return self.points[idx]

    def translate(self, dx: float, dy: float, dz: float) -> Sketch2D:
        """Translates all points in profile sketch."""
        offset = Vector3D(dx, dy, dz)
        return Sketch2D([p + offset for p in self.points])

    def to_plane(self, plane_name: str = "XY") -> Sketch2D:
        """
        Reorients a 2D profile from XY plane into target plane ('XY', 'XZ', 'YZ').
        """
        p_name = plane_name.upper()
        if p_name == "XY":
            return self
        elif p_name == "XZ":
            # Map (x, y, 0) -> (x, 0, y)
            return Sketch2D([Vector3D(p.x, 0.0, p.y) for p in self.points])
        elif p_name == "YZ":
            # Map (x, y, 0) -> (0, x, y)
            return Sketch2D([Vector3D(0.0, p.x, p.y) for p in self.points])
        else:
            raise ValueError(f"Unknown plane {plane_name}; expected 'XY', 'XZ', or 'YZ'")

    @classmethod
    def rectangle(
        cls, width: float, height: float, center: bool = False
    ) -> Sketch2D:
        """Creates a rectangular profile in the XY plane."""
        w, h = float(width), float(height)
        if center:
            x0, x1 = -0.5 * w, 0.5 * w
            y0, y1 = -0.5 * h, 0.5 * h
        else:
            x0, x1 = 0.0, w
            y0, y1 = 0.0, h
        pts = [
            Vector3D(x0, y0, 0.0),
            Vector3D(x1, y0, 0.0),
            Vector3D(x1, y1, 0.0),
            Vector3D(x0, y1, 0.0),
        ]
        return cls(pts)

    @classmethod
    def circle(cls, radius: float, segments: int = 32) -> Sketch2D:
        """Creates a circular profile in the XY plane."""
        r = float(radius)
        n = max(8, int(segments))
        step = 2.0 * math.pi / n
        pts = [
            Vector3D(r * math.cos(i * step), r * math.sin(i * step), 0.0)
            for i in range(n)
        ]
        return cls(pts)

    @classmethod
    def regular_polygon(cls, radius: float, sides: int = 6) -> Sketch2D:
        """Creates a regular polygon (e.g. hexagon, octagon) in the XY plane."""
        r = float(radius)
        n = max(3, int(sides))
        step = 2.0 * math.pi / n
        pts = [
            Vector3D(r * math.cos(i * step), r * math.sin(i * step), 0.0)
            for i in range(n)
        ]
        return cls(pts)

    @classmethod
    def star(cls, outer_radius: float, inner_radius: float, points: int = 5) -> Sketch2D:
        """Creates a star-shaped polygon profile in the XY plane."""
        r_out = float(outer_radius)
        r_in = float(inner_radius)
        n = max(3, int(points))
        step = math.pi / n
        pts: List[Vector3D] = []
        for i in range(2 * n):
            r = r_out if i % 2 == 0 else r_in
            theta = i * step
            pts.append(Vector3D(r * math.cos(theta), r * math.sin(theta), 0.0))
        return cls(pts)

    @classmethod
    def naca_airfoil(
        cls,
        code: str = "2412",
        chord: float = 1.0,
        num_points: int = 30,
    ) -> Sketch2D:
        """
        Generates a 4-digit NACA airfoil profile in the XY plane.
        Leading edge at origin, trailing edge at (chord, 0).
        """
        c = float(chord)
        n_pts = max(10, int(num_points))

        # Digits: M = max camber / 100, P = max camber position / 10, T = thickness / 100
        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0

        upper: List[Vector3D] = []
        lower: List[Vector3D] = []

        # Half-cosine distribution for fine leading-edge resolution
        for i in range(n_pts + 1):
            beta = i * math.pi / n_pts
            x = c * 0.5 * (1.0 - math.cos(beta))
            xc = x / c if c > 0 else 0.0

            # Thickness distribution yt
            yt = (
                5.0 * t * c * (
                    0.2969 * math.sqrt(max(0.0, xc))
                    - 0.1260 * xc
                    - 0.3516 * (xc ** 2)
                    + 0.2843 * (xc ** 3)
                    - 0.1015 * (xc ** 4)
                )
            )

            # Camber line yc and slope theta
            if xc < p and p > 0:
                yc = (m * c / (p ** 2)) * (2.0 * p * xc - xc ** 2)
                dy_dx = (2.0 * m / (p ** 2)) * (p - xc)
            elif p < 1.0:
                yc = (m * c / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc ** 2)
                dy_dx = (2.0 * m / ((1.0 - p) ** 2)) * (p - xc)
            else:
                yc = 0.0
                dy_dx = 0.0

            theta = math.atan(dy_dx)
            xu = x - yt * math.sin(theta)
            yu = yc + yt * math.cos(theta)
            xl = x + yt * math.sin(theta)
            yl = yc - yt * math.cos(theta)

            upper.append(Vector3D(xu, yu, 0.0))
            if i > 0 and i < n_pts:
                lower.append(Vector3D(xl, yl, 0.0))

        # Closed profile: upper surface LE -> TE, then lower surface TE -> LE
        pts = upper + list(reversed(lower))
        return cls(pts)


def extrude(
    profile: Sketch2D,
    direction: Optional[Vector3D] = None,
    height: Optional[float] = None,
    draft_angle_deg: float = 0.0,
    twist_angle_deg: float = 0.0,
    steps: int = 1,
    name: str = "Extrusion",
) -> Solid:
    """
    Sweeps a 2D planar profile sketch along a direction vector to create a 3D solid.
    Supports linear extrusion, draft angles, and axial twist.
    """
    if direction is None:
        h = 1.0 if height is None else float(height)
        dir_vec = Vector3D(0.0, 0.0, h)
    else:
        dir_vec = direction

    n_steps = max(1, int(steps))
    pts_base = list(profile.points)
    n_pts = len(pts_base)

    # Compute base polygon centroid
    cx = sum(p.x for p in pts_base) / n_pts
    cy = sum(p.y for p in pts_base) / n_pts
    cz = sum(p.z for p in pts_base) / n_pts
    centroid = Vector3D(cx, cy, cz)

    # Generate profile levels
    levels: List[List[Vector3D]] = []
    draft_rad = math.radians(draft_angle_deg)
    twist_rad = math.radians(twist_angle_deg)

    for s in range(n_steps + 1):
        t = s / n_steps
        offset = dir_vec * t
        scale = 1.0 - t * math.tan(draft_rad)
        rot_angle = t * twist_rad

        cos_rot = math.cos(rot_angle)
        sin_rot = math.sin(rot_angle)

        level_pts: List[Vector3D] = []
        for p in pts_base:
            # Shift to centroid, scale, twist, and shift back + offset
            d = p - centroid
            dx = d.x * scale
            dy = d.y * scale
            rx = dx * cos_rot - dy * sin_rot
            ry = dx * sin_rot + dy * cos_rot
            level_pts.append(centroid + Vector3D(rx, ry, d.z * scale) + offset)
        levels.append(level_pts)

    faces: List[List[Vector3D]] = []

    # Bottom cap (reversed winding for outward pointing normal)
    faces.append(list(reversed(levels[0])))

    # Lateral quad strips between adjacent levels
    for s in range(n_steps):
        l_curr = levels[s]
        l_next = levels[s + 1]
        for i in range(n_pts):
            p0 = l_curr[i]
            p1 = l_curr[(i + 1) % n_pts]
            p2 = l_next[(i + 1) % n_pts]
            p3 = l_next[i]
            faces.append([p0, p1, p2, p3])

    # Top cap (standard CCW winding)
    faces.append(list(levels[-1]))

    return build_solid_from_polygons(faces, name=name)


def revolve(
    profile: Sketch2D,
    axis_origin: Optional[Vector3D] = None,
    axis_dir: Optional[Vector3D] = None,
    angle_deg: float = 360.0,
    segments: int = 32,
    name: str = "Revolve",
) -> Solid:
    """
    Sweeps a planar profile sketch around an axis of rotation.
    If angle == 360 deg, creates a closed rotationally symmetric solid.
    If angle < 360 deg, creates a solid bounded by sector cap faces.
    """
    a_orig = Vector3D(0, 0, 0) if axis_origin is None else axis_origin
    a_dir = Vector3D(0, 0, 1) if axis_dir is None else axis_dir.normalized()

    total_angle = math.radians(min(360.0, max(1.0, float(angle_deg))))
    is_full_circle = abs(total_angle - 2.0 * math.pi) < 1e-4
    n_segs = max(6, int(segments))

    step_angle = total_angle / n_segs
    profile_pts = list(profile.points)
    n_pts = len(profile_pts)

    # Generate rotated profile rings
    rings: List[List[Vector3D]] = []
    num_rings = n_segs if is_full_circle else (n_segs + 1)

    for s in range(num_rings):
        theta = s * step_angle
        # Rotation about arbitrary axis using Matrix4x4
        rot_mat = (
            Matrix4x4.translation(a_orig.x, a_orig.y, a_orig.z)
            @ Matrix4x4.rotation_axis_angle(a_dir, theta)
            @ Matrix4x4.translation(-a_orig.x, -a_orig.y, -a_orig.z)
        )
        rings.append([rot_mat.transform_point(p) for p in profile_pts])

    faces: List[List[Vector3D]] = []

    # Lateral quad strips
    for s in range(n_segs):
        r_curr = rings[s]
        r_next = rings[(s + 1) % len(rings)]
        for i in range(n_pts):
            p0 = r_curr[i]
            p1 = r_curr[(i + 1) % n_pts]
            p2 = r_next[(i + 1) % n_pts]
            p3 = r_next[i]
            faces.append([p0, p1, p2, p3])

    # End caps for partial revolution
    if not is_full_circle:
        faces.append(list(reversed(rings[0])))
        faces.append(list(rings[-1]))

    return build_solid_from_polygons(faces, name=name)


def loft(
    profiles: Sequence[Sketch2D],
    name: str = "Loft",
) -> Solid:
    """
    Creates a transition solid through two or more planar profile sketches.
    Connects corresponding vertices with quadrilateral lateral faces.
    """
    if len(profiles) < 2:
        raise ValueError("Loft requires at least two profile sketches")

    # Resample all profiles to uniform point count
    target_count = len(profiles[0].points)
    for p in profiles[1:]:
        if len(p.points) != target_count:
            raise ValueError("Currently loft profiles must share identical vertex count")

    faces: List[List[Vector3D]] = []

    # Bottom cap
    faces.append(list(reversed(profiles[0].points)))

    # Lateral skins between adjacent stations
    for s in range(len(profiles) - 1):
        p_curr = profiles[s].points
        p_next = profiles[s + 1].points
        for i in range(target_count):
            p0 = p_curr[i]
            p1 = p_curr[(i + 1) % target_count]
            p2 = p_next[(i + 1) % target_count]
            p3 = p_next[i]
            faces.append([p0, p1, p2, p3])

    # Top cap
    faces.append(list(profiles[-1].points))

    return build_solid_from_polygons(faces, name=name)
