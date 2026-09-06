"""
Solida: 3D Exact B-Rep Solid Primitives.
Generators for Box, Cylinder, Sphere, Cone, and Torus.
Guarantees outward-pointing normals, 2-manifold half-edge topology, and Euler invariants.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Optional
from solida.geometry import Vector3D
from solida.brep import Solid, build_solid_from_polygons


def make_box(
    dx: float,
    dy: float,
    dz: float,
    center: bool = False,
    origin: Optional[Vector3D] = None,
    name: str = "Box",
) -> Solid:
    """
    Generates a 3D rectangular cuboid solid.
    V = 8, E = 12, F = 6, Euler chi = 2.
    """
    ox, oy, oz = (0.0, 0.0, 0.0) if origin is None else (origin.x, origin.y, origin.z)
    if center:
        x0, x1 = ox - 0.5 * dx, ox + 0.5 * dx
        y0, y1 = oy - 0.5 * dy, oy + 0.5 * dy
        z0, z1 = oz - 0.5 * dz, oz + 0.5 * dz
    else:
        x0, x1 = ox, ox + dx
        y0, y1 = oy, oy + dy
        z0, z1 = oz, oz + dz

    # 8 vertices
    p000 = Vector3D(x0, y0, z0)
    p100 = Vector3D(x1, y0, z0)
    p110 = Vector3D(x1, y1, z0)
    p010 = Vector3D(x0, y1, z0)

    p001 = Vector3D(x0, y0, z1)
    p101 = Vector3D(x1, y0, z1)
    p111 = Vector3D(x1, y1, z1)
    p011 = Vector3D(x0, y1, z1)

    # 6 outward oriented faces (CCW from outside)
    faces = [
        [p000, p010, p110, p100],  # Bottom (-Z)
        [p001, p101, p111, p011],  # Top (+Z)
        [p000, p100, p101, p001],  # Front (-Y)
        [p110, p010, p011, p111],  # Back (+Y)
        [p000, p001, p011, p010],  # Left (-X)
        [p100, p110, p111, p101],  # Right (+X)
    ]
    return build_solid_from_polygons(faces, name=name)


def make_cylinder(
    radius: float,
    height: float,
    segments: int = 32,
    center: bool = False,
    name: str = "Cylinder",
) -> Solid:
    """
    Generates a 3D right circular cylinder solid.
    Comprises bottom disk, top disk, and lateral quadrilateral faces.
    V = 2 * segs, E = 3 * segs, F = segs + 2, Euler chi = 2.
    """
    r = float(radius)
    h = float(height)
    n = max(8, int(segments))

    z0 = -0.5 * h if center else 0.0
    z1 = 0.5 * h if center else h

    bottom_ring: List[Vector3D] = []
    top_ring: List[Vector3D] = []

    angle_step = 2.0 * math.pi / n
    for i in range(n):
        theta = i * angle_step
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        bottom_ring.append(Vector3D(x, y, z0))
        top_ring.append(Vector3D(x, y, z1))

    faces: List[List[Vector3D]] = []

    # Bottom face: reversed ring for outward normal pointing down (-Z)
    faces.append(list(reversed(bottom_ring)))

    # Top face: standard CCW ring for outward normal pointing up (+Z)
    faces.append(list(top_ring))

    # Lateral quad faces
    for i in range(n):
        b0 = bottom_ring[i]
        b1 = bottom_ring[(i + 1) % n]
        t1 = top_ring[(i + 1) % n]
        t0 = top_ring[i]
        faces.append([b0, b1, t1, t0])

    return build_solid_from_polygons(faces, name=name)


def make_sphere(
    radius: float,
    rings: int = 16,
    sectors: int = 32,
    center: Optional[Vector3D] = None,
    name: str = "Sphere",
) -> Solid:
    """
    Generates a 3D UV Sphere solid.
    V = (rings - 1) * sectors + 2, Euler chi = 2.
    """
    r = float(radius)
    n_rings = max(4, int(rings))
    n_sectors = max(6, int(sectors))
    c = Vector3D(0, 0, 0) if center is None else center

    # Generate vertices
    # South pole at -Z
    south_pole = c + Vector3D(0.0, 0.0, -r)
    # North pole at +Z
    north_pole = c + Vector3D(0.0, 0.0, r)

    grid: List[List[Vector3D]] = []
    phi_step = math.pi / n_rings
    theta_step = 2.0 * math.pi / n_sectors

    for i in range(1, n_rings):
        phi = -0.5 * math.pi + i * phi_step
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        ring: List[Vector3D] = []
        for j in range(n_sectors):
            theta = j * theta_step
            x = r * cos_phi * math.cos(theta)
            y = r * cos_phi * math.sin(theta)
            z = r * sin_phi
            ring.append(c + Vector3D(x, y, z))
        grid.append(ring)

    faces: List[List[Vector3D]] = []

    # South pole triangle fan
    first_ring = grid[0]
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors
        faces.append([south_pole, first_ring[j_next], first_ring[j]])

    # Intermediate quad bands
    for i in range(len(grid) - 1):
        r_curr = grid[i]
        r_next = grid[i + 1]
        for j in range(n_sectors):
            j_next = (j + 1) % n_sectors
            faces.append([r_curr[j], r_curr[j_next], r_next[j_next], r_next[j]])

    # North pole triangle fan
    last_ring = grid[-1]
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors
        faces.append([north_pole, last_ring[j], last_ring[j_next]])

    return build_solid_from_polygons(faces, name=name)


def make_cone(
    radius: float,
    height: float,
    segments: int = 32,
    center: bool = False,
    name: str = "Cone",
) -> Solid:
    """
    Generates a 3D right circular cone solid.
    V = segs + 1, E = 2 * segs, F = segs + 1, Euler chi = 2.
    """
    r = float(radius)
    h = float(height)
    n = max(8, int(segments))

    z0 = -0.5 * h if center else 0.0
    z1 = 0.5 * h if center else h
    apex = Vector3D(0.0, 0.0, z1)

    base_ring: List[Vector3D] = []
    angle_step = 2.0 * math.pi / n
    for i in range(n):
        theta = i * angle_step
        base_ring.append(Vector3D(r * math.cos(theta), r * math.sin(theta), z0))

    faces: List[List[Vector3D]] = []

    # Base cap (reversed for outward normal pointing -Z)
    faces.append(list(reversed(base_ring)))

    # Lateral triangular faces meeting at apex
    for i in range(n):
        p0 = base_ring[i]
        p1 = base_ring[(i + 1) % n]
        faces.append([p0, p1, apex])

    return build_solid_from_polygons(faces, name=name)


def make_torus(
    major_radius: float,
    minor_radius: float,
    major_segs: int = 24,
    minor_segs: int = 16,
    name: str = "Torus",
) -> Solid:
    """
    Generates a 3D Torus solid (genus-1 donut).
    Euler characteristic chi = V - E + F = 0 (topological torus).
    """
    R = float(major_radius)
    r = float(minor_radius)
    n_u = max(8, int(major_segs))
    n_v = max(6, int(minor_segs))

    step_u = 2.0 * math.pi / n_u
    step_v = 2.0 * math.pi / n_v

    grid: List[List[Vector3D]] = []
    for i in range(n_u):
        u = i * step_u
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        ring: List[Vector3D] = []
        for j in range(n_v):
            v = j * step_v
            cos_v = math.cos(v)
            sin_v = math.sin(v)
            x = (R + r * cos_v) * cos_u
            y = (R + r * cos_v) * sin_u
            z = r * sin_v
            ring.append(Vector3D(x, y, z))
        grid.append(ring)

    faces: List[List[Vector3D]] = []
    for i in range(n_u):
        i_next = (i + 1) % n_u
        for j in range(n_v):
            j_next = (j + 1) % n_v
            p00 = grid[i][j]
            p01 = grid[i][j_next]
            p11 = grid[i_next][j_next]
            p10 = grid[i_next][j]
            faces.append([p00, p10, p11, p01])

    return build_solid_from_polygons(faces, name=name)
