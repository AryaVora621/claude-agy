"""
Solida: 3D Mesh Tessellation & Industry-Standard STL / Wavefront OBJ Exporters.
Includes Ear-Clipping triangulation, ASCII STL, Binary STL, and OBJ codecs.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
import struct
from typing import List, Tuple, Sequence, Optional, Dict, Union, BinaryIO, TextIO
from solida.geometry import Vector3D
from solida.brep import Solid, Face


class Facet3D:
    """Oriented 3D triangular facet with unit normal."""
    __slots__ = ("v0", "v1", "v2", "normal")

    def __init__(
        self,
        v0: Vector3D,
        v1: Vector3D,
        v2: Vector3D,
        normal: Optional[Vector3D] = None,
    ) -> None:
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        if normal is None:
            e1 = v1 - v0
            e2 = v2 - v0
            cross = e1.cross(e2)
            self.normal = cross.normalized() if cross.norm_sq() > 1e-15 else Vector3D(0, 0, 1)
        else:
            self.normal = normal.normalized()

    def __repr__(self) -> str:
        return f"Facet3D(n={self.normal}, v0={self.v0}, v1={self.v1}, v2={self.v2})"

    def area(self) -> float:
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        return 0.5 * e1.cross(e2).norm()

    def centroid(self) -> Vector3D:
        return (self.v0 + self.v1 + self.v2) / 3.0


def triangulate_polygon_2d(pts_2d: Sequence[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """
    Ear-clipping triangulation for simple non-self-intersecting 2D polygons.
    Returns list of vertex index triplets (i, j, k).
    """
    n = len(pts_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Compute 2D signed area (shoelace formula) to check winding
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += pts_2d[i][0] * pts_2d[j][1] - pts_2d[j][0] * pts_2d[i][1]

    # Ensure counter-clockwise ordering
    indices = list(range(n))
    if area2 < 0:
        indices.reverse()

    def is_convex(i: int, j: int, k: int) -> bool:
        pi, pj, pk = pts_2d[i], pts_2d[j], pts_2d[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pj[1]) - (pj[1] - pi[1]) * (pk[0] - pj[0])
        return cross >= 0.0

    def point_in_triangle(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        has_neg = (d1 < -1e-7) or (d2 < -1e-7) or (d3 < -1e-7)
        has_pos = (d1 > 1e-7) or (d2 > 1e-7) or (d3 > 1e-7)
        return not (has_neg and has_pos)

    triangles: List[Tuple[int, int, int]] = []
    # Ear clipping iteration
    count = 2 * len(indices)
    curr = 0
    while len(indices) > 3 and count > 0:
        count -= 1
        num = len(indices)
        prev_idx = indices[(curr - 1 + num) % num]
        curr_idx = indices[curr % num]
        next_idx = indices[(curr + 1) % num]

        if is_convex(prev_idx, curr_idx, next_idx):
            # Check if any other vertex lies inside this candidate ear
            is_ear = True
            for other_idx in indices:
                if other_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if point_in_triangle(pts_2d[other_idx], pts_2d[prev_idx], pts_2d[curr_idx], pts_2d[next_idx]):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((prev_idx, curr_idx, next_idx))
                indices.pop(curr % num)
                continue

        curr = (curr + 1) % num

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    else:
        # Fallback fan triangulation if polygon has degenerate edges
        for i in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[i], indices[i + 1]))

    return triangles


def triangulate_polygon_3d(points: Sequence[Vector3D]) -> List[Facet3D]:
    """
    Projects 3D planar polygon onto dominant 2D axis plane, triangulates,
    and reconstructs 3D Facet3D objects.
    """
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [Facet3D(points[0], points[1], points[2])]

    # Determine dominant normal axis using Newell's method
    nx, ny, nz = 0.0, 0.0, 0.0
    for i in range(n):
        curr = points[i]
        nxt = points[(i + 1) % n]
        nx += (curr.y - nxt.y) * (curr.z + nxt.z)
        ny += (curr.z - nxt.z) * (curr.x + nxt.x)
        nz += (curr.x - nxt.x) * (curr.y + nxt.y)

    abs_x, abs_y, abs_z = abs(nx), abs(ny), abs(nz)
    normal = Vector3D(nx, ny, nz).normalized()

    # Project to 2D
    pts_2d: List[Tuple[float, float]] = []
    if abs_z >= abs_x and abs_z >= abs_y:
        # Project onto XY plane
        pts_2d = [(p.x, p.y) for p in points]
    elif abs_y >= abs_x and abs_y >= abs_z:
        # Project onto XZ plane
        pts_2d = [(p.x, p.z) for p in points]
    else:
        # Project onto YZ plane
        pts_2d = [(p.y, p.z) for p in points]

    tri_indices = triangulate_polygon_2d(pts_2d)
    facets: List[Facet3D] = []
    for i, j, k in tri_indices:
        f = Facet3D(points[i], points[j], points[k], normal=normal)
        # Ensure facet normal matches Newell normal
        if f.normal.dot(normal) < 0:
            f = Facet3D(points[i], points[k], points[j], normal=normal)
        facets.append(f)

    return facets


def tessellate_solid(solid: Solid) -> List[Facet3D]:
    """Tessellates entire 3D B-Rep solid into triangular facets."""
    facets: List[Facet3D] = []
    for face in solid.faces:
        pts = [v.point for v in face.vertices()]
        face_facets = triangulate_polygon_3d(pts)
        facets.extend(face_facets)
    return facets


def export_stl_ascii(solid: Solid, solid_name: Optional[str] = None) -> str:
    """Exports B-Rep solid to ASCII STL format string."""
    name = solid.name if solid_name is None else solid_name
    facets = tessellate_solid(solid)
    lines: List[str] = [f"solid {name}"]

    for f in facets:
        n = f.normal
        lines.append(f"  facet normal {n.x:.6e} {n.y:.6e} {n.z:.6e}")
        lines.append("    outer loop")
        lines.append(f"      vertex {f.v0.x:.6e} {f.v0.y:.6e} {f.v0.z:.6e}")
        lines.append(f"      vertex {f.v1.x:.6e} {f.v1.y:.6e} {f.v1.z:.6e}")
        lines.append(f"      vertex {f.v2.x:.6e} {f.v2.y:.6e} {f.v2.z:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")

    lines.append(f"endsolid {name}\n")
    return "\n".join(lines)


def export_stl_binary(solid: Solid, header_text: str = "Solida B-Rep CAD Kernel") -> bytes:
    """
    Exports B-Rep solid to standard 80-byte header Binary STL bytes.
    Structure:
      80 bytes: header string (ASCII, space-padded)
      4 bytes: unsigned int32 triangle count N
      N * 50 bytes: (3x float32 normal, 3x float32 v0, 3x float32 v1, 3x float32 v2, uint16 attr)
    """
    facets = tessellate_solid(solid)
    header = header_text.encode("ascii")[:80].ljust(80, b" ")
    num_triangles = len(facets)

    chunks: List[bytes] = [header, struct.pack("<I", num_triangles)]

    for f in facets:
        record = struct.pack(
            "<12fH",
            f.normal.x, f.normal.y, f.normal.z,
            f.v0.x, f.v0.y, f.v0.z,
            f.v1.x, f.v1.y, f.v1.z,
            f.v2.x, f.v2.y, f.v2.z,
            0,  # attribute byte count
        )
        chunks.append(record)

    return b"".join(chunks)


def export_obj(solid: Solid, name: Optional[str] = None) -> str:
    """
    Exports B-Rep solid to Wavefront OBJ format string with shared vertices.
    """
    obj_name = solid.name if name is None else name
    facets = tessellate_solid(solid)

    vertex_list: List[Vector3D] = []
    vertex_map: Dict[Tuple[float, float, float], int] = {}

    def get_vertex_idx(p: Vector3D) -> int:
        key = (round(p.x, 6), round(p.y, 6), round(p.z, 6))
        if key not in vertex_map:
            idx = len(vertex_list) + 1  # OBJ is 1-indexed
            vertex_map[key] = idx
            vertex_list.append(p)
            return idx
        return vertex_map[key]

    face_indices: List[Tuple[int, int, int]] = []
    for f in facets:
        i0 = get_vertex_idx(f.v0)
        i1 = get_vertex_idx(f.v1)
        i2 = get_vertex_idx(f.v2)
        face_indices.append((i0, i1, i2))

    lines: List[str] = [
        f"# Solida Wavefront OBJ Export: {obj_name}",
        f"o {obj_name}",
    ]

    for v in vertex_list:
        lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")

    for i0, i1, i2 in face_indices:
        lines.append(f"f {i0} {i1} {i2}")

    return "\n".join(lines) + "\n"


def parse_stl_ascii(content: str) -> List[Facet3D]:
    """Parses ASCII STL format string into Facet3D list."""
    facets: List[Facet3D] = []
    lines = content.splitlines()
    i = 0
    curr_normal = Vector3D(0, 0, 1)
    curr_verts: List[Vector3D] = []

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("facet normal"):
            parts = line.split()
            curr_normal = Vector3D(float(parts[2]), float(parts[3]), float(parts[4]))
            curr_verts = []
        elif line.startswith("vertex"):
            parts = line.split()
            curr_verts.append(Vector3D(float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("endfacet"):
            if len(curr_verts) == 3:
                facets.append(Facet3D(curr_verts[0], curr_verts[1], curr_verts[2], curr_normal))
        i += 1
    return facets


def parse_stl_binary(data: bytes) -> List[Facet3D]:
    """Parses Binary STL byte buffer into Facet3D list."""
    if len(data) < 84:
        raise ValueError("Binary STL buffer too short")

    num_triangles = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + num_triangles * 50
    if len(data) < expected_len:
        raise ValueError(f"Binary STL truncated: expected {expected_len} bytes, got {len(data)}")

    facets: List[Facet3D] = []
    offset = 84
    for _ in range(num_triangles):
        nx, ny, nz, x0, y0, z0, x1, y1, z1, x2, y2, z2, _ = struct.unpack(
            "<12fH", data[offset : offset + 50]
        )
        offset += 50
        facets.append(
            Facet3D(
                Vector3D(x0, y0, z0),
                Vector3D(x1, y1, z1),
                Vector3D(x2, y2, z2),
                Vector3D(nx, ny, nz),
            )
        )
    return facets
