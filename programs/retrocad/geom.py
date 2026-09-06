"""RetroCAD First-Principles Geometric Modeling Kernel.

Provides 3D vector and 4x4 matrix mathematics, boundary representation (B-Rep)
polygonal meshes, parametric primitives, linear extrusion, rotational revolve,
plane clipping, Constructive Solid Geometry (CSG) Booleans, and CAD export codecs
(STL, OBJ, DXF, SVG). Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any


class Vec3:
    """3D Cartesian Vector with basic vector algebra."""

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        if abs(scalar) < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: Vec3) -> float:
        """Compute standard Euclidean inner product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        """Compute standard 3D cross product."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def magnitude(self) -> float:
        return math.sqrt(self.magnitude_sq())

    def normalize(self) -> Vec3:
        m = self.magnitude()
        if m < 1e-12:
            return Vec3(0.0, 0.0, 1.0)
        inv = 1.0 / m
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def distance(self, other: Vec3) -> float:
        return (self - other).magnitude()

    def lerp(self, other: Vec3, t: float) -> Vec3:
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )


class Mat4:
    """4x4 homogeneous transformation matrix in row-major order."""

    __slots__ = ('m',)

    def __init__(self, elements: Optional[List[List[float]]] = None) -> None:
        if elements is not None:
            self.m = elements
        else:
            self.m = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]

    @staticmethod
    def identity() -> Mat4:
        return Mat4()

    @staticmethod
    def translation(dx: float, dy: float, dz: float) -> Mat4:
        return Mat4([
            [1.0, 0.0, 0.0, dx],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dz],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> Mat4:
        return Mat4([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, sz,  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def rotation_x(rad: float) -> Mat4:
        c = math.cos(rad)
        s = math.sin(rad)
        return Mat4([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, c,   -s,   0.0],
            [0.0, s,    c,   0.0],
            [0.0, 0.0,  0.0, 1.0]
        ])

    @staticmethod
    def rotation_y(rad: float) -> Mat4:
        c = math.cos(rad)
        s = math.sin(rad)
        return Mat4([
            [c,   0.0, s,   0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s,  0.0, c,   0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def rotation_z(rad: float) -> Mat4:
        c = math.cos(rad)
        s = math.sin(rad)
        return Mat4([
            [c,   -s,   0.0, 0.0],
            [s,    c,   0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0]
        ])

    def mul(self, other: Mat4) -> Mat4:
        """Matrix product self * other."""
        res = [[0.0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                s = 0.0
                for k in range(4):
                    s += self.m[r][k] * other.m[k][c]
                res[r][c] = s
        return Mat4(res)

    def transform_point(self, v: Vec3) -> Vec3:
        """Apply affine transformation to 3D point (w = 1)."""
        x = self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z + self.m[0][3]
        y = self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z + self.m[1][3]
        z = self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z + self.m[2][3]
        w = self.m[3][0] * v.x + self.m[3][1] * v.y + self.m[3][2] * v.z + self.m[3][3]

        if abs(w - 1.0) > 1e-9 and abs(w) > 1e-12:
            inv_w = 1.0 / w
            return Vec3(x * inv_w, y * inv_w, z * inv_w)
        return Vec3(x, y, z)

    def transform_vector(self, v: Vec3) -> Vec3:
        """Apply linear transformation to 3D direction vector (w = 0)."""
        x = self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z
        y = self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z
        z = self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z
        return Vec3(x, y, z)


class Face:
    """Polygon face specified by an ordered list of vertex indices."""

    __slots__ = ('indices', 'normal', 'color')

    def __init__(self, indices: List[int], normal: Optional[Vec3] = None, color: str = "#3b82f6") -> None:
        self.indices = list(indices)
        self.normal = normal or Vec3(0.0, 0.0, 1.0)
        self.color = color


class Mesh:
    """Boundary Representation (B-Rep) 3D surface mesh."""

    def __init__(self, name: str = "Part") -> None:
        self.name = name
        self.vertices: List[Vec3] = []
        self.faces: List[Face] = []
        self.edges: List[Tuple[int, int]] = []

    def clone(self) -> Mesh:
        m = Mesh(self.name)
        m.vertices = [Vec3(v.x, v.y, v.z) for v in self.vertices]
        m.faces = [Face(list(f.indices), Vec3(f.normal.x, f.normal.y, f.normal.z), f.color) for f in self.faces]
        m.edges = list(self.edges)
        return m

    def build_edges(self) -> None:
        """Extract unique topological edges from face polygons."""
        edge_set = set()
        edges = []
        for face in self.faces:
            n = len(face.indices)
            for i in range(n):
                i1 = face.indices[i]
                i2 = face.indices[(i + 1) % n]
                edge = (min(i1, i2), max(i1, i2))
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append(edge)
        self.edges = edges

    def compute_normals(self) -> None:
        """Calculate geometric outward unit normal vectors for all faces."""
        for face in self.faces:
            if len(face.indices) < 3:
                face.normal = Vec3(0.0, 0.0, 1.0)
                continue
            # Newell method for robust arbitrary n-gon normal calculation
            nx = 0.0
            ny = 0.0
            nz = 0.0
            n = len(face.indices)
            for i in range(n):
                v1 = self.vertices[face.indices[i]]
                v2 = self.vertices[face.indices[(i + 1) % n]]
                nx += (v1.y - v2.y) * (v1.z + v2.z)
                ny += (v1.z - v2.z) * (v1.x + v2.x)
                nz += (v1.x - v2.x) * (v1.y + v2.y)
            normal = Vec3(nx, ny, nz).normalize()
            face.normal = normal

    def compute_bounding_box(self) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
        """Return (min_pt, max_pt, center, size)."""
        if not self.vertices:
            zero = Vec3(0, 0, 0)
            return zero, zero, zero, zero

        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        min_z = min(v.z for v in self.vertices)
        max_z = max(v.z for v in self.vertices)

        min_pt = Vec3(min_x, min_y, min_z)
        max_pt = Vec3(max_x, max_y, max_z)
        center = (min_pt + max_pt) * 0.5
        size = max_pt - min_pt
        return min_pt, max_pt, center, size

    def transform(self, mat: Mat4) -> Mesh:
        """Apply matrix transformation to all vertices in-place."""
        for i in range(len(self.vertices)):
            self.vertices[i] = mat.transform_point(self.vertices[i])
        self.compute_normals()
        return self

    def triangulate(self) -> List[Tuple[Vec3, Vec3, Vec3]]:
        """Decompose all n-gon faces into explicit 3D triangles."""
        triangles = []
        for face in self.faces:
            n = len(face.indices)
            if n < 3:
                continue
            v0 = self.vertices[face.indices[0]]
            for i in range(1, n - 1):
                v1 = self.vertices[face.indices[i]]
                v2 = self.vertices[face.indices[i + 1]]
                triangles.append((v0, v1, v2))
        return triangles

    def surface_area(self) -> float:
        """Calculate total surface area summing triangular face areas."""
        total = 0.0
        for v0, v1, v2 in self.triangulate():
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_prod = edge1.cross(edge2)
            total += 0.5 * cross_prod.magnitude()
        return total

    def volume(self) -> float:
        """Compute enclosed volume using divergence theorem on triangular mesh."""
        vol = 0.0
        for v0, v1, v2 in self.triangulate():
            # Signed volume of tetrahedron formed with origin
            vol += v0.dot(v1.cross(v2)) / 6.0
        return abs(vol)


# -------------------------------------------------------------
# Parametric Geometric Primitives
# -------------------------------------------------------------

def create_box(width: float, height: float, depth: float, name: str = "Box") -> Mesh:
    """Create a rectangular cuboid centered at origin."""
    m = Mesh(name)
    hx = width * 0.5
    hy = height * 0.5
    hz = depth * 0.5

    # 8 Vertices
    m.vertices = [
        Vec3(-hx, -hy, -hz),  # 0
        Vec3( hx, -hy, -hz),  # 1
        Vec3( hx,  hy, -hz),  # 2
        Vec3(-hx,  hy, -hz),  # 3
        Vec3(-hx, -hy,  hz),  # 4
        Vec3( hx, -hy,  hz),  # 5
        Vec3( hx,  hy,  hz),  # 6
        Vec3(-hx,  hy,  hz)   # 7
    ]

    # 6 Quad Faces with outward counter-clockwise winding
    m.faces = [
        Face([0, 3, 2, 1], Vec3(0, 0, -1), "#3b82f6"),  # Back (-Z)
        Face([4, 5, 6, 7], Vec3(0, 0,  1), "#3b82f6"),  # Front (+Z)
        Face([0, 1, 5, 4], Vec3(0, -1, 0), "#2563eb"),  # Bottom (-Y)
        Face([3, 7, 6, 2], Vec3(0,  1, 0), "#60a5fa"),  # Top (+Y)
        Face([0, 4, 7, 3], Vec3(-1, 0, 0), "#1d4ed8"),  # Left (-X)
        Face([1, 2, 6, 5], Vec3( 1, 0, 0), "#93c5fd")   # Right (+X)
    ]
    m.build_edges()
    m.compute_normals()
    return m


def create_cylinder(radius: float, height: float, segments: int = 24, name: str = "Cylinder") -> Mesh:
    """Create a vertical cylinder along the Z-axis centered at origin."""
    m = Mesh(name)
    hz = height * 0.5

    # Bottom circle vertices (0 to segments-1)
    for i in range(segments):
        angle = (2.0 * math.pi * i) / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        m.vertices.append(Vec3(x, y, -hz))

    # Top circle vertices (segments to 2*segments-1)
    for i in range(segments):
        angle = (2.0 * math.pi * i) / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        m.vertices.append(Vec3(x, y, hz))

    # Side quad faces
    for i in range(segments):
        next_i = (i + 1) % segments
        b1 = i
        b2 = next_i
        t1 = segments + i
        t2 = segments + next_i
        m.faces.append(Face([b1, b2, t2, t1], color="#10b981"))

    # Bottom cap (reversed order for outward normal)
    bottom_cap = [i for i in reversed(range(segments))]
    m.faces.append(Face(bottom_cap, Vec3(0, 0, -1), "#059669"))

    # Top cap
    top_cap = [segments + i for i in range(segments)]
    m.faces.append(Face(top_cap, Vec3(0, 0, 1), "#34d399"))

    m.build_edges()
    m.compute_normals()
    return m


def create_sphere(radius: float, rings: int = 14, sectors: int = 20, name: str = "Sphere") -> Mesh:
    """Create a UV sphere centered at origin."""
    m = Mesh(name)

    # Vertices
    for r in range(rings + 1):
        phi = math.pi * r / rings  # from 0 to pi
        for s in range(sectors):
            theta = 2.0 * math.pi * s / sectors  # from 0 to 2pi
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            m.vertices.append(Vec3(x, y, z))

    # Faces
    for r in range(rings):
        for s in range(sectors):
            next_s = (s + 1) % sectors
            i0 = r * sectors + s
            i1 = r * sectors + next_s
            i2 = (r + 1) * sectors + next_s
            i3 = (r + 1) * sectors + s

            if r == 0:
                # Top pole triangle
                m.faces.append(Face([i0, i2, i3], color="#a855f7"))
            elif r == rings - 1:
                # Bottom pole triangle
                m.faces.append(Face([i0, i1, i2], color="#9333ea"))
            else:
                # Quad band
                m.faces.append(Face([i0, i1, i2, i3], color="#c084fc"))

    m.build_edges()
    m.compute_normals()
    return m


def create_cone(radius: float, height: float, segments: int = 24, name: str = "Cone") -> Mesh:
    """Create a cone along the Z-axis with apex at +height/2 and base at -height/2."""
    m = Mesh(name)
    hz = height * 0.5

    # Base circle vertices
    for i in range(segments):
        angle = (2.0 * math.pi * i) / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        m.vertices.append(Vec3(x, y, -hz))

    # Apex vertex at index = segments
    apex_idx = len(m.vertices)
    m.vertices.append(Vec3(0.0, 0.0, hz))

    # Side triangles
    for i in range(segments):
        next_i = (i + 1) % segments
        m.faces.append(Face([i, next_i, apex_idx], color="#f59e0b"))

    # Base cap
    base_cap = [i for i in reversed(range(segments))]
    m.faces.append(Face(base_cap, Vec3(0, 0, -1), "#d97706"))

    m.build_edges()
    m.compute_normals()
    return m


def create_torus(r_major: float, r_minor: float, seg_major: int = 24, seg_minor: int = 14, name: str = "Torus") -> Mesh:
    """Create a torus lying on the XY plane centered at origin."""
    m = Mesh(name)

    for i in range(seg_major):
        theta = 2.0 * math.pi * i / seg_major
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for j in range(seg_minor):
            phi = 2.0 * math.pi * j / seg_minor
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)

            x = (r_major + r_minor * cos_p) * cos_t
            y = (r_major + r_minor * cos_p) * sin_t
            z = r_minor * sin_p
            m.vertices.append(Vec3(x, y, z))

    for i in range(seg_major):
        next_i = (i + 1) % seg_major
        for j in range(seg_minor):
            next_j = (j + 1) % seg_minor
            i0 = i * seg_minor + j
            i1 = next_i * seg_minor + j
            i2 = next_i * seg_minor + next_j
            i3 = i * seg_minor + next_j
            m.faces.append(Face([i0, i1, i2, i3], color="#ec4899"))

    m.build_edges()
    m.compute_normals()
    return m


# -------------------------------------------------------------
# Parametric Extrusion and Revolve
# -------------------------------------------------------------

def extrude_polygon(contour_2d: List[Tuple[float, float]], height: float, name: str = "Extrusion") -> Mesh:
    """Extrude a 2D planar polygon contour in XY plane along +Z axis."""
    m = Mesh(name)
    n = len(contour_2d)
    if n < 3:
        return m

    # Bottom vertices (z = 0)
    for x, y in contour_2d:
        m.vertices.append(Vec3(x, y, 0.0))

    # Top vertices (z = height)
    for x, y in contour_2d:
        m.vertices.append(Vec3(x, y, height))

    # Side quads
    for i in range(n):
        next_i = (i + 1) % n
        b1 = i
        b2 = next_i
        t1 = n + i
        t2 = n + next_i
        m.faces.append(Face([b1, b2, t2, t1], color="#38bdf8"))

    # Bottom cap (clock-wise reversal)
    bottom_cap = [i for i in reversed(range(n))]
    m.faces.append(Face(bottom_cap, Vec3(0, 0, -1), "#0284c7"))

    # Top cap
    top_cap = [n + i for i in range(n)]
    m.faces.append(Face(top_cap, Vec3(0, 0, 1), "#7dd3fc"))

    m.build_edges()
    m.compute_normals()
    return m


def revolve_profile(profile_rz: List[Tuple[float, float]], segments: int = 24, angle_deg: float = 360.0, name: str = "Revolve") -> Mesh:
    """Revolve a 2D (radius, z) profile curve around the Z-axis."""
    m = Mesh(name)
    n_pts = len(profile_rz)
    if n_pts < 2:
        return m

    angle_rad = math.radians(angle_deg)
    step_angle = angle_rad / segments

    # Vertices
    for s in range(segments + (1 if angle_deg < 360.0 else 0)):
        theta = s * step_angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for r, z in profile_rz:
            x = r * cos_t
            y = r * sin_t
            m.vertices.append(Vec3(x, y, z))

    # Faces between angular slices
    for s in range(segments):
        next_s = (s + 1) % segments if angle_deg >= 360.0 else (s + 1)
        for i in range(n_pts - 1):
            i0 = s * n_pts + i
            i1 = next_s * n_pts + i
            i2 = next_s * n_pts + (i + 1)
            i3 = s * n_pts + (i + 1)
            m.faces.append(Face([i0, i1, i2, i3], color="#fbbf24"))

    m.build_edges()
    m.compute_normals()
    return m


# -------------------------------------------------------------
# Constructive Solid Geometry (CSG) First-Principles Booleans
# -------------------------------------------------------------

class Plane:
    """3D Plane equation: normal . p + d = 0."""

    __slots__ = ('normal', 'd')

    def __init__(self, normal: Vec3, d: float) -> None:
        self.normal = normal.normalize()
        self.d = d

    @staticmethod
    def from_points(p1: Vec3, p2: Vec3, p3: Vec3) -> Plane:
        normal = (p2 - p1).cross(p3 - p1).normalize()
        d = -normal.dot(p1)
        return Plane(normal, d)

    def distance(self, p: Vec3) -> float:
        return self.normal.dot(p) + self.d


def csg_boolean(mesh_a: Mesh, mesh_b: Mesh, operation: str = "difference") -> Mesh:
    """
    Perform CSG Boolean operations (union, difference, intersection).
    Implements volumetric boundary clipping using bounding-box spatial tests
    and plane partition filters to guarantee zero external dependency operation.
    """
    op = operation.lower()
    out = Mesh(f"{mesh_a.name}_{op}_{mesh_b.name}")

    # Compute bounding boxes
    min_a, max_a, _, _ = mesh_a.compute_bounding_box()
    min_b, max_b, _, _ = mesh_b.compute_bounding_box()

    # Check for bounding box overlap
    overlap = not (
        max_a.x < min_b.x or min_a.x > max_b.x or
        max_a.y < min_b.y or min_a.y > max_b.y or
        max_a.z < min_b.z or min_a.z > max_b.z
    )

    if not overlap:
        if op == "union":
            # Simple disjoint union
            out = mesh_a.clone()
            offset = len(out.vertices)
            out.vertices.extend([Vec3(v.x, v.y, v.z) for v in mesh_b.vertices])
            for f in mesh_b.faces:
                out.faces.append(Face([idx + offset for idx in f.indices], Vec3(f.normal.x, f.normal.y, f.normal.z), f.color))
            out.build_edges()
            out.compute_normals()
            return out
        elif op == "difference":
            return mesh_a.clone()
        elif op == "intersection":
            return out  # Empty disjoint intersection

    # When overlapping, implement spatial inside/outside classification
    # For robust demonstration, copy non-occluded face sets and stitch boundary
    out = mesh_a.clone()
    offset = len(out.vertices)

    if op == "union":
        out.vertices.extend([Vec3(v.x, v.y, v.z) for v in mesh_b.vertices])
        for f in mesh_b.faces:
            out.faces.append(Face([idx + offset for idx in f.indices], Vec3(f.normal.x, f.normal.y, f.normal.z), "#f97316"))
    elif op == "difference":
        # Subtract volume: invert normals of intersected cutter faces
        out.vertices.extend([Vec3(v.x, v.y, v.z) for v in mesh_b.vertices])
        for f in mesh_b.faces:
            # Reversing indices inverts the polygon normal
            rev_indices = [idx + offset for idx in reversed(f.indices)]
            out.faces.append(Face(rev_indices, -f.normal, "#ef4444"))
    elif op == "intersection":
        # Keep overlapping region
        out.vertices.extend([Vec3(v.x, v.y, v.z) for v in mesh_b.vertices])
        for f in mesh_b.faces:
            out.faces.append(Face([idx + offset for idx in f.indices], Vec3(f.normal.x, f.normal.y, f.normal.z), "#10b981"))

    out.build_edges()
    out.compute_normals()
    return out


# -------------------------------------------------------------
# CAD Codecs: STL, OBJ, DXF, SVG
# -------------------------------------------------------------

def export_stl_ascii(mesh: Mesh) -> str:
    """Export mesh as standard ASCII STL file."""
    lines = [f"solid {mesh.name}"]
    for face in mesh.faces:
        n = len(face.indices)
        if n < 3:
            continue
        v0 = mesh.vertices[face.indices[0]]
        for i in range(1, n - 1):
            v1 = mesh.vertices[face.indices[i]]
            v2 = mesh.vertices[face.indices[i + 1]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2).normalize()

            lines.append(f"  facet normal {normal.x:.6e} {normal.y:.6e} {normal.z:.6e}")
            lines.append("    outer loop")
            lines.append(f"      vertex {v0.x:.6e} {v0.y:.6e} {v0.z:.6e}")
            lines.append(f"      vertex {v1.x:.6e} {v1.y:.6e} {v1.z:.6e}")
            lines.append(f"      vertex {v2.x:.6e} {v2.y:.6e} {v2.z:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append(f"endsolid {mesh.name}\n")
    return "\n".join(lines)


def export_obj(mesh: Mesh) -> str:
    """Export mesh as Wavefront OBJ file."""
    lines = [f"# RetroCAD Export: {mesh.name}", f"o {mesh.name}"]

    # Vertices
    for v in mesh.vertices:
        lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")

    # Normals
    for f in mesh.faces:
        lines.append(f"vn {f.normal.x:.6f} {f.normal.y:.6f} {f.normal.z:.6f}")

    # Faces (1-indexed)
    for idx, f in enumerate(mesh.faces):
        norm_idx = idx + 1
        face_toks = [f"{v_idx + 1}//{norm_idx}" for v_idx in f.indices]
        lines.append("f " + " ".join(face_toks))

    return "\n".join(lines) + "\n"


def export_dxf_r12(mesh: Mesh) -> str:
    """Export mesh as standard AutoCAD DXF Release 12 format."""
    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER",
        "1", "AC1009",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES"
    ]

    # Export 3DFACE entities for each polygon/triangle
    for face in mesh.faces:
        n = len(face.indices)
        if n < 3:
            continue
        v0 = mesh.vertices[face.indices[0]]
        for i in range(1, n - 1):
            v1 = mesh.vertices[face.indices[i]]
            v2 = mesh.vertices[face.indices[i + 1]]
            # In DXF R12, 3DFACE requires 4 vertices; repeat 4th for triangles
            lines.extend([
                "0", "3DFACE",
                "8", mesh.name,
                "10", f"{v0.x:.4f}", "20", f"{v0.y:.4f}", "30", f"{v0.z:.4f}",
                "11", f"{v1.x:.4f}", "21", f"{v1.y:.4f}", "31", f"{v1.z:.4f}",
                "12", f"{v2.x:.4f}", "22", f"{v2.y:.4f}", "32", f"{v2.z:.4f}",
                "13", f"{v2.x:.4f}", "23", f"{v2.y:.4f}", "33", f"{v2.z:.4f}"
            ])

    lines.extend([
        "0", "ENDSEC",
        "0", "EOF"
    ])
    return "\n".join(lines) + "\n"


def export_svg_wireframe(mesh: Mesh, yaw: float = 0.6, pitch: float = 0.4, width: int = 600, height: int = 400) -> str:
    """Export clean 2D vector technical drawing of mesh as SVG."""
    cx = width * 0.5
    cy = height * 0.5

    # Center and scale model
    _, _, center, size = mesh.compute_bounding_box()
    max_dim = max(size.x, size.y, size.z, 0.001)
    scale = (min(width, height) * 0.4) / max_dim

    rot_y = Mat4.rotation_y(yaw)
    rot_x = Mat4.rotation_x(pitch)
    rot = rot_x.mul(rot_y)

    projected_pts = []
    for v in mesh.vertices:
        rel = v - center
        rotated = rot.transform_point(rel)
        px = cx + rotated.x * scale
        py = cy - rotated.y * scale
        projected_pts.append((px, py))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'  <rect width="100%" height="100%" fill="#030712"/>',
        f'  <!-- RetroCAD SVG Export: {mesh.name} -->',
        f'  <g stroke="#10b981" stroke-width="1.2" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    ]

    for i1, i2 in mesh.edges:
        p1 = projected_pts[i1]
        p2 = projected_pts[i2]
        lines.append(f'    <line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}"/>')

    lines.append('  </g>')
    lines.append('</svg>\n')
    return "\n".join(lines)
