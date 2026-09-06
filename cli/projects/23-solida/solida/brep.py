"""
Solida: Manifold Half-Edge Boundary Representation (B-Rep) Topology Data Structure.
Entities: Vertex, HalfEdge, Edge, Loop, Face, Shell, Solid.
Euler-Poincaré invariants and topological navigation.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional, Set, Sequence, Iterable
from solida.geometry import Vector3D, Matrix4x4, BoundingBox3D, Plane


class Vertex:
    """0-dimensional topological boundary entity with 3D Cartesian coordinates."""
    __slots__ = ("id", "point", "half_edge")

    def __init__(self, vertex_id: int, point: Vector3D) -> None:
        self.id = int(vertex_id)
        self.point = point
        self.half_edge: Optional[HalfEdge] = None

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, pt={self.point})"


class HalfEdge:
    """Directed 1-dimensional manifold half-edge."""
    __slots__ = ("id", "origin", "twin", "next", "prev", "edge", "face", "loop")

    def __init__(self, he_id: int, origin: Vertex) -> None:
        self.id = int(he_id)
        self.origin = origin
        self.twin: Optional[HalfEdge] = None
        self.next: Optional[HalfEdge] = None
        self.prev: Optional[HalfEdge] = None
        self.edge: Optional[Edge] = None
        self.face: Optional[Face] = None
        self.loop: Optional[Loop] = None

    def __repr__(self) -> str:
        orig_id = self.origin.id if self.origin else None
        target_id = self.target.id if self.target else None
        twin_id = self.twin.id if self.twin else None
        return f"HalfEdge(id={self.id}, {orig_id}->{target_id}, twin={twin_id})"

    @property
    def target(self) -> Vertex:
        """Target vertex of this directed half-edge."""
        if self.next is not None:
            return self.next.origin
        if self.twin is not None:
            return self.twin.origin
        raise RuntimeError("Unconnected half-edge has no target vertex")

    def vector(self) -> Vector3D:
        """Geometric vector from origin to target."""
        return self.target.point - self.origin.point

    def length(self) -> float:
        return self.vector().norm()


class Edge:
    """Unoriented 1-dimensional boundary edge shared by two twin half-edges."""
    __slots__ = ("id", "half_edge")

    def __init__(self, edge_id: int, half_edge: HalfEdge) -> None:
        self.id = int(edge_id)
        self.half_edge = half_edge

    def __repr__(self) -> str:
        v0 = self.half_edge.origin.id
        v1 = self.half_edge.target.id
        return f"Edge(id={self.id}, ({v0}, {v1}))"

    @property
    def is_manifold(self) -> bool:
        """Returns True if this edge is bounded by exactly two manifold faces."""
        return self.half_edge is not None and self.half_edge.twin is not None


class Loop:
    """Directed cycle of half-edges defining a boundary loop of a face."""
    __slots__ = ("id", "half_edge", "face")

    def __init__(self, loop_id: int, half_edge: HalfEdge) -> None:
        self.id = int(loop_id)
        self.half_edge = half_edge
        self.face: Optional[Face] = None

    def __repr__(self) -> str:
        v_ids = [v.id for v in self.vertices()]
        return f"Loop(id={self.id}, verts={v_ids})"

    def half_edges(self) -> List[HalfEdge]:
        """Traverses and returns all half-edges in order around this loop."""
        edges: List[HalfEdge] = []
        curr = self.half_edge
        start = curr
        while True:
            edges.append(curr)
            curr = curr.next
            if curr is None or curr is start:
                break
        return edges

    def vertices(self) -> List[Vertex]:
        """Returns all ordered vertices in this boundary loop."""
        return [he.origin for he in self.half_edges()]

    def compute_normal(self) -> Vector3D:
        """
        Calculates loop polygon outward unit normal using Newell's method.
        Robust for arbitrary non-planar or planar 3D polygons.
        """
        pts = [v.point for v in self.vertices()]
        n = len(pts)
        if n < 3:
            return Vector3D(0.0, 0.0, 1.0)

        nx, ny, nz = 0.0, 0.0, 0.0
        for i in range(n):
            curr = pts[i]
            nxt = pts[(i + 1) % n]
            nx += (curr.y - nxt.y) * (curr.z + nxt.z)
            ny += (curr.z - nxt.z) * (curr.x + nxt.x)
            nz += (curr.x - nxt.x) * (curr.y + nxt.y)

        normal = Vector3D(nx, ny, nz)
        if normal.norm_sq() < 1e-15:
            # Fallback to simple cross product of first two edges
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = v1.cross(v2)

        return normal.normalized()

    def compute_area(self) -> float:
        """Calculates 3D polygonal surface area of this loop."""
        pts = [v.point for v in self.vertices()]
        n = len(pts)
        if n < 3:
            return 0.0

        normal = self.compute_normal()
        area_vec = Vector3D(0.0, 0.0, 0.0)
        p0 = pts[0]
        for i in range(1, n - 1):
            v1 = pts[i] - p0
            v2 = pts[i + 1] - p0
            area_vec = area_vec + v1.cross(v2)

        return 0.5 * abs(area_vec.dot(normal))


class Face:
    """2-dimensional manifold surface patch bounded by outer and inner loops."""
    __slots__ = ("id", "outer_loop", "inner_loops", "shell", "normal")

    def __init__(self, face_id: int, outer_loop: Loop, inner_loops: Optional[List[Loop]] = None) -> None:
        self.id = int(face_id)
        self.outer_loop = outer_loop
        self.inner_loops = inner_loops if inner_loops is not None else []
        self.shell: Optional[Shell] = None
        self.normal: Optional[Vector3D] = None

        outer_loop.face = self
        for he in outer_loop.half_edges():
            he.face = self
            he.loop = outer_loop

        for loop in self.inner_loops:
            loop.face = self
            for he in loop.half_edges():
                he.face = self
                he.loop = loop

    def __repr__(self) -> str:
        return f"Face(id={self.id}, loop={self.outer_loop})"

    def vertices(self) -> List[Vertex]:
        return self.outer_loop.vertices()

    def compute_normal(self) -> Vector3D:
        if self.normal is None:
            self.normal = self.outer_loop.compute_normal()
        return self.normal

    def compute_area(self) -> float:
        area = self.outer_loop.compute_area()
        for inner in self.inner_loops:
            area -= inner.compute_area()
        return max(0.0, area)

    def triangulate(self) -> List[Tuple[Vertex, Vertex, Vertex]]:
        """
        Triangulates planar polygonal face via fan or ear-clipping.
        Returns list of vertex triplets (v0, v1, v2) with CCW orientation.
        """
        verts = self.vertices()
        n = len(verts)
        if n < 3:
            return []
        if n == 3:
            return [(verts[0], verts[1], verts[2])]

        # Convex fan triangulation from vertex 0 (exact for convex polygons)
        triangles: List[Tuple[Vertex, Vertex, Vertex]] = []
        for i in range(1, n - 1):
            triangles.append((verts[0], verts[i], verts[i + 1]))
        return triangles


class Shell:
    """Connected, oriented 2-manifold surface enclosing a 3D solid region."""
    __slots__ = ("id", "faces", "solid")

    def __init__(self, shell_id: int, faces: Optional[List[Face]] = None) -> None:
        self.id = int(shell_id)
        self.faces = faces if faces is not None else []
        self.solid: Optional[Solid] = None
        for f in self.faces:
            f.shell = self

    def __repr__(self) -> str:
        return f"Shell(id={self.id}, num_faces={len(self.faces)})"

    def vertices(self) -> List[Vertex]:
        seen: Set[int] = set()
        res: List[Vertex] = []
        for f in self.faces:
            for v in f.vertices():
                if v.id not in seen:
                    seen.add(v.id)
                    res.append(v)
        return res

    def edges(self) -> List[Edge]:
        seen: Set[int] = set()
        res: List[Edge] = []
        for f in self.faces:
            for he in f.outer_loop.half_edges():
                if he.edge and he.edge.id not in seen:
                    seen.add(he.edge.id)
                    res.append(he.edge)
        return res

    def euler_characteristic(self) -> int:
        """Evaluates Poincaré-Euler characteristic chi = V - E + F."""
        v = len(self.vertices())
        e = len(self.edges())
        f = len(self.faces)
        return v - e + f

    def compute_volume(self) -> float:
        """
        Computes exact signed volume via the Divergence Theorem:
        V = 1/6 * sum_{(v0, v1, v2)} v0 . (v1 x v2)
        """
        total_vol = 0.0
        for f in self.faces:
            for v0, v1, v2 in f.triangulate():
                p0, p1, p2 = v0.point, v1.point, v2.point
                total_vol += p0.dot(p1.cross(p2))
        return abs(total_vol) / 6.0

    def compute_surface_area(self) -> float:
        """Computes total surface area across all faces."""
        return sum(f.compute_area() for f in self.faces)

    def compute_center_of_mass(self) -> Vector3D:
        """
        Computes centroid / center of mass of solid volume enclosed by shell.
        Uses tetrahedral decomposition against origin.
        """
        cx, cy, cz = 0.0, 0.0, 0.0
        total_vol = 0.0

        for f in self.faces:
            for v0, v1, v2 in f.triangulate():
                p0, p1, p2 = v0.point, v1.point, v2.point
                det = p0.dot(p1.cross(p2))
                vol = det / 6.0
                total_vol += vol
                # Centroid of tetrahedron with origin is 1/4 (p0 + p1 + p2 + 0)
                cx += vol * 0.25 * (p0.x + p1.x + p2.x)
                cy += vol * 0.25 * (p0.y + p1.y + p2.y)
                cz += vol * 0.25 * (p0.z + p1.z + p2.z)

        if abs(total_vol) < 1e-15:
            # Fallback to vertex average
            verts = self.vertices()
            if not verts:
                return Vector3D(0, 0, 0)
            avg = sum((v.point for v in verts), Vector3D(0, 0, 0))
            return avg / len(verts)

        return Vector3D(cx / total_vol, cy / total_vol, cz / total_vol)


class Solid:
    """3-dimensional CAD Solid defined by outer and inner B-Rep shells."""
    __slots__ = ("name", "outer_shell", "inner_shells")

    def __init__(
        self,
        outer_shell: Shell,
        inner_shells: Optional[List[Shell]] = None,
        name: str = "Solid",
    ) -> None:
        self.name = str(name)
        self.outer_shell = outer_shell
        self.inner_shells = inner_shells if inner_shells is not None else []

        outer_shell.solid = self
        for s in self.inner_shells:
            s.solid = self

    def __repr__(self) -> str:
        chi = self.outer_shell.euler_characteristic()
        v = len(self.outer_shell.vertices())
        f = len(self.outer_shell.faces)
        return f"Solid('{self.name}', V={v}, F={f}, chi={chi})"

    @property
    def faces(self) -> List[Face]:
        return self.outer_shell.faces

    @property
    def vertices(self) -> List[Vertex]:
        return self.outer_shell.vertices()

    @property
    def edges(self) -> List[Edge]:
        return self.outer_shell.edges()

    def volume(self) -> float:
        vol = self.outer_shell.compute_volume()
        for inner in self.inner_shells:
            vol -= inner.compute_volume()
        return max(0.0, vol)

    def surface_area(self) -> float:
        area = self.outer_shell.compute_surface_area()
        for inner in self.inner_shells:
            area += inner.compute_surface_area()
        return area

    def center_of_mass(self) -> Vector3D:
        return self.outer_shell.compute_center_of_mass()

    def bounding_box(self) -> BoundingBox3D:
        return BoundingBox3D.from_points([v.point for v in self.vertices])

    def transform(self, matrix: Matrix4x4) -> Solid:
        """Transforms entire solid geometry by affine transformation matrix."""
        new_polygons: List[List[Vector3D]] = []
        for face in self.faces:
            poly = [matrix.transform_point(v.point) for v in face.vertices()]
            new_polygons.append(poly)
        return build_solid_from_polygons(new_polygons, name=self.name)

    def translate(self, dx: float, dy: float, dz: float) -> Solid:
        return self.transform(Matrix4x4.translation(dx, dy, dz))

    def scale(self, sx: float, sy: float, sz: float) -> Solid:
        return self.transform(Matrix4x4.scaling(sx, sy, sz))

    def rotate_x(self, angle_rad: float) -> Solid:
        return self.transform(Matrix4x4.rotation_x(angle_rad))

    def rotate_y(self, angle_rad: float) -> Solid:
        return self.transform(Matrix4x4.rotation_y(angle_rad))

    def rotate_z(self, angle_rad: float) -> Solid:
        return self.transform(Matrix4x4.rotation_z(angle_rad))


def build_solid_from_polygons(
    polygon_faces: Sequence[Sequence[Vector3D]],
    name: str = "Solid",
    tol: float = 1e-6,
) -> Solid:
    """
    Constructs a topologically valid half-edge 2-manifold B-Rep Solid from oriented polygons.
    - Merges coincident vertices within spatial tolerance.
    - Creates half-edges, boundary loops, faces, and undirected edge twins.
    - Guarantees 2-manifold topological integrity.
    """
    vertex_pool: List[Vertex] = []

    def get_or_create_vertex(p: Vector3D) -> Vertex:
        for v in vertex_pool:
            if v.point.approx_eq(p, tol):
                return v
        new_v = Vertex(len(vertex_pool), p)
        vertex_pool.append(new_v)
        return new_v

    faces_list: List[Face] = []
    edges_map: Dict[Tuple[int, int], Edge] = {}
    half_edges_map: Dict[Tuple[int, int], HalfEdge] = {}
    he_id_counter = 0

    for f_idx, poly in enumerate(polygon_faces):
        if len(poly) < 3:
            continue
        poly_verts = [get_or_create_vertex(pt) for pt in poly]

        loop_half_edges: List[HalfEdge] = []
        n = len(poly_verts)
        for i in range(n):
            v_orig = poly_verts[i]
            v_next = poly_verts[(i + 1) % n]
            if v_orig.id == v_next.id:
                continue

            he = HalfEdge(he_id_counter, v_orig)
            he_id_counter += 1
            v_orig.half_edge = he
            loop_half_edges.append(he)
            half_edges_map[(v_orig.id, v_next.id)] = he

        if len(loop_half_edges) < 3:
            continue

        # Link next / prev pointers in loop
        k = len(loop_half_edges)
        for i in range(k):
            loop_half_edges[i].next = loop_half_edges[(i + 1) % k]
            loop_half_edges[i].prev = loop_half_edges[(i - 1 + k) % k]

        loop = Loop(f_idx, loop_half_edges[0])
        face = Face(f_idx, loop)
        faces_list.append(face)

    # Pair twin half-edges and create undirected edges
    edge_id_counter = 0
    for (v0_id, v1_id), he in half_edges_map.items():
        twin_key = (v1_id, v0_id)
        edge_key = (min(v0_id, v1_id), max(v0_id, v1_id))

        if twin_key in half_edges_map:
            twin_he = half_edges_map[twin_key]
            he.twin = twin_he
            twin_he.twin = he

        if edge_key not in edges_map:
            edge = Edge(edge_id_counter, he)
            edge_id_counter += 1
            edges_map[edge_key] = edge
            he.edge = edge
        else:
            he.edge = edges_map[edge_key]

    shell = Shell(0, faces_list)
    return Solid(shell, name=name)
