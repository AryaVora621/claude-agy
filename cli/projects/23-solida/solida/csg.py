"""
Solida: Constructive Solid Geometry (CSG) 3D Boolean Engine.
Implements Binary Space Partitioning (BSP) Tree polygon clipping for exact
Union, Difference, and Intersection operations on 3D B-Rep solids.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Sequence
from solida.geometry import Vector3D, Plane
from solida.brep import Solid, build_solid_from_polygons


class CSGPolygon:
    """Convex 3D polygon with oriented plane for BSP-tree partitioning."""
    __slots__ = ("vertices", "plane")

    def __init__(self, vertices: Sequence[Vector3D], plane: Optional[Plane] = None) -> None:
        self.vertices = list(vertices)
        if plane is None:
            self.plane = self._compute_plane()
        else:
            self.plane = plane

    def _compute_plane(self) -> Plane:
        pts = self.vertices
        n = len(pts)
        if n < 3:
            return Plane(Vector3D(0, 0, 1), 0.0)

        # Newell's method for robust normal evaluation
        nx, ny, nz = 0.0, 0.0, 0.0
        for i in range(n):
            curr = pts[i]
            nxt = pts[(i + 1) % n]
            nx += (curr.y - nxt.y) * (curr.z + nxt.z)
            ny += (curr.z - nxt.z) * (curr.x + nxt.x)
            nz += (curr.x - nxt.x) * (curr.y + nxt.y)

        normal = Vector3D(nx, ny, nz)
        if normal.norm_sq() < 1e-15:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = v1.cross(v2)

        norm_vec = normal.normalized()
        d = -norm_vec.dot(pts[0])
        return Plane(norm_vec, d)

    def flip(self) -> None:
        """Inverts polygon winding and surface normal."""
        self.vertices.reverse()
        self.plane = Plane(-self.plane.normal, -self.plane.d)

    def clone(self) -> CSGPolygon:
        return CSGPolygon(list(self.vertices), self.plane)

    def split_by_plane(
        self,
        plane: Plane,
        coplanar_front: List[CSGPolygon],
        coplanar_back: List[CSGPolygon],
        front: List[CSGPolygon],
        back: List[CSGPolygon],
        epsilon: float = 1e-5,
    ) -> None:
        """
        Splits this polygon by a splitting plane into front, back, coplanar_front,
        or coplanar_back lists. Spanning polygons are cut into sub-polygons.
        """
        # Classify each vertex relative to the plane
        types: List[int] = []
        # -1 = BACK, 0 = COPLANAR, 1 = FRONT
        polygon_type = 0
        for v in self.vertices:
            t = plane.normal.dot(v) + plane.d
            v_type = 0
            if t < -epsilon:
                v_type = -1
            elif t > epsilon:
                v_type = 1
            polygon_type |= v_type if v_type != 0 else 0
            types.append(v_type)

        has_front = 1 in types
        has_back = -1 in types

        # 1. Fully coplanar
        if not has_front and not has_back:
            # Check if polygon normal aligns with plane normal
            alignment = self.plane.normal.dot(plane.normal)
            if alignment > 0:
                coplanar_front.append(self)
            else:
                coplanar_back.append(self)
            return

        # 2. Fully front
        if has_front and not has_back:
            front.append(self)
            return

        # 3. Fully back
        if has_back and not has_front:
            back.append(self)
            return

        # 4. Spanning: polygon crosses the plane -> cut into front and back parts
        f_pts: List[Vector3D] = []
        b_pts: List[Vector3D] = []
        num_v = len(self.vertices)

        for i in range(num_v):
            j = (i + 1) % num_v
            ti = types[i]
            tj = types[j]
            vi = self.vertices[i]
            vj = self.vertices[j]

            if ti != -1:
                f_pts.append(vi)
            if ti != 1:
                b_pts.append(vi)

            # Edge intersects plane
            if (ti == 1 and tj == -1) or (ti == -1 and tj == 1):
                denom = plane.normal.dot(vj - vi)
                if abs(denom) > 1e-15:
                    t_val = (-plane.d - plane.normal.dot(vi)) / denom
                    t_clamped = max(0.0, min(1.0, t_val))
                    v_cut = vi.lerp(vj, t_clamped)
                    f_pts.append(v_cut)
                    b_pts.append(v_cut)

        if len(f_pts) >= 3:
            front.append(CSGPolygon(f_pts, self.plane))
        if len(b_pts) >= 3:
            back.append(CSGPolygon(b_pts, self.plane))


class CSGNode:
    """
    Binary Space Partitioning (BSP) Tree node for CSG 3D Boolean modeling.
    Each node divides solid space into front and back half-spaces.
    """
    __slots__ = ("plane", "front", "back", "polygons")

    def __init__(self, polygons: Optional[Sequence[CSGPolygon]] = None) -> None:
        self.plane: Optional[Plane] = None
        self.front: Optional[CSGNode] = None
        self.back: Optional[CSGNode] = None
        self.polygons: List[CSGPolygon] = []

        if polygons:
            self.build(polygons)

    def clone(self) -> CSGNode:
        node = CSGNode()
        node.plane = self.plane
        node.front = self.front.clone() if self.front is not None else None
        node.back = self.back.clone() if self.back is not None else None
        node.polygons = [p.clone() for p in self.polygons]
        return node

    def invert(self) -> None:
        """Inverts solid space (interior becomes exterior, exterior becomes interior)."""
        for p in self.polygons:
            p.flip()
        if self.plane is not None:
            self.plane = Plane(-self.plane.normal, -self.plane.d)
        if self.front is not None:
            self.front.invert()
        if self.back is not None:
            self.back.invert()
        # Swap front and back child subtrees
        self.front, self.back = self.back, self.front

    def clip_polygons(self, polygons: List[CSGPolygon]) -> List[CSGPolygon]:
        """Recursively clips list of polygons against this BSP tree."""
        if self.plane is None:
            return list(polygons)

        front: List[CSGPolygon] = []
        back: List[CSGPolygon] = []

        for p in polygons:
            p.split_by_plane(self.plane, front, back, front, back)

        if self.front is not None:
            front = self.front.clip_polygons(front)

        if self.back is not None:
            back = self.back.clip_polygons(back)
        else:
            # Inside solid: discarded
            back = []

        return front + back

    def clip_to(self, other: CSGNode) -> None:
        """Clips all polygons of this tree against another BSP tree."""
        self.polygons = other.clip_polygons(self.polygons)
        if self.front is not None:
            self.front.clip_to(other)
        if self.back is not None:
            self.back.clip_to(other)

    def all_polygons(self) -> List[CSGPolygon]:
        """Recursively gathers all polygons from this BSP tree."""
        res = list(self.polygons)
        if self.front is not None:
            res.extend(self.front.all_polygons())
        if self.back is not None:
            res.extend(self.back.all_polygons())
        return res

    def build(self, polygons: Sequence[CSGPolygon]) -> None:
        """Constructs BSP tree from a list of polygons."""
        if not polygons:
            return

        if self.plane is None:
            self.plane = polygons[0].plane

        front: List[CSGPolygon] = []
        back: List[CSGPolygon] = []

        for p in polygons:
            p.split_by_plane(
                self.plane,
                self.polygons,
                self.polygons,
                front,
                back,
            )

        if front:
            if self.front is None:
                self.front = CSGNode()
            self.front.build(front)

        if back:
            if self.back is None:
                self.back = CSGNode()
            self.back.build(back)


def solid_to_csg(solid: Solid) -> CSGNode:
    """Converts a B-Rep Solid into a BSP-tree CSG representation."""
    polys: List[CSGPolygon] = []
    for face in solid.faces:
        # Triangulate face so each polygon is guaranteed planar & convex
        for v0, v1, v2 in face.triangulate():
            polys.append(CSGPolygon([v0.point, v1.point, v2.point]))
    return CSGNode(polys)


def csg_to_solid(polygons: Sequence[CSGPolygon], name: str = "CSG_Solid") -> Solid:
    """Converts BSP-tree CSG polygons back into a manifold B-Rep Solid."""
    poly_pts = [p.vertices for p in polygons if len(p.vertices) >= 3]
    return build_solid_from_polygons(poly_pts, name=name)


def csg_union(solid_a: Solid, solid_b: Solid, name: str = "Union") -> Solid:
    """
    Constructive Solid Geometry Union (A union B).
    Preserves outer boundary, removes all interior shared volume.
    """
    a = solid_to_csg(solid_a)
    b = solid_to_csg(solid_b)

    a.clip_to(b)
    b.clip_to(a)
    b.invert()
    b.clip_to(a)
    b.invert()
    a.build(b.all_polygons())

    return csg_to_solid(a.all_polygons(), name=name)


def csg_difference(solid_a: Solid, solid_b: Solid, name: str = "Difference") -> Solid:
    """
    Constructive Solid Geometry Difference (A minus B).
    Carves cavity of B out of A.
    """
    a = solid_to_csg(solid_a)
    b = solid_to_csg(solid_b)

    a.invert()
    a.clip_to(b)
    b.clip_to(a)
    b.invert()
    b.clip_to(a)
    b.invert()
    a.build(b.all_polygons())
    a.invert()

    return csg_to_solid(a.all_polygons(), name=name)


def csg_intersection(solid_a: Solid, solid_b: Solid, name: str = "Intersection") -> Solid:
    """
    Constructive Solid Geometry Intersection (A intersect B).
    Keeps only the volume shared by both A and B.
    """
    a = solid_to_csg(solid_a)
    b = solid_to_csg(solid_b)

    a.invert()
    b.clip_to(a)
    b.invert()
    a.clip_to(b)
    b.clip_to(a)
    a.build(b.all_polygons())
    a.invert()

    return csg_to_solid(a.all_polygons(), name=name)
