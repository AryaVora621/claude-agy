"""
GeoPrism: Computational Geometry Engine.
Implements Convex Hull (Andrew's Monotone Chain), Bowyer-Watson Delaunay Triangulation,
Voronoi Diagram dual graph construction, and polygon spatial predicates.
"""

import math
from typing import List, Tuple, Dict, Set, Optional, Union

Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]


def orient2d(p: Point2D, q: Point2D, r: Point2D) -> float:
    """
    2D orientation test (cross product of vectors pq and pr).
    Returns:
      > 0: counter-clockwise turn (left)
      < 0: clockwise turn (right)
      = 0: collinear points
    """
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def euclidean_dist(p: Point2D, q: Point2D) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def convex_hull(points: List[Point2D]) -> List[Point2D]:
    """
    Computes the Convex Hull of a 2D point set using Andrew's Monotone Chain algorithm.
    Time Complexity: O(N log N).
    Returns vertices in counter-clockwise order.
    """
    unique_pts = sorted(list(set(points)))
    if len(unique_pts) <= 2:
        return unique_pts

    # Lower hull
    lower: List[Point2D] = []
    for p in unique_pts:
        while len(lower) >= 2 and orient2d(lower[-2], lower[-1], p) <= 1e-11:
            lower.pop()
        lower.append(p)

    # Upper hull
    upper: List[Point2D] = []
    for p in reversed(unique_pts):
        while len(upper) >= 2 and orient2d(upper[-2], upper[-1], p) <= 1e-11:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hull, omitting last point of each list (duplicates)
    return lower[:-1] + upper[:-1]


def polygon_area(polygon: List[Point2D]) -> float:
    """
    Computes the area of a 2D simple polygon using the Shoelace formula.
    Positive for CCW winding, negative for CW.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return 0.5 * area


def point_in_polygon(point: Point2D, polygon: List[Point2D]) -> bool:
    """
    Determines whether point is inside simple polygon using Ray-Casting algorithm.
    Points on the boundary are treated as inside.
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if point lies on segment
        d1 = euclidean_dist((x, y), (x1, y1))
        d2 = euclidean_dist((x, y), (x2, y2))
        line_len = euclidean_dist((x1, y1), (x2, y2))
        if abs(d1 + d2 - line_len) < 1e-9:
            return True

        # Ray casting horizontal to the right
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= x_intersect:
                inside = not inside

    return inside


class Triangle:
    """Represents a 2D Delaunay triangle with its vertices and circumcircle."""
    __slots__ = ("a", "b", "c", "circumcenter", "circumradius_sq", "circumradius")

    def __init__(self, a: Point2D, b: Point2D, c: Point2D) -> None:
        # Enforce counter-clockwise vertex ordering
        if orient2d(a, b, c) < 0:
            b, c = c, b
        self.a = a
        self.b = b
        self.c = c
        self.circumcenter, self.circumradius_sq = self._compute_circumcircle(a, b, c)
        self.circumradius = math.sqrt(self.circumradius_sq) if self.circumradius_sq < float("inf") else float("inf")

    @staticmethod
    def _compute_circumcircle(a: Point2D, b: Point2D, c: Point2D) -> Tuple[Point2D, float]:
        ax, ay = a
        bx, by = b
        cx, cy = c

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return (0.0, 0.0), float("inf")

        a_sq = ax * ax + ay * ay
        b_sq = bx * bx + by * by
        c_sq = cx * cx + cy * cy

        ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
        uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
        radius_sq = (ax - ux) ** 2 + (ay - uy) ** 2
        return (ux, uy), radius_sq

    def in_circumcircle(self, p: Point2D, eps: float = 1e-9) -> bool:
        """Returns True if point p lies strictly inside or on this triangle's circumcircle."""
        ux, uy = self.circumcenter
        dist_sq = (p[0] - ux) ** 2 + (p[1] - uy) ** 2
        return dist_sq <= self.circumradius_sq + eps

    def shares_vertex(self, other_vertices: Set[Point2D]) -> bool:
        """Checks if any triangle vertex is in other_vertices."""
        return (self.a in other_vertices or self.b in other_vertices or self.c in other_vertices)

    def edges(self) -> List[Tuple[Point2D, Point2D]]:
        """Returns the 3 canonical undirected edges of this triangle."""
        edges = []
        for u, v in ((self.a, self.b), (self.b, self.c), (self.c, self.a)):
            if u > v:
                u, v = v, u
            edges.append((u, v))
        return edges

    def __repr__(self) -> str:
        return f"Triangle({self.a}, {self.b}, {self.c})"


def delaunay_triangulation(points: List[Point2D]) -> List[Triangle]:
    """
    Computes Delaunay Triangulation of 2D points using the incremental Bowyer-Watson algorithm.
    Time Complexity: O(N log N) on average, O(N^2) worst case.
    """
    unique_pts = list(dict.fromkeys(points))
    if len(unique_pts) < 3:
        return []

    # Check for collinearity
    all_collinear = True
    p0, p1 = unique_pts[0], unique_pts[1]
    for p in unique_pts[2:]:
        if abs(orient2d(p0, p1, p)) > 1e-9:
            all_collinear = False
            break
    if all_collinear:
        return []

    # Compute bounding box
    min_x = min(p[0] for p in unique_pts)
    max_x = max(p[0] for p in unique_pts)
    min_y = min(p[1] for p in unique_pts)
    max_y = max(p[1] for p in unique_pts)

    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy, 1.0)
    mid_x = (min_x + max_x) * 0.5
    mid_y = (min_y + max_y) * 0.5

    # Create super-triangle enclosing all points
    p_sup1 = (mid_x - 20.0 * delta_max, mid_y - delta_max)
    p_sup2 = (mid_x, mid_y + 20.0 * delta_max)
    p_sup3 = (mid_x + 20.0 * delta_max, mid_y - delta_max)
    super_vertices = {p_sup1, p_sup2, p_sup3}

    triangles: List[Triangle] = [Triangle(p_sup1, p_sup2, p_sup3)]

    for pt in unique_pts:
        bad_triangles: List[Triangle] = []
        for t in triangles:
            if t.in_circumcircle(pt):
                bad_triangles.append(t)

        # Find boundary of cavity (edges belonging to exactly one bad triangle)
        edge_counts: Dict[Tuple[Point2D, Point2D], int] = {}
        for t in bad_triangles:
            for e in t.edges():
                edge_counts[e] = edge_counts.get(e, 0) + 1

        polygon_edges = [edge for edge, count in edge_counts.items() if count == 1]

        # Remove bad triangles
        triangles = [t for t in triangles if t not in bad_triangles]

        # Form new triangles connecting point to boundary edges
        for u, v in polygon_edges:
            triangles.append(Triangle(u, v, pt))

    # Remove triangles that share vertices with super-triangle
    valid_triangles = [t for t in triangles if not t.shares_vertex(super_vertices)]
    return valid_triangles


class VoronoiDiagram:
    """
    Represents the Voronoi Diagram constructed as the geometric dual of a Delaunay Triangulation.
    """
    def __init__(self, seeds: List[Point2D], triangles: List[Triangle]) -> None:
        self.seeds = seeds
        self.triangles = triangles
        self.vertices: List[Point2D] = []
        self.cells: Dict[Point2D, List[Point2D]] = {}
        self.edges: List[Edge2D] = []
        self._build_diagram()

    def _build_diagram(self) -> None:
        # Collect circumcenters as Voronoi vertices
        vertex_set: Set[Point2D] = set()
        seed_to_triangles: Dict[Point2D, List[Triangle]] = {s: [] for s in self.seeds}

        for t in self.triangles:
            cc = t.circumcenter
            vertex_set.add(cc)
            for v in (t.a, t.b, t.c):
                if v in seed_to_triangles:
                    seed_to_triangles[v].append(t)

        self.vertices = list(vertex_set)

        # For each seed point, sort the incident triangle circumcenters angularly
        for seed, incident in seed_to_triangles.items():
            if len(incident) < 3:
                # Unbounded or boundary cell
                self.cells[seed] = [t.circumcenter for t in incident]
                continue

            sx, sy = seed
            # Sort circumcenters by angle relative to seed point
            sorted_circumcenters = sorted(
                [t.circumcenter for t in incident],
                key=lambda cc: math.atan2(cc[1] - sy, cc[0] - sx)
            )
            self.cells[seed] = sorted_circumcenters

        # Build Voronoi edges (connecting circumcenters of triangles sharing a Delaunay edge)
        edge_to_triangles: Dict[Tuple[Point2D, Point2D], List[Triangle]] = {}
        for t in self.triangles:
            for e in t.edges():
                edge_to_triangles.setdefault(e, []).append(t)

        for e, tri_list in edge_to_triangles.items():
            if len(tri_list) == 2:
                v1 = tri_list[0].circumcenter
                v2 = tri_list[1].circumcenter
                if v1 != v2:
                    self.edges.append((v1, v2))


def voronoi_diagram(points: List[Point2D]) -> VoronoiDiagram:
    """
    Constructs the Voronoi Diagram for a set of 2D seed points.
    """
    triangles = delaunay_triangulation(points)
    return VoronoiDiagram(points, triangles)
