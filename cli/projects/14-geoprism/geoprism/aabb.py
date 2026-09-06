"""
GeoPrism: N-Dimensional Axis-Aligned Bounding Box (AABB).
Implements spatial volume, margin, intersection, union, and MinDist metrics.
"""

import math
from typing import Tuple, List, Optional, Union

Point = Tuple[float, ...]


class AABB:
    """
    N-Dimensional Axis-Aligned Bounding Box defined by lower and upper coordinate tuples.
    """
    __slots__ = ("lower", "upper", "dimensions")

    def __init__(self, lower: Union[List[float], Tuple[float, ...]], upper: Union[List[float], Tuple[float, ...]]) -> None:
        if len(lower) != len(upper):
            raise ValueError("Lower and upper coordinate bounds must have matching dimensions")
        self.dimensions = len(lower)
        # Ensure lower <= upper for all axes
        low = []
        up = []
        for l, u in zip(lower, upper):
            if l <= u:
                low.append(float(l))
                up.append(float(u))
            else:
                low.append(float(u))
                up.append(float(l))
        self.lower: Tuple[float, ...] = tuple(low)
        self.upper: Tuple[float, ...] = tuple(up)

    @classmethod
    def from_point(cls, point: Union[List[float], Tuple[float, ...]], radius: float = 0.0) -> "AABB":
        """Constructs an AABB around a point with optional expansion radius."""
        lower = tuple(float(x) - radius for x in point)
        upper = tuple(float(x) + radius for x in point)
        return cls(lower, upper)

    @classmethod
    def from_points(cls, points: List[Union[List[float], Tuple[float, ...]]]) -> "AABB":
        """Constructs the minimum bounding box enclosing a collection of points."""
        if not points:
            raise ValueError("Cannot create AABB from empty points list")
        dims = len(points[0])
        low = [float("inf")] * dims
        up = [float("-inf")] * dims
        for p in points:
            for d in range(dims):
                if p[d] < low[d]:
                    low[d] = p[d]
                if p[d] > up[d]:
                    up[d] = p[d]
        return cls(tuple(low), tuple(up))

    def volume(self) -> float:
        """Computes the hyper-volume (area in 2D) of the bounding box."""
        vol = 1.0
        for l, u in zip(self.lower, self.upper):
            vol *= max(0.0, u - l)
        return vol

    def margin(self) -> float:
        """Computes the sum of edge lengths / perimeter (proportional to surface margin)."""
        return sum(max(0.0, u - l) for l, u in zip(self.lower, self.upper))

    def center(self) -> Tuple[float, ...]:
        """Returns the geometric midpoint coordinates of the box."""
        return tuple(0.5 * (l + u) for l, u in zip(self.lower, self.upper))

    def contains_point(self, point: Union[List[float], Tuple[float, ...]]) -> bool:
        """Returns True if point lies inside or on the boundary of the box."""
        for x, l, u in zip(point, self.lower, self.upper):
            if x < l or x > u:
                return False
        return True

    def contains_box(self, other: "AABB") -> bool:
        """Returns True if this box completely encloses other."""
        for l1, u1, l2, u2 in zip(self.lower, self.upper, other.lower, other.upper):
            if l1 > l2 or u1 < u2:
                return False
        return True

    def intersects(self, other: "AABB") -> bool:
        """Returns True if this box overlaps or touches other box."""
        for l1, u1, l2, u2 in zip(self.lower, self.upper, other.lower, other.upper):
            if l1 > u2 or u1 < l2:
                return False
        return True

    def intersection(self, other: "AABB") -> Optional["AABB"]:
        """Returns the intersection AABB or None if disjoint."""
        if not self.intersects(other):
            return None
        low = tuple(max(l1, l2) for l1, l2 in zip(self.lower, other.lower))
        up = tuple(min(u1, u2) for u1, u2 in zip(self.upper, other.upper))
        return AABB(low, up)

    def union(self, other: "AABB") -> "AABB":
        """Returns the minimum bounding box enclosing both self and other."""
        low = tuple(min(l1, l2) for l1, l2 in zip(self.lower, other.lower))
        up = tuple(max(u1, u2) for u1, u2 in zip(self.upper, other.upper))
        return AABB(low, up)

    def overlap_volume(self, other: "AABB") -> float:
        """Computes the volume of intersection between self and other."""
        inter = self.intersection(other)
        return inter.volume() if inter is not None else 0.0

    def min_dist(self, point: Union[List[float], Tuple[float, ...]]) -> float:
        """
        Computes the exact minimum Euclidean distance from point to the closest surface of this AABB.
        Used for branch-and-bound k-NN pruning.
        """
        dist_sq = 0.0
        for x, l, u in zip(point, self.lower, self.upper):
            if x < l:
                d = l - x
                dist_sq += d * d
            elif x > u:
                d = x - u
                dist_sq += d * d
        return math.sqrt(dist_sq)

    def min_max_dist(self, point: Union[List[float], Tuple[float, ...]]) -> float:
        """
        Computes the MinMaxDist metric: the minimum of the maximum distances to the faces.
        Guarantees that at least one point in the box lies within this distance.
        """
        # In N-D: min_k ( (dist to furthest face along k)^2 + sum_{i != k} (dist to closest face along i)^2 )
        d = self.dimensions
        r = [0.0] * d
        p = [0.0] * d

        for i in range(d):
            x = point[i]
            l = self.lower[i]
            u = self.upper[i]
            if x <= 0.5 * (l + u):
                r[i] = (l - x) if x < l else 0.0
                p[i] = u - x
            else:
                r[i] = (x - u) if x > u else 0.0
                p[i] = x - l

        min_val = float("inf")
        sum_r_sq = sum(val * val for val in r)

        for k in range(d):
            val = p[k] * p[k] + (sum_r_sq - r[k] * r[k])
            if val < min_val:
                min_val = val

        return math.sqrt(min_val)

    def __repr__(self) -> str:
        low_s = ", ".join(f"{x:.2f}" for x in self.lower)
        up_s = ", ".join(f"{x:.2f}" for x in self.upper)
        return f"AABB([{low_s}] -> [{up_s}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AABB):
            return False
        return self.lower == other.lower and self.upper == other.upper
