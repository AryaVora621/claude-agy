"""
PhotonPBR: Geometric Primitives, Hit Records, and Ray Intersection Engines.
Implements Axis-Aligned Bounding Boxes (AABB) using the Kay-Kajiya slab method,
exact quadric Ray-Sphere intersection, and the Möller-Trumbore Ray-Triangle algorithm.
"""

import math
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from photon.vec3 import Vec3, Ray


class HitRecord:
    """Carries geometric and material properties at a ray-primitive intersection point."""
    __slots__ = ("point", "normal", "material", "t", "u", "v", "front_face")

    def __init__(
        self,
        point: Vec3 = Vec3(),
        normal: Vec3 = Vec3(),
        material: Optional[object] = None,
        t: float = 0.0,
        u: float = 0.0,
        v: float = 0.0,
        front_face: bool = True
    ):
        self.point = point
        self.normal = normal
        self.material = material
        self.t = t
        self.u = u
        self.v = v
        self.front_face = front_face

    def set_face_normal(self, ray: Ray, outward_normal: Vec3) -> None:
        """Ensure the surface normal always opposes the incoming incident ray."""
        self.front_face = ray.direction.dot(outward_normal) < 0.0
        self.normal = outward_normal if self.front_face else -outward_normal


class AABB:
    """
    Axis-Aligned Bounding Box (AABB) for spatial partitioning.
    Uses Andrew Kensler's optimized slab intersection test.
    """
    __slots__ = ("min_pt", "max_pt")

    def __init__(self, min_pt: Vec3, max_pt: Vec3):
        # Ensure min and max bounds are ordered along all axes
        self.min_pt = Vec3(
            min(min_pt.x, max_pt.x),
            min(min_pt.y, max_pt.y),
            min(min_pt.z, max_pt.z)
        )
        self.max_pt = Vec3(
            max(min_pt.x, max_pt.x),
            max(min_pt.y, max_pt.y),
            max(min_pt.z, max_pt.z)
        )

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        """
        Fast Kay-Kajiya slab test along axes X, Y, and Z.
        Returns True if the ray traverses the box interval.
        """
        for a in range(3):
            inv_d = 1.0 / (ray.direction[a] if abs(ray.direction[a]) > 1e-12 else 1e-12)
            t0 = (self.min_pt[a] - ray.origin[a]) * inv_d
            t1 = (self.max_pt[a] - ray.origin[a]) * inv_d
            if inv_d < 0.0:
                t0, t1 = t1, t0
            t_min = t0 if t0 > t_min else t_min
            t_max = t1 if t1 < t_max else t_max
            if t_max <= t_min:
                return False
        return True

    def surface_area(self) -> float:
        """Compute surface area for Surface Area Heuristic (SAH) cost estimation."""
        dx = max(0.0, self.max_pt.x - self.min_pt.x)
        dy = max(0.0, self.max_pt.y - self.min_pt.y)
        dz = max(0.0, self.max_pt.z - self.min_pt.z)
        return 2.0 * (dx * dy + dy * dz + dz * dx)

    @staticmethod
    def surrounding_box(box0: "AABB", box1: "AABB") -> "AABB":
        """Compute smallest bounding box containing both box0 and box1."""
        small = Vec3(
            min(box0.min_pt.x, box1.min_pt.x),
            min(box0.min_pt.y, box1.min_pt.y),
            min(box0.min_pt.z, box1.min_pt.z)
        )
        big = Vec3(
            max(box0.max_pt.x, box1.max_pt.x),
            max(box0.max_pt.y, box1.max_pt.y),
            max(box0.max_pt.z, box1.max_pt.z)
        )
        return AABB(small, big)


class Hittable(ABC):
    """Abstract base class for all 3D scene geometry."""

    @abstractmethod
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Determine if ray intersects geometry within distance interval [t_min, t_max]."""
        pass

    @abstractmethod
    def bounding_box(self) -> AABB:
        """Compute conservative axis-aligned bounding volume."""
        pass


class Sphere(Hittable):
    """Analytic Quadric Sphere: |P - Center|^2 = Radius^2."""
    __slots__ = ("center", "radius", "material", "_bbox")

    def __init__(self, center: Vec3, radius: float, material: object):
        self.center = center
        self.radius = float(radius)
        self.material = material
        r_vec = Vec3(radius, radius, radius)
        self._bbox = AABB(center - r_vec, center + r_vec)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = ray.direction.length_squared()
        half_b = oc.dot(ray.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0.0:
            return None

        sqrtd = math.sqrt(discriminant)
        # Find nearest root that lies within the valid distance interval
        root = (-half_b - sqrtd) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrtd) / a
            if root < t_min or root > t_max:
                return None

        rec = HitRecord()
        rec.t = root
        rec.point = ray.at(rec.t)
        outward_normal = (rec.point - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.material = self.material
        return rec

    def bounding_box(self) -> AABB:
        return self._bbox


class Triangle(Hittable):
    """
    3D Planar Triangle with Möller-Trumbore Ray-Triangle intersection.
    Directly evaluates barycentric coordinates without explicit plane solving.
    """
    __slots__ = ("v0", "v1", "v2", "material", "normal", "_bbox")

    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3, material: object):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        edge1 = v1 - v0
        edge2 = v2 - v0
        self.normal = edge1.cross(edge2).normalized()

        # Compute tight bounding box with epsilon padding
        eps = 1e-4
        min_p = Vec3(
            min(v0.x, v1.x, v2.x) - eps,
            min(v0.y, v1.y, v2.y) - eps,
            min(v0.z, v1.z, v2.z) - eps
        )
        max_p = Vec3(
            max(v0.x, v1.x, v2.x) + eps,
            max(v0.y, v1.y, v2.y) + eps,
            max(v0.z, v1.z, v2.z) + eps
        )
        self._bbox = AABB(min_p, max_p)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Möller-Trumbore algorithm."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        pvec = ray.direction.cross(edge2)
        det = edge1.dot(pvec)

        # Ray parallel to triangle
        if abs(det) < 1e-8:
            return None

        inv_det = 1.0 / det
        tvec = ray.origin - self.v0
        u = tvec.dot(pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None

        qvec = tvec.cross(edge1)
        v = ray.direction.dot(qvec) * inv_det
        if v < 0.0 or (u + v) > 1.0:
            return None

        t = edge2.dot(qvec) * inv_det
        if t < t_min or t > t_max:
            return None

        rec = HitRecord()
        rec.t = t
        rec.point = ray.at(t)
        rec.set_face_normal(ray, self.normal)
        rec.material = self.material
        rec.u = u
        rec.v = v
        return rec

    def bounding_box(self) -> AABB:
        return self._bbox


class HittableList(Hittable):
    """Container holding a collection of Hittable primitives with linear sweep."""
    __slots__ = ("objects", "_bbox")

    def __init__(self, objects: Optional[List[Hittable]] = None):
        self.objects: List[Hittable] = objects if objects is not None else []
        self._recompute_bbox()

    def add(self, obj: Hittable) -> None:
        self.objects.append(obj)
        self._recompute_bbox()

    def _recompute_bbox(self) -> None:
        if not self.objects:
            self._bbox = AABB(Vec3(0, 0, 0), Vec3(0, 0, 0))
            return
        box = self.objects[0].bounding_box()
        for obj in self.objects[1:]:
            box = AABB.surrounding_box(box, obj.bounding_box())
        self._bbox = box

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_rec = None
        closest_so_far = t_max

        for obj in self.objects:
            rec = obj.hit(ray, t_min, closest_so_far)
            if rec is not None:
                closest_so_far = rec.t
                closest_rec = rec

        return closest_rec

    def bounding_box(self) -> AABB:
        return self._bbox
