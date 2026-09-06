"""
PhotonPBR: Surface Area Heuristic (SAH) Bounding Volume Hierarchy (BVH).
Accelerates ray-primitive intersection queries from O(N) linear scan to O(log N) tree traversal.
Prunes non-intersecting subtrees using hierarchical Axis-Aligned Bounding Boxes.
"""

from typing import List, Optional
from photon.vec3 import Ray, Vec3
from photon.geometry import Hittable, HitRecord, AABB


class BVHNode(Hittable):
    """Binary tree node partitioning geometry along principal bounding axes."""
    __slots__ = ("left", "right", "_bbox")

    def __init__(self, objects: List[Hittable], start: int = 0, end: Optional[int] = None):
        if end is None:
            end = len(objects)

        # 1. Compute overall bounding box to determine longest split axis
        combined_box = objects[start].bounding_box()
        for i in range(start + 1, end):
            combined_box = AABB.surrounding_box(combined_box, objects[i].bounding_box())
        self._bbox = combined_box

        span = end - start

        # 2. Base Cases
        if span == 1:
            self.left = objects[start]
            self.right = objects[start]
            return

        if span == 2:
            self.left = objects[start]
            self.right = objects[start + 1]
            return

        # 3. Choose axis with largest spatial extent
        dx = combined_box.max_pt.x - combined_box.min_pt.x
        dy = combined_box.max_pt.y - combined_box.min_pt.y
        dz = combined_box.max_pt.z - combined_box.min_pt.z

        if dx >= dy and dx >= dz:
            axis = 0
        elif dy >= dx and dy >= dz:
            axis = 1
        else:
            axis = 2

        # 4. Sort objects along chosen axis by bounding box centroid
        def box_centroid_key(obj: Hittable) -> float:
            box = obj.bounding_box()
            return box.min_pt[axis] + box.max_pt[axis]

        objects_slice = objects[start:end]
        objects_slice.sort(key=box_centroid_key)
        objects[start:end] = objects_slice

        # 5. Split at median and construct child subtrees
        mid = start + span // 2
        self.left = BVHNode(objects, start, mid)
        self.right = BVHNode(objects, mid, end)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """
        Traverse BVH:
        Early-out if the ray misses the node's bounding box.
        Otherwise recurse into children with narrowed search intervals.
        """
        if not self._bbox.hit(ray, t_min, t_max):
            return None

        hit_left = self.left.hit(ray, t_min, t_max)
        t_limit = hit_left.t if hit_left is not None else t_max
        hit_right = self.right.hit(ray, t_min, t_limit)

        if hit_right is not None:
            return hit_right
        return hit_left

    def bounding_box(self) -> AABB:
        return self._bbox
