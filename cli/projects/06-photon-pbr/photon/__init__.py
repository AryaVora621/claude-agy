"""
PhotonPBR: Physically-Based Monte Carlo Path Tracer with BVH Acceleration.
Built from first principles in zero-dependency Python.
"""

from photon.vec3 import Vec3, Ray, Color
from photon.geometry import HitRecord, AABB, Hittable, Sphere, Triangle, HittableList
from photon.bvh import BVHNode
from photon.material import Material, Lambertian, Metal, Dielectric, DiffuseLight
from photon.camera import Camera
from photon.tracer import PathTracer, ray_color
from photon.framebuffer import Framebuffer

__all__ = [
    "Vec3", "Ray", "Color", "HitRecord", "AABB", "Hittable", "Sphere",
    "Triangle", "HittableList", "BVHNode", "Material", "Lambertian", "Metal",
    "Dielectric", "DiffuseLight", "Camera", "PathTracer", "ray_color", "Framebuffer"
]
