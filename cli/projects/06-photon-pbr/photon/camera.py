"""
PhotonPBR: Thin-Lens Virtual Pinhole/Defocus Camera.
Calculates orthonormal camera basis vectors (u, v, w), field of view,
and generates primary rays with optional depth-of-field lens aperture sampling.
"""

import math
from photon.vec3 import Vec3, Ray


class Camera:
    """Virtual camera projecting viewport rays into 3D world space."""

    def __init__(
        self,
        lookfrom: Vec3 = Vec3(0, 0, 0),
        lookat: Vec3 = Vec3(0, 0, -1),
        vup: Vec3 = Vec3(0, 1, 0),
        vfov_degrees: float = 90.0,
        aspect_ratio: float = 16.0 / 9.0,
        aperture: float = 0.0,
        focus_dist: float = 1.0
    ):
        self.origin = lookfrom
        self.lens_radius = aperture / 2.0

        # Viewport geometry
        theta = math.radians(vfov_degrees)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        # Orthonormal camera coordinate frame (u=right, v=up, w=backwards)
        self.w = (lookfrom - lookat).normalized()
        self.u = vup.cross(self.w).normalized()
        self.v = self.w.cross(self.u)

        self.horizontal = focus_dist * viewport_width * self.u
        self.vertical = focus_dist * viewport_height * self.v
        self.lower_left_corner = (
            self.origin
            - self.horizontal / 2.0
            - self.vertical / 2.0
            - focus_dist * self.w
        )

    def get_ray(self, s: float, t: float) -> Ray:
        """
        Generate primary ray directed towards normalized viewport coordinates (s, t).
        Supports defocus blur if aperture > 0.
        """
        if self.lens_radius > 0.0:
            rd = self.lens_radius * Vec3.random_in_unit_disk()
            offset = self.u * rd.x + self.v * rd.y
        else:
            offset = Vec3(0.0, 0.0, 0.0)

        ray_origin = self.origin + offset
        ray_direction = (
            self.lower_left_corner
            + s * self.horizontal
            + t * self.vertical
            - self.origin
            - offset
        )
        return Ray(ray_origin, ray_direction)
