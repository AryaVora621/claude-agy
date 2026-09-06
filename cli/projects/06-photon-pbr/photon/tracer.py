"""
PhotonPBR: Monte Carlo Path Tracing Core with Russian Roulette Termination.
Solves Kajiya's Rendering Equation by recursively sampling scattered radiance paths
and computing global illumination, soft area shadows, color bleeding, and caustics.
"""

import math
import random
import time
from typing import Optional, List, Tuple, Callable
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import Hittable, HitRecord


def ray_color(
    ray: Ray,
    world: Hittable,
    depth: int,
    max_depth: int = 8,
    background: Color = Color(0.0, 0.0, 0.0)
) -> Color:
    """
    Recursively evaluate path radiance along ray.
    t_min is set to 0.001 to eradicate numerical shadow acne.
    """
    if depth <= 0:
        return Color(0.0, 0.0, 0.0)

    # 1. Test scene geometry
    rec = world.hit(ray, 0.001, float("inf"))
    if rec is None:
        return background

    # 2. Emitted light
    emitted = rec.material.emitted(rec.u, rec.v, rec.point)

    # 3. Surface scattering
    did_scatter, attenuation, scattered = rec.material.scatter(ray, rec)
    if not did_scatter:
        return emitted

    # 4. Russian Roulette Path Termination (unbiased efficiency optimization)
    if depth < max_depth - 3:
        # Survival probability based on maximum albedo channel
        p_survive = min(max(attenuation.x, attenuation.y, attenuation.z, 0.05), 0.95)
        if random.random() > p_survive:
            return emitted
        attenuation = attenuation * (1.0 / p_survive)

    # 5. Recursive Radiance Accumulation
    incoming_radiance = ray_color(scattered, world, depth - 1, max_depth, background)
    return emitted + (attenuation * incoming_radiance)


class PathTracer:
    """Multi-sample Monte Carlo Path Tracer Engine."""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.rays_cast = 0

    def render(
        self,
        world: Hittable,
        camera: object,
        width: int,
        height: int,
        samples_per_pixel: int = 16,
        background: Color = Color(0.0, 0.0, 0.0),
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[Color]]:
        """
        Render 2D image framebuffer.
        Yields (height x width) grid of HDR Colors.
        """
        self.rays_cast = 0
        framebuffer: List[List[Color]] = []
        t0 = time.perf_counter()

        for j in range(height - 1, -1, -1):
            row: List[Color] = []
            for i in range(width):
                pixel_color = Color(0.0, 0.0, 0.0)
                for _ in range(samples_per_pixel):
                    u = (i + random.random()) / width
                    v = (j + random.random()) / height
                    r = camera.get_ray(u, v)
                    self.rays_cast += 1
                    sample_col = ray_color(r, world, self.max_depth, self.max_depth, background)
                    pixel_color = pixel_color + sample_col
                row.append(pixel_color)

            framebuffer.append(row)
            if progress_callback is not None:
                progress_callback(height - j, height)

        elapsed = time.perf_counter() - t0
        return framebuffer
