"""
PhotonPBR: Physically-Based Materials and Light Transport BRDFs.
Implements energy-conserving Lambertian diffuse, roughness-fuzzed metallic reflection,
dielectric glass refraction with Schlick's Fresnel approximation, and emissive area lights.
"""

import math
import random
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import HitRecord


class Material(ABC):
    """Abstract base class for surface light scattering models."""

    @abstractmethod
    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[bool, Color, Ray]:
        """
        Evaluate bidirectional surface scattering:
        Returns (did_scatter, attenuation, scattered_ray).
        """
        pass

    def emitted(self, u: float, v: float, p: Vec3) -> Color:
        """Light emission from self-illuminated surfaces."""
        return Color(0.0, 0.0, 0.0)


class Lambertian(Material):
    """Ideal diffuse reflection obeying Lambert's cosine law."""
    __slots__ = ("albedo",)

    def __init__(self, albedo: Color):
        self.albedo = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[bool, Color, Ray]:
        scatter_direction = rec.normal + Vec3.random_unit_vector()
        if scatter_direction.near_zero():
            scatter_direction = rec.normal

        scattered = Ray(rec.point, scatter_direction)
        return True, self.albedo, scattered


class Metal(Material):
    """Specular metallic reflection with microfacet roughness fuzz."""
    __slots__ = ("albedo", "fuzz")

    def __init__(self, albedo: Color, fuzz: float = 0.0):
        self.albedo = albedo
        self.fuzz = min(1.0, max(0.0, float(fuzz)))

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[bool, Color, Ray]:
        reflected = ray_in.direction.reflect(rec.normal)
        if self.fuzz > 0.0:
            reflected = reflected + self.fuzz * Vec3.random_in_unit_sphere()

        scattered = Ray(rec.point, reflected)
        # Only reflect outward away from surface
        did_scatter = scattered.direction.dot(rec.normal) > 0.0
        return did_scatter, self.albedo, scattered


class Dielectric(Material):
    """
    Transparent glass/water dielectric with Snell's law refraction
    and Schlick's polynomial approximation for angular Fresnel reflectance.
    """
    __slots__ = ("ir",)

    def __init__(self, index_of_refraction: float = 1.5):
        self.ir = float(index_of_refraction)

    @staticmethod
    def _reflectance(cosine: float, ref_idx: float) -> float:
        """Schlick's approximation for Fresnel reflectance."""
        r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
        r0 = r0 * r0
        return r0 + (1.0 - r0) * math.pow(1.0 - cosine, 5)

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[bool, Color, Ray]:
        attenuation = Color(1.0, 1.0, 1.0)
        refraction_ratio = (1.0 / self.ir) if rec.front_face else self.ir

        unit_direction = ray_in.direction
        cos_theta = min((-unit_direction).dot(rec.normal), 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        # Check for Total Internal Reflection (TIR)
        cannot_refract = (refraction_ratio * sin_theta) > 1.0
        reflectance = self._reflectance(cos_theta, refraction_ratio)

        if cannot_refract or reflectance > random.random():
            # Total internal reflection or Fresnel reflection
            direction = unit_direction.reflect(rec.normal)
        else:
            # Snell refraction
            success, direction = unit_direction.refract(rec.normal, refraction_ratio)
            if not success:
                direction = unit_direction.reflect(rec.normal)

        scattered = Ray(rec.point, direction)
        return True, attenuation, scattered


class DiffuseLight(Material):
    """Area light source emitting radiant energy."""
    __slots__ = ("emit_color",)

    def __init__(self, emit_color: Color):
        self.emit_color = emit_color

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[bool, Color, Ray]:
        # Light sources absorb incident rays without scattering
        return False, Color(0.0, 0.0, 0.0), Ray(Vec3(), Vec3(1, 0, 0))

    def emitted(self, u: float, v: float, p: Vec3) -> Color:
        return self.emit_color
