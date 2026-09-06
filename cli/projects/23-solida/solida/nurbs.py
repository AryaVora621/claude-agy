"""
Solida: Non-Uniform Rational B-Splines (NURBS) Curves, Surfaces & Curvatures.
Cox-de Boor recursive basis evaluation with exact rational conic weights.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Sequence, Optional
from solida.geometry import Vector3D


def create_clamped_knot_vector(num_ctrl_pts: int, degree: int) -> List[float]:
    """
    Generates an open clamped knot vector where multiplicity of first and last
    knots equals (degree + 1), ensuring curve/surface interpolates endpoints.
    Total length = num_ctrl_pts + degree + 1.
    """
    if num_ctrl_pts <= degree:
        raise ValueError(f"Number of control points ({num_ctrl_pts}) must exceed degree ({degree})")

    p = degree
    knots = [0.0] * (p + 1)
    interior_count = num_ctrl_pts - p - 1
    if interior_count > 0:
        step = 1.0 / (interior_count + 1)
        for i in range(1, interior_count + 1):
            knots.append(i * step)
    knots.extend([1.0] * (p + 1))
    return knots


def basis_function(i: int, p: int, u: float, knots: Sequence[float]) -> float:
    """
    Evaluates i-th B-spline basis function N_{i, p}(u) via Cox-de Boor recursion.
    Handles division by zero as 0.0 according to standard NURBS definitions.
    """
    u_min = knots[0]
    u_max = knots[-1]
    u = max(u_min, min(u_max, float(u)))

    if p == 0:
        # Handle rightmost endpoint
        if abs(u - u_max) < 1e-12:
            return 1.0 if knots[i] < u_max and abs(knots[i + 1] - u_max) < 1e-12 else 0.0
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0

    c1 = 0.0
    denom1 = knots[i + p] - knots[i]
    if abs(denom1) > 1e-15:
        c1 = ((u - knots[i]) / denom1) * basis_function(i, p - 1, u, knots)

    c2 = 0.0
    denom2 = knots[i + p + 1] - knots[i + 1]
    if abs(denom2) > 1e-15:
        c2 = ((knots[i + p + 1] - u) / denom2) * basis_function(i + 1, p - 1, u, knots)

    return c1 + c2


def basis_derivative(i: int, p: int, u: float, knots: Sequence[float]) -> float:
    """Evaluates analytical first derivative d/du N_{i, p}(u)."""
    if p == 0:
        return 0.0

    d1 = 0.0
    denom1 = knots[i + p] - knots[i]
    if abs(denom1) > 1e-15:
        d1 = (p / denom1) * basis_function(i, p - 1, u, knots)

    d2 = 0.0
    denom2 = knots[i + p + 1] - knots[i + 1]
    if abs(denom2) > 1e-15:
        d2 = (p / denom2) * basis_function(i + 1, p - 1, u, knots)

    return d1 - d2


def basis_second_derivative(i: int, p: int, u: float, knots: Sequence[float]) -> float:
    """Evaluates analytical second derivative d^2/du^2 N_{i, p}(u)."""
    if p < 2:
        return 0.0

    d1 = 0.0
    denom1 = knots[i + p] - knots[i]
    if abs(denom1) > 1e-15:
        d1 = (p / denom1) * basis_derivative(i, p - 1, u, knots)

    d2 = 0.0
    denom2 = knots[i + p + 1] - knots[i + 1]
    if abs(denom2) > 1e-15:
        d2 = (p / denom2) * basis_derivative(i + 1, p - 1, u, knots)

    return d1 - d2


class NURBSCurve:
    """
    Parametric Non-Uniform Rational B-Spline (NURBS) 3D curve.
    Defined by degree p, control points P_i, weights w_i, and knot vector U.
    """
    __slots__ = ("degree", "control_points", "weights", "knots")

    def __init__(
        self,
        degree: int,
        control_points: Sequence[Vector3D],
        weights: Optional[Sequence[float]] = None,
        knots: Optional[Sequence[float]] = None,
    ) -> None:
        self.degree = int(degree)
        self.control_points = list(control_points)
        n = len(self.control_points)

        if n <= self.degree:
            raise ValueError(f"Control points count ({n}) must exceed degree ({self.degree})")

        if weights is None:
            self.weights = [1.0] * n
        else:
            if len(weights) != n:
                raise ValueError("Weights count must match control points count")
            self.weights = [float(w) for w in weights]

        if knots is None:
            self.knots = create_clamped_knot_vector(n, self.degree)
        else:
            expected_knots = n + self.degree + 1
            if len(knots) != expected_knots:
                raise ValueError(f"Expected {expected_knots} knots, got {len(knots)}")
            self.knots = [float(k) for k in knots]

    def point_at(self, u: float) -> Vector3D:
        """Evaluates point C(u) = sum(N_i * w_i * P_i) / sum(N_i * w_i)."""
        num = Vector3D(0.0, 0.0, 0.0)
        denom = 0.0

        for i, (pt, w) in enumerate(zip(self.control_points, self.weights)):
            b = basis_function(i, self.degree, u, self.knots)
            wb = b * w
            num = num + pt * wb
            denom += wb

        if abs(denom) < 1e-15:
            return self.control_points[0]
        return num / denom

    def derivative_at(self, u: float) -> Vector3D:
        """Evaluates first analytical derivative C'(u) via rational quotient rule."""
        a = Vector3D(0.0, 0.0, 0.0)
        da = Vector3D(0.0, 0.0, 0.0)
        w = 0.0
        dw = 0.0

        for i, (pt, wt) in enumerate(zip(self.control_points, self.weights)):
            b = basis_function(i, self.degree, u, self.knots)
            db = basis_derivative(i, self.degree, u, self.knots)

            wb = wt * b
            wdb = wt * db

            a = a + pt * wb
            da = da + pt * wdb

            w += wb
            dw += wdb

        if abs(w) < 1e-15:
            return Vector3D(0.0, 0.0, 0.0)

        # C'(u) = (A' * w - A * w') / w^2
        return (da * w - a * dw) / (w * w)

    def tangent_at(self, u: float) -> Vector3D:
        """Unit tangent vector T(u) = C'(u) / ||C'(u)||."""
        d = self.derivative_at(u)
        return d.normalized()

    def second_derivative_at(self, u: float) -> Vector3D:
        """Evaluates second analytical derivative C''(u)."""
        # Central difference approximation of C'(u) for high numerical robustness
        h = 1e-5
        u_min = self.knots[0]
        u_max = self.knots[-1]
        u1 = max(u_min, u - h)
        u2 = min(u_max, u + h)
        d1 = self.derivative_at(u1)
        d2 = self.derivative_at(u2)
        return (d2 - d1) / (u2 - u1)

    def curvature_at(self, u: float) -> float:
        """Scalar curvature kappa(u) = ||C' x C''|| / ||C'||^3."""
        d1 = self.derivative_at(u)
        d2 = self.second_derivative_at(u)
        speed = d1.norm()
        if speed < 1e-9:
            return 0.0
        cross = d1.cross(d2).norm()
        return cross / (speed * speed * speed)

    def sample_points(self, num_samples: int = 50) -> List[Vector3D]:
        """Evenly samples points along parameter u in [0, 1]."""
        u_min = self.knots[0]
        u_max = self.knots[-1]
        step = (u_max - u_min) / (num_samples - 1)
        return [self.point_at(u_min + i * step) for i in range(num_samples)]

    def arc_length(self, samples: int = 100) -> float:
        """Numerical approximation of curve arc length."""
        pts = self.sample_points(samples)
        length = 0.0
        for i in range(len(pts) - 1):
            length += pts[i].distance_to(pts[i + 1])
        return length

    @classmethod
    def create_line(cls, p0: Vector3D, p1: Vector3D) -> NURBSCurve:
        """Creates a degree-1 linear NURBS segment."""
        return cls(degree=1, control_points=[p0, p1])

    @classmethod
    def create_circular_arc(
        cls,
        radius: float,
        start_angle_rad: float = 0.0,
        sweep_angle_rad: float = math.pi * 0.5,
        center: Optional[Vector3D] = None,
        normal: Vector3D = Vector3D(0, 0, 1),
    ) -> NURBSCurve:
        c = Vector3D(0, 0, 0) if center is None else center
        return cls.create_arc(
            center=c,
            radius=radius,
            start_angle_rad=start_angle_rad,
            end_angle_rad=start_angle_rad + sweep_angle_rad,
            normal=normal,
        )

    @classmethod
    def create_arc(
        cls,
        center: Vector3D,
        radius: float,
        start_angle_rad: float,
        end_angle_rad: float,
        normal: Vector3D = Vector3D(0, 0, 1),
    ) -> NURBSCurve:
        """
        Creates an exact rational circular arc of angle <= pi.
        Uses 3 control points with midpoint weight w_1 = cos(theta / 2).
        """
        theta = end_angle_rad - start_angle_rad
        if theta < 0:
            theta += 2.0 * math.pi
        if theta > math.pi + 1e-6:
            raise ValueError("Single NURBS arc span cannot exceed pi radians; subdivide into multi-span")

        n = normal.normalized()
        # Create orthonormal basis
        ref = Vector3D(1, 0, 0) if abs(n.x) < 0.9 else Vector3D(0, 1, 0)
        u_axis = ref.cross(n).normalized()
        v_axis = n.cross(u_axis).normalized()

        half_theta = 0.5 * theta
        mid_angle = start_angle_rad + half_theta

        # Start point
        p0 = center + (u_axis * math.cos(start_angle_rad) + v_axis * math.sin(start_angle_rad)) * radius
        # End point
        p2 = center + (u_axis * math.cos(end_angle_rad) + v_axis * math.sin(end_angle_rad)) * radius
        # Tangent intersection control point
        p1_dist = radius / math.cos(half_theta)
        p1 = center + (u_axis * math.cos(mid_angle) + v_axis * math.sin(mid_angle)) * p1_dist

        w1 = math.cos(half_theta)
        return cls(
            degree=2,
            control_points=[p0, p1, p2],
            weights=[1.0, w1, 1.0],
            knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )


class NURBSSurface:
    """
    Parametric Tensor-Product Non-Uniform Rational B-Spline (NURBS) 3D surface.
    Defined by degrees (p, q), 2D control point grid, weights grid, and knot vectors (U, V).
    """
    __slots__ = (
        "degree_u",
        "degree_v",
        "control_points",
        "weights",
        "knots_u",
        "knots_v",
        "num_u",
        "num_v",
    )

    def __init__(
        self,
        degree_u: int,
        degree_v: int,
        control_points: Sequence[Sequence[Vector3D]],
        weights: Optional[Sequence[Sequence[float]]] = None,
        knots_u: Optional[Sequence[float]] = None,
        knots_v: Optional[Sequence[float]] = None,
    ) -> None:
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)
        self.control_points = [list(row) for row in control_points]
        self.num_u = len(self.control_points)
        self.num_v = len(self.control_points[0]) if self.num_u > 0 else 0

        if self.num_u <= self.degree_u:
            raise ValueError(f"Control points in U ({self.num_u}) must exceed degree ({self.degree_u})")
        if self.num_v <= self.degree_v:
            raise ValueError(f"Control points in V ({self.num_v}) must exceed degree ({self.degree_v})")

        if weights is None:
            self.weights = [[1.0] * self.num_v for _ in range(self.num_u)]
        else:
            self.weights = [[float(w) for w in row] for row in weights]

        if knots_u is None:
            self.knots_u = create_clamped_knot_vector(self.num_u, self.degree_u)
        else:
            self.knots_u = [float(k) for k in knots_u]

        if knots_v is None:
            self.knots_v = create_clamped_knot_vector(self.num_v, self.degree_v)
        else:
            self.knots_v = [float(k) for k in knots_v]

    def point_at(self, u: float, v: float) -> Vector3D:
        """
        Evaluates surface point S(u, v):
        S(u, v) = sum_i sum_j (N_i * N_j * w_ij * P_ij) / sum_i sum_j (N_i * N_j * w_ij).
        """
        num = Vector3D(0.0, 0.0, 0.0)
        denom = 0.0

        # Precompute basis in U
        basis_u = [basis_function(i, self.degree_u, u, self.knots_u) for i in range(self.num_u)]
        basis_v = [basis_function(j, self.degree_v, v, self.knots_v) for j in range(self.num_v)]

        for i in range(self.num_u):
            bu = basis_u[i]
            if abs(bu) < 1e-15:
                continue
            for j in range(self.num_v):
                bv = basis_v[j]
                if abs(bv) < 1e-15:
                    continue

                w = self.weights[i][j]
                pt = self.control_points[i][j]
                wb = bu * bv * w
                num = num + pt * wb
                denom += wb

        if abs(denom) < 1e-15:
            return self.control_points[0][0]
        return num / denom

    def partial_derivatives_at(self, u: float, v: float) -> Tuple[Vector3D, Vector3D]:
        """Evaluates partial derivatives S_u(u, v) and S_v(u, v)."""
        h = 1e-5
        u_min, u_max = self.knots_u[0], self.knots_u[-1]
        v_min, v_max = self.knots_v[0], self.knots_v[-1]

        u1 = max(u_min, u - h)
        u2 = min(u_max, u + h)
        s_u1 = self.point_at(u1, v)
        s_u2 = self.point_at(u2, v)
        s_u = (s_u2 - s_u1) / (u2 - u1)

        v1 = max(v_min, v - h)
        v2 = min(v_max, v + h)
        s_v1 = self.point_at(u, v1)
        s_v2 = self.point_at(u, v2)
        s_v = (s_v2 - s_v1) / (v2 - v1)

        return (s_u, s_v)

    def normal_at(self, u: float, v: float) -> Vector3D:
        """Unit surface normal vector n(u, v) = (S_u x S_v) / ||S_u x S_v||."""
        su, sv = self.partial_derivatives_at(u, v)
        n = su.cross(sv)
        if n.norm_sq() < 1e-15:
            return Vector3D(0.0, 0.0, 1.0)
        return n.normalized()

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        """Calculates metric coefficients E, F, G."""
        su, sv = self.partial_derivatives_at(u, v)
        e = su.dot(su)
        f = su.dot(sv)
        g = sv.dot(sv)
        return (e, f, g)

    def gaussian_curvature_at(self, u: float, v: float) -> float:
        """Gaussian curvature K = (L*N - M^2) / (E*G - F^2)."""
        h = 1e-4
        su, sv = self.partial_derivatives_at(u, v)
        n = su.cross(sv).normalized()

        su_plus, _ = self.partial_derivatives_at(u + h, v)
        su_minus, _ = self.partial_derivatives_at(u - h, v)
        suu = (su_plus - su_minus) / (2.0 * h)

        _, sv_plus = self.partial_derivatives_at(u, v + h)
        _, sv_minus = self.partial_derivatives_at(u, v - h)
        svv = (sv_plus - sv_minus) / (2.0 * h)

        su_vplus, _ = self.partial_derivatives_at(u, v + h)
        su_vminus, _ = self.partial_derivatives_at(u, v - h)
        suv = (su_vplus - su_vminus) / (2.0 * h)

        l = suu.dot(n)
        m = suv.dot(n)
        n_coeff = svv.dot(n)

        e = su.dot(su)
        f = su.dot(sv)
        g = sv.dot(sv)

        denom = e * g - f * f
        if abs(denom) < 1e-12:
            return 0.0
        return (l * n_coeff - m * m) / denom

    def sample_mesh(
        self, samples_u: int = 16, samples_v: int = 16
    ) -> Tuple[List[Vector3D], List[Tuple[int, int, int]]]:
        """
        Discretizes NURBS surface into triangular mesh (vertices, triangle index triplets).
        """
        vertices: List[Vector3D] = []
        triangles: List[Tuple[int, int, int]] = []

        u_min, u_max = self.knots_u[0], self.knots_u[-1]
        v_min, v_max = self.knots_v[0], self.knots_v[-1]

        step_u = (u_max - u_min) / (samples_u - 1)
        step_v = (v_max - v_min) / (samples_v - 1)

        # Generate vertex grid
        for i in range(samples_u):
            u = u_min + i * step_u
            for j in range(samples_v):
                v = v_min + j * step_v
                vertices.append(self.point_at(u, v))

        # Generate triangle pairs for each quad cell
        for i in range(samples_u - 1):
            for j in range(samples_v - 1):
                idx00 = i * samples_v + j
                idx01 = i * samples_v + (j + 1)
                idx10 = (i + 1) * samples_v + j
                idx11 = (i + 1) * samples_v + (j + 1)

                triangles.append((idx00, idx10, idx11))
                triangles.append((idx00, idx11, idx01))

        return (vertices, triangles)

    @classmethod
    def create_bilinear_patch(
        cls, p00: Vector3D, p10: Vector3D, p01: Vector3D, p11: Vector3D
    ) -> NURBSSurface:
        """Creates a degree (1, 1) planar/ruled bilinear surface patch."""
        ctrl_pts = [
            [p00, p01],
            [p10, p11],
        ]
        return cls(degree_u=1, degree_v=1, control_points=ctrl_pts)

    @classmethod
    def create_extruded_surface(
        cls, profile: NURBSCurve, direction: Vector3D
    ) -> NURBSSurface:
        """Linear extrusion of a NURBS curve into a ruled NURBS surface."""
        ctrl_pts = [
            profile.control_points,
            [pt + direction for pt in profile.control_points],
        ]
        weights = [
            profile.weights,
            profile.weights,
        ]
        return cls(
            degree_u=1,
            degree_v=profile.degree,
            control_points=ctrl_pts,
            weights=weights,
            knots_u=[0.0, 0.0, 1.0, 1.0],
            knots_v=profile.knots,
        )
