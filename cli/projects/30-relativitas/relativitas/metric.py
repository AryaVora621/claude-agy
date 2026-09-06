"""Spacetime Metric Tensors and Christoffel Connection Symbols.

Provides pseudo-Riemannian manifold geometry for:
1. Schwarzschild Metric (Static, non-rotating spherical black hole)
2. Kerr Metric in Boyer-Lindquist Coordinates (Rotating black hole with frame dragging)

Features:
- Metric tensor g_uv and contravariant inverse g^uv
- Christoffel connection symbols of the second kind Gamma^u_ab
- Outer/inner event horizons, ergosphere boundary, and ISCO radii
- Conservation invariants: energy E, axial angular momentum L, and 4-velocity norm
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List, Tuple


class SpacetimeMetric(ABC):
    """Abstract base class for 4D pseudo-Riemannian spacetime metrics (-, +, +, +)."""

    def __init__(self, mass: float = 1.0) -> None:
        self.mass = mass

    @abstractmethod
    def metric_tensor(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        """Return 4x4 covariant metric tensor g_uv at coordinates x = (t, r, theta, phi)."""
        pass

    @abstractmethod
    def inverse_metric(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        """Return 4x4 contravariant inverse metric tensor g^uv."""
        pass

    @abstractmethod
    def horizon_radius(self) -> float:
        """Return outer event horizon radius r_+."""
        pass

    @abstractmethod
    def isco_radius(self) -> float:
        """Return innermost stable circular orbit (ISCO) radius."""
        pass

    def christoffel_symbols(
        self,
        x: Tuple[float, float, float, float],
        eps: float = 1e-6,
    ) -> List[List[List[float]]]:
        """Compute Christoffel connection symbols Gamma^u_ab at coordinate point x.

        Evaluated via central finite-differences of metric tensor components:
        Gamma^u_ab = 0.5 * g^us * (d_a g_bs + d_b g_as - d_s g_ab)
        """
        g_inv = self.inverse_metric(x)

        # Precompute partial derivatives: dg[alpha][beta][sigma] = d_(alpha) g_(beta, sigma)
        dg: List[List[List[float]]] = [
            [[0.0 for _ in range(4)] for _ in range(4)] for _ in range(4)
        ]

        # Numerical differentiation step sizes
        steps = (eps, eps, eps, eps)

        for alpha in range(4):
            x_plus = list(x)
            x_minus = list(x)
            x_plus[alpha] += steps[alpha]
            x_minus[alpha] -= steps[alpha]

            g_plus = self.metric_tensor(tuple(x_plus))  # type: ignore
            g_minus = self.metric_tensor(tuple(x_minus)) # type: ignore
            inv_2h = 0.5 / steps[alpha]

            for beta in range(4):
                for sigma in range(4):
                    dg[alpha][beta][sigma] = (g_plus[beta][sigma] - g_minus[beta][sigma]) * inv_2h

        # Assemble Gamma[mu][alpha][beta]
        gamma: List[List[List[float]]] = [
            [[0.0 for _ in range(4)] for _ in range(4)] for _ in range(4)
        ]

        for mu in range(4):
            for alpha in range(4):
                for beta in range(alpha, 4):
                    val = 0.0
                    for sigma in range(4):
                        term = dg[alpha][beta][sigma] + dg[beta][alpha][sigma] - dg[sigma][alpha][beta]
                        val += g_inv[mu][sigma] * term
                    val *= 0.5
                    gamma[mu][alpha][beta] = val
                    gamma[mu][beta][alpha] = val

        return gamma

    def scalar_norm(
        self,
        x: Tuple[float, float, float, float],
        p: Tuple[float, float, float, float],
    ) -> float:
        """Compute 4-momentum scalar invariant: g_uv * p^u * p^v."""
        g = self.metric_tensor(x)
        res = 0.0
        for mu in range(4):
            for nu in range(4):
                res += g[mu][nu] * p[mu] * p[nu]
        return res


class SchwarzschildMetric(SpacetimeMetric):
    """Static spherically symmetric Schwarzschild spacetime."""

    def __init__(self, mass: float = 1.0) -> None:
        super().__init__(mass)

    def horizon_radius(self) -> float:
        """Event horizon r_s = 2M."""
        return 2.0 * self.mass

    def photon_sphere_radius(self) -> float:
        """Photon sphere r_ph = 3M."""
        return 3.0 * self.mass

    def isco_radius(self) -> float:
        """Innermost Stable Circular Orbit r_isco = 6M."""
        return 6.0 * self.mass

    def metric_tensor(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        _, r, theta, _ = x
        m = self.mass
        r = max(r, 2.0001 * m)
        sin_th = math.sin(theta)

        f = 1.0 - (2.0 * m / r)
        g = [[0.0 for _ in range(4)] for _ in range(4)]
        g[0][0] = -f
        g[1][1] = 1.0 / f
        g[2][2] = r * r
        g[3][3] = r * r * sin_th * sin_th
        return g

    def inverse_metric(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        _, r, theta, _ = x
        m = self.mass
        r = max(r, 2.0001 * m)
        sin_th = math.sin(theta)
        sin_sq = sin_th * sin_th
        sin_sq = max(sin_sq, 1e-12)

        f = 1.0 - (2.0 * m / r)
        g_inv = [[0.0 for _ in range(4)] for _ in range(4)]
        g_inv[0][0] = -1.0 / f
        g_inv[1][1] = f
        g_inv[2][2] = 1.0 / (r * r)
        g_inv[3][3] = 1.0 / (r * r * sin_sq)
        return g_inv

    def christoffel_symbols(
        self,
        x: Tuple[float, float, float, float],
        eps: float = 1e-6,
    ) -> List[List[List[float]]]:
        """Exact analytical Christoffel symbols for Schwarzschild metric."""
        _, r, theta, _ = x
        m = self.mass
        r = max(r, 2.0001 * m)
        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        cot_th = cos_th / sin_th if abs(sin_th) > 1e-9 else 0.0

        r_minus_2m = r - 2.0 * m
        r_sq = r * r
        r_cu = r_sq * r

        gamma: List[List[List[float]]] = [
            [[0.0 for _ in range(4)] for _ in range(4)] for _ in range(4)
        ]

        # Gamma^t_tr = Gamma^t_rt = M / (r(r - 2M))
        gamma[0][0][1] = gamma[0][1][0] = m / (r * r_minus_2m)

        # Gamma^r_tt = M(r - 2M) / r^3
        gamma[1][0][0] = (m * r_minus_2m) / r_cu

        # Gamma^r_rr = -M / (r(r - 2M))
        gamma[1][1][1] = -m / (r * r_minus_2m)

        # Gamma^r_th_th = -(r - 2M)
        gamma[1][2][2] = -r_minus_2m

        # Gamma^r_ph_ph = -(r - 2M) * sin^2(theta)
        gamma[1][3][3] = -r_minus_2m * sin_th * sin_th

        # Gamma^th_r_th = Gamma^th_th_r = 1/r
        gamma[2][1][2] = gamma[2][2][1] = 1.0 / r

        # Gamma^th_ph_ph = -sin(theta) * cos(theta)
        gamma[2][3][3] = -sin_th * cos_th

        # Gamma^ph_r_ph = Gamma^ph_ph_r = 1/r
        gamma[3][1][3] = gamma[3][3][1] = 1.0 / r

        # Gamma^ph_th_ph = Gamma^ph_ph_th = cot(theta)
        gamma[3][2][3] = gamma[3][3][2] = cot_th

        return gamma


class KerrMetric(SpacetimeMetric):
    """Rotating Kerr black hole spacetime in Boyer-Lindquist coordinates."""

    def __init__(self, mass: float = 1.0, spin: float = 0.9) -> None:
        super().__init__(mass)
        if abs(spin) >= mass:
            raise ValueError(f"Spin parameter |a|={abs(spin)} must be strictly less than M={mass}")
        self.spin = spin  # a = J/M

    def horizon_radius(self) -> float:
        """Outer event horizon r_+ = M + sqrt(M^2 - a^2)."""
        return self.mass + math.sqrt(self.mass * self.mass - self.spin * self.spin)

    def inner_horizon_radius(self) -> float:
        """Inner (Cauchy) horizon r_- = M - sqrt(M^2 - a^2)."""
        return self.mass - math.sqrt(self.mass * self.mass - self.spin * self.spin)

    def ergosphere_radius(self, theta: float) -> float:
        """Outer boundary of ergosphere r_ergo(theta) = M + sqrt(M^2 - a^2 * cos^2(theta))."""
        cos_th = math.cos(theta)
        return self.mass + math.sqrt(self.mass * self.mass - self.spin * self.spin * cos_th * cos_th)

    def isco_radius(self, prograde: bool = True) -> float:
        """Innermost Stable Circular Orbit r_isco via Bardeen-Press-Teukolsky (1972)."""
        m = self.mass
        a = self.spin
        sign = 1.0 if prograde else -1.0

        a_bar = a / m
        z1 = 1.0 + (1.0 - a_bar * a_bar)**(1.0 / 3.0) * (
            (1.0 + a_bar)**(1.0 / 3.0) + (1.0 - a_bar)**(1.0 / 3.0)
        )
        z2 = math.sqrt(3.0 * a_bar * a_bar + z1 * z1)
        term = math.sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)))

        return m * (3.0 + z2 - sign * term)

    def frame_dragging_angular_velocity(self, r: float, theta: float) -> float:
        """Frame dragging frequency omega = -g_t_phi / g_phi_phi."""
        m = self.mass
        a = self.spin
        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        sigma = r * r + a * a * cos_th * cos_th
        denom = (r * r + a * a) * sigma + 2.0 * m * r * a * a * sin_th * sin_th
        return (2.0 * m * a * r) / denom if denom > 0 else 0.0

    def metric_tensor(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        _, r, theta, _ = x
        m = self.mass
        a = self.spin
        r_plus = self.horizon_radius()
        r = max(r, r_plus + 1e-4)

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        sin_sq = sin_th * sin_th

        sigma = r * r + a * a * cos_th * cos_th
        delta = r * r - 2.0 * m * r + a * a

        g = [[0.0 for _ in range(4)] for _ in range(4)]

        # g_tt = -(1 - 2Mr / Sigma)
        g[0][0] = -(1.0 - (2.0 * m * r) / sigma)

        # g_t_phi = g_phi_t = -2Mar * sin^2(theta) / Sigma
        g_t_phi = -(2.0 * m * a * r * sin_sq) / sigma
        g[0][3] = g[3][0] = g_t_phi

        # g_rr = Sigma / Delta
        g[1][1] = sigma / delta

        # g_th_th = Sigma
        g[2][2] = sigma

        # g_phi_phi = (r^2 + a^2 + 2Ma^2 r sin^2(theta) / Sigma) * sin^2(theta)
        g[3][3] = (r * r + a * a + (2.0 * m * a * a * r * sin_sq) / sigma) * sin_sq

        return g

    def inverse_metric(self, x: Tuple[float, float, float, float]) -> List[List[float]]:
        _, r, theta, _ = x
        m = self.mass
        a = self.spin
        r_plus = self.horizon_radius()
        r = max(r, r_plus + 1e-4)

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        sin_sq = max(sin_th * sin_th, 1e-12)

        sigma = r * r + a * a * cos_th * cos_th
        delta = max(r * r - 2.0 * m * r + a * a, 1e-9)

        g_inv = [[0.0 for _ in range(4)] for _ in range(4)]

        # Determinant of t-phi block: g_tt * g_phi_phi - (g_t_phi)^2 = -Delta * sin^2(theta)
        # g^tt = - ((r^2 + a^2)^2 - a^2 Delta sin^2(theta)) / (Sigma * Delta)
        r2_a2 = r * r + a * a
        g_inv[0][0] = - (r2_a2 * r2_a2 - a * a * delta * sin_sq) / (sigma * delta)

        # g^t_phi = g^phi_t = -2Mar / (Sigma * Delta)
        g_inv[0][3] = g_inv[3][0] = - (2.0 * m * a * r) / (sigma * delta)

        # g^rr = Delta / Sigma
        g_inv[1][1] = delta / sigma

        # g^th_th = 1 / Sigma
        g_inv[2][2] = 1.0 / sigma

        # g^phi_phi = (Delta - a^2 sin^2(theta)) / (Sigma * Delta * sin^2(theta))
        g_inv[3][3] = (delta - a * a * sin_sq) / (sigma * delta * sin_sq)

        return g_inv
