"""Relativistic Accretion Disk Physics & Gravitational Redshift Engine.

Models:
1. Geometrically thin, optically thick equatorial Keplerian accretion disk (Novikov-Thorne / Shakura-Sunyaev)
2. Circular equatorial orbit 4-velocities u_em^u in Schwarzschild and Kerr spacetimes
3. Relativistic frequency shift factor g = nu_obs / nu_em
4. Relativistic Doppler beaming (I_obs = g^4 * I_em)
5. Thermal radiating profile and TrueColor RGB color mapping based on redshift g and bolometric flux
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from relativitas.metric import SpacetimeMetric, KerrMetric, SchwarzschildMetric


@dataclass
class RadiativeTransferResult:
    """Photometric and relativistic spectral telemetry of an accretion disk emission."""
    r: float
    phi: float
    redshift_factor_g: float
    rest_frame_intensity: float
    observed_intensity: float
    rgb: Tuple[int, int, int]
    is_beamed: bool


class AccretionDisk:
    """Equatorial Keplerian accretion disk surrounding a black hole."""

    def __init__(
        self,
        metric: SpacetimeMetric,
        r_in: Optional[float] = None,
        r_out: Optional[float] = None,
        peak_intensity: float = 1.0,
    ) -> None:
        self.metric = metric
        self.r_in = r_in if r_in is not None else metric.isco_radius()
        self.r_out = r_out if r_out is not None else 15.0 * metric.mass
        self.peak_intensity = peak_intensity

    def keplerian_angular_velocity(self, r: float) -> float:
        """Angular velocity Omega_K = dphi/dt of circular equatorial orbits."""
        m = self.metric.mass
        if isinstance(self.metric, KerrMetric):
            a = self.metric.spin
            denom = r**1.5 + a * math.sqrt(m)
            return math.sqrt(m) / denom if denom > 0 else 0.0
        else:
            return math.sqrt(m / (r * r * r))

    def emitter_four_velocity(self, r: float) -> Tuple[float, float, float, float]:
        """Contravariant 4-velocity u_em^u = (u^t, 0, 0, Omega_K * u^t) of disk material."""
        omega = self.keplerian_angular_velocity(r)
        theta = 0.5 * math.pi
        g = self.metric.metric_tensor((0.0, r, theta, 0.0))

        # Normalization: g_uv * u^u * u^v = -1
        # u^t = 1 / sqrt(-(g_tt + 2*Omega*g_t_phi + Omega^2*g_phi_phi))
        denom = -(g[0][0] + 2.0 * omega * g[0][3] + omega * omega * g[3][3])
        if denom <= 1e-9:
            ut = 1.0
        else:
            ut = 1.0 / math.sqrt(denom)

        uphi = omega * ut
        return (ut, 0.0, 0.0, uphi)

    def rest_frame_profile(self, r: float) -> float:
        """Intrinsic radiative flux profile I_em(r) of the accretion disk.

        Approximated via Shakura-Sunyaev / Novikov-Thorne profile:
        I(r) proportional to (r_in / r)^3 * (1 - sqrt(r_in / r))
        """
        if r < self.r_in or r > self.r_out:
            return 0.0

        ratio = self.r_in / r
        inner_factor = max(0.0, 1.0 - math.sqrt(ratio))
        # Normalization factor ~4.0 to scale peak to ~1.0
        raw = (ratio**3) * inner_factor * 4.5 * self.peak_intensity
        return max(0.0, min(1.0, raw))

    def compute_redshift_factor(
        self,
        r: float,
        p_contra: Tuple[float, float, float, float],
    ) -> float:
        """Compute relativistic frequency ratio g = nu_obs / nu_em.

        Observer at coordinate rest at infinity: u_obs = (1, 0, 0, 0) => nu_obs = -p_t
        Emitter in Keplerian orbit: u_em = (ut, 0, 0, uphi) => nu_em = - (p_t * ut + p_phi * uphi)
        g = p_t / (ut * p_t + uphi * p_phi)
        """
        theta = 0.5 * math.pi
        g_cov = self.metric.metric_tensor((0.0, r, theta, 0.0))

        # Covariant 4-momenta: p_mu = g_mu_nu * p^nu
        pt_cov = g_cov[0][0] * p_contra[0] + g_cov[0][3] * p_contra[3]
        pphi_cov = g_cov[3][0] * p_contra[0] + g_cov[3][3] * p_contra[3]

        ut, _, _, uphi = self.emitter_four_velocity(r)

        denom = ut * pt_cov + uphi * pphi_cov
        if abs(denom) < 1e-12:
            return 1.0

        g_factor = pt_cov / denom
        # Constrain physical range to prevent numerical infinities near horizon
        return max(0.01, min(5.0, g_factor))

    def evaluate_radiative_transfer(
        self,
        r: float,
        phi: float,
        p_contra: Tuple[float, float, float, float],
    ) -> RadiativeTransferResult:
        """Calculate complete relativistic emission, Doppler boost, and RGB TrueColor."""
        i_em = self.rest_frame_profile(r)
        if i_em <= 0.0:
            return RadiativeTransferResult(
                r=r,
                phi=phi,
                redshift_factor_g=0.0,
                rest_frame_intensity=0.0,
                observed_intensity=0.0,
                rgb=(0, 0, 0),
                is_beamed=False,
            )

        g = self.compute_redshift_factor(r, p_contra)

        # Liouville theorem: I_obs = g^4 * I_em
        i_obs = (g**4) * i_em

        # Color mapping based on relativistic frequency shift g and flux i_obs
        rgb = self.map_redshift_to_rgb(g, i_obs)
        is_beamed = g > 1.1

        return RadiativeTransferResult(
            r=r,
            phi=phi,
            redshift_factor_g=g,
            rest_frame_intensity=i_em,
            observed_intensity=i_obs,
            rgb=rgb,
            is_beamed=is_beamed,
        )

    @staticmethod
    def map_redshift_to_rgb(g: float, intensity: float) -> Tuple[int, int, int]:
        """Map relativistic frequency ratio g and observed flux to TrueColor RGB."""
        # Brightness factor with soft compression
        brightness = math.tanh(intensity * 1.5)

        if g >= 1.3:
            # Extreme blue Doppler boosting (approaching gas): intense blue-white
            base_r, base_g, base_b = 0.7, 0.85, 1.0
        elif g >= 1.1:
            # Moderate blue shift: brilliant pale cyan/white
            base_r, base_g, base_b = 0.9, 0.95, 1.0
        elif g >= 0.95:
            # Quasi-neutral emission: bright golden yellow
            base_r, base_g, base_b = 1.0, 0.85, 0.4
        elif g >= 0.8:
            # Moderate redshift: warm amber / orange
            base_r, base_g, base_b = 1.0, 0.55, 0.15
        elif g >= 0.6:
            # Strong gravitational redshift: deep crimson
            base_r, base_g, base_b = 0.85, 0.2, 0.1
        else:
            # Extreme redshift near ISCO: faint dark infrared/maroon
            base_r, base_g, base_b = 0.5, 0.05, 0.05

        r_byte = int(max(0.0, min(255.0, base_r * brightness * 255.0)))
        g_byte = int(max(0.0, min(255.0, base_g * brightness * 255.0)))
        b_byte = int(max(0.0, min(255.0, base_b * brightness * 255.0)))

        return (r_byte, g_byte, b_byte)
