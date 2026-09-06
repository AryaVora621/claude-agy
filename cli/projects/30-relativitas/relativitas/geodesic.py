"""Curved Spacetime Geodesic Integrator for General Relativity.

Integrates the 8-state first-order geodesic equations of motion:
    dx^u / dlambda = p^u
    dp^u / dlambda = -Gamma^u_ab * p^a * p^b

Supports:
- Null geodesics (massless photons, kappa = 0)
- Timelike geodesics (massive matter particles, kappa = -1)
- 4th-Order Runge-Kutta (RK4) integration with geometry-adaptive step sizing
- Exact conservation checking: 4-momentum norm, energy E = -p_t, and angular momentum L = p_phi
- Ray-matter intersection events: event horizon capture, equatorial accretion disk crossing, and celestial escape
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from relativitas.metric import SpacetimeMetric


class GeodesicStatus(Enum):
    """Termination status of an integrated geodesic trajectory."""
    IN_FLIGHT = "IN_FLIGHT"
    HORIZON_CAPTURED = "HORIZON_CAPTURED"
    DISK_INTERSECTED = "DISK_INTERSECTED"
    ESCAPED = "ESCAPED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"


@dataclass
class GeodesicState:
    """8-state coordinate and 4-momentum vector in curved spacetime."""
    t: float
    r: float
    theta: float
    phi: float
    pt: float
    pr: float
    ptheta: float
    pphi: float
    affine_lambda: float = 0.0

    @property
    def coords(self) -> Tuple[float, float, float, float]:
        """Position 4-vector x^u = (t, r, theta, phi)."""
        return (self.t, self.r, self.theta, self.phi)

    @property
    def momentum(self) -> Tuple[float, float, float, float]:
        """Contravariant 4-momentum p^u = (pt, pr, ptheta, pphi)."""
        return (self.pt, self.pr, self.ptheta, self.pphi)

    def copy(self) -> GeodesicState:
        """Create a detached shallow copy of this geodesic state."""
        return GeodesicState(
            t=self.t,
            r=self.r,
            theta=self.theta,
            phi=self.phi,
            pt=self.pt,
            pr=self.pr,
            ptheta=self.ptheta,
            pphi=self.pphi,
            affine_lambda=self.affine_lambda,
        )


@dataclass
class DiskIntersection:
    """Telemetry recorded when a geodesic pierces the equatorial plane."""
    r_cross: float
    phi_cross: float
    t_cross: float
    state: GeodesicState
    step_index: int


@dataclass
class GeodesicResult:
    """Complete trajectory and physical telemetry of an integrated geodesic."""
    final_state: GeodesicState
    status: GeodesicStatus
    steps: int
    trajectory: List[GeodesicState]
    disk_intersections: List[DiskIntersection]
    initial_norm: float
    final_norm: float
    energy_conserved: bool
    angular_momentum_conserved: bool


class GeodesicIntegrator:
    """Runge-Kutta 4th-order integrator for curved spacetime geodesics."""

    def __init__(
        self,
        metric: SpacetimeMetric,
        is_null: bool = True,
        horizon_buffer: float = 0.05,
        escape_radius: float = 40.0,
        max_steps: int = 1500,
        base_step_size: float = 0.1,
    ) -> None:
        self.metric = metric
        self.is_null = is_null
        self.target_norm = 0.0 if is_null else -1.0
        self.horizon_buffer = horizon_buffer
        self.escape_radius = escape_radius
        self.max_steps = max_steps
        self.base_step_size = base_step_size

    def compute_derivatives(
        self,
        state: GeodesicState,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """Evaluate system derivatives: dx^u/dlambda and dp^u/dlambda."""
        x = state.coords
        p = state.momentum
        gamma = self.metric.christoffel_symbols(x)

        # dx^u / dlambda = p^u
        dxdl = p

        # dp^u / dlambda = - Gamma^u_ab * p^a * p^b
        dpdl = [0.0, 0.0, 0.0, 0.0]
        for mu in range(4):
            acc = 0.0
            for a in range(4):
                pa = p[a]
                if abs(pa) < 1e-18:
                    continue
                # Exploit symmetry Gamma^mu_ab = Gamma^mu_ba
                acc += gamma[mu][a][a] * pa * pa
                for b in range(a + 1, 4):
                    acc += 2.0 * gamma[mu][a][b] * pa * p[b]
            dpdl[mu] = -acc

        return (dxdl, (dpdl[0], dpdl[1], dpdl[2], dpdl[3]))

    def adaptive_step_size(self, r: float, direction: float = 1.0) -> float:
        """Scale step size h dynamically based on proximity to the event horizon."""
        r_h = self.metric.horizon_radius()
        dr = max(0.01, r - r_h)

        # Scale smaller near strong-field horizon, larger in weak-field asymptotic regions
        scale = min(1.0, dr / (4.0 * self.metric.mass))
        h = max(0.005, self.base_step_size * scale)
        return direction * h

    def step_rk4(self, state: GeodesicState, h: float) -> GeodesicState:
        """Perform a single 4th-order Runge-Kutta integration step."""
        # State vector representation: [t, r, th, ph, pt, pr, pth, pph]
        y0 = [
            state.t, state.r, state.theta, state.phi,
            state.pt, state.pr, state.ptheta, state.pphi
        ]

        def get_derivs(y: List[float]) -> List[float]:
            st = GeodesicState(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])
            dxdl, dpdl = self.compute_derivatives(st)
            return [
                dxdl[0], dxdl[1], dxdl[2], dxdl[3],
                dpdl[0], dpdl[1], dpdl[2], dpdl[3]
            ]

        # k1
        k1 = get_derivs(y0)

        # k2
        y1 = [y0[i] + 0.5 * h * k1[i] for i in range(8)]
        k2 = get_derivs(y1)

        # k3
        y2 = [y0[i] + 0.5 * h * k2[i] for i in range(8)]
        k3 = get_derivs(y2)

        # k4
        y3 = [y0[i] + h * k3[i] for i in range(8)]
        k4 = get_derivs(y3)

        # Update
        y_next = [
            y0[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(8)
        ]

        # Coordinate clamping to avoid polar singularities
        theta_next = y_next[2]
        if theta_next <= 0.001:
            theta_next = 0.001
        elif theta_next >= math.pi - 0.001:
            theta_next = math.pi - 0.001

        # Wrap phi into [0, 2*pi)
        phi_next = y_next[3] % (2.0 * math.pi)

        return GeodesicState(
            t=y_next[0],
            r=y_next[1],
            theta=theta_next,
            phi=phi_next,
            pt=y_next[4],
            pr=y_next[5],
            ptheta=y_next[6],
            pphi=y_next[7],
            affine_lambda=state.affine_lambda + h,
        )

    def integrate(
        self,
        initial_state: GeodesicState,
        backward: bool = False,
        stop_on_disk: bool = True,
        disk_r_in: Optional[float] = None,
        disk_r_out: Optional[float] = None,
        store_trajectory: bool = False,
    ) -> GeodesicResult:
        """Integrate a geodesic trajectory until a physical termination event occurs."""
        r_h = self.metric.horizon_radius()
        r_capture = r_h + self.horizon_buffer

        if disk_r_in is None:
            disk_r_in = self.metric.isco_radius()
        if disk_r_out is None:
            disk_r_out = 15.0 * self.metric.mass

        direction = -1.0 if backward else 1.0
        current_state = initial_state.copy()

        trajectory: List[GeodesicState] = []
        if store_trajectory:
            trajectory.append(current_state.copy())

        disk_intersections: List[DiskIntersection] = []
        status = GeodesicStatus.IN_FLIGHT

        initial_norm = self.metric.scalar_norm(current_state.coords, current_state.momentum)

        for step in range(self.max_steps):
            # Check event horizon capture
            if current_state.r <= r_capture:
                status = GeodesicStatus.HORIZON_CAPTURED
                break

            # Check escape to celestial sphere
            if current_state.r >= self.escape_radius:
                status = GeodesicStatus.ESCAPED
                break

            h = self.adaptive_step_size(current_state.r, direction=direction)
            prev_state = current_state
            current_state = self.step_rk4(prev_state, h)

            if store_trajectory:
                trajectory.append(current_state.copy())

            # Detect equatorial accretion disk crossing (theta crosses pi/2)
            half_pi = 0.5 * math.pi
            prev_diff = prev_state.theta - half_pi
            curr_diff = current_state.theta - half_pi

            if prev_diff * curr_diff <= 0.0 and abs(curr_diff - prev_diff) > 1e-12:
                # Linear interpolation to exact disk plane crossing
                fraction = abs(prev_diff) / (abs(prev_diff) + abs(curr_diff))
                r_cross = prev_state.r + fraction * (current_state.r - prev_state.r)
                phi_cross = (prev_state.phi + fraction * (current_state.phi - prev_state.phi)) % (2.0 * math.pi)
                t_cross = prev_state.t + fraction * (current_state.t - prev_state.t)

                if disk_r_in <= r_cross <= disk_r_out:
                    intersection = DiskIntersection(
                        r_cross=r_cross,
                        phi_cross=phi_cross,
                        t_cross=t_cross,
                        state=current_state.copy(),
                        step_index=step,
                    )
                    disk_intersections.append(intersection)

                    if stop_on_disk:
                        status = GeodesicStatus.DISK_INTERSECTED
                        break
        else:
            status = GeodesicStatus.MAX_STEPS_REACHED

        final_norm = self.metric.scalar_norm(current_state.coords, current_state.momentum)

        # Invariant checks
        # In stationary, axisymmetric spacetime: covariant p_t and p_phi are Killing constants of motion
        g_init = self.metric.metric_tensor(initial_state.coords)
        pt_cov_init = g_init[0][0] * initial_state.pt + g_init[0][3] * initial_state.pphi
        pphi_cov_init = g_init[3][0] * initial_state.pt + g_init[3][3] * initial_state.pphi

        g_curr = self.metric.metric_tensor(current_state.coords)
        pt_cov_curr = g_curr[0][0] * current_state.pt + g_curr[0][3] * current_state.pphi
        pphi_cov_curr = g_curr[3][0] * current_state.pt + g_curr[3][3] * current_state.pphi

        energy_conserved = abs(pt_cov_curr - pt_cov_init) < 1e-2
        ang_mom_conserved = abs(pphi_cov_curr - pphi_cov_init) < 1e-2

        return GeodesicResult(
            final_state=current_state,
            status=status,
            steps=len(trajectory) if store_trajectory else step + 1,
            trajectory=trajectory,
            disk_intersections=disk_intersections,
            initial_norm=initial_norm,
            final_norm=final_norm,
            energy_conserved=energy_conserved,
            angular_momentum_conserved=ang_mom_conserved,
        )
