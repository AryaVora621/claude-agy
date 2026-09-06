"""
Symplectic Boris Particle Pusher & Neoclassical Tokamak Orbit Engine.

Integrates charged particle equations of motion in 3D toroidal magnetic and electric
fields using the volume-preserving, unconditionally stable Boris algorithm (Boris 1970).

Captures full Lorentz gyro-motion, magnetic mirror reflection, neoclassical banana
orbits for trapped ions/alphas, passing particle drift surfaces, and guiding center drifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

from .equilibrium import ELEMENTARY_CHARGE, KILO_ELECTRON_VOLT
from .magnetic import MagneticField, MagneticFieldEvaluator


# Physical mass constants (kg)
MASS_PROTON: float = 1.67262192369e-27
MASS_DEUTERON: float = 2.01410177811 * 1.66053906660e-27
MASS_TRITON: float = 3.01604928199 * 1.66053906660e-27
MASS_ALPHA: float = 4.001506179127 * 1.66053906660e-27
MASS_ELECTRON: float = 9.1093837015e-31


class ParticleSpecies(Enum):
    """Charged particle species in magnetic confinement fusion plasmas."""

    DEUTERIUM = "Deuterium (D+)"
    TRITIUM = "Tritium (T+)"
    ALPHA = "Alpha Particle (He2+)"
    ELECTRON = "Electron (e-)"
    NORMALIZED_ION = "Normalized Test Ion"


@dataclass
class SpeciesProperties:
    """Charge and rest mass characteristics of a plasma species."""

    charge: float  # Coulombs
    mass: float  # Kilograms
    name: str

    @classmethod
    def from_species(cls, species: ParticleSpecies) -> SpeciesProperties:
        if species == ParticleSpecies.DEUTERIUM:
            return cls(ELEMENTARY_CHARGE, MASS_DEUTERON, "D+")
        if species == ParticleSpecies.TRITIUM:
            return cls(ELEMENTARY_CHARGE, MASS_TRITON, "T+")
        if species == ParticleSpecies.ALPHA:
            return cls(2.0 * ELEMENTARY_CHARGE, MASS_ALPHA, "He2+")
        if species == ParticleSpecies.ELECTRON:
            return cls(-ELEMENTARY_CHARGE, MASS_ELECTRON, "e-")
        return cls(1.0, 1.0, "Test Particle")


@dataclass
class ParticleState:
    """Cartesian 6D phase space state (x, y, z, vx, vy, vz) and simulation time."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    time: float = 0.0

    @property
    def r_cylindrical(self) -> float:
        """Cylindrical major radial position R = sqrt(x^2 + y^2)."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def phi_cylindrical(self) -> float:
        """Toroidal angle phi in [0, 2*pi)."""
        phi = math.atan2(self.y, self.x)
        return phi if phi >= 0.0 else phi + 2.0 * math.pi

    @property
    def speed(self) -> float:
        """Speed magnitude |v| in m/s."""
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def kinetic_energy_joules(self, mass: float) -> float:
        """Kinetic energy in Joules: 0.5 * m * v^2."""
        return 0.5 * mass * (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def kinetic_energy_kev(self, mass: float) -> float:
        """Kinetic energy in kilo-electron-volts (keV)."""
        return self.kinetic_energy_joules(mass) / KILO_ELECTRON_VOLT


class BorisParticlePusher:
    """
    Symplectic Boris integrator for charged particle dynamics in toroidal geometries.

    Advances Lorentz equations:
        dx/dt = v
        dv/dt = (q / m) * (E + v x B)
    preserving phase space volume and maintaining exact kinetic energy conservation when E = 0.
    """

    def __init__(
        self,
        evaluator: MagneticFieldEvaluator,
        species: ParticleSpecies = ParticleSpecies.DEUTERIUM,
        electric_field_func: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None,
    ) -> None:
        self.evaluator = evaluator
        self.props = SpeciesProperties.from_species(species)
        self.electric_field_func = electric_field_func

    def step(self, state: ParticleState, dt: float) -> ParticleState:
        """
        Execute one Boris integration step of duration dt.
        """
        r_cyl = state.r_cylindrical
        phi_cyl = state.phi_cylindrical

        # Query magnetic field at current position
        mag_cyl = self.evaluator.evaluate_at(r_cyl, state.z)
        bx, by, bz = mag_cyl.to_cartesian(phi_cyl)

        # Query electric field if present, otherwise default to zero
        if self.electric_field_func is not None:
            ex, ey, ez = self.electric_field_func(state.x, state.y, state.z)
        else:
            ex, ey, ez = 0.0, 0.0, 0.0

        q_over_m = self.props.charge / self.props.mass
        half_dt = 0.5 * dt

        # 1. First half-step electric acceleration: v_minus = v_k + (q * dt / (2 * m)) * E
        vx_minus = state.vx + q_over_m * half_dt * ex
        vy_minus = state.vy + q_over_m * half_dt * ey
        vz_minus = state.vz + q_over_m * half_dt * ez

        # 2. Boris magnetic rotation
        # Rotation vector t = (q * dt / (2 * m)) * B
        tx = q_over_m * half_dt * bx
        ty = q_over_m * half_dt * by
        tz = q_over_m * half_dt * bz

        t_mag2 = tx * tx + ty * ty + tz * tz
        s_factor = 2.0 / (1.0 + t_mag2)
        sx = s_factor * tx
        sy = s_factor * ty
        sz = s_factor * tz

        # v_prime = v_minus + v_minus x t
        v_cross_t_x = vy_minus * tz - vz_minus * ty
        v_cross_t_y = vz_minus * tx - vx_minus * tz
        v_cross_t_z = vx_minus * ty - vy_minus * tx

        vx_prime = vx_minus + v_cross_t_x
        vy_prime = vy_minus + v_cross_t_y
        vz_prime = vz_minus + v_cross_t_z

        # v_plus = v_minus + v_prime x s
        v_cross_s_x = vy_prime * sz - vz_prime * sy
        v_cross_s_y = vz_prime * sx - vx_prime * sz
        v_cross_s_z = vx_prime * sy - vy_prime * sx

        vx_plus = vx_minus + v_cross_s_x
        vy_plus = vy_minus + v_cross_s_y
        vz_plus = vz_minus + v_cross_s_z

        # 3. Second half-step electric acceleration: v_{k+1} = v_plus + (q * dt / (2 * m)) * E
        vx_next = vx_plus + q_over_m * half_dt * ex
        vy_next = vy_plus + q_over_m * half_dt * ey
        vz_next = vz_plus + q_over_m * half_dt * ez

        # 4. Position update: x_{k+1} = x_k + v_{k+1} * dt
        x_next = state.x + vx_next * dt
        y_next = state.y + vy_next * dt
        z_next = state.z + vz_next * dt

        return ParticleState(
            x=x_next,
            y=y_next,
            z=z_next,
            vx=vx_next,
            vy=vy_next,
            vz=vz_next,
            time=state.time + dt,
        )

    def trace_trajectory(
        self,
        initial_state: ParticleState,
        dt: float,
        num_steps: int,
        stride: int = 1,
    ) -> List[ParticleState]:
        """Integrate trajectory over num_steps, recording every stride states."""
        history: List[ParticleState] = [initial_state]
        curr = initial_state
        for step_idx in range(1, num_steps + 1):
            curr = self.step(curr, dt)
            if step_idx % stride == 0:
                history.append(curr)
        return history

    def cyclotron_frequency(self, r: float, z: float) -> float:
        """Cyclotron gyro-frequency omega_c = |q| * |B| / m in radians/sec."""
        mag = self.evaluator.evaluate_at(r, z).magnitude
        return abs(self.props.charge) * mag / self.props.mass

    def larmor_radius(self, speed_perp: float, r: float, z: float) -> float:
        """Larmor gyration radius rho_L = m * v_perp / (|q| * |B|) in meters."""
        omega_c = self.cyclotron_frequency(r, z)
        return speed_perp / omega_c if omega_c > 1e-12 else 0.0

    def analyze_orbit_trapping(
        self,
        initial_state: ParticleState,
        r_axis: float,
        dt: float,
        num_steps: int,
    ) -> Tuple[bool, int, float, float]:
        """
        Analyze whether particle trajectory is trapped (banana orbit) or passing.

        Returns:
            (is_trapped, bounce_count, bounce_frequency_hz, banana_width_meters)
        """
        history = self.trace_trajectory(initial_state, dt, num_steps, stride=1)
        r_history = [s.r_cylindrical for s in history]

        # Inboard and outboard radial excursions
        r_min_val = min(r_history)
        r_max_val = max(r_history)
        banana_width = r_max_val - r_min_val

        # Detect turning points in parallel velocity or cylindrical vertical motion Z
        # A trapped banana particle reverses its toroidal / poloidal circulation direction
        vz_history = [s.vz for s in history]
        sign_changes = 0
        for i in range(1, len(vz_history)):
            if vz_history[i - 1] * vz_history[i] < 0.0:
                sign_changes += 1

        bounce_count = sign_changes // 2
        total_time = num_steps * dt
        bounce_freq = bounce_count / total_time if total_time > 0.0 else 0.0

        # Trapped particle if it bounces repeatedly and does not encompass the inboard axis
        # Passing particles circumnavigate R_axis, so r_min < R_axis < r_max
        is_passing = (r_min_val < r_axis) and (r_max_val > r_axis)
        is_trapped = (not is_passing) and (bounce_count >= 2)

        return is_trapped, bounce_count, bounce_freq, banana_width
