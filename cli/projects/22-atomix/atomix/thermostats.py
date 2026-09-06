"""
Atomix: Statistical Mechanics Ensembles, Thermostats, and Barostats.
Implements:
  - Maxwell-Boltzmann velocity sampling with zero net momentum
  - Berendsen weak-coupling thermostat (canonical NVT)
  - Andersen stochastic collision thermostat
  - Nosé-Hoover extended phase space dynamical thermostat
  - Berendsen isotropic barostat (isothermal-isobaric NPT)
Zero external dependencies.
"""

from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple

from atomix.types import Vector3D, Atom, SimulationBox, KB_REDUCED


def initialize_maxwell_boltzmann_velocities(
    atoms: List[Atom],
    target_temperature: float,
    kb: float = KB_REDUCED,
    remove_drift: bool = True,
    seed: Optional[int] = 42,
) -> None:
    """
    Sample atomic velocities from the Maxwell-Boltzmann distribution at target_temperature.
    Removes center-of-mass linear momentum and rescales exactly to match target T.
    """
    if target_temperature <= 0.0:
        for a in atoms:
            a.velocity = Vector3D(0.0, 0.0, 0.0)
        return

    rng = random.Random(seed)

    # Box-Muller Gaussian sampling for each Cartesian component
    for atom in atoms:
        sigma_v = math.sqrt(kb * target_temperature / atom.mass)

        # Generate 3 independent standard normal variates
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        u3 = max(1e-12, rng.random())
        u4 = rng.random()

        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        z2 = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)

        atom.velocity = Vector3D(z0 * sigma_v, z1 * sigma_v, z2 * sigma_v)

    # Remove net center-of-mass linear momentum
    if remove_drift and len(atoms) > 1:
        total_mass = sum(a.mass for a in atoms)
        p_cm_x = sum(a.mass * a.velocity.x for a in atoms) / total_mass
        p_cm_y = sum(a.mass * a.velocity.y for a in atoms) / total_mass
        p_cm_z = sum(a.mass * a.velocity.z for a in atoms) / total_mass
        v_drift = Vector3D(p_cm_x, p_cm_y, p_cm_z)

        for a in atoms:
            a.velocity = a.velocity - v_drift

    # Exact kinetic temperature normalization
    n_atoms = len(atoms)
    n_dofs = max(1, 3 * n_atoms - 3) if remove_drift else 3 * n_atoms
    curr_ek = 0.5 * sum(a.mass * a.velocity.norm_sq() for a in atoms)
    curr_temp = (2.0 * curr_ek) / (n_dofs * kb)

    if curr_temp > 1e-12:
        scale = math.sqrt(target_temperature / curr_temp)
        for a in atoms:
            a.velocity = a.velocity * scale


class BerendsenThermostat:
    """
    Berendsen weak-coupling thermostat (Canonical NVT ensemble).
    Exponentially relaxes the system towards target_temperature with time constant tau_t.
    lambda = sqrt(1 + (dt / tau_t) * (T_target / T_curr - 1))
    """
    def __init__(
        self,
        target_temperature: float,
        tau_t: float = 0.1,
        kb: float = KB_REDUCED,
    ) -> None:
        if target_temperature < 0.0:
            raise ValueError(f"Target temperature must be non-negative, got {target_temperature}")
        if tau_t <= 0.0:
            raise ValueError(f"Thermostat coupling constant tau_t must be positive, got {tau_t}")
        self.target_temperature = float(target_temperature)
        self.tau_t = float(tau_t)
        self.kb = float(kb)

    def apply(
        self,
        atoms: List[Atom],
        dt: float,
        current_temperature: float,
    ) -> float:
        """
        Rescale atomic velocities. Returns the applied scaling factor lambda.
        """
        if current_temperature < 1e-6 or self.target_temperature < 1e-6:
            return 1.0

        ratio = self.target_temperature / current_temperature
        term = 1.0 + (dt / self.tau_t) * (ratio - 1.0)
        # Guard against unphysical negative values during extreme shocks
        term = max(0.2, min(5.0, term))
        scale = math.sqrt(term)

        # Clamp scaling factor to avoid destabilizing dynamics
        scale = max(0.8, min(1.25, scale))

        for atom in atoms:
            atom.velocity = atom.velocity * scale

        return scale


class AndersenThermostat:
    """
    Andersen stochastic collision thermostat.
    Simulates random impulsive collisions with a fictional heat bath at collision rate nu.
    """
    def __init__(
        self,
        target_temperature: float,
        collision_rate: float = 1.0,
        kb: float = KB_REDUCED,
        seed: Optional[int] = None,
    ) -> None:
        self.target_temperature = float(target_temperature)
        self.collision_rate = float(collision_rate)
        self.kb = float(kb)
        self.rng = random.Random(seed)

    def apply(self, atoms: List[Atom], dt: float) -> int:
        """
        Randomly re-draw velocities of collided atoms. Returns number of collisions.
        """
        prob_collision = 1.0 - math.exp(-self.collision_rate * dt)
        collisions = 0

        for atom in atoms:
            if self.rng.random() < prob_collision:
                collisions += 1
                sigma_v = math.sqrt(self.kb * self.target_temperature / atom.mass)

                u1 = max(1e-12, self.rng.random())
                u2 = self.rng.random()
                u3 = max(1e-12, self.rng.random())
                u4 = self.rng.random()

                z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                z2 = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)

                atom.velocity = Vector3D(z0 * sigma_v, z1 * sigma_v, z2 * sigma_v)

        return collisions


class NoseHooverThermostat:
    """
    Nosé-Hoover dynamical thermostat with extended phase space friction variable xi.
    Samples the true canonical (NVT) ensemble in phase space.
    d(xi)/dt = (1 / Q) * (sum(m_i * v_i^2) - N_df * kb * T_0)
    """
    def __init__(
        self,
        target_temperature: float,
        tau_nh: float = 0.2,
        kb: float = KB_REDUCED,
    ) -> None:
        self.target_temperature = float(target_temperature)
        self.tau_nh = float(tau_nh)
        self.kb = float(kb)
        self.xi = 0.0          # Friction coefficient
        self.thermostat_energy = 0.0

    def step(
        self,
        atoms: List[Atom],
        dt: float,
        current_kinetic_energy: float,
    ) -> None:
        """
        Advance the Nosé-Hoover friction variable and scale velocities.
        """
        n_atoms = len(atoms)
        n_dofs = max(1, 3 * n_atoms - 3)
        target_ke = 0.5 * n_dofs * self.kb * self.target_temperature

        # Thermostat inertial mass Q = N_df * kb * T0 * tau^2
        q_mass = n_dofs * self.kb * self.target_temperature * (self.tau_nh * self.tau_nh)
        if q_mass < 1e-12:
            return

        # Acceleration of friction variable: d(xi)/dt = (2*Ke - 2*target_ke) / Q
        xi_dot = (2.0 * current_kinetic_energy - 2.0 * target_ke) / q_mass

        # Update friction coefficient
        self.xi += xi_dot * dt
        # Clamp xi to prevent numerical divergence
        self.xi = max(-10.0, min(10.0, self.xi))

        # Velocity damping factor: exp(-xi * dt)
        decay = math.exp(-self.xi * dt)
        decay = max(0.5, min(1.5, decay))

        for atom in atoms:
            atom.velocity = atom.velocity * decay


class BerendsenBarostat:
    """
    Berendsen isotropic barostat (Isothermal-Isobaric NPT ensemble).
    Adjusts simulation box volume and atomic coordinates to maintain target pressure.
    mu = [1 - beta * (dt / tau_p) * (P_target - P_curr)]^(1/3)
    """
    def __init__(
        self,
        target_pressure: float = 0.0,
        tau_p: float = 1.0,
        compressibility: float = 4.5e-5,
    ) -> None:
        self.target_pressure = float(target_pressure)
        self.tau_p = float(tau_p)
        self.compressibility = float(compressibility)

    def apply(
        self,
        atoms: List[Atom],
        box: SimulationBox,
        dt: float,
        current_pressure: float,
    ) -> float:
        """
        Scale box dimensions and atomic coordinates to regulate pressure.
        Returns the linear scaling factor mu.
        """
        delta_p = self.target_pressure - current_pressure
        # Volume scale factor: 1 - beta * (dt / tau_p) * (P_target - P_curr)
        term = 1.0 - self.compressibility * (dt / self.tau_p) * delta_p
        term = max(0.9, min(1.1, term))

        mu = term ** (1.0 / 3.0)

        # Scale box dimensions
        box.scale(mu)

        # Scale atomic coordinates with respect to box center
        for atom in atoms:
            atom.position = Vector3D(
                atom.position.x * mu,
                atom.position.y * mu,
                atom.position.z * mu,
            )

        return mu
