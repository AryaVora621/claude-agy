"""
NovaPhysics: Sequential Impulse Constraint Solver.
Solves velocity constraints, Coulomb friction cones, Baumgarte stabilization, and warm starting.
"""

import math
from typing import Dict, List, Optional, Tuple
from .math2d import Vec2
from .contact import ContactManifold, ContactPoint
from .body import RigidBody
from .joints import Joint

BAUMGARTE_FACTOR = 0.2
PENETRATION_SLOP = 0.005
RESTITUTION_THRESHOLD = 0.5


class ContactConstraint:
    """Solves normal non-penetration and tangential friction constraints for a manifold."""
    __slots__ = ("manifold", "body_a", "body_b", "normal", "tangent")

    def __init__(self, manifold: ContactManifold) -> None:
        self.manifold = manifold
        self.body_a = manifold.body_a
        self.body_b = manifold.body_b
        self.normal = manifold.normal
        # Tangent vector perpendicular to normal (t = (-ny, nx))
        self.tangent = Vec2(-self.normal.y, self.normal.x)

    def pre_solve(self, dt: float, inv_dt: float) -> None:
        """Precomputes constraint effective masses and applies warm-starting impulses."""
        b_a = self.body_a
        b_b = self.body_b
        normal = self.normal
        tangent = self.tangent

        for cp in self.manifold.points:
            # Vectors from center of mass to contact point in world space
            cp.r_a = cp.position - b_a.position
            cp.r_b = cp.position - b_b.position

            rn_a = cp.r_a.cross(normal)
            rn_b = cp.r_b.cross(normal)
            k_normal = b_a.inv_mass + b_b.inv_mass + (rn_a * rn_a * b_a.inv_inertia) + (rn_b * rn_b * b_b.inv_inertia)
            cp.normal_mass = 1.0 / k_normal if k_normal > 1e-9 else 0.0

            rt_a = cp.r_a.cross(tangent)
            rt_b = cp.r_b.cross(tangent)
            k_tangent = b_a.inv_mass + b_b.inv_mass + (rt_a * rt_a * b_a.inv_inertia) + (rt_b * rt_b * b_b.inv_inertia)
            cp.tangent_mass = 1.0 / k_tangent if k_tangent > 1e-9 else 0.0

            # Compute relative velocity for restitution velocity bias once in pre-solve
            v_a = b_a.get_velocity_at_world_point(cp.position)
            v_b = b_b.get_velocity_at_world_point(cp.position)
            v_rel = v_b - v_a
            v_normal = v_rel.dot(normal)

            if v_normal < -RESTITUTION_THRESHOLD:
                cp.velocity_bias = -self.manifold.restitution * v_normal
                cp.position_bias = 0.0
            else:
                cp.velocity_bias = 0.0
                overlap = max(0.0, cp.penetration - PENETRATION_SLOP)
                cp.position_bias = (BAUMGARTE_FACTOR * inv_dt) * overlap

            # Warm starting: apply accumulated impulses from previous frame
            p = normal * cp.normal_impulse + tangent * cp.tangent_impulse
            b_a.apply_impulse(-p, cp.position)
            b_b.apply_impulse(p, cp.position)

    def solve_velocity(self, dt: float, inv_dt: float) -> None:
        """Sequential impulse iteration for normal non-penetration and Coulomb friction."""
        b_a = self.body_a
        b_b = self.body_b
        normal = self.normal
        tangent = self.tangent

        for cp in self.manifold.points:
            # 1. Solve Friction (Tangential) Impulse First
            v_a = b_a.get_velocity_at_world_point(cp.position)
            v_b = b_b.get_velocity_at_world_point(cp.position)
            v_rel = v_b - v_a

            v_tangent = v_rel.dot(tangent)
            delta_lambda_t = -cp.tangent_mass * v_tangent

            # Coulomb friction cone limit: |lambda_t| <= mu * lambda_n
            max_friction = self.manifold.dynamic_friction * cp.normal_impulse
            new_lambda_t = max(-max_friction, min(max_friction, cp.tangent_impulse + delta_lambda_t))
            actual_delta_t = new_lambda_t - cp.tangent_impulse
            cp.tangent_impulse = new_lambda_t

            p_t = tangent * actual_delta_t
            b_a.apply_impulse(-p_t, cp.position)
            b_b.apply_impulse(p_t, cp.position)

            # 2. Solve Normal Non-Penetration Impulse
            v_a = b_a.get_velocity_at_world_point(cp.position)
            v_b = b_b.get_velocity_at_world_point(cp.position)
            v_rel = v_b - v_a
            v_normal = v_rel.dot(normal)

            delta_lambda_n = -cp.normal_mass * (v_normal - cp.velocity_bias - cp.position_bias)
            # Normal impulse can only push (lambda_n >= 0)
            new_lambda_n = max(0.0, cp.normal_impulse + delta_lambda_n)
            actual_delta_n = new_lambda_n - cp.normal_impulse
            cp.normal_impulse = new_lambda_n

            p_n = normal * actual_delta_n
            b_a.apply_impulse(-p_n, cp.position)
            b_b.apply_impulse(p_n, cp.position)


class SequentialImpulseSolver:
    """
    Projected Gauss-Seidel sequential impulse solver.
    Maintains contact history for warm starting and coordinates joint constraints.
    """
    __slots__ = ("contact_history", "warm_starting")

    def __init__(self, warm_starting: bool = True) -> None:
        self.warm_starting = warm_starting
        # Key: (id(body_a), id(body_b)), Value: list of (position, normal_impulse, tangent_impulse)
        self.contact_history: Dict[Tuple[int, int], List[Tuple[Vec2, float, float]]] = {}

    def solve(
        self,
        manifolds: List[ContactManifold],
        joints: List[Joint],
        dt: float,
        velocity_iterations: int = 8
    ) -> None:
        """Solves velocity constraints for contacts and joints over multiple iterations."""
        if dt <= 1e-9:
            return

        inv_dt = 1.0 / dt
        constraints: List[ContactConstraint] = []

        # 1. Initialize contact constraints and match warm starting impulses
        new_history: Dict[Tuple[int, int], List[Tuple[Vec2, float, float]]] = {}

        for m in manifolds:
            pair_key = (id(m.body_a), id(m.body_b))
            if self.warm_starting and pair_key in self.contact_history:
                old_points = self.contact_history[pair_key]
                for cp in m.points:
                    # Match by closest contact point
                    best_dist = 0.05  # Matching radius threshold
                    matched = None
                    for old_pos, old_ni, old_ti in old_points:
                        d = (cp.position - old_pos).length()
                        if d < best_dist:
                            best_dist = d
                            matched = (old_ni, old_ti)
                    if matched is not None:
                        cp.normal_impulse = matched[0]
                        cp.tangent_impulse = matched[1]

            cc = ContactConstraint(m)
            constraints.append(cc)

        # 2. Pre-solve step (compute effective masses and apply warm starting)
        for j in joints:
            j.pre_solve(dt, inv_dt)

        for c in constraints:
            c.pre_solve(dt, inv_dt)

        # 3. Gauss-Seidel velocity iterations
        for _ in range(velocity_iterations):
            for j in joints:
                j.solve_velocity(dt, inv_dt)

            for c in constraints:
                c.solve_velocity(dt, inv_dt)

        # 4. Record contact history for next frame
        for m in manifolds:
            pair_key = (id(m.body_a), id(m.body_b))
            records = [(cp.position, cp.normal_impulse, cp.tangent_impulse) for cp in m.points]
            new_history[pair_key] = records

        self.contact_history = new_history
