"""
NovaPhysics: Physics World Simulation Manager.
Coordinates symplectic integration, broadphase pruning, narrowphase dispatch, and constraint solving.
"""

from typing import List, Optional, Tuple, Set
from .math2d import Vec2
from .body import RigidBody, BodyType
from .shapes import AABB
from .broadphase import DynamicAABBTree
from .contact import ContactManifold, find_collision
from .joints import Joint
from .solver import SequentialImpulseSolver


class World:
    """
    Simulates a 2D physics universe with rigid bodies, joints, and contact dynamics.
    Supports symplectic integration and sub-stepping for extreme numerical stability.
    """
    __slots__ = (
        "gravity", "bodies", "joints", "broadphase",
        "solver", "sub_steps", "velocity_iterations",
        "body_tree_ids", "_step_count", "_time"
    )

    def __init__(
        self,
        gravity: Vec2 = Vec2(0.0, -9.81),
        sub_steps: int = 1,
        velocity_iterations: int = 8,
        warm_starting: bool = True
    ) -> None:
        self.gravity = gravity
        self.sub_steps = max(1, sub_steps)
        self.velocity_iterations = max(1, velocity_iterations)
        self.bodies: List[RigidBody] = []
        self.joints: List[Joint] = []
        self.broadphase = DynamicAABBTree(fat_margin=0.1)
        self.solver = SequentialImpulseSolver(warm_starting=warm_starting)
        self.body_tree_ids: dict = {}
        self._step_count = 0
        self._time = 0.0

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def time(self) -> float:
        return self._time

    def add_body(self, body: RigidBody) -> RigidBody:
        """Adds a rigid body to the simulation world."""
        if body not in self.bodies:
            self.bodies.append(body)
            # Dynamic and Kinematic bodies are added to broadphase; static bodies also added so dynamics can hit them
            aabb = body.get_aabb()
            node_id = self.broadphase.insert(aabb, body)
            self.body_tree_ids[id(body)] = node_id
        return body

    def remove_body(self, body: RigidBody) -> None:
        """Removes a rigid body and its attached joints from the world."""
        if body in self.bodies:
            self.bodies.remove(body)
            body_id = id(body)
            if body_id in self.body_tree_ids:
                node_id = self.body_tree_ids.pop(body_id)
                self.broadphase.remove(node_id)

            # Remove associated joints
            self.joints = [j for j in self.joints if j.body_a != body and j.body_b != body]

    def add_joint(self, joint: Joint) -> Joint:
        """Adds a mechanical constraint joint between bodies."""
        if joint not in self.joints:
            self.joints.append(joint)
        return joint

    def remove_joint(self, joint: Joint) -> None:
        """Removes a joint from the simulation."""
        if joint in self.joints:
            self.joints.remove(joint)

    def step(self, dt: float) -> None:
        """Advances physics simulation by dt seconds, split across sub_steps."""
        if dt <= 1e-9:
            return

        sub_dt = dt / float(self.sub_steps)
        for _ in range(self.sub_steps):
            self._single_step(sub_dt)

        self._step_count += 1
        self._time += dt

    def _single_step(self, dt: float) -> None:
        """Executes a single symplectic integration step."""
        # 1. Integrate forces and accelerations -> velocities
        for body in self.bodies:
            if body.body_type == BodyType.DYNAMIC:
                body.integrate_forces(self.gravity, dt)

        # 2. Broadphase collision detection
        candidate_pairs = self.broadphase.query_pairs()

        # Filter out disabled pairs (e.g. static-static, or connected joints where collide_connected is False)
        connected_pairs: Set[Tuple[int, int]] = set()
        for j in self.joints:
            if not j.collide_connected:
                id_a, id_b = id(j.body_a), id(j.body_b)
                connected_pairs.add((min(id_a, id_b), max(id_a, id_b)))

        # 3. Narrowphase collision detection
        manifolds: List[ContactManifold] = []
        for body_a, body_b in candidate_pairs:
            # Skip two static/kinematic bodies
            if body_a.body_type != BodyType.DYNAMIC and body_b.body_type != BodyType.DYNAMIC:
                continue

            # Skip connected joint bodies if disabled
            id_a, id_b = id(body_a), id(body_b)
            if (min(id_a, id_b), max(id_a, id_b)) in connected_pairs:
                continue

            manifold = find_collision(body_a, body_b)
            if manifold is not None and len(manifold.points) > 0:
                manifolds.append(manifold)

        # 4. Constraint and impulse solve
        self.solver.solve(manifolds, self.joints, dt, velocity_iterations=self.velocity_iterations)

        # 5. Integrate velocities -> positions (Symplectic Euler)
        for body in self.bodies:
            if body.body_type != BodyType.STATIC:
                body.integrate_velocities(dt)
                # Update broadphase tree
                body_id = id(body)
                if body_id in self.body_tree_ids:
                    node_id = self.body_tree_ids[body_id]
                    self.broadphase.update(node_id, body.get_aabb())

        # 6. Clear accumulated external forces
        for body in self.bodies:
            body.clear_forces()
