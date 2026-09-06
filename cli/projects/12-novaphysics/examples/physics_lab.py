#!/usr/bin/env python3
"""
NovaPhysics: Interactive Terminal Physics Laboratory.
Demonstrates 2D rigid body dynamics, GJK/EPA contact resolution, joints, cloth, and ANSI Braille rasterization.

Usage:
    python3 examples/physics_lab.py --scene jenga
    python3 examples/physics_lab.py --scene cradle
    python3 examples/physics_lab.py --scene double_pendulum
    python3 examples/physics_lab.py --scene cloth
    python3 examples/physics_lab.py --scene avalanche
    python3 examples/physics_lab.py --all --steps 50
"""

import os
import sys
import time
import math
import argparse
from typing import Tuple, Optional

# Add package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from novaphysics import (
    Vec2, Box, Circle, Polygon, RigidBody, BodyType, Material,
    World, DistanceJoint, RevoluteJoint, SpringJoint, ClothMesh,
    PhysicsRenderer
)


def create_jenga_scene() -> Tuple[World, PhysicsRenderer, Optional[ClothMesh]]:
    """Stable tower of stacked boxes under gravity."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=4, velocity_iterations=12, warm_starting=True)
    renderer = PhysicsRenderer(char_width=76, char_height=24, view_center=Vec2(0.0, 4.5), view_scale=9.0)

    # Static ground
    ground = RigidBody(Box(14.0, 1.0), position=Vec2(0.0, 0.0), body_type=BodyType.STATIC)
    world.add_body(ground)

    # 7 stacked boxes
    box_mat = Material(density=1.0, restitution=0.05, dynamic_friction=0.6)
    for i in range(7):
        b = RigidBody(Box(1.8, 0.8), position=Vec2(0.0, 0.9 + i * 0.85), material=box_mat)
        world.add_body(b)

    # A falling projectile ball hitting the upper half
    proj_mat = Material(density=2.0, restitution=0.4, dynamic_friction=0.3)
    proj = RigidBody(Circle(0.45), position=Vec2(-6.0, 4.0), material=proj_mat)
    proj.velocity = Vec2(8.0, 1.0)
    world.add_body(proj)

    return world, renderer, None


def create_cradle_scene() -> Tuple[World, PhysicsRenderer, Optional[ClothMesh]]:
    """Newton's Cradle with 5 steel balls on distance joints."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=4, velocity_iterations=14)
    renderer = PhysicsRenderer(char_width=76, char_height=24, view_center=Vec2(0.0, 4.0), view_scale=10.0)

    # Overhead ceiling
    ceiling = RigidBody(Box(10.0, 0.5), position=Vec2(0.0, 7.0), body_type=BodyType.STATIC)
    world.add_body(ceiling)

    ball_radius = 0.4
    steel = Material(density=2.0, restitution=0.98, dynamic_friction=0.05)

    for i in range(5):
        x = (i - 2) * (ball_radius * 2.0)
        # Pull leftmost ball back to initiate momentum transfer
        if i == 0:
            ball_pos = Vec2(x - 2.5, 5.5)
        else:
            ball_pos = Vec2(x, 3.5)

        ball = RigidBody(Circle(ball_radius), position=ball_pos, material=steel)
        world.add_body(ball)

        anchor = Vec2(x, 7.0)
        joint = DistanceJoint(ceiling, ball, anchor_a=anchor, anchor_b=ball_pos, distance=3.5)
        world.add_joint(joint)

    return world, renderer, None


def create_double_pendulum_scene() -> Tuple[World, PhysicsRenderer, Optional[ClothMesh]]:
    """Chaotic double pendulum linked via two revolute pin joints."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=4, velocity_iterations=12)
    renderer = PhysicsRenderer(char_width=76, char_height=24, view_center=Vec2(0.0, 3.5), view_scale=9.0)

    # Static base pin at (0, 6)
    base = RigidBody(Circle(0.2), position=Vec2(0.0, 6.0), body_type=BodyType.STATIC)
    world.add_body(base)

    # Beam 1
    beam1 = RigidBody(Box(2.6, 0.3), position=Vec2(1.3, 6.0))
    world.add_body(beam1)
    j1 = RevoluteJoint(base, beam1, pivot=Vec2(0.0, 6.0))
    world.add_joint(j1)

    # Beam 2
    beam2 = RigidBody(Box(2.4, 0.3), position=Vec2(3.8, 6.0))
    world.add_body(beam2)
    j2 = RevoluteJoint(beam1, beam2, pivot=Vec2(2.6, 6.0))
    world.add_joint(j2)

    return world, renderer, None


def create_cloth_scene() -> Tuple[World, PhysicsRenderer, Optional[ClothMesh]]:
    """10x8 cloth fluttering in turbulent wind with an obstacle."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=2)
    renderer = PhysicsRenderer(char_width=76, char_height=24, view_center=Vec2(0.0, 4.0), view_scale=9.0)

    cloth = ClothMesh(
        cols=10,
        rows=8,
        origin=Vec2(-2.25, 6.5),
        spacing=0.5,
        pin_corners=True,
        iterations=5
    )

    # Add a rigid sphere obstacle in world
    obstacle = RigidBody(Circle(1.0), position=Vec2(0.0, 3.8), body_type=BodyType.STATIC)
    world.add_body(obstacle)

    return world, renderer, cloth


def create_avalanche_scene() -> Tuple[World, PhysicsRenderer, Optional[ClothMesh]]:
    """Rigid bodies sliding down angled funnels into an accumulation container."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=3, velocity_iterations=10)
    renderer = PhysicsRenderer(char_width=76, char_height=24, view_center=Vec2(0.0, 4.5), view_scale=8.5)

    # Angled ramp 1 (left)
    ramp1 = RigidBody(Box(7.0, 0.5), position=Vec2(-3.0, 6.5), angle=-0.35, body_type=BodyType.STATIC)
    world.add_body(ramp1)

    # Angled ramp 2 (right)
    ramp2 = RigidBody(Box(7.0, 0.5), position=Vec2(3.0, 4.0), angle=0.35, body_type=BodyType.STATIC)
    world.add_body(ramp2)

    # Catching bin bottom and walls
    bin_bottom = RigidBody(Box(8.0, 0.6), position=Vec2(0.0, 0.5), body_type=BodyType.STATIC)
    bin_left = RigidBody(Box(0.6, 3.0), position=Vec2(-4.0, 1.8), body_type=BodyType.STATIC)
    bin_right = RigidBody(Box(0.6, 3.0), position=Vec2(4.0, 1.8), body_type=BodyType.STATIC)
    world.add_body(bin_bottom)
    world.add_body(bin_left)
    world.add_body(bin_right)

    # Falling assorted objects
    mat = Material(density=1.0, restitution=0.2, dynamic_friction=0.4)
    for i in range(8):
        pos = Vec2(-4.5 + (i % 3) * 0.8, 8.0 + i * 0.8)
        if i % 2 == 0:
            b = RigidBody(Circle(0.35), position=pos, material=mat)
        else:
            b = RigidBody(Box(0.7, 0.7), position=pos, material=mat)
        world.add_body(b)

    return world, renderer, None


def run_scene(scene_name: str, steps: int = 120, fps_delay: float = 0.0) -> None:
    scenes = {
        "jenga": create_jenga_scene,
        "cradle": create_cradle_scene,
        "double_pendulum": create_double_pendulum_scene,
        "cloth": create_cloth_scene,
        "avalanche": create_avalanche_scene
    }

    if scene_name not in scenes:
        print(f"Unknown scene: {scene_name}. Choices: {list(scenes.keys())}")
        return

    world, renderer, cloth = scenes[scene_name]()
    dt = 1.0 / 60.0

    print(f"\n\033[1;32m[NovaPhysics]\033[0m Running scene '{scene_name}' for {steps} steps...")

    for step in range(steps):
        # Interactive wind or tearing for cloth
        if cloth is not None:
            wind_x = 3.0 * math.sin(step * 0.1)
            cloth.set_wind(Vec2(wind_x, 0.5))
            cloth.step(dt)
            cloth.apply_obstacle_sphere(Vec2(0.0, 3.8), 1.0)
            # Tear cloth at step 60
            if step == 60:
                cloth.tear_at(Vec2(0.0, 5.0), radius=0.8)

        world.step(dt)

        # Print rendered frame at intervals or live
        if fps_delay > 0.0:
            print("\033[H", end="")  # Move cursor to home
            frame = renderer.render_world(world, cloth)
            print(frame)
            time.sleep(fps_delay)
        elif step == 0 or step == steps // 2 or step == steps - 1:
            frame = renderer.render_world(world, cloth)
            print(f"\n--- Snapshot at Step {step} (t = {world.time:.2f}s) ---")
            print(frame)

    print(f"\033[1;32m[NovaPhysics]\033[0m Scene '{scene_name}' simulation completed successfully.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaPhysics: 2D Rigid Body Dynamics & Constraint Solver Lab")
    parser.add_argument("--scene", choices=["jenga", "cradle", "double_pendulum", "cloth", "avalanche"], default="jenga")
    parser.add_argument("--all", action="store_true", help="Run all scenes sequentially")
    parser.add_argument("--steps", type=int, default=60, help="Number of physics steps per scene")
    parser.add_argument("--live", action="store_true", help="Run animated live view in terminal")

    args = parser.parse_args()
    fps_delay = 0.03 if args.live else 0.0

    if args.all:
        for s in ["jenga", "cradle", "double_pendulum", "cloth", "avalanche"]:
            run_scene(s, steps=args.steps, fps_delay=fps_delay)
    else:
        run_scene(args.scene, steps=args.steps, fps_delay=fps_delay)


if __name__ == "__main__":
    main()
