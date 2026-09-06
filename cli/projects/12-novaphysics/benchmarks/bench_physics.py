#!/usr/bin/env python3
"""
NovaPhysics: Performance Benchmark Suite.
Measures vector math throughput, Dynamic AABB tree operations, GJK/EPA queries,
full rigid body world steps, and cloth relaxation iterations.
"""

import os
import sys
import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from novaphysics import (
    Vec2, Box, Circle, Polygon, RigidBody, BodyType, Material,
    World, DynamicAABBTree, AABB, gjk_intersect, epa_penetration,
    ClothMesh
)


def bench_vector_math(iterations: int = 500_000) -> float:
    """Benchmark raw 2D vector algebra operations."""
    v1 = Vec2(1.23, 4.56)
    v2 = Vec2(7.89, -2.34)

    start = time.perf_counter()
    for _ in range(iterations):
        v3 = v1 + v2
        d = v1.dot(v2)
        c = v1.cross(v2)
        v4 = v1.rotated(0.01)
    elapsed = time.perf_counter() - start

    ops_per_sec = (iterations * 4) / elapsed
    print(f"  Vector Math (dot, cross, add, rotate): {iterations * 4:,} ops in {elapsed:.3f}s -> {ops_per_sec:,.0f} ops/sec")
    return ops_per_sec


def bench_broadphase(num_bodies: int = 100, iterations: int = 2_000) -> float:
    """Benchmark Dynamic AABB Tree insertions, queries, and pair pruning."""
    tree = DynamicAABBTree(fat_margin=0.2)
    nodes = []

    # Insert 100 boxes in a 10x10 grid
    for i in range(10):
        for j in range(10):
            box = AABB(Vec2(i * 2.0, j * 2.0), Vec2(i * 2.0 + 1.0, j * 2.0 + 1.0))
            nid = tree.insert(box, f"body_{i}_{j}")
            nodes.append(nid)

    start = time.perf_counter()
    for it in range(iterations):
        # Update a moving query box
        q_box = AABB(Vec2(it % 15, (it * 2) % 15), Vec2(it % 15 + 2.5, (it * 2) % 15 + 2.5))
        overlaps = tree.query(q_box)
    elapsed = time.perf_counter() - start

    queries_per_sec = iterations / elapsed
    print(f"  Broadphase BVH Queries (100 leaves):  {iterations:,} queries in {elapsed:.3f}s -> {queries_per_sec:,.0f} queries/sec")
    return queries_per_sec


def bench_gjk_epa(iterations: int = 20_000) -> float:
    """Benchmark convex polygon-polygon GJK intersection and EPA penetration tests."""
    b1 = RigidBody(Box(2.0, 2.0), position=Vec2(0.0, 0.0), angle=0.2)
    b2 = RigidBody(Box(2.0, 2.0), position=Vec2(1.5, 0.4), angle=-0.3)

    start = time.perf_counter()
    for _ in range(iterations):
        colliding, simplex = gjk_intersect(b1, b2)
        if colliding:
            pen, normal = epa_penetration(b1, b2, simplex)
    elapsed = time.perf_counter() - start

    tests_per_sec = iterations / elapsed
    print(f"  GJK + EPA Convex Narrowphase:          {iterations:,} tests in {elapsed:.3f}s -> {tests_per_sec:,.0f} tests/sec")
    return tests_per_sec


def bench_rigid_body_world(num_bodies: int = 40, steps: int = 500) -> float:
    """Benchmark complete physics world step (integration, broadphase, narrowphase, solver)."""
    world = World(gravity=Vec2(0.0, -9.81), sub_steps=2, velocity_iterations=8, warm_starting=True)

    # Static ground and walls
    ground = RigidBody(Box(20.0, 1.0), position=Vec2(0.0, 0.0), body_type=BodyType.STATIC)
    wall_l = RigidBody(Box(1.0, 10.0), position=Vec2(-10.0, 5.0), body_type=BodyType.STATIC)
    wall_r = RigidBody(Box(1.0, 10.0), position=Vec2(10.0, 5.0), body_type=BodyType.STATIC)
    world.add_body(ground)
    world.add_body(wall_l)
    world.add_body(wall_r)

    # 40 dynamic bodies (mixed boxes and circles)
    for i in range(num_bodies):
        pos = Vec2(-6.0 + (i % 6) * 2.0, 2.0 + (i // 6) * 1.5)
        if i % 2 == 0:
            b = RigidBody(Circle(0.4), position=pos)
        else:
            b = RigidBody(Box(0.8, 0.8), position=pos)
        world.add_body(b)

    dt = 1.0 / 60.0
    start = time.perf_counter()
    for _ in range(steps):
        world.step(dt)
    elapsed = time.perf_counter() - start

    steps_per_sec = steps / elapsed
    body_steps_per_sec = (steps * num_bodies) / elapsed
    print(f"  Full World Step ({num_bodies} bodies):         {steps} steps in {elapsed:.3f}s -> {steps_per_sec:,.0f} steps/sec ({body_steps_per_sec:,.0f} body*steps/sec)")
    return steps_per_sec


def bench_cloth_simulation(cols: int = 10, rows: int = 8, steps: int = 500) -> float:
    """Benchmark Position-Based Dynamics (PBD) cloth mesh simulation."""
    cloth = ClothMesh(cols=cols, rows=rows, origin=Vec2(0.0, 5.0), spacing=0.4, iterations=4)
    dt = 1.0 / 60.0

    start = time.perf_counter()
    for _ in range(steps):
        cloth.step(dt)
    elapsed = time.perf_counter() - start

    steps_per_sec = steps / elapsed
    particle_steps_per_sec = (steps * cols * rows) / elapsed
    print(f"  PBD Cloth Mesh ({cols*rows} particles):       {steps} steps in {elapsed:.3f}s -> {steps_per_sec:,.0f} steps/sec ({particle_steps_per_sec:,.0f} part*steps/sec)")
    return steps_per_sec


def run_all_benchmarks() -> None:
    print("================================================================================")
    print("           NovaPhysics Engine: High-Performance Benchmarks                     ")
    print("================================================================================")
    bench_vector_math()
    bench_broadphase()
    bench_gjk_epa()
    bench_rigid_body_world()
    bench_cloth_simulation()
    print("================================================================================")


if __name__ == "__main__":
    run_all_benchmarks()
