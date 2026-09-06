#!/usr/bin/env python3
"""
VeloSLAM: Autonomous Robotics & SLAM Interactive Laboratory.
Demonstrates:
  1. Simultaneous Localization and Mapping (EKF-SLAM) with landmark tracking and covariance ellipses.
  2. 2D Probabilistic Log-Odds Occupancy Grid Mapping with Bresenham raycasting.
  3. Analytical Dubins Paths and Kinodynamic Hybrid A* non-holonomic global motion planning.
  4. Real-time Dynamic Window Approach (DWA) local collision avoidance.
  5. High-resolution sub-pixel Unicode Braille terminal map renderer.
"""

import math
import random
import time
from typing import List, Tuple
from veloslam.linalg import normalize_angle
from veloslam.ekf_slam import EKFSLAM
from veloslam.occupancy import OccupancyGrid, LaserScan
from veloslam.dubins import dubins_shortest_path
from veloslam.hybrid_astar import HybridAStar, Pose2D
from veloslam.dwa import DWAPlanner, DWAConfig, RobotState
from veloslam.visualizer import render_braille_map


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def demo_ekf_slam_and_occupancy():
    print_banner("1. EKF-SLAM & Occupancy Grid Mapping Simulation")

    # True world landmarks
    true_landmarks = [
        (3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, -3.0),
        (3.0, -3.0),
        (5.0, 0.0),
        (-5.0, 0.0),
        (0.0, 5.0),
        (0.0, -5.0),
    ]

    # Initialize EKF-SLAM and Occupancy Grid
    slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)
    grid = OccupancyGrid(width_m=16.0, height_m=16.0, resolution=0.2, origin_x=-8.0, origin_y=-8.0)

    # Insert true obstacles into grid for LiDAR simulation
    for lx, ly in true_landmarks:
        gx, gy = grid.world_to_grid(lx, ly)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if grid.is_valid_grid(gx + dx, gy + dy):
                    grid.cells[grid._get_index(gx + dx, gy + dy)] = grid.l_max

    # True robot ground-truth pose
    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    dt = 0.1
    v = 0.8
    omega = 0.25  # Drive in circle

    print("Simulating circular trajectory with noisy sensors (50 steps)...")

    for step in range(50):
        # 1. Kinematic Ground Truth update
        true_x += v * dt * math.cos(true_theta)
        true_y += v * dt * math.sin(true_theta)
        true_theta = normalize_angle(true_theta + omega * dt)

        # 2. Control input with noise for odometry
        noisy_v = v + random.gauss(0.0, 0.03)
        noisy_w = omega + random.gauss(0.0, 0.015)

        # EKF Predict
        slam.predict(noisy_v, noisy_w, dt)

        # 3. Simulate LiDAR landmark observations
        observations: List[Tuple[float, float]] = []
        for lx, ly in true_landmarks:
            dx = lx - true_x
            dy = ly - true_y
            dist = math.hypot(dx, dy)
            if dist < 6.0:  # Sensor max range
                true_bearing = normalize_angle(math.atan2(dy, dx) - true_theta)
                # Add sensor noise
                z_r = dist + random.gauss(0.0, 0.08)
                z_phi = normalize_angle(true_bearing + random.gauss(0.0, 0.02))
                observations.append((z_r, z_phi))

        # EKF Update
        if observations:
            slam.update(observations)

    est_pose = slam.get_robot_pose()
    pos_error = math.hypot(est_pose[0] - true_x, est_pose[1] - true_y)
    ang_error = abs(normalize_angle(est_pose[2] - true_theta))

    print(f"\nGround Truth Pose:  x={true_x:+.3f}m, y={true_y:+.3f}m, theta={math.degrees(true_theta):+.1f} deg")
    print(f"EKF Estimated Pose: x={est_pose[0]:+.3f}m, y={est_pose[1]:+.3f}m, theta={math.degrees(est_pose[2]):+.1f} deg")
    print(f"Estimation Errors:  Position={pos_error:.3f}m, Heading={math.degrees(ang_error):.2f} deg")
    print(f"Tracked Landmarks:  {slam.num_landmarks} landmarks discovered and mapped")

    # Render Braille Map
    lm_data = slam.get_landmarks()
    print("\nTerminal Braille Map (Green=Robot, Orange=Landmarks/Ellipses, Gray=Obstacles):")
    canvas = render_braille_map(
        grid=grid,
        robot_pose=est_pose,
        landmarks=lm_data,
        width_chars=60,
        height_chars=20
    )
    print(canvas)


def demo_motion_planning_and_dwa():
    print_banner("2. Kinodynamic Hybrid A* & Reactive DWA Local Navigation")

    # Environment setup
    grid = OccupancyGrid(width_m=20.0, height_m=16.0, resolution=0.25, origin_x=-10.0, origin_y=-8.0)

    # Obstacle wall with a narrow gap
    for y_m in [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        gx, gy = grid.world_to_grid(0.0, y_m)
        for dx in range(-1, 2):
            if grid.is_valid_grid(gx + dx, gy):
                grid.cells[grid._get_index(gx + dx, gy)] = grid.l_max

    # Global Kinodynamic Planning
    start_pose = Pose2D(-6.0, 0.0, 0.0)
    goal_pose = Pose2D(6.0, 0.0, 0.0)

    print(f"Planning global path from ({start_pose.x}, {start_pose.y}) to ({goal_pose.x}, {goal_pose.y})...")
    planner = HybridAStar(grid, min_turning_radius=1.2, step_size=0.4)
    t0 = time.perf_counter()
    global_path = planner.plan(start_pose, goal_pose)
    t_plan = (time.perf_counter() - t0) * 1000

    if global_path:
        print(f"Hybrid A* Path Found: {len(global_path)} waypoints in {t_plan:.2f} ms")
    else:
        print("Falling back to analytical Dubins path...")
        dub = dubins_shortest_path(start_pose.to_tuple(), goal_pose.to_tuple(), rho=1.2)
        global_path = dub.sample_path(step_size=0.2)

    # Local DWA Reactive Control simulation
    dwa = DWAPlanner(DWAConfig(max_speed=1.0, robot_radius=0.3))
    robot_state = RobotState(x=start_pose.x, y=start_pose.y, theta=start_pose.theta)

    target_idx = 0
    executed_traj: List[Tuple[float, float, float]] = []

    print("\nExecuting DWA local tracking...")
    for cycle in range(60):
        executed_traj.append(robot_state.pose_tuple())
        # Find lookahead target along global path
        if global_path and target_idx < len(global_path):
            while target_idx < len(global_path) - 1:
                wp = global_path[target_idx]
                if math.hypot(wp[0] - robot_state.x, wp[1] - robot_state.y) > 1.2:
                    break
                target_idx += 1
            local_goal = (global_path[target_idx][0], global_path[target_idx][1])
        else:
            local_goal = (goal_pose.x, goal_pose.y)

        # Plan DWA velocity
        cmd_v, cmd_w, _ = dwa.plan(robot_state, local_goal, grid)

        # Step forward
        robot_state.theta = normalize_angle(robot_state.theta + cmd_w * 0.1)
        robot_state.x += cmd_v * math.cos(robot_state.theta) * 0.1
        robot_state.y += cmd_v * math.sin(robot_state.theta) * 0.1
        robot_state.v = cmd_v
        robot_state.omega = cmd_w

        # Check goal reached
        if math.hypot(robot_state.x - goal_pose.x, robot_state.y - goal_pose.y) < 0.4:
            print(f"Goal successfully reached at cycle {cycle}! Final distance: {math.hypot(robot_state.x - goal_pose.x, robot_state.y - goal_pose.y):.3f}m")
            break

    # Visualizer
    print("\nTerminal Braille Map (Magenta=Planned Path, Green=Current Robot Pose, Gray=Obstacles):")
    canvas = render_braille_map(
        grid=grid,
        robot_pose=robot_state.pose_tuple(),
        path=global_path,
        width_chars=60,
        height_chars=20
    )
    print(canvas)


def main():
    print("=" * 70)
    print("      VELOSLAM: AUTONOMOUS ROBOTICS & SLAM SHOWCASE LAB")
    print("   First-Principles Python Standard Library SLAM & Motion Planning")
    print("=" * 70)
    demo_ekf_slam_and_occupancy()
    demo_motion_planning_and_dwa()
    print("\n" + "=" * 70)
    print("      LABORATORY DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
