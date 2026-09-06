"""
VeloSLAM: Comprehensive Performance Benchmark Suite.
Measures execution throughput and latency across linear algebra, EKF-SLAM state estimation,
probabilistic occupancy mapping, Dubins analytical curves, Hybrid A*, and DWA local planning.
"""

import time
import math
import random
from veloslam.linalg import Matrix, Vector, mahalanobis_distance
from veloslam.ekf_slam import EKFSLAM
from veloslam.occupancy import OccupancyGrid, LaserScan
from veloslam.dubins import dubins_shortest_path
from veloslam.hybrid_astar import HybridAStar, Pose2D
from veloslam.dwa import DWAPlanner, DWAConfig, RobotState


def bench_matrix_multiplication():
    print("\n--- 1. Linear Algebra & Matrix Multiplication ---")
    size = 20
    a = Matrix([[random.random() for _ in range(size)] for _ in range(size)])
    b = Matrix([[random.random() for _ in range(size)] for _ in range(size)])

    iterations = 2000
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
    elapsed = time.perf_counter() - t0
    ops_sec = iterations / elapsed
    print(f"20x20 Matrix Multiply: {iterations} iterations in {elapsed*1000:.2f} ms ({ops_sec:,.0f} mults/sec)")


def bench_ekf_slam():
    print("\n--- 2. EKF-SLAM State Estimation & Landmark Scalability ---")
    slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)

    # Initialize 10 landmarks
    obs = [(5.0 + i * 0.5, 0.1 * i - 0.5) for i in range(10)]
    slam.update(obs)

    iterations = 2000
    t0 = time.perf_counter()
    for _ in range(iterations):
        slam.predict(v=1.0, omega=0.05, dt=0.05)
    elapsed_predict = time.perf_counter() - t0
    predicts_sec = iterations / elapsed_predict
    print(f"EKF-SLAM Predict (10 landmarks): {iterations} steps in {elapsed_predict*1000:.2f} ms ({predicts_sec:,.0f} predicts/sec)")

    update_steps = 200
    t0 = time.perf_counter()
    for _ in range(update_steps):
        slam.update([(5.2, 0.12)])
    elapsed_update = time.perf_counter() - t0
    updates_sec = update_steps / elapsed_update
    print(f"EKF-SLAM Update: {update_steps} observation steps in {elapsed_update*1000:.2f} ms ({updates_sec:,.0f} updates/sec)")


def bench_occupancy_grid():
    print("\n--- 3. Probabilistic Occupancy Grid & Bresenham Raycasting ---")
    grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution=0.1, origin_x=-10.0, origin_y=-10.0)
    num_beams = 180
    ranges = [random.uniform(2.0, 8.0) for _ in range(num_beams)]
    scan = LaserScan(
        ranges=ranges,
        angle_min=-math.pi / 2.0,
        angle_max=math.pi / 2.0,
        range_min=0.1,
        range_max=10.0
    )

    scans = 200
    t0 = time.perf_counter()
    for _ in range(scans):
        grid.update_with_scan((0.0, 0.0, 0.0), scan)
    elapsed = time.perf_counter() - t0
    rays_sec = (scans * num_beams) / elapsed
    print(f"Occupancy Raycasting: {scans * num_beams} rays across 200x200 grid in {elapsed*1000:.2f} ms ({rays_sec:,.0f} rays/sec)")


def bench_dubins_curves():
    print("\n--- 4. Analytical Dubins Shortest-Path Solver ---")
    iterations = 20000
    rho = 1.2
    queries = [
        (
            (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-math.pi, math.pi)),
            (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-math.pi, math.pi)),
        )
        for _ in range(iterations)
    ]

    t0 = time.perf_counter()
    for start, goal in queries:
        _ = dubins_shortest_path(start, goal, rho)
    elapsed = time.perf_counter() - t0
    evals_sec = iterations / elapsed
    print(f"Dubins Analytical Solver: {iterations} paths solved in {elapsed*1000:.2f} ms ({evals_sec:,.0f} paths/sec)")


def bench_hybrid_astar():
    print("\n--- 5. Kinodynamic Hybrid A* Motion Planning ---")
    grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution=0.4, origin_x=-10.0, origin_y=-10.0)
    # Add scattered obstacles
    for ox in [-2.0, 0.0, 2.0]:
        for oy in [-2.0, 0.0, 2.0]:
            gx, gy = grid.world_to_grid(ox, oy)
            grid.cells[grid._get_index(gx, gy)] = grid.l_max

    planner = HybridAStar(grid, min_turning_radius=1.0, step_size=0.5, analytic_shot_freq=4)

    start = Pose2D(-6.0, -6.0, 0.0)
    goal = Pose2D(6.0, 6.0, math.pi / 2.0)

    plans = 20
    t0 = time.perf_counter()
    for _ in range(plans):
        path = planner.plan(start, goal, max_iterations=800)
    elapsed = time.perf_counter() - t0
    plans_sec = plans / elapsed
    avg_latency = (elapsed / plans) * 1000
    print(f"Hybrid A* (Non-Holonomic 3D): {plans} plans in {elapsed*1000:.2f} ms (avg {avg_latency:.2f} ms/plan, {plans_sec:.1f} plans/sec)")


def bench_dwa_local_planner():
    print("\n--- 6. Dynamic Window Approach (DWA) Trajectory Rollouts ---")
    grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.1, origin_x=-5.0, origin_y=-5.0)
    # Add a nearby obstacle
    gx, gy = grid.world_to_grid(1.5, 0.2)
    grid.cells[grid._get_index(gx, gy)] = grid.l_max

    planner = DWAPlanner(DWAConfig())
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.5, omega=0.0)
    goal = (5.0, 1.0)

    cycles = 500
    t0 = time.perf_counter()
    for _ in range(cycles):
        _ = planner.plan(state, goal, grid)
    elapsed = time.perf_counter() - t0
    hz = cycles / elapsed
    print(f"DWA Local Planning Cycles: {cycles} iterations in {elapsed*1000:.2f} ms ({hz:,.0f} Hz / cycles/sec)")


def run_all_benchmarks():
    print("=" * 65)
    print("      VELOSLAM: FIRST-PRINCIPLES ROBOTICS PERFORMANCE BENCHMARK")
    print("=" * 65)
    bench_matrix_multiplication()
    bench_ekf_slam()
    bench_occupancy_grid()
    bench_dubins_curves()
    bench_hybrid_astar()
    bench_dwa_local_planner()
    print("\n" + "=" * 65)
    print("      ALL VELOSLAM BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_all_benchmarks()
