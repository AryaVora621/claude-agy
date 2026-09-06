"""High-throughput performance microbenchmarks for Avionix aerial robotics subsystems.

Measures operational throughput and latency across:
1. 6-DOF Rigid-Body Dynamics RK4 Integration (steps/s)
2. Quaternion to SO(3) Rotation Matrix & Euler Conversions (conversions/s)
3. Minimum-Snap Quintic Spline 3D Trajectory Solving (solves/s)
4. Differential Flatness SE(3) State & Rate Recovery (evals/s)
5. SE(3) Geometric Tracking Controller Update on SO(3) (cycles/s)
6. Quadrotor Motor Mixer with Anti-Saturation (mixes/s)
7. Multi-Rate 15-State Error-State EKF Prediction & Joseph Update (updates/s)
8. Sub-Pixel Unicode Braille 3D Trajectory Renderer (FPS)
9. EFIS Primary Flight Display (PFD) Frame Generation (FPS)
10. Full Closed-Loop Simulation Step (Sensors + EKF + Control + Dynamics + DOB) (steps/s)
"""

import math
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.autopilot import Autopilot
from avionix.avionics_pfd import (
    BrailleFlightCanvas,
    render_3d_flight_path,
    render_primary_flight_display,
)
from avionix.control import MotorMixer, SE3GeometricController
from avionix.dynamics import (
    Matrix3x3,
    QuadrotorDynamics,
    QuadrotorParams,
    QuadrotorState,
    Quaternion,
    Vector3,
)
from avionix.estimation import GPSReading, IMUReading, MultiRateESEKF
from avionix.trajectory import (
    DifferentialFlatness,
    FlatOutput,
    FullTrajectoryState,
    MinimumSnapTrajectory,
)


def benchmark(name: str, fn: Callable[[], int], iterations: int = 5) -> Tuple[float, float]:
    """Run benchmark function over multiple iterations and report throughput and latency."""
    # Warmup
    fn()

    times = []
    total_ops = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        ops = fn()
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        total_ops += ops

    avg_time = sum(times) / len(times)
    throughput = (total_ops / iterations) / avg_time
    latency_us = (avg_time / (total_ops / iterations)) * 1e6
    return throughput, latency_us


def bench_rk4_dynamics() -> int:
    """Benchmark 6-DOF RK4 numerical integration throughput."""
    dynamics = QuadrotorDynamics()
    state = QuadrotorState()
    cmds = [500.0, 500.0, 500.0, 500.0]
    dt = 0.005
    n_steps = 20000

    for _ in range(n_steps):
        state = dynamics.step_rk4(state, dt, cmds)

    return n_steps


def bench_quaternion_transforms() -> int:
    """Benchmark Quaternion to SO(3) matrix and Euler angle conversion throughput."""
    n_ops = 50000
    q = Quaternion.from_euler(0.2, -0.3, 1.2)
    v = Vector3(1.0, 2.0, 3.0)

    for i in range(n_ops):
        R = q.to_rotation_matrix()
        r, p, y = q.to_euler()
        v_rot = q.rotate_vec(v)

    return n_ops


def bench_minimum_snap_solving() -> int:
    """Benchmark piecewise quintic spline linear system solving across waypoints."""
    wps = [
        Vector3(0.0, 0.0, 0.0),
        Vector3(2.0, 3.0, 1.5),
        Vector3(5.0, -1.0, 2.5),
        Vector3(7.0, 2.0, 1.0),
    ]
    durations = [2.0, 2.5, 3.0]
    n_solves = 400

    for _ in range(n_solves):
        traj = MinimumSnapTrajectory(wps, durations)

    return n_solves


def bench_differential_flatness() -> int:
    """Benchmark differential flatness mapping from flat output to full 6-DOF state."""
    flatness = DifferentialFlatness()
    flat = FlatOutput(
        position=Vector3(2.0, 3.0, 4.0),
        velocity=Vector3(1.5, -0.5, 0.2),
        acceleration=Vector3(0.8, 1.2, -0.3),
        jerk=Vector3(0.2, -0.1, 0.05),
        snap=Vector3(0.05, 0.02, -0.01),
        yaw_rad=0.785,
        yaw_rate_rad_s=0.1,
    )
    n_evals = 30000

    for _ in range(n_evals):
        full = flatness.flat_to_full_state(flat, 1.0)

    return n_evals


def bench_geometric_controller() -> int:
    """Benchmark SE(3) geometric tracking controller update cycle."""
    ctrl = SE3GeometricController()
    state = QuadrotorState(
        position=Vector3(1.0, -0.5, 2.2),
        velocity=Vector3(0.5, 0.2, -0.1),
        attitude=Quaternion.from_euler(0.1, -0.15, 0.3),
    )
    ref = FullTrajectoryState(
        time_s=1.0,
        position=Vector3(1.1, -0.4, 2.3),
        velocity=Vector3(0.6, 0.25, 0.0),
        acceleration=Vector3(0.1, 0.05, 0.0),
        thrust_n=9.81,
        attitude=Quaternion.identity(),
        rotation_matrix=Matrix3x3.identity(),
        angular_velocity=Vector3(0.0, 0.0, 0.0),
        angular_acceleration=Vector3(0.0, 0.0, 0.0),
        body_rates=Vector3(0.0, 0.0, 0.0),
    )
    n_cycles = 25000

    for _ in range(n_cycles):
        wrench = ctrl.update(state, ref, dt_s=0.01)

    return n_cycles


def bench_motor_mixer() -> int:
    """Benchmark motor mixer with attitude priority anti-saturation."""
    mixer = MotorMixer()
    torques = Vector3(0.05, -0.08, 0.02)
    n_mixes = 50000

    for _ in range(n_mixes):
        speeds = mixer.mix(12.5, torques)

    return n_mixes


def bench_ekf_estimation() -> int:
    """Benchmark 15-state Error-State EKF predict and Joseph-form GPS measurement updates."""
    ekf = MultiRateESEKF()
    imu = IMUReading(
        timestamp_s=0.01,
        accel_m_s2=Vector3(0.02, -0.01, 9.83),
        gyro_rad_s=Vector3(0.005, -0.003, 0.002),
    )
    gps = GPSReading(
        timestamp_s=0.1,
        position_m=Vector3(1.5, 2.0, 3.0),
        velocity_m_s=Vector3(0.2, 0.1, 0.0),
    )
    n_steps = 10000

    for i in range(n_steps):
        ekf.predict(imu, dt_s=0.01)
        if i % 10 == 0:
            ekf.update_gps(gps)

    return n_steps


def bench_braille_3d_rendering() -> int:
    """Benchmark 3D sub-pixel Braille trajectory renderer frame generation."""
    wps = [Vector3(0.0, 0.0, 0.0), Vector3(2.0, 3.0, 1.5), Vector3(5.0, 0.0, 2.5)]
    path = [Vector3(i * 0.1, math.sin(i * 0.2), 1.0 + i * 0.02) for i in range(50)]
    state = QuadrotorState(position=Vector3(5.0, 0.0, 2.0))
    n_frames = 150

    for _ in range(n_frames):
        txt = render_3d_flight_path(wps, path, state, char_width=50, char_height=16)

    return n_frames


def bench_pfd_rendering() -> int:
    """Benchmark Primary Flight Display (PFD) synthetic horizon frame generation."""
    state = QuadrotorState(
        position=Vector3(1.5, -2.0, 4.5),
        velocity=Vector3(3.2, 0.5, 0.8),
        attitude=Quaternion.from_euler(0.12, -0.08, 1.45),
    )
    target = Vector3(0.0, 0.0, 5.0)
    n_frames = 150

    for _ in range(n_frames):
        pfd = render_primary_flight_display(state, target_pos=target, flight_mode="NAV SE(3)")

    return n_frames


def bench_closed_loop_simulation() -> int:
    """Benchmark full closed-loop autopilot simulation step throughput."""
    ap = Autopilot(use_ekf=False, seed=42)
    ap.takeoff(target_altitude_m=2.0)
    n_steps = 3000

    for _ in range(n_steps):
        ap.step(dt_s=0.01)

    return n_steps


def main() -> None:
    print("=" * 82)
    print("           AVIONIX: AERIAL ROBOTICS SUBSYSTEM MICROBENCHMARKS")
    print("=" * 82)
    print(f"{'Benchmark Target':<48} | {'Throughput':>16} | {'Latency':>10}")
    print("-" * 82)

    benchmarks = [
        ("6-DOF RK4 Numerical Dynamics Integration", bench_rk4_dynamics, "steps/s"),
        ("Quaternion to SO(3) & Euler Transformations", bench_quaternion_transforms, "ops/s"),
        ("Minimum-Snap Quintic Spline 3D Solving", bench_minimum_snap_solving, "solves/s"),
        ("Differential Flatness SE(3) State Recovery", bench_differential_flatness, "evals/s"),
        ("SE(3) Geometric Tracking Controller on SO(3)", bench_geometric_controller, "cycles/s"),
        ("Motor Mixer with Priority Anti-Saturation", bench_motor_mixer, "mixes/s"),
        ("15-State ES-EKF Multi-Rate Sensor Fusion", bench_ekf_estimation, "updates/s"),
        ("Sub-Pixel Braille 3D Trajectory Renderer", bench_braille_3d_rendering, "FPS"),
        ("EFIS Primary Flight Display (PFD) Engine", bench_pfd_rendering, "FPS"),
        ("Full Closed-Loop Autopilot Simulation Step", bench_closed_loop_simulation, "steps/s"),
    ]

    for name, fn, unit in benchmarks:
        thru, lat = benchmark(name, fn, iterations=3)
        if thru >= 1e6:
            thru_str = f"{thru/1e6:7.2f} M {unit}"
        elif thru >= 1e3:
            thru_str = f"{thru/1e3:7.1f} k {unit}"
        else:
            thru_str = f"{thru:7.1f} {unit}"

        lat_str = f"{lat:7.2f} µs" if lat < 1000.0 else f"{lat/1000.0:7.2f} ms"
        print(f"{name:<48} | {thru_str:>16} | {lat_str:>10}")

    print("=" * 82)


if __name__ == "__main__":
    main()
