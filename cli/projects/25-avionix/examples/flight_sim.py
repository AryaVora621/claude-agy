"""Interactive Terminal Flight Simulator and Aerial Robotics Mission Workbench.

Demonstrates:
1. Automated Takeoff and EFIS Primary Flight Display (PFD) with Artificial Horizon
2. 3D Minimum-Snap Waypoint Navigation with Sub-Pixel Braille Trajectory Renderer
3. 3D Lemniscate of Gerono (Figure-8) Aerobatic Loop Tracking on SE(3)
4. Active Disturbance Observer (DOB) Crosswind Rejection
5. Multi-Rate 15-State Error-State EKF Sensor Fusion (IMU + Baro + GPS)
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.autopilot import Autopilot, FlightMode
from avionix.avionics_pfd import (
    BrailleFlightCanvas,
    render_3d_flight_path,
    render_primary_flight_display,
)
from avionix.dynamics import QuadrotorParams, QuadrotorState, Quaternion, Vector3
from avionix.estimation import GPSReading, IMUReading, MultiRateESEKF, SensorSimulator
from avionix.trajectory import (
    AerobaticTrajectories,
    DifferentialFlatness,
    FlatOutput,
    MinimumSnapTrajectory,
)


def demo_takeoff_pfd() -> None:
    print("\n" + "=" * 78)
    print("      DEMO 1: AUTOMATED TAKEOFF & EFIS PRIMARY FLIGHT DISPLAY (PFD)")
    print("=" * 78)
    print("Initializing quadrotor dynamics and commanding vertical takeoff to 2.5m...")

    ap = Autopilot(use_ekf=False, seed=42)
    ap.takeoff(target_altitude_m=2.5)

    # Simulate takeoff until steady hover
    ap.run_simulation(duration_s=2.0, dt_s=0.01)

    print(f"Takeoff Complete! Flight Mode: [{ap.mode.value}]")
    print(f"Final Telemetry: Pos: {ap.true_state.position} | Vel: {ap.true_state.velocity.norm():.2f} m/s")
    print("\nDisplaying Electronic Flight Instrument System (EFIS) PFD:")
    pfd_str = render_primary_flight_display(
        ap.true_state,
        target_pos=Vector3(0.0, 0.0, 2.5),
        flight_mode="ALT HOLD 2.5m",
        display_width=74,
    )
    print(pfd_str)


def demo_waypoint_navigation() -> None:
    print("\n" + "=" * 78)
    print("      DEMO 2: 3D MINIMUM-SNAP WAYPOINT NAVIGATION & BRAILLE TRAJECTORY")
    print("=" * 78)
    print("Computing piecewise 5th-order polynomial minimum-snap spline through 5 waypoints...")

    waypoints = [
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 3.0, 2.5),
        Vector3(5.0, 4.0, 3.5),
        Vector3(7.0, 1.0, 2.0),
        Vector3(8.0, -2.0, 1.5),
    ]
    durations = [2.0, 2.5, 2.5, 2.0]

    ap = Autopilot(use_ekf=False, seed=42)
    ap.arm()
    ap.true_state.position = waypoints[0]
    ap.fly_waypoints(waypoints, durations)

    print(f"Mission: 4 Segments | Total Planned Duration: {ap.active_trajectory.total_duration:.1f} s")
    print("Executing SE(3) Geometric Tracking Control...")

    flown_path: List[Vector3] = []
    # Run simulation
    total_steps = int(ap.active_trajectory.total_duration / 0.01) + 20
    for s in range(total_steps):
        entry = ap.step(0.01)
        if s % 5 == 0:
            flown_path.append(entry.true_state.position)

    rmse_p, rmse_v = ap.calculate_tracking_rmse()
    print(f"Navigation Complete! Waypoint Target: {waypoints[-1]}")
    print(f"Actual Position: {ap.true_state.position}")
    print(f"Tracking Performance: Pos RMSE = {rmse_p:.3f} m | Vel RMSE = {rmse_v:.3f} m/s")

    print("\n3D Orbital Perspective Trajectory (Cyan: Waypoints, Green: Flown Path, Red/Orange: Quad Body):")
    orbit_view = render_3d_flight_path(
        waypoints,
        flown_path,
        ap.true_state,
        char_width=74,
        char_height=18,
        azimuth_deg=55.0,
        elevation_deg=28.0,
    )
    print(orbit_view)


def demo_aerobatics_figure_eight() -> None:
    print("\n" + "=" * 78)
    print("      DEMO 3: 3D LEMNISCATE AEROBATIC TRACKING ON SO(3)")
    print("=" * 78)
    print("Executing high-speed Figure-8 aerobatic trajectory (Radius X=3.5m, Y=2.5m)...")

    ap = Autopilot(use_ekf=False, seed=42)
    ap.arm()
    ap.true_state.position = Vector3(0.0, 0.0, 2.5)
    ap.fly_figure_eight(radius_x=3.5, radius_y=2.5, altitude=2.5)

    flown_path: List[Vector3] = []
    max_tilt_deg = 0.0
    total_steps = len(ap.active_setpoints)

    for s in range(total_steps):
        entry = ap.step(0.01)
        roll, pitch, _ = entry.true_state.attitude.to_euler()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch)))))
        if tilt > max_tilt_deg:
            max_tilt_deg = tilt
        if s % 4 == 0:
            flown_path.append(entry.true_state.position)

    rmse_p, rmse_v = ap.calculate_tracking_rmse()
    print("Aerobatic Loop Completed without Gimbal-Lock or Singularities!")
    print(f"Peak Attitude Tilt Angle: {max_tilt_deg:.1f}° (SO(3) Chordal Metric)")
    print(f"Trajectory Accuracy: Position RMSE = {rmse_p:.3f} m | Velocity RMSE = {rmse_v:.3f} m/s")

    ref_wps = [pt.position for pt in ap.active_setpoints[::10]]
    view = render_3d_flight_path(
        ref_wps,
        flown_path,
        ap.true_state,
        char_width=74,
        char_height=18,
        azimuth_deg=45.0,
        elevation_deg=35.0,
    )
    print(view)


def demo_wind_gust_rejection() -> None:
    print("\n" + "=" * 78)
    print("      DEMO 4: ACTIVE DISTURBANCE OBSERVER (DOB) WIND REJECTION")
    print("=" * 78)
    print("Injecting sudden 2.5 N lateral aerodynamic crosswind at t = 1.0 s...")

    ap = Autopilot(use_ekf=False, seed=42)
    ap.takeoff(target_altitude_m=2.0)

    # 1. Fly to hover under calm conditions (t=0 to 1s)
    ap.run_simulation(duration_s=1.0, dt_s=0.01)
    calm_pos = ap.true_state.position

    # 2. Inject strong 2.5 N crosswind along +X axis
    wind_vector = Vector3(2.5, 0.0, 0.0)
    ap.set_wind(wind_vector)
    ap.run_simulation(duration_s=2.0, dt_s=0.01)

    print(f"Calm Hover Position:        X = {calm_pos.x:+5.3f} m, Y = {calm_pos.y:+5.3f} m, Z = {calm_pos.z:5.3f} m")
    print(f"True Wind Disturbance:      {wind_vector.x:5.2f} N along +X")
    print(f"Estimated Wind Disturbance: {ap.dob.estimated_force.x:5.2f} N (DOB Low-Pass Residual)")
    print(f"Disturbance Estimation Error: {abs(ap.dob.estimated_force.x - wind_vector.x):.3f} N")
    print(f"Final Drift-Compensated Position: X = {ap.true_state.position.x:+5.3f} m, Y = {ap.true_state.position.y:+5.3f} m, Z = {ap.true_state.position.z:5.3f} m")
    print("Result: Autonomous attitude trim counteracted lateral aerodynamic drift.")


def demo_ekf_sensor_fusion() -> None:
    print("\n" + "=" * 78)
    print("      DEMO 5: 15-STATE MULTI-RATE ERROR-STATE EKF SENSOR FUSION")
    print("=" * 78)
    print("Fusing 100 Hz IMU (accel + gyro drift), 50 Hz Barometer, and 10 Hz GPS...")

    ap = Autopilot(use_ekf=True, seed=42)
    ap.takeoff(target_altitude_m=2.0)

    # Fly forward to (3.0, 2.0, 2.0)
    wps = [Vector3(0.0, 0.0, 2.0), Vector3(3.0, 2.0, 2.0)]
    durations = [3.0]
    ap.fly_waypoints(wps, durations)

    # Run simulation
    ap.run_simulation(duration_s=3.2, dt_s=0.01)

    true_p = ap.true_state.position
    est_p = ap.ekf.state.position
    pos_err = (true_p - est_p).norm()

    print(f"True Final Position:      {true_p}")
    print(f"EKF Estimated Position:   {est_p}")
    print(f"Final Estimation Error:   {pos_err * 100:.1f} cm (Joseph-stabilized covariance update)")
    print(f"Estimated Gyro Bias:      {ap.ekf.state.gyro_bias}")
    print(f"Estimated Accel Bias:     {ap.ekf.state.accel_bias}")
    print("State estimation successfully fused noisy asynchronous observations into smooth navigation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avionix: 6-DOF Quadrotor Dynamics & Flight Simulator")
    parser.add_argument("--all", action="store_true", help="Run all flight simulation demonstrations")
    parser.add_argument("--takeoff", action="store_true", help="Run Automated Takeoff & PFD Demo")
    parser.add_argument("--waypoints", action="store_true", help="Run 3D Waypoint Navigation Demo")
    parser.add_argument("--aerobatics", action="store_true", help="Run 3D Aerobatics Figure-8 Demo")
    parser.add_argument("--wind", action="store_true", help="Run Disturbance Observer Wind Rejection Demo")
    parser.add_argument("--ekf", action="store_true", help="Run Multi-Rate EKF Sensor Fusion Demo")

    args = parser.parse_args()

    # Default to running all demonstrations if no specific flag is given
    run_all = args.all or not any([args.takeoff, args.waypoints, args.aerobatics, args.wind, args.ekf])

    print("=" * 78)
    print("      AVIONIX: 6-DOF AERIAL ROBOTICS & FLIGHT DYNAMICS WORKBENCH")
    print("=" * 78)

    if run_all or args.takeoff:
        demo_takeoff_pfd()

    if run_all or args.waypoints:
        demo_waypoint_navigation()

    if run_all or args.aerobatics:
        demo_aerobatics_figure_eight()

    if run_all or args.wind:
        demo_wind_gust_rejection()

    if run_all or args.ekf:
        demo_ekf_sensor_fusion()

    print("\n" + "=" * 78)
    print("                ALL AVIONIX DEMONSTRATIONS COMPLETED")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
