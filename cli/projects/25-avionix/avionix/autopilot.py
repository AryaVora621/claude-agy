"""Flight Autopilot, Disturbance Observer & Mission Executive for Quadrotor UAVs.

Implements:
1. High-level Autopilot state machine (DISARMED, ARMED, TAKEOFF, HOVER, WAYPOINT_NAV,
   TRAJECTORY_TRACK, RTL, LAND).
2. Disturbance Observer (DOB) estimating external wind forces and active trimming.
3. Closed-loop flight executive integrating dynamics, EKF state estimation,
   SE(3) geometric control, and telemetry logging.
"""

from __future__ import annotations
import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .avionics_pfd import render_3d_flight_path, render_primary_flight_display
from .control import ControlWrench, SE3GeometricController
from .dynamics import Matrix3x3, QuadrotorDynamics, QuadrotorParams, QuadrotorState, Quaternion, Vector3
from .estimation import MultiRateESEKF, SensorSimulator
from .trajectory import (
    AerobaticTrajectories,
    DifferentialFlatness,
    FlatOutput,
    FullTrajectoryState,
    MinimumSnapTrajectory,
)


class FlightMode(enum.Enum):
    """Autopilot operational flight mode states."""

    DISARMED = "DISARMED"
    ARMED = "ARMED"
    TAKEOFF = "TAKEOFF"
    HOVER = "HOVER"
    WAYPOINT_NAV = "WAYPOINT_NAV"
    TRAJECTORY_TRACK = "TRAJECTORY_TRACK"
    RTL = "RTL"
    LAND = "LAND"


class DisturbanceObserver:
    """Estimates unmodeled external wind aerodynamic forces for active rejection."""

    def __init__(self, mass_kg: float, cutoff_freq_hz: float = 5.0) -> None:
        self.mass = mass_kg
        self.gain = 2.0 * math.pi * cutoff_freq_hz
        self.estimated_force = Vector3(0.0, 0.0, 0.0)

    def update(
        self,
        measured_accel_w: Vector3,
        commanded_thrust_w: Vector3,
        dt_s: float,
    ) -> Vector3:
        """Update force disturbance estimate via low-pass filtered momentum residual."""
        # Residual = m * a_measured - (F_thrust - m * g * e3)
        g_term = Vector3(0.0, 0.0, self.mass * 9.81)
        expected_force_w = commanded_thrust_w - g_term
        actual_force_w = measured_accel_w * self.mass
        residual = actual_force_w - expected_force_w

        # First-order low pass filter
        alpha = self.gain * dt_s / (1.0 + self.gain * dt_s)
        self.estimated_force = self.estimated_force + (residual - self.estimated_force) * alpha
        return self.estimated_force


@dataclass
class FlightLogEntry:
    """Single telemetry time sample of the flight."""

    time_s: float
    true_state: QuadrotorState
    estimated_state: QuadrotorState
    reference_pos: Vector3
    reference_vel: Vector3
    commanded_thrust: float
    commanded_torques: Vector3
    wind_disturbance: Vector3
    estimated_wind: Vector3


class Autopilot:
    """Unified Flight Autopilot and mission execution engine."""

    def __init__(
        self,
        params: Optional[QuadrotorParams] = None,
        use_ekf: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        self.params = params or QuadrotorParams()
        self.dynamics = QuadrotorDynamics(self.params)
        self.controller = SE3GeometricController(self.params)
        self.flatness = DifferentialFlatness(self.params)
        self.use_ekf = use_ekf

        self.sensors = SensorSimulator(seed=seed)
        self.ekf = MultiRateESEKF()
        self.dob = DisturbanceObserver(self.params.mass_kg)

        self.mode = FlightMode.DISARMED
        self.current_time_s = 0.0
        self.true_state = QuadrotorState()
        self.home_pos = Vector3(0.0, 0.0, 0.0)

        # Active mission setpoints
        self.active_trajectory: Optional[MinimumSnapTrajectory] = None
        self.active_setpoints: Optional[List[FlatOutput]] = None
        self.trajectory_index = 0
        self.target_hover_pos = Vector3(0.0, 0.0, 1.5)

        # Active external disturbances
        self.active_wind = Vector3(0.0, 0.0, 0.0)

        # Telemetry history
        self.log: List[FlightLogEntry] = []

    def arm(self) -> None:
        """Arm motors to idle spinning speed."""
        self.mode = FlightMode.ARMED
        idle_speed = self.params.omega_min_rad_s
        self.true_state.rotor_speeds = [idle_speed] * 4

    def disarm(self) -> None:
        """Cut motor power completely."""
        self.mode = FlightMode.DISARMED
        self.true_state.rotor_speeds = [0.0] * 4

    def takeoff(self, target_altitude_m: float = 2.0) -> None:
        """Initiate automated takeoff sequence."""
        self.arm()
        self.mode = FlightMode.TAKEOFF
        self.target_hover_pos = Vector3(self.true_state.position.x, self.true_state.position.y, target_altitude_m)

    def set_wind(self, wind_vector_n: Vector3) -> None:
        """Inject aerodynamic wind gust forces."""
        self.active_wind = wind_vector_n

    def fly_waypoints(self, waypoints: List[Vector3], durations: List[float]) -> None:
        """Plan and execute minimum-snap trajectory through waypoints."""
        self.active_trajectory = MinimumSnapTrajectory(waypoints, durations)
        self.mode = FlightMode.WAYPOINT_NAV
        self.trajectory_start_time = self.current_time_s

    def fly_figure_eight(self, radius_x: float = 3.0, radius_y: float = 2.0, altitude: float = 2.5) -> None:
        """Execute 3D Lemniscate aerobatic trajectory."""
        self.active_setpoints = AerobaticTrajectories.figure_eight(radius_x, radius_y, altitude)
        self.trajectory_index = 0
        self.mode = FlightMode.TRAJECTORY_TRACK

    def get_current_setpoint(self) -> FullTrajectoryState:
        """Retrieve active setpoint based on current flight mode."""
        if self.mode in (FlightMode.ARMED, FlightMode.TAKEOFF, FlightMode.HOVER, FlightMode.LAND):
            flat = FlatOutput(
                position=self.target_hover_pos,
                velocity=Vector3(0.0, 0.0, 0.0),
                acceleration=Vector3(0.0, 0.0, 0.0),
            )
            return self.flatness.flat_to_full_state(flat, self.current_time_s)

        elif self.mode == FlightMode.WAYPOINT_NAV and self.active_trajectory:
            rel_t = self.current_time_s - self.trajectory_start_time
            flat = self.active_trajectory.evaluate(rel_t)
            return self.flatness.flat_to_full_state(flat, self.current_time_s)

        elif self.mode == FlightMode.TRAJECTORY_TRACK and self.active_setpoints:
            idx = min(self.trajectory_index, len(self.active_setpoints) - 1)
            flat = self.active_setpoints[idx]
            return self.flatness.flat_to_full_state(flat, self.current_time_s)

        # Default hover
        flat = FlatOutput(position=self.target_hover_pos)
        return self.flatness.flat_to_full_state(flat, self.current_time_s)

    def step(self, dt_s: float = 0.01) -> FlightLogEntry:
        """Execute one complete 100 Hz simulation step (Sensors -> EKF -> Control -> Dynamics)."""
        # 1. State estimation (Multi-Rate EKF or Perfect State Feedback)
        est_state = self.true_state.copy()
        if self.use_ekf:
            # Evaluate true acceleration from physical equations of motion
            _, true_acc, _, _, _ = self.dynamics.derivatives(
                self.true_state, self.true_state.rotor_speeds, external_force_w=self.active_wind
            )
            imu = self.sensors.generate_imu(self.true_state, true_acc, self.current_time_s, dt_s)
            self.ekf.predict(imu, dt_s)

            # 50 Hz Barometer
            if int(self.current_time_s * 100) % 2 == 0:
                baro = self.sensors.generate_barometer(self.true_state, self.current_time_s)
                self.ekf.update_barometer(baro)

            # 10 Hz GPS
            if int(self.current_time_s * 100) % 10 == 0:
                gps = self.sensors.generate_gps(self.true_state, self.current_time_s)
                self.ekf.update_gps(gps)

            est_state.position = self.ekf.state.position
            est_state.velocity = self.ekf.state.velocity
            est_state.attitude = self.ekf.state.attitude

        # 2. Get reference setpoint
        ref = self.get_current_setpoint()

        # 3. Compute control wrench via SE(3) geometric controller
        if self.mode == FlightMode.DISARMED:
            wrench = ControlWrench(0.0, Vector3(0.0, 0.0, 0.0), [0.0] * 4)
        else:
            wrench = self.controller.update(est_state, ref, dt_s)

        # 4. Advance physical dynamics via RK4
        self.true_state = self.dynamics.step_rk4(
            self.true_state, dt_s, wrench.rotor_commands, external_force_w=self.active_wind
        )

        # 5. Disturbance Observer update
        R = self.true_state.attitude.to_rotation_matrix()
        thrust_w = R.dot_vec(Vector3(0.0, 0.0, wrench.collective_thrust))
        est_wind = self.dob.update(self.true_state.velocity, thrust_w, dt_s)

        # 6. Mode transitions
        if self.mode == FlightMode.TAKEOFF:
            if abs(self.true_state.position.z - self.target_hover_pos.z) < 0.1:
                self.mode = FlightMode.HOVER

        elif self.mode == FlightMode.TRAJECTORY_TRACK and self.active_setpoints:
            self.trajectory_index += 1
            if self.trajectory_index >= len(self.active_setpoints):
                self.mode = FlightMode.HOVER
                self.target_hover_pos = self.true_state.position

        elif self.mode == FlightMode.WAYPOINT_NAV and self.active_trajectory:
            rel_t = self.current_time_s - self.trajectory_start_time
            if rel_t >= self.active_trajectory.total_duration:
                self.mode = FlightMode.HOVER
                self.target_hover_pos = self.active_trajectory.waypoints[-1]

        # Record telemetry
        entry = FlightLogEntry(
            time_s=self.current_time_s,
            true_state=self.true_state.copy(),
            estimated_state=est_state,
            reference_pos=ref.position,
            reference_vel=ref.velocity,
            commanded_thrust=wrench.collective_thrust,
            commanded_torques=wrench.moments,
            wind_disturbance=self.active_wind,
            estimated_wind=est_wind,
        )
        self.log.append(entry)
        self.current_time_s += dt_s
        return entry

    def run_simulation(self, duration_s: float, dt_s: float = 0.01) -> None:
        """Run continuous simulation for specified duration."""
        steps = int(duration_s / dt_s)
        for _ in range(steps):
            self.step(dt_s)

    def calculate_tracking_rmse(self) -> Tuple[float, float]:
        """Compute position and velocity tracking Root Mean Squared Error (RMSE)."""
        if not self.log:
            return 0.0, 0.0

        pos_sq_sum = 0.0
        vel_sq_sum = 0.0
        for entry in self.log:
            p_err = (entry.true_state.position - entry.reference_pos).norm_sq()
            v_err = (entry.true_state.velocity - entry.reference_vel).norm_sq()
            pos_sq_sum += p_err
            vel_sq_sum += v_err

        n = len(self.log)
        return math.sqrt(pos_sq_sum / n), math.sqrt(vel_sq_sum / n)
