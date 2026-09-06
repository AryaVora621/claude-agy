"""Avionix: 6-DOF Quadrotor Dynamics, Differential Flatness & SE(3) Geometric Tracking Control Engine.

Pure Python 3.10+ standard library implementation of:
1. 6-DOF Rigid-Body Flight Dynamics on SE(3) with Quaternion Kinematics & RK4.
2. Differential Flatness & Piecewise Quintic Polynomial Trajectory Generation.
3. Lee-Leok-McClamroch (2010) Geometric Tracking Controller on SO(3).
4. Multi-Rate Error-State Extended Kalman Filter (15-state ES-EKF).
5. Sub-pixel Braille 3D Trajectory Visualizer & Primary Flight Display (PFD).
6. Autonomous Mission Executive & Disturbance Observer (DOB).
"""

from .autopilot import Autopilot, DisturbanceObserver, FlightLogEntry, FlightMode
from .avionics_pfd import (
    BrailleFlightCanvas,
    render_3d_flight_path,
    render_primary_flight_display,
)
from .control import (
    CascadedPIDController,
    ControlWrench,
    GeometricGains,
    MotorMixer,
    SE3GeometricController,
)
from .dynamics import (
    Matrix3x3,
    QuadrotorDynamics,
    QuadrotorParams,
    QuadrotorState,
    Quaternion,
    Vector3,
)
from .estimation import (
    BaroReading,
    EKFState,
    GPSReading,
    IMUReading,
    MagReading,
    MultiRateESEKF,
    SensorSimulator,
)
from .trajectory import (
    AerobaticTrajectories,
    DifferentialFlatness,
    FlatOutput,
    FullTrajectoryState,
    MinimumSnapTrajectory,
    PiecewisePolynomial1D,
    PolynomialSegment,
)

__version__ = "0.1.0"
__all__ = [
    "Vector3",
    "Quaternion",
    "Matrix3x3",
    "QuadrotorParams",
    "QuadrotorState",
    "QuadrotorDynamics",
    "FlatOutput",
    "FullTrajectoryState",
    "DifferentialFlatness",
    "PolynomialSegment",
    "PiecewisePolynomial1D",
    "MinimumSnapTrajectory",
    "AerobaticTrajectories",
    "GeometricGains",
    "ControlWrench",
    "MotorMixer",
    "SE3GeometricController",
    "CascadedPIDController",
    "IMUReading",
    "BaroReading",
    "MagReading",
    "GPSReading",
    "EKFState",
    "MultiRateESEKF",
    "SensorSimulator",
    "BrailleFlightCanvas",
    "render_3d_flight_path",
    "render_primary_flight_display",
    "FlightMode",
    "DisturbanceObserver",
    "FlightLogEntry",
    "Autopilot",
]
