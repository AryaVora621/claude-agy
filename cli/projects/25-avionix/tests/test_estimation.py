"""Unit tests for Sensor Simulation and Multi-Rate Error-State Extended Kalman Filter (ES-EKF)."""

import math
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.dynamics import QuadrotorState, Quaternion, Vector3
from avionix.estimation import (
    BaroReading,
    EKFState,
    GPSReading,
    IMUReading,
    MagReading,
    MultiRateESEKF,
    SensorSimulator,
)


class TestSensorSimulator(unittest.TestCase):
    """Test synthetic flight sensors with noise models and bias drift."""

    def setUp(self) -> None:
        self.sim = SensorSimulator(seed=123)

    def test_imu_hover_specific_force(self) -> None:
        # Hovering quadrotor: acceleration is zero, specific force sensed by accelerometer = R^T * [0, 0, g]
        state = QuadrotorState(position=Vector3(0.0, 0.0, 5.0))
        acc_world = Vector3(0.0, 0.0, 0.0)

        imu = self.sim.generate_imu(state, acc_world, timestamp_s=0.0, dt_s=0.01)
        # Sensed vertical specific force should be ~ +9.81 m/s^2 (upward normal force)
        self.assertAlmostEqual(imu.accel_m_s2.z, 9.81, delta=0.2)
        self.assertAlmostEqual(imu.accel_m_s2.x, 0.0, delta=0.2)
        self.assertAlmostEqual(imu.accel_m_s2.y, 0.0, delta=0.2)

    def test_barometer_reading(self) -> None:
        state = QuadrotorState(position=Vector3(0.0, 0.0, 12.5))
        baro = self.sim.generate_barometer(state, timestamp_s=0.0)
        self.assertAlmostEqual(baro.altitude_m, 12.5, delta=0.5)

    def test_gps_fix(self) -> None:
        state = QuadrotorState(
            position=Vector3(10.0, -15.0, 8.0),
            velocity=Vector3(2.0, 1.0, 0.0),
        )
        gps = self.sim.generate_gps(state, timestamp_s=1.0)
        self.assertAlmostEqual(gps.position_m.x, 10.0, delta=1.0)
        self.assertAlmostEqual(gps.position_m.y, -15.0, delta=1.0)
        self.assertAlmostEqual(gps.position_m.z, 8.0, delta=1.5)


class TestMultiRateESEKF(unittest.TestCase):
    """Test 15-state Error-State EKF estimation, prediction, and Joseph updates."""

    def setUp(self) -> None:
        self.ekf = MultiRateESEKF()

    def test_initial_state_defaults(self) -> None:
        self.assertAlmostEqual(self.ekf.state.position.norm(), 0.0)
        self.assertAlmostEqual(self.ekf.state.velocity.norm(), 0.0)
        self.assertAlmostEqual(self.ekf.state.attitude.norm(), 1.0)

    def test_imu_prediction_step(self) -> None:
        # High-rate IMU prediction step with constant upward specific force of 9.81 (hover)
        # Net acceleration = 9.81 - 9.81 = 0
        imu = IMUReading(
            timestamp_s=0.01,
            accel_m_s2=Vector3(0.0, 0.0, 9.81),
            gyro_rad_s=Vector3(0.0, 0.0, 0.0),
        )
        self.ekf.predict(imu, dt_s=0.01)

        # Position and velocity should remain zero in hover
        self.assertAlmostEqual(self.ekf.state.velocity.z, 0.0, places=4)
        self.assertAlmostEqual(self.ekf.state.position.z, 0.0, places=4)

    def test_barometer_measurement_update(self) -> None:
        # Observe altitude 5.0m via barometer
        baro = BaroReading(timestamp_s=0.1, altitude_m=5.0)
        self.ekf.update_barometer(baro)

        # Altitude estimate should shift towards measured 5.0m
        self.assertGreater(self.ekf.state.position.z, 0.0)

    def test_gps_measurement_fusion(self) -> None:
        # Fuse multiple GPS measurements to verify filter convergence
        true_pos = Vector3(4.0, 3.0, 6.0)
        for step in range(15):
            gps = GPSReading(
                timestamp_s=step * 0.1,
                position_m=true_pos,
                velocity_m_s=Vector3(0.0, 0.0, 0.0),
            )
            self.ekf.update_gps(gps)

        # Estimate should converge close to true position
        pos_err = (self.ekf.state.position - true_pos).norm()
        self.assertLess(pos_err, 0.5)


if __name__ == "__main__":
    unittest.main()
