"""
Unit tests for VeloSLAM Extended Kalman Filter SLAM (EKF-SLAM).
"""

import math
import unittest
from veloslam.ekf_slam import EKFSLAM


class TestEKFSLAM(unittest.TestCase):
    def test_motion_prediction_linear(self):
        slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)
        # Drive forward at 2.0 m/s for 1.0s
        slam.predict(v=2.0, omega=0.0, dt=1.0)
        x, y, theta = slam.get_robot_pose()

        self.assertAlmostEqual(x, 2.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(theta, 0.0, places=3)

        # Covariance in x should have grown due to motion noise
        cov = slam.get_robot_covariance()
        self.assertGreater(cov[0][0], 0.0)

    def test_motion_prediction_turning(self):
        slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)
        # Turn 90 degrees (pi/2 rad/s) for 1.0s with no forward speed
        slam.predict(v=0.0, omega=math.pi * 0.5, dt=1.0)
        x, y, theta = slam.get_robot_pose()

        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(theta, math.pi * 0.5, places=3)

    def test_landmark_initialization_and_update(self):
        slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)
        self.assertEqual(slam.num_landmarks, 0)

        # First observation of landmark at range 5.0m, bearing 0 rad (straight ahead)
        matched_ids = slam.update([(5.0, 0.0)])
        self.assertEqual(len(matched_ids), 1)
        self.assertEqual(slam.num_landmarks, 1)

        lms = slam.get_landmarks()
        lm_id, lx, ly, s_maj, s_min, ang = lms[0]
        self.assertEqual(lm_id, 0)
        self.assertAlmostEqual(lx, 5.0, places=2)
        self.assertAlmostEqual(ly, 0.0, places=2)

        # Landmark uncertainty before repeated measurements
        init_uncertainty = s_maj * s_min

        # Drive forward 1 meter
        slam.predict(v=1.0, omega=0.0, dt=1.0)
        # Re-observe landmark at range 4.0m, bearing 0 rad
        slam.update([(4.0, 0.0)])

        self.assertEqual(slam.num_landmarks, 1)  # Landmark correctly associated, not duplicated!

        # Observe again several times to fuse information
        for _ in range(5):
            slam.update([(4.0, 0.0)])

        updated_lms = slam.get_landmarks()
        _, _, _, updated_s_maj, updated_s_min, _ = updated_lms[0]
        updated_uncertainty = updated_s_maj * updated_s_min

        # Repeated measurements should reduce landmark covariance uncertainty
        self.assertLess(updated_uncertainty, init_uncertainty)

    def test_multi_landmark_tracking(self):
        slam = EKFSLAM(init_x=0.0, init_y=0.0, init_theta=0.0)

        # Landmark 1 at 3m directly ahead (0 deg)
        # Landmark 2 at 4m directly to the left (90 deg)
        slam.update([(3.0, 0.0), (4.0, math.pi * 0.5)])
        self.assertEqual(slam.num_landmarks, 2)

        lms = slam.get_landmarks()
        self.assertAlmostEqual(lms[0][1], 3.0, places=2)
        self.assertAlmostEqual(lms[0][2], 0.0, places=2)
        self.assertAlmostEqual(lms[1][1], 0.0, places=2)
        self.assertAlmostEqual(lms[1][2], 4.0, places=2)


if __name__ == "__main__":
    unittest.main()
