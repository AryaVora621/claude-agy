"""
Automated Unit Test Suite for KineMatix 3D Kinematics Engine.
16 unit tests verifying forward kinematics, geometric Jacobians,
DLS inverse kinematics, manipulability, and Stewart platforms.
Standard library Python unittest: zero external dependencies.
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.kinematix.kinematics_engine import (
    Vector3,
    Matrix4,
    DHLink,
    SerialRobotArm,
    StewartPlatform,
)
from programs.kinematix.presets import (
    preset_puma560,
    preset_ur5,
    preset_scara,
    preset_stanford_arm,
    preset_anthropomorphic_7dof,
    preset_stewart_hexapod,
)


class TestKineMatix(unittest.TestCase):
    """Test suite for 3D robotics kinematics and parallel platforms."""

    def test_vector3_operations(self):
        """Verify vector addition, cross product orthogonality, and normalization."""
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        v3 = v1.cross(v2)

        self.assertAlmostEqual(v3.x, 0.0)
        self.assertAlmostEqual(v3.y, 0.0)
        self.assertAlmostEqual(v3.z, 1.0)
        self.assertAlmostEqual(v3.norm(), 1.0)
        self.assertAlmostEqual(v1.dot(v3), 0.0)
        self.assertAlmostEqual(v2.dot(v3), 0.0)

    def test_matrix4_dh_composition(self):
        """Verify Denavit-Hartenberg homogeneous matrix construction and properties."""
        m = Matrix4.from_dh(theta=math.pi / 2, d=0.5, a=0.2, alpha=0.0)
        p = Vector3(0.0, 0.0, 0.0)
        p_trans = m.transform_point(p)

        self.assertAlmostEqual(p_trans.x, 0.0, places=5)
        self.assertAlmostEqual(p_trans.y, 0.2, places=5)
        self.assertAlmostEqual(p_trans.z, 0.5, places=5)

    def test_forward_kinematics_planar_reach(self):
        """Verify 2-link planar arm forward kinematics matches simple trigonometry."""
        links = [
            DHLink(a=0.5, alpha=0.0, name="Link1"),
            DHLink(a=0.3, alpha=0.0, name="Link2"),
        ]
        arm = SerialRobotArm(links)

        # Fully extended at theta = [0, 0]: X should be 0.5 + 0.3 = 0.8, Y = 0
        pos_ext, _ = arm.get_end_effector_pose([0.0, 0.0])
        self.assertAlmostEqual(pos_ext.x, 0.8, delta=1e-4)
        self.assertAlmostEqual(pos_ext.y, 0.0, delta=1e-4)

        # At theta = [pi/2, 0]: X should be 0, Y should be 0.8
        pos_rot, _ = arm.get_end_effector_pose([math.pi / 2, 0.0])
        self.assertAlmostEqual(pos_rot.x, 0.0, delta=1e-4)
        self.assertAlmostEqual(pos_rot.y, 0.8, delta=1e-4)

    def test_geometric_jacobian_dimensions(self):
        """Verify Jacobian matrix structure has 6 rows and N columns."""
        robot, _ = preset_puma560()
        q = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
        j = robot.compute_jacobian(q)

        self.assertEqual(len(j), 6, "Jacobian must have 6 rows (3 linear, 3 angular)")
        self.assertEqual(len(j[0]), 6, "Jacobian must have 6 columns for 6 joints")

    def test_numerical_jacobian_agreement(self):
        """Verify analytical geometric Jacobian matches finite difference approximation."""
        robot, _ = preset_puma560()
        q0 = [0.2, -0.3, 0.5, 0.1, 0.4, -0.2]
        j_geom = robot.compute_jacobian(q0)

        eps = 1e-6
        p0, _ = robot.get_end_effector_pose(q0)

        for i in range(robot.num_joints):
            q_plus = list(q0)
            q_plus[i] += eps
            p_plus, _ = robot.get_end_effector_pose(q_plus)

            dp_num = (p_plus - p0) / eps
            self.assertAlmostEqual(dp_num.x, j_geom[0][i], delta=1e-3)
            self.assertAlmostEqual(dp_num.y, j_geom[1][i], delta=1e-3)
            self.assertAlmostEqual(dp_num.z, j_geom[2][i], delta=1e-3)

    def test_inverse_kinematics_puma_convergence(self):
        """Verify DLS inverse kinematics converges to within 2 mm of reachable target."""
        robot, _ = preset_puma560()
        target = Vector3(0.38, 0.18, 0.32)
        initial_q = [0.0, -0.4, 1.2, 0.0, 0.8, 0.0]

        solved_q, converged, err = robot.solve_inverse_kinematics_dls(
            target,
            initial_angles=initial_q,
            max_iter=60,
            pos_tolerance=0.002,
        )

        self.assertTrue(converged, f"PUMA IK should converge, final error: {err:.4f} m")
        self.assertLess(err, 0.002, "Error must be below 2 mm tolerance")

        actual_pos, _ = robot.get_end_effector_pose(solved_q)
        dist = (target - actual_pos).norm()
        self.assertLess(dist, 0.002)

    def test_singularity_damping_robustness(self):
        """Verify Damped Least-Squares prevents numerical explosion at singular configurations."""
        robot, _ = preset_puma560()
        # Fully outstretched singular configuration
        singular_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        w_sing = robot.compute_manipulability(singular_q)

        # Target slightly beyond reach
        target_out = Vector3(1.5, 0.0, 0.5)
        solved_q, _, _ = robot.solve_inverse_kinematics_dls(
            target_out,
            initial_angles=singular_q,
            max_iter=20,
            damping=0.1,
        )

        # Solved joint angles must remain finite and bounded without NaN or Inf
        for angle in solved_q:
            self.assertFalse(math.isnan(angle), "Joint angle must not be NaN near singularity")
            self.assertFalse(math.isinf(angle), "Joint angle must not be Inf near singularity")

    def test_joint_limit_clamping(self):
        """Verify solved joint angles strictly respect physical joint limits."""
        links = [
            DHLink(a=0.4, joint_min=-0.5, joint_max=0.5, name="Constrained Joint 1"),
            DHLink(a=0.3, joint_min=-0.8, joint_max=0.8, name="Constrained Joint 2"),
        ]
        robot = SerialRobotArm(links)
        target = Vector3(1.0, 1.0, 0.0)  # Extreme target

        solved_q, _, _ = robot.solve_inverse_kinematics_dls(target, max_iter=30)
        self.assertGreaterEqual(solved_q[0], -0.5)
        self.assertLessEqual(solved_q[0], 0.5)
        self.assertGreaterEqual(solved_q[1], -0.8)
        self.assertLessEqual(solved_q[1], 0.8)

    def test_manipulability_index_positivity(self):
        """Verify Yoshikawa manipulability index is non-negative and positive in regular configurations."""
        robot, _ = preset_puma560()
        nominal_q = [0.0, -0.5, 1.0, 0.0, 0.5, 0.0]
        w = robot.compute_manipulability(nominal_q)

        self.assertGreater(w, 0.001, "Manipulability index must be positive in dexterous posture")

    def test_manipulability_drop_near_boundary(self):
        """Verify manipulability index decreases significantly as arm approaches singular posture."""
        robot, _ = preset_puma560()
        dexterous_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        singular_q = [0.0, 0.0, 1.57, 0.0, 0.0, 0.0]

        w_dexterous = robot.compute_manipulability(dexterous_q)
        w_singular = robot.compute_manipulability(singular_q)

        self.assertGreater(w_dexterous, w_singular * 10.0, "Dexterous posture must have significantly higher manipulability than singular")

    def test_scara_prismatic_joint(self):
        """Verify SCARA kinematics properly handles prismatic linear vertical Z displacement."""
        scara, _ = preset_scara()
        q1 = [0.0, 0.0, -0.05, 0.0]
        q2 = [0.0, 0.0, -0.15, 0.0]

        p1, _ = scara.get_end_effector_pose(q1)
        p2, _ = scara.get_end_effector_pose(q2)

        # Vertical height displacement magnitude should match prismatic displacement difference: 0.10 m
        self.assertAlmostEqual(abs(p1.z - p2.z), 0.10, delta=1e-4)

    def test_ur5_forward_kinematics(self):
        """Verify UR5 forward kinematics computes realistic end-effector coordinates."""
        ur5, meta = preset_ur5()
        q = meta["initial_q"]
        pos, _ = ur5.get_end_effector_pose(q)

        self.assertTrue(0.3 < pos.norm() < 0.85, "UR5 reach distance should be within physical workspace")
        self.assertTrue(-0.2 < pos.z < 0.8, "UR5 Z position should be above ground")

    def test_stanford_arm_kinematics(self):
        """Verify Stanford Arm forward kinematics with prismatic extension."""
        stanford, meta = preset_stanford_arm()
        q = meta["initial_q"]
        pos, _ = stanford.get_end_effector_pose(q)

        self.assertIsNotNone(pos)
        self.assertGreater(pos.norm(), 0.2)

    def test_anthropomorphic_7dof_ik(self):
        """Verify 7-DOF redundant arm successfully reaches spatial targets."""
        robot, meta = preset_anthropomorphic_7dof()
        target = meta["default_target"]

        solved_q, converged, err = robot.solve_inverse_kinematics_dls(
            target,
            initial_angles=meta["initial_q"],
            max_iter=50,
            pos_tolerance=0.003,
        )

        self.assertTrue(converged, f"7-DOF arm should reach target, error: {err:.4f}")
        self.assertLess(err, 0.003)

    def test_stewart_hexapod_symmetry(self):
        """Verify Stewart-Gough platform yields symmetric leg lengths at pure vertical home displacement."""
        hexapod, meta = preset_stewart_hexapod()
        lengths, pts = hexapod.inverse_kinematics(
            translation=Vector3(0.0, 0.0, 0.0),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

        self.assertEqual(len(lengths), 6, "Platform must compute 6 leg lengths")
        self.assertEqual(len(pts), 6, "Platform must compute 6 platform joint coordinates")

        # In symmetric home state, all 6 actuator struts must have equal length
        for l in lengths[1:]:
            self.assertAlmostEqual(l, lengths[0], delta=1e-4)

    def test_presets_catalog_loading(self):
        """Verify all curated robotic presets instantiate correctly without exceptions."""
        presets = [
            preset_puma560,
            preset_ur5,
            preset_scara,
            preset_stanford_arm,
            preset_anthropomorphic_7dof,
            preset_stewart_hexapod,
        ]
        for p_fn in presets:
            system, meta = p_fn()
            self.assertIsNotNone(system)
            self.assertIn("name", meta)
            self.assertIn("type", meta)


if __name__ == "__main__":
    unittest.main()
