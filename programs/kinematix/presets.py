"""
Curated Robotic Presets for KineMatix 3D Studio.
Standard library Python: zero external dependencies.
"""

import math
from typing import Tuple, Dict, Any
from programs.kinematix.kinematics_engine import DHLink, SerialRobotArm, StewartPlatform, Vector3


def preset_puma560() -> Tuple[SerialRobotArm, Dict[str, Any]]:
    """
    Unimation PUMA 560 6-DOF Industrial Arm.
    Standard industrial manipulator with shoulder, elbow, and spherical wrist.
    """
    links = [
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=math.pi / 2, name="Base Waist (J1)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.4318, alpha=0.0, name="Shoulder (J2)"),
        DHLink(theta_offset=0.0, d_offset=0.1500, a=0.0203, alpha=-math.pi / 2, name="Elbow (J3)"),
        DHLink(theta_offset=0.0, d_offset=0.4318, a=0.0, alpha=math.pi / 2, name="Wrist Pitch (J4)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=-math.pi / 2, name="Wrist Yaw (J5)"),
        DHLink(theta_offset=0.0, d_offset=0.1000, a=0.0, alpha=0.0, name="Wrist Roll (J6)"),
    ]
    robot = SerialRobotArm(links, base_name="PUMA 560")
    initial_q = [0.0, -0.4, 1.2, 0.0, 0.8, 0.0]

    metadata = {
        "name": "PUMA 560 6-DOF Industrial Manipulator",
        "description": "Canonical 6-axis articulated serial arm with orthogonal shoulder and 3-axis spherical wrist.",
        "type": "serial",
        "initial_q": initial_q,
        "default_target": Vector3(0.35, 0.20, 0.35),
    }
    return robot, metadata


def preset_ur5() -> Tuple[SerialRobotArm, Dict[str, Any]]:
    """
    Universal Robots UR5 6-DOF Collaborative Cobot.
    Zero-offset elbow architecture widely used in modern flexible automation.
    """
    links = [
        DHLink(theta_offset=0.0, d_offset=0.089159, a=0.0, alpha=math.pi / 2, name="Base (J1)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=-0.42500, alpha=0.0, name="Shoulder (J2)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=-0.39225, alpha=0.0, name="Elbow (J3)"),
        DHLink(theta_offset=0.0, d_offset=0.10915, a=0.0, alpha=math.pi / 2, name="Wrist 1 (J4)"),
        DHLink(theta_offset=0.0, d_offset=0.09465, a=0.0, alpha=-math.pi / 2, name="Wrist 2 (J5)"),
        DHLink(theta_offset=0.0, d_offset=0.08230, a=0.0, alpha=0.0, name="Wrist 3 (J6)"),
    ]
    robot = SerialRobotArm(links, base_name="UR5 Cobot")
    initial_q = [0.0, -math.pi / 3, math.pi / 2, -math.pi / 6, 0.0, 0.0]

    metadata = {
        "name": "UR5 6-DOF Collaborative Cobot",
        "description": "Six rotating joints with orthogonal axes providing spherical reach and collaborative assembly dexterity.",
        "type": "serial",
        "initial_q": initial_q,
        "default_target": Vector3(0.40, 0.15, 0.30),
    }
    return robot, metadata


def preset_scara() -> Tuple[SerialRobotArm, Dict[str, Any]]:
    """
    SCARA (Selective Compliance Assembly Robot Arm) 4-DOF.
    High-speed planar articulation with vertical Z plunge and tool rotation.
    """
    links = [
        DHLink(theta_offset=0.0, d_offset=0.20, a=0.30, alpha=0.0, name="Arm 1 (J1)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.25, alpha=math.pi, name="Arm 2 (J2)"),
        DHLink(theta_offset=0.0, d_offset=0.10, a=0.0, alpha=0.0, is_revolute=False, joint_min=-0.20, joint_max=0.05, name="Z Plunge (J3)"),
        DHLink(theta_offset=0.0, d_offset=0.05, a=0.0, alpha=0.0, name="Tool Roll (J4)"),
    ]
    robot = SerialRobotArm(links, base_name="SCARA 4-DOF")
    initial_q = [0.5, -0.8, -0.05, 0.0]

    metadata = {
        "name": "SCARA 4-DOF Assembly Robot",
        "description": "Selective compliance kinematics for high-throughput pick-and-place with vertical linear Z quill.",
        "type": "serial",
        "initial_q": initial_q,
        "default_target": Vector3(0.35, 0.25, 0.10),
    }
    return robot, metadata


def preset_stanford_arm() -> Tuple[SerialRobotArm, Dict[str, Any]]:
    """
    Stanford Arm (1969 Scheiman).
    First computer-controlled robot arm with spherical joint kinematics and prismatic boom.
    """
    links = [
        DHLink(theta_offset=0.0, d_offset=0.25, a=0.0, alpha=-math.pi / 2, name="Base Pan (J1)"),
        DHLink(theta_offset=0.0, d_offset=0.15, a=0.0, alpha=math.pi / 2, name="Shoulder Tilt (J2)"),
        DHLink(theta_offset=0.0, d_offset=0.30, a=0.0, alpha=0.0, is_revolute=False, joint_min=0.1, joint_max=0.6, name="Prismatic Boom (J3)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=-math.pi / 2, name="Wrist Roll (J4)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=math.pi / 2, name="Wrist Pitch (J5)"),
        DHLink(theta_offset=0.0, d_offset=0.10, a=0.0, alpha=0.0, name="Wrist Yaw (J6)"),
    ]
    robot = SerialRobotArm(links, base_name="Stanford Arm")
    initial_q = [0.2, 0.4, 0.25, 0.0, 0.0, 0.0]

    metadata = {
        "name": "Stanford Arm 6-DOF (RRPRRR)",
        "description": "Historical 1969 manipulator featuring prismatic boom extension and 3-axis spherical wrist.",
        "type": "serial",
        "initial_q": initial_q,
        "default_target": Vector3(0.30, 0.20, 0.35),
    }
    return robot, metadata


def preset_anthropomorphic_7dof() -> Tuple[SerialRobotArm, Dict[str, Any]]:
    """
    7-DOF Kinematically Redundant Anthropomorphic Arm.
    Shoulder (3-DOF) + Elbow (1-DOF) + Wrist (3-DOF) with elbow swivel redundancy.
    """
    links = [
        DHLink(theta_offset=0.0, d_offset=0.20, a=0.0, alpha=-math.pi / 2, name="Shoulder Roll (J1)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=math.pi / 2, name="Shoulder Pitch (J2)"),
        DHLink(theta_offset=0.0, d_offset=0.30, a=0.0, alpha=-math.pi / 2, name="Shoulder Yaw (J3)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=math.pi / 2, name="Elbow Flex (J4)"),
        DHLink(theta_offset=0.0, d_offset=0.28, a=0.0, alpha=-math.pi / 2, name="Forearm Roll (J5)"),
        DHLink(theta_offset=0.0, d_offset=0.0, a=0.0, alpha=math.pi / 2, name="Wrist Pitch (J6)"),
        DHLink(theta_offset=0.0, d_offset=0.10, a=0.0, alpha=0.0, name="Wrist Roll (J7)"),
    ]
    robot = SerialRobotArm(links, base_name="7-DOF Humanoid Arm")
    initial_q = [0.0, 0.5, 0.0, 1.1, 0.0, 0.4, 0.0]

    metadata = {
        "name": "7-DOF Redundant Anthropomorphic Arm",
        "description": "Human arm kinematic model featuring 1 degree of redundancy enabling obstacle avoidance via null-space swivel.",
        "type": "serial",
        "initial_q": initial_q,
        "default_target": Vector3(0.35, 0.20, 0.30),
    }
    return robot, metadata


def preset_stewart_hexapod() -> Tuple[StewartPlatform, Dict[str, Any]]:
    """
    6-DOF Stewart-Gough Parallel Kinematic Platform.
    6 linear actuators driving top mobile stage in translation and rotation.
    """
    platform = StewartPlatform(base_radius=0.35, platform_radius=0.22, home_height=0.38)
    metadata = {
        "name": "Stewart-Gough 6-DOF Parallel Hexapod",
        "description": "High-stiffness 6-axis parallel platform with 6 linear prismatic struts for flight simulation and precision machining.",
        "type": "parallel",
        "default_translation": Vector3(0.0, 0.0, 0.0),
        "default_rpy": (0.0, 0.0, 0.0),
    }
    return platform, metadata


PRESETS = {
    "puma560": preset_puma560,
    "ur5": preset_ur5,
    "scara": preset_scara,
    "stanford": preset_stanford_arm,
    "humanoid7": preset_anthropomorphic_7dof,
    "hexapod": preset_stewart_hexapod,
}
