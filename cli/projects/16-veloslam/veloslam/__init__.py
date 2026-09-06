"""
VeloSLAM: Autonomous Robotics, SLAM & Motion Planning Engine.
Zero-dependency first-principles Python standard library implementation.
"""

from veloslam.linalg import (
    Matrix,
    Vector,
    normalize_angle,
    mahalanobis_distance,
    covariance_ellipse_2d,
)
from veloslam.ekf_slam import (
    EKFSLAM,
    Landmark,
)
from veloslam.occupancy import (
    OccupancyGrid,
    LaserScan,
    bresenham_line,
)
from veloslam.dubins import (
    DubinsPath,
    dubins_shortest_path,
)
from veloslam.hybrid_astar import (
    Pose2D,
    HybridNode,
    HybridAStar,
)
from veloslam.dwa import (
    RobotState,
    DWAConfig,
    DWAPlanner,
)
from veloslam.visualizer import (
    render_braille_map,
)

__all__ = [
    "Matrix",
    "Vector",
    "normalize_angle",
    "mahalanobis_distance",
    "covariance_ellipse_2d",
    "EKFSLAM",
    "Landmark",
    "OccupancyGrid",
    "LaserScan",
    "bresenham_line",
    "DubinsPath",
    "dubins_shortest_path",
    "Pose2D",
    "HybridNode",
    "HybridAStar",
    "RobotState",
    "DWAConfig",
    "DWAPlanner",
    "render_braille_map",
]
