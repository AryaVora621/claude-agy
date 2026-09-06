"""
AeroFlow: Computational Fluid Dynamics, Lattice Boltzmann & Navier-Stokes Engine.
Zero external dependencies. Pure Python 3.10+ standard library implementation.
"""

from aeroflow.types import (
    Vector2D,
    Grid2D,
    VectorField2D,
    ObstacleMask,
    FluidParams,
)
from aeroflow.lbm import (
    LatticeD2Q9,
    LBMSolver,
)
from aeroflow.navier_stokes import (
    NavierStokesSolver,
)
from aeroflow.obstacles import (
    create_cylinder_obstacle,
    create_ellipse_obstacle,
    create_naca_airfoil_obstacle,
    create_plate_obstacle,
)
from aeroflow.aerodynamics import (
    AerodynamicForces,
    AerodynamicTracker,
)
from aeroflow.analysis import (
    FlowStatistics,
    compute_stream_function,
    trace_streamline,
    compute_enstrophy,
    compute_q_criterion,
)
from aeroflow.visualizer import (
    BrailleFlowCanvas,
    render_vorticity_field,
    render_velocity_vector_field,
    render_cfd_telemetry_hud,
)

__all__ = [
    "Vector2D",
    "Grid2D",
    "VectorField2D",
    "ObstacleMask",
    "FluidParams",
    "LatticeD2Q9",
    "LBMSolver",
    "NavierStokesSolver",
    "create_cylinder_obstacle",
    "create_ellipse_obstacle",
    "create_naca_airfoil_obstacle",
    "create_plate_obstacle",
    "AerodynamicForces",
    "AerodynamicTracker",
    "FlowStatistics",
    "compute_stream_function",
    "trace_streamline",
    "compute_enstrophy",
    "compute_q_criterion",
    "BrailleFlowCanvas",
    "render_vorticity_field",
    "render_velocity_vector_field",
    "render_cfd_telemetry_hud",
]
