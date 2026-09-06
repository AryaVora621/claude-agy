"""
Structura: First-Principles Finite Element Analysis (FEA) & Modal Dynamics Engine.
Pure Python standard library implementation with zero external dependencies.
"""

from structura.types import (
    Node2D,
    Material,
    STRUCTURAL_STEEL,
    ALUMINUM_6061,
    TITANIUM_TI6AL4V,
    StressTensor2D,
    StrainTensor2D,
    BoundaryCondition,
    NodalLoad,
)
from structura.elements import (
    Element,
    Truss2D,
    Beam2D,
    TriangleCST,
    QuadQ4,
)
from structura.sparse import (
    DOKMatrix,
    CSRMatrix,
    solve_pcg,
)
from structura.mesh import (
    Mesh2D,
    generate_rectangular_mesh,
    generate_truss_bridge_mesh,
    generate_perforated_plate_mesh,
)
from structura.solver import (
    FEASolver,
    FEAResult,
)
from structura.modal import (
    ModalSolver,
    ModalResult,
    ModalMode,
)
from structura.visualizer import (
    BrailleFEACanvas,
    render_mesh_deformation,
    render_stress_field,
    render_fea_telemetry_hud,
    turbo_colormap,
)

__all__ = [
    "Node2D",
    "Material",
    "STRUCTURAL_STEEL",
    "ALUMINUM_6061",
    "TITANIUM_TI6AL4V",
    "StressTensor2D",
    "StrainTensor2D",
    "BoundaryCondition",
    "NodalLoad",
    "Element",
    "Truss2D",
    "Beam2D",
    "TriangleCST",
    "QuadQ4",
    "DOKMatrix",
    "CSRMatrix",
    "solve_pcg",
    "Mesh2D",
    "generate_rectangular_mesh",
    "generate_truss_bridge_mesh",
    "generate_perforated_plate_mesh",
    "FEASolver",
    "FEAResult",
    "ModalSolver",
    "ModalResult",
    "ModalMode",
    "BrailleFEACanvas",
    "render_mesh_deformation",
    "render_stress_field",
    "render_fea_telemetry_hud",
    "turbo_colormap",
]
