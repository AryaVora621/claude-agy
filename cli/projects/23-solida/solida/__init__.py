"""
Solida: 3D Boundary Representation (B-Rep) Solid Modeling CAD & NURBS Geometric Modeling Kernel.
Pure Python standard library. Zero external dependencies.
"""

from solida.geometry import (
    Vector3D,
    Matrix4x4,
    Quaternion,
    Ray3D,
    Plane,
    BoundingBox3D,
)
from solida.nurbs import (
    create_clamped_knot_vector,
    basis_function,
    basis_derivative,
    NURBSCurve,
    NURBSSurface,
)
from solida.brep import (
    Vertex,
    HalfEdge,
    Edge,
    Loop,
    Face,
    Shell,
    Solid,
    build_solid_from_polygons,
)
from solida.primitives import (
    make_box,
    make_cylinder,
    make_sphere,
    make_cone,
    make_torus,
)
from solida.features import (
    Sketch2D,
    extrude,
    revolve,
    loft,
)
from solida.csg import (
    CSGPolygon,
    CSGNode,
    solid_to_csg,
    csg_to_solid,
    csg_union,
    csg_difference,
    csg_intersection,
)
from solida.tessellation import (
    Facet3D,
    triangulate_polygon_2d,
    triangulate_polygon_3d,
    tessellate_solid,
    export_stl_ascii,
    export_stl_binary,
    export_obj,
    parse_stl_ascii,
    parse_stl_binary,
)
from solida.visualizer import (
    Camera3D,
    BrailleCADCanvas,
    render_solid_cad,
    render_cad_hud,
)
from solida.kernel import (
    Part,
    Workplane,
    Assembly,
)

__version__ = "1.0.0"
__all__ = [
    # Geometry
    "Vector3D",
    "Matrix4x4",
    "Quaternion",
    "Ray3D",
    "Plane",
    "BoundingBox3D",
    # NURBS
    "create_clamped_knot_vector",
    "basis_function",
    "basis_derivative",
    "NURBSCurve",
    "NURBSSurface",
    # B-Rep Topology
    "Vertex",
    "HalfEdge",
    "Edge",
    "Loop",
    "Face",
    "Shell",
    "Solid",
    "build_solid_from_polygons",
    # Primitives
    "make_box",
    "make_cylinder",
    "make_sphere",
    "make_cone",
    "make_torus",
    # Features
    "Sketch2D",
    "extrude",
    "revolve",
    "loft",
    # CSG Booleans
    "CSGPolygon",
    "CSGNode",
    "solid_to_csg",
    "csg_to_solid",
    "csg_union",
    "csg_difference",
    "csg_intersection",
    # Tessellation & Codecs
    "Facet3D",
    "triangulate_polygon_2d",
    "triangulate_polygon_3d",
    "tessellate_solid",
    "export_stl_ascii",
    "export_stl_binary",
    "export_obj",
    "parse_stl_ascii",
    "parse_stl_binary",
    # Visualizer
    "Camera3D",
    "BrailleCADCanvas",
    "render_solid_cad",
    "render_cad_hud",
    # Kernel & Assembly
    "Part",
    "Workplane",
    "Assembly",
]
