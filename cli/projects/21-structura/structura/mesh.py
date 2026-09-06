"""
Structura: 2D Finite Element Mesh Data Structure and Parametric Generators.
Implements:
  - Mesh2D container
  - generate_rectangular_mesh (Quad Q4 or Triangle CST)
  - generate_truss_bridge_mesh (Pratt/Warren truss bridge)
  - generate_perforated_plate_mesh (plate with central circular hole for Kirsch benchmark)
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from structura.types import (
    Node2D,
    Material,
    BoundaryCondition,
    NodalLoad,
    STRUCTURAL_STEEL,
)
from structura.elements import Element, Truss2D, Beam2D, TriangleCST, QuadQ4


class Mesh2D:
    """
    Container for finite element nodes, elements, boundary conditions, and loads.
    """
    def __init__(self):
        self.nodes: Dict[int, Node2D] = {}
        self.elements: List[Element] = []
        self.boundary_conditions: List[BoundaryCondition] = []
        self.nodal_loads: List[NodalLoad] = []

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def num_dofs(self) -> int:
        """
        Total degrees of freedom in the mesh.
        Continuum / Truss: 2 DOFs per node (Ux, Uy).
        Beams / Frames: 3 DOFs per node (Ux, Uy, Theta_z).
        """
        has_beams = any(isinstance(elem, Beam2D) for elem in self.elements)
        dofs_per_node = 3 if has_beams else 2
        return self.num_nodes * dofs_per_node

    def add_node(self, x: float, y: float, node_id: Optional[int] = None) -> int:
        """Add a node and return its ID."""
        if node_id is None:
            node_id = len(self.nodes)
        if node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists.")
        self.nodes[node_id] = Node2D(id=node_id, x=x, y=y)
        return node_id

    def add_element(self, element: Element) -> None:
        """Add an element to the mesh."""
        self.elements.append(element)

    def add_boundary_condition(self, node_id: int, dof: int, prescribed_value: float = 0.0) -> None:
        """Add displacement constraint at node DOF (0: Ux, 1: Uy, 2: Theta_z)."""
        self.boundary_conditions.append(BoundaryCondition(node_id, dof, prescribed_value))

    def add_nodal_load(self, node_id: int, fx: float = 0.0, fy: float = 0.0, moment: float = 0.0) -> None:
        """Add external point load at node."""
        self.nodal_loads.append(NodalLoad(node_id, fx, fy, moment))

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Compute bounding box (min_x, max_x, min_y, max_y)."""
        if not self.nodes:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return (min(xs), max(xs), min(ys), max(ys))

    def find_nodes_at_x(self, target_x: float, tol: float = 1e-4) -> List[int]:
        """Find all node IDs with x-coordinate close to target_x."""
        return [nid for nid, n in self.nodes.items() if abs(n.x - target_x) <= tol]

    def find_nodes_at_y(self, target_y: float, tol: float = 1e-4) -> List[int]:
        """Find all node IDs with y-coordinate close to target_y."""
        return [nid for nid, n in self.nodes.items() if abs(n.y - target_y) <= tol]

    def find_node_nearest(self, x: float, y: float) -> int:
        """Find node ID closest to (x, y)."""
        best_id = -1
        min_d = float("inf")
        for nid, n in self.nodes.items():
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < min_d:
                min_d = d
                best_id = nid
        return best_id


def generate_rectangular_mesh(
    length: float,
    height: float,
    nx: int,
    ny: int,
    material: Material = STRUCTURAL_STEEL,
    elem_type: str = "quad",
) -> Mesh2D:
    """
    Generate a 2D structured rectangular domain [0, length] x [0, height].
    elem_type: 'quad' for 4-node QuadQ4, or 'tri' for 3-node TriangleCST.
    """
    mesh = Mesh2D()
    dx = length / nx
    dy = height / ny

    # Create nodes
    for j in range(ny + 1):
        y = j * dy
        for i in range(nx + 1):
            x = i * dx
            mesh.add_node(x, y)

    def node_idx(ix: int, iy: int) -> int:
        return iy * (nx + 1) + ix

    elem_id = 0
    for j in range(ny):
        for i in range(nx):
            n0 = node_idx(i, j)
            n1 = node_idx(i + 1, j)
            n2 = node_idx(i + 1, j + 1)
            n3 = node_idx(i, j + 1)

            if elem_type.lower() == "quad":
                quad = QuadQ4(elem_id, (n0, n1, n2, n3), material)
                mesh.add_element(quad)
                elem_id += 1
            elif elem_type.lower() == "tri":
                # Split quad into two CST triangles
                t1 = TriangleCST(elem_id, (n0, n1, n2), material)
                elem_id += 1
                t2 = TriangleCST(elem_id, (n0, n2, n3), material)
                elem_id += 1
                mesh.add_element(t1)
                mesh.add_element(t2)
            else:
                raise ValueError(f"Unknown elem_type '{elem_type}'. Use 'quad' or 'tri'.")

    return mesh


def generate_truss_bridge_mesh(
    span: float,
    height: float,
    num_bays: int = 4,
    material: Material = STRUCTURAL_STEEL,
    area: float = 0.005,
) -> Mesh2D:
    """
    Generate a 2D Pratt/Warren planar truss bridge.
    Consists of bottom chord, top chord, vertical posts, and diagonals.
    """
    mesh = Mesh2D()
    bay_len = span / num_bays

    # Bottom chord nodes (y = 0): 0 to num_bays
    bottom_nodes = []
    for i in range(num_bays + 1):
        nid = mesh.add_node(x=i * bay_len, y=0.0)
        bottom_nodes.append(nid)

    # Top chord nodes (y = height): 1 to num_bays - 1 (trapezoidal / Warren)
    top_nodes = []
    for i in range(num_bays + 1):
        nid = mesh.add_node(x=i * bay_len, y=height)
        top_nodes.append(nid)

    elem_id = 0
    # 1. Bottom chord elements
    for i in range(num_bays):
        mesh.add_element(Truss2D(elem_id, (bottom_nodes[i], bottom_nodes[i + 1]), material, area))
        elem_id += 1

    # 2. Top chord elements
    for i in range(num_bays):
        mesh.add_element(Truss2D(elem_id, (top_nodes[i], top_nodes[i + 1]), material, area))
        elem_id += 1

    # 3. Vertical members
    for i in range(num_bays + 1):
        mesh.add_element(Truss2D(elem_id, (bottom_nodes[i], top_nodes[i]), material, area))
        elem_id += 1

    # 4. Diagonal cross members
    for i in range(num_bays):
        if i % 2 == 0:
            mesh.add_element(Truss2D(elem_id, (bottom_nodes[i], top_nodes[i + 1]), material, area))
        else:
            mesh.add_element(Truss2D(elem_id, (top_nodes[i], bottom_nodes[i + 1]), material, area))
        elem_id += 1

    return mesh


def generate_perforated_plate_mesh(
    width: float = 2.0,
    height: float = 2.0,
    hole_radius: float = 0.4,
    n_radial: int = 5,
    n_tangential: int = 8,
    material: Material = STRUCTURAL_STEEL,
) -> Mesh2D:
    """
    Generate quarter-symmetry finite element mesh of a plate with a central circular hole.
    Covers quadrant [0, width] x [0, height] with circular inner hole boundary r = hole_radius.
    This is the standard Kirsch benchmark problem for elastic stress concentration.
    """
    mesh = Mesh2D()

    # Angles from 0 to pi/2 (first quadrant)
    theta_vals = [i * (0.5 * math.pi / n_tangential) for i in range(n_tangential + 1)]

    # Generate nodes row by row from inner radius to outer boundary
    for j in range(n_radial + 1):
        s = j / float(n_radial)  # 0.0 (hole) to 1.0 (outer boundary)
        for i, theta in enumerate(theta_vals):
            r_inner = hole_radius
            # Outer boundary coordinates along rays
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Ray intersection with outer rectangle [0, width] x [0, height]
            if cos_t > 1e-6 and (height * cos_t) > (width * sin_t):
                r_outer = width / cos_t
            else:
                r_outer = height / max(1e-6, sin_t)

            r = r_inner + s * (r_outer - r_inner)
            x = r * cos_t
            y = r * sin_t
            mesh.add_node(x, y)

    def node_id(radial_idx: int, tang_idx: int) -> int:
        return radial_idx * (n_tangential + 1) + tang_idx

    elem_id = 0
    for j in range(n_radial):
        for i in range(n_tangential):
            # Counter-clockwise ordering: n0(r_j, theta_i) -> n1(r_{j+1}, theta_i) -> n2(r_{j+1}, theta_{i+1}) -> n3(r_j, theta_{i+1})
            n0 = node_id(j, i)
            n1 = node_id(j + 1, i)
            n2 = node_id(j + 1, i + 1)
            n3 = node_id(j, i + 1)
            quad = QuadQ4(elem_id, (n0, n1, n2, n3), material)
            mesh.add_element(quad)
            elem_id += 1

    return mesh
