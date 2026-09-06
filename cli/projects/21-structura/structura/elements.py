"""
Structura: Finite Element Formulations.
Implements:
  - Truss2D (2-node axial tension/compression)
  - Beam2D (2-node Euler-Bernoulli frame element with bending)
  - TriangleCST (3-node Constant Strain Triangle / T3 continuum)
  - QuadQ4 (4-node bilinear isoparametric quadrilateral with 2x2 Gauss quadrature)
Zero external dependencies.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

from structura.types import (
    Node2D,
    Material,
    StressTensor2D,
    StrainTensor2D,
)


def mat_mul_3x3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two 3x3 matrices."""
    return [
        [
            a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]
            for c in range(3)
        ]
        for r in range(3)
    ]


def mat_mul_rect(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """General matrix multiplication: C = A * B."""
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    c = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols_b):
                c[i][j] += aik * b[k][j]
    return c


def transpose(a: List[List[float]]) -> List[List[float]]:
    """Matrix transpose."""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] for r in range(rows)] for c in range(cols)]


class Element(ABC):
    """
    Abstract base class for 2D finite elements.
    """
    def __init__(self, element_id: int, node_ids: Tuple[int, ...], material: Material):
        self.element_id = element_id
        self.node_ids = node_ids
        self.material = material

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_dofs_per_node(self) -> int:
        pass

    @abstractmethod
    def get_dofs(self, nodes_dict: Dict[int, Node2D]) -> List[int]:
        """Return list of global DOF indices associated with this element."""
        pass

    @abstractmethod
    def compute_stiffness_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        plane_strain: bool = False,
    ) -> List[List[float]]:
        """Compute local-to-global element stiffness matrix K_e."""
        pass

    @abstractmethod
    def compute_mass_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        lumped: bool = True,
    ) -> List[List[float]]:
        """Compute element mass matrix M_e."""
        pass

    @abstractmethod
    def compute_stress_strain(
        self,
        nodes_dict: Dict[int, Node2D],
        element_displacements: List[float],
        plane_strain: bool = False,
    ) -> Tuple[StressTensor2D, StrainTensor2D]:
        """Compute element Cauchy stress and engineering strain."""
        pass


class Truss2D(Element):
    """
    2D 2-node bar/truss element carrying purely axial loads.
    DOFs per node: 2 (Ux, Uy).
    Total element DOFs: 4.
    """
    def __init__(
        self,
        element_id: int,
        node_ids: Tuple[int, int],
        material: Material,
        cross_section_area: float = 0.001,
    ):
        super().__init__(element_id, node_ids, material)
        self.cross_section_area = cross_section_area

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def num_dofs_per_node(self) -> int:
        return 2

    def get_dofs(self, nodes_dict: Dict[int, Node2D]) -> List[int]:
        n1, n2 = self.node_ids
        return [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]

    def _get_geometry(self, nodes_dict: Dict[int, Node2D]) -> Tuple[float, float, float]:
        n1 = nodes_dict[self.node_ids[0]]
        n2 = nodes_dict[self.node_ids[1]]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            raise ValueError(f"Truss element {self.element_id} has zero length.")
        c = dx / length
        s = dy / length
        return length, c, s

    def compute_stiffness_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        plane_strain: bool = False,
    ) -> List[List[float]]:
        length, c, s = self._get_geometry(nodes_dict)
        k = (self.material.elastic_modulus * self.cross_section_area) / length

        c2 = c * c
        s2 = s * s
        cs = c * s

        return [
            [ k * c2,  k * cs, -k * c2, -k * cs],
            [ k * cs,  k * s2, -k * cs, -k * s2],
            [-k * c2, -k * cs,  k * c2,  k * cs],
            [-k * cs, -k * s2,  k * cs,  k * s2],
        ]

    def compute_mass_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        lumped: bool = True,
    ) -> List[List[float]]:
        length, _, _ = self._get_geometry(nodes_dict)
        total_mass = self.material.density * self.cross_section_area * length
        half_m = 0.5 * total_mass

        if lumped:
            return [
                [half_m, 0.0, 0.0, 0.0],
                [0.0, half_m, 0.0, 0.0],
                [0.0, 0.0, half_m, 0.0],
                [0.0, 0.0, 0.0, half_m],
            ]
        else:
            # Consistent mass matrix
            m6 = total_mass / 6.0
            return [
                [2 * m6, 0.0, 1 * m6, 0.0],
                [0.0, 2 * m6, 0.0, 1 * m6],
                [1 * m6, 0.0, 2 * m6, 0.0],
                [0.0, 1 * m6, 0.0, 2 * m6],
            ]

    def compute_stress_strain(
        self,
        nodes_dict: Dict[int, Node2D],
        element_displacements: List[float],
        plane_strain: bool = False,
    ) -> Tuple[StressTensor2D, StrainTensor2D]:
        length, c, s = self._get_geometry(nodes_dict)
        u1, v1, u2, v2 = element_displacements

        # Elongation = delta_L = (u2 - u1)*c + (v2 - v1)*s
        delta_l = (u2 - u1) * c + (v2 - v1) * s
        axial_strain = delta_l / length
        axial_stress = self.material.elastic_modulus * axial_strain

        # Project axial stress onto Cartesian directions
        stress = StressTensor2D(
            sigma_x=axial_stress * c * c,
            sigma_y=axial_stress * s * s,
            tau_xy=axial_stress * c * s,
        )
        strain = StrainTensor2D(
            eps_x=axial_strain * c * c,
            eps_y=axial_strain * s * s,
            gamma_xy=2.0 * axial_strain * c * s,
        )
        return stress, strain


class Beam2D(Element):
    """
    2D 2-node Euler-Bernoulli frame element carrying axial tension/compression,
    shear force, and bending moments.
    DOFs per node: 3 (Ux, Uy, Theta_z).
    Total element DOFs: 6.
    """
    def __init__(
        self,
        element_id: int,
        node_ids: Tuple[int, int],
        material: Material,
        cross_section_area: float = 0.01,
        moment_of_inertia: float = 1e-5,
    ):
        super().__init__(element_id, node_ids, material)
        self.cross_section_area = cross_section_area
        self.moment_of_inertia = moment_of_inertia

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def num_dofs_per_node(self) -> int:
        return 3

    def get_dofs(self, nodes_dict: Dict[int, Node2D]) -> List[int]:
        n1, n2 = self.node_ids
        return [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2]

    def _get_geometry(self, nodes_dict: Dict[int, Node2D]) -> Tuple[float, float, float]:
        n1 = nodes_dict[self.node_ids[0]]
        n2 = nodes_dict[self.node_ids[1]]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            raise ValueError(f"Beam element {self.element_id} has zero length.")
        return length, dx / length, dy / length

    def compute_stiffness_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        plane_strain: bool = False,
    ) -> List[List[float]]:
        length, c, s = self._get_geometry(nodes_dict)
        e = self.material.elastic_modulus
        a = self.cross_section_area
        i_z = self.moment_of_inertia
        l2 = length * length
        l3 = l2 * length

        # Local stiffness matrix k_local (6x6)
        ka = e * a / length
        kb1 = 12.0 * e * i_z / l3
        kb2 = 6.0 * e * i_z / l2
        kb3 = 4.0 * e * i_z / length
        kb4 = 2.0 * e * i_z / length

        k_local = [
            [ ka,   0.0,   0.0, -ka,   0.0,   0.0],
            [0.0,   kb1,   kb2,  0.0,  -kb1,   kb2],
            [0.0,   kb2,   kb3,  0.0,  -kb2,   kb4],
            [-ka,   0.0,   0.0,  ka,   0.0,   0.0],
            [0.0,  -kb1,  -kb2,  0.0,   kb1,  -kb2],
            [0.0,   kb2,   kb4,  0.0,  -kb2,   kb3],
        ]

        # Transformation matrix T (6x6)
        t_mat = [
            [  c,   s, 0.0, 0.0, 0.0, 0.0],
            [ -s,   c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,   c,   s, 0.0],
            [0.0, 0.0, 0.0,  -s,   c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        # K_global = T^T * k_local * T
        tt = transpose(t_mat)
        temp = mat_mul_rect(k_local, t_mat)
        return mat_mul_rect(tt, temp)

    def compute_mass_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        lumped: bool = True,
    ) -> List[List[float]]:
        length, _, _ = self._get_geometry(nodes_dict)
        total_mass = self.material.density * self.cross_section_area * length
        half_m = 0.5 * total_mass
        rot_inertia = total_mass * length * length / 24.0

        m = [[0.0] * 6 for _ in range(6)]
        m[0][0] = half_m
        m[1][1] = half_m
        m[2][2] = rot_inertia
        m[3][3] = half_m
        m[4][4] = half_m
        m[5][5] = rot_inertia
        return m

    def compute_stress_strain(
        self,
        nodes_dict: Dict[int, Node2D],
        element_displacements: List[float],
        plane_strain: bool = False,
    ) -> Tuple[StressTensor2D, StrainTensor2D]:
        length, c, s = self._get_geometry(nodes_dict)
        u1, v1, theta1, u2, v2, theta2 = element_displacements

        # Local displacements u_local = T * u_global
        u1_loc =  c * u1 + s * v1
        v1_loc = -s * u1 + c * v1
        u2_loc =  c * u2 + s * v2
        v2_loc = -s * u2 + c * v2

        axial_strain = (u2_loc - u1_loc) / length
        axial_stress = self.material.elastic_modulus * axial_strain

        # Max bending moment at ends: M = E*I * curvature
        # Estimate bending stress sigma_b = M * (h / 2) / I
        stress = StressTensor2D(
            sigma_x=axial_stress * c * c,
            sigma_y=axial_stress * s * s,
            tau_xy=axial_stress * c * s,
        )
        strain = StrainTensor2D(
            eps_x=axial_strain * c * c,
            eps_y=axial_strain * s * s,
            gamma_xy=2.0 * axial_strain * c * s,
        )
        return stress, strain


class TriangleCST(Element):
    """
    Constant Strain Triangle (CST / T3) element for 2D plane stress and plane strain.
    DOFs per node: 2 (Ux, Uy).
    Total element DOFs: 6.
    """
    def __init__(self, element_id: int, node_ids: Tuple[int, int, int], material: Material):
        super().__init__(element_id, node_ids, material)

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def num_dofs_per_node(self) -> int:
        return 2

    def get_dofs(self, nodes_dict: Dict[int, Node2D]) -> List[int]:
        n1, n2, n3 = self.node_ids
        return [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1]

    def _get_geometry(self, nodes_dict: Dict[int, Node2D]) -> Tuple[float, List[float], List[float]]:
        n1 = nodes_dict[self.node_ids[0]]
        n2 = nodes_dict[self.node_ids[1]]
        n3 = nodes_dict[self.node_ids[2]]

        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y

        # Twice the area of the triangle
        two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(two_area) < 1e-12:
            raise ValueError(f"Triangle CST element {self.element_id} is degenerate/collinear.")

        # If winding is clockwise, flip sign to keep area positive
        area = 0.5 * abs(two_area)

        # Gradients b_i = y_j - y_k, c_i = x_k - x_j
        b = [y2 - y3, y3 - y1, y1 - y2]
        c = [x3 - x2, x1 - x3, x2 - x1]

        # If negative winding, adjust signs
        if two_area < 0:
            b = [-val for val in b]
            c = [-val for val in c]

        return area, b, c

    def _get_b_matrix(self, area: float, b: List[float], c: List[float]) -> List[List[float]]:
        """
        Build 3x6 strain-displacement matrix B.
        epsilon = B * u_e
        """
        inv_2a = 1.0 / (2.0 * area)
        return [
            [b[0] * inv_2a, 0.0,           b[1] * inv_2a, 0.0,           b[2] * inv_2a, 0.0],
            [0.0,           c[0] * inv_2a, 0.0,           c[1] * inv_2a, 0.0,           c[2] * inv_2a],
            [c[0] * inv_2a, b[0] * inv_2a, c[1] * inv_2a, b[1] * inv_2a, c[2] * inv_2a, b[2] * inv_2a],
        ]

    def compute_stiffness_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        plane_strain: bool = False,
    ) -> List[List[float]]:
        area, b, c = self._get_geometry(nodes_dict)
        b_mat = self._get_b_matrix(area, b, c)
        d_mat = self.material.get_elasticity_matrix(plane_strain=plane_strain)

        # K_e = t * Area * B^T * D * B
        bt = transpose(b_mat)
        db = mat_mul_rect(d_mat, b_mat)
        btdb = mat_mul_rect(bt, db)

        thickness = self.material.thickness
        scale = thickness * area

        for r in range(6):
            for col in range(6):
                btdb[r][col] *= scale

        return btdb

    def compute_mass_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        lumped: bool = True,
    ) -> List[List[float]]:
        area, _, _ = self._get_geometry(nodes_dict)
        total_mass = self.material.density * self.material.thickness * area
        one_third_m = total_mass / 3.0

        m = [[0.0] * 6 for _ in range(6)]
        for i in range(6):
            m[i][i] = one_third_m
        return m

    def compute_stress_strain(
        self,
        nodes_dict: Dict[int, Node2D],
        element_displacements: List[float],
        plane_strain: bool = False,
    ) -> Tuple[StressTensor2D, StrainTensor2D]:
        area, b, c = self._get_geometry(nodes_dict)
        b_mat = self._get_b_matrix(area, b, c)
        d_mat = self.material.get_elasticity_matrix(plane_strain=plane_strain)

        # eps = B * u
        eps = [0.0, 0.0, 0.0]
        for r in range(3):
            for col in range(6):
                eps[r] += b_mat[r][col] * element_displacements[col]

        # sigma = D * eps
        sig = [0.0, 0.0, 0.0]
        for r in range(3):
            for col in range(3):
                sig[r] += d_mat[r][col] * eps[col]

        stress = StressTensor2D(sigma_x=sig[0], sigma_y=sig[1], tau_xy=sig[2])
        strain = StrainTensor2D(eps_x=eps[0], eps_y=eps[1], gamma_xy=eps[2])
        return stress, strain


class QuadQ4(Element):
    """
    4-node bilinear isoparametric quadrilateral for 2D plane stress and plane strain.
    DOFs per node: 2 (Ux, Uy).
    Total element DOFs: 8.
    Numerical integration: 2x2 Gauss-Legendre quadrature (exact for bilinear quads).
    """
    # 2x2 Gauss integration points and weights
    _GAUSS_COORD = 1.0 / math.sqrt(3.0)  # ~0.5773502691896257
    GAUSS_POINTS = [
        (-_GAUSS_COORD, -_GAUSS_COORD),
        ( _GAUSS_COORD, -_GAUSS_COORD),
        ( _GAUSS_COORD,  _GAUSS_COORD),
        (-_GAUSS_COORD,  _GAUSS_COORD),
    ]
    GAUSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

    # Node local coordinate signs in standard CCW order: (-1, -1), (1, -1), (1, 1), (-1, 1)
    XI_NODES = (-1.0, 1.0, 1.0, -1.0)
    ETA_NODES = (-1.0, -1.0, 1.0, 1.0)

    def __init__(self, element_id: int, node_ids: Tuple[int, int, int, int], material: Material):
        super().__init__(element_id, node_ids, material)

    @property
    def num_nodes(self) -> int:
        return 4

    @property
    def num_dofs_per_node(self) -> int:
        return 2

    def get_dofs(self, nodes_dict: Dict[int, Node2D]) -> List[int]:
        dofs = []
        for nid in self.node_ids:
            dofs.extend([2 * nid, 2 * nid + 1])
        return dofs

    def _evaluate_shape_derivatives(
        self,
        xi: float,
        eta: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute partial derivatives of shape functions dN_i / d_xi and dN_i / d_eta.
        N_i = 0.25 * (1 + xi_i * xi) * (1 + eta_i * eta)
        """
        dn_dxi = [0.25 * self.XI_NODES[i] * (1.0 + self.ETA_NODES[i] * eta) for i in range(4)]
        dn_deta = [0.25 * self.ETA_NODES[i] * (1.0 + self.XI_NODES[i] * xi) for i in range(4)]
        return dn_dxi, dn_deta

    def _compute_jacobian_and_b(
        self,
        xi: float,
        eta: float,
        x_coords: List[float],
        y_coords: List[float],
    ) -> Tuple[float, List[List[float]]]:
        """
        Compute Jacobian determinant det(J) and 3x8 strain-displacement matrix B at (xi, eta).
        """
        dn_dxi, dn_deta = self._evaluate_shape_derivatives(xi, eta)

        # Jacobian components: J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
        j11 = sum(dn_dxi[i] * x_coords[i] for i in range(4))
        j12 = sum(dn_dxi[i] * y_coords[i] for i in range(4))
        j21 = sum(dn_deta[i] * x_coords[i] for i in range(4))
        j22 = sum(dn_deta[i] * y_coords[i] for i in range(4))

        det_j = j11 * j22 - j12 * j21
        if det_j <= 0.0:
            raise ValueError(f"Quad Q4 element {self.element_id} has non-positive Jacobian {det_j:.4e} (inverted or distorted elements).")

        # Inverse Jacobian J^-1 = (1/det_j) * [[j22, -j12], [-j21, j11]]
        inv_det = 1.0 / det_j
        inv_j11 =  j22 * inv_det
        inv_j12 = -j12 * inv_det
        inv_j21 = -j21 * inv_det
        inv_j22 =  j11 * inv_det

        # Cartesian spatial derivatives dN_i/dx, dN_i/dy
        dn_dx = [inv_j11 * dn_dxi[i] + inv_j12 * dn_deta[i] for i in range(4)]
        dn_dy = [inv_j21 * dn_dxi[i] + inv_j22 * dn_deta[i] for i in range(4)]

        # Assemble 3x8 B-matrix
        b_mat = [[0.0] * 8 for _ in range(3)]
        for i in range(4):
            c0 = 2 * i
            c1 = 2 * i + 1
            b_mat[0][c0] = dn_dx[i]
            b_mat[1][c1] = dn_dy[i]
            b_mat[2][c0] = dn_dy[i]
            b_mat[2][c1] = dn_dx[i]

        return det_j, b_mat

    def compute_stiffness_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        plane_strain: bool = False,
    ) -> List[List[float]]:
        nodes = [nodes_dict[nid] for nid in self.node_ids]
        x_coords = [n.x for n in nodes]
        y_coords = [n.y for n in nodes]

        d_mat = self.material.get_elasticity_matrix(plane_strain=plane_strain)
        thickness = self.material.thickness

        k_elem = [[0.0] * 8 for _ in range(8)]

        # 2x2 Gauss Quadrature Integration
        for p in range(4):
            xi, eta = self.GAUSS_POINTS[p]
            w = self.GAUSS_WEIGHTS[p]

            det_j, b_mat = self._compute_jacobian_and_b(xi, eta, x_coords, y_coords)
            bt = transpose(b_mat)
            db = mat_mul_rect(d_mat, b_mat)
            btdb = mat_mul_rect(bt, db)

            weight_factor = w * det_j * thickness
            for r in range(8):
                for c in range(8):
                    k_elem[r][c] += btdb[r][c] * weight_factor

        return k_elem

    def compute_mass_matrix(
        self,
        nodes_dict: Dict[int, Node2D],
        lumped: bool = True,
    ) -> List[List[float]]:
        nodes = [nodes_dict[nid] for nid in self.node_ids]
        x_coords = [n.x for n in nodes]
        y_coords = [n.y for n in nodes]

        # Integrate area over Gauss points
        total_area = 0.0
        for p in range(4):
            xi, eta = self.GAUSS_POINTS[p]
            det_j, _ = self._compute_jacobian_and_b(xi, eta, x_coords, y_coords)
            total_area += self.GAUSS_WEIGHTS[p] * det_j

        total_mass = self.material.density * self.material.thickness * total_area
        quarter_m = total_mass / 4.0

        m = [[0.0] * 8 for _ in range(8)]
        for i in range(8):
            m[i][i] = quarter_m
        return m

    def compute_stress_strain(
        self,
        nodes_dict: Dict[int, Node2D],
        element_displacements: List[float],
        plane_strain: bool = False,
    ) -> Tuple[StressTensor2D, StrainTensor2D]:
        """
        Compute stress and strain evaluated at element centroid (xi=0, eta=0).
        """
        nodes = [nodes_dict[nid] for nid in self.node_ids]
        x_coords = [n.x for n in nodes]
        y_coords = [n.y for n in nodes]

        _, b_center = self._compute_jacobian_and_b(0.0, 0.0, x_coords, y_coords)
        d_mat = self.material.get_elasticity_matrix(plane_strain=plane_strain)

        # eps = B * u
        eps = [0.0, 0.0, 0.0]
        for r in range(3):
            for col in range(8):
                eps[r] += b_center[r][col] * element_displacements[col]

        # sigma = D * eps
        sig = [0.0, 0.0, 0.0]
        for r in range(3):
            for col in range(3):
                sig[r] += d_mat[r][col] * eps[col]

        stress = StressTensor2D(sigma_x=sig[0], sigma_y=sig[1], tau_xy=sig[2])
        strain = StrainTensor2D(eps_x=eps[0], eps_y=eps[1], gamma_xy=eps[2])
        return stress, strain
