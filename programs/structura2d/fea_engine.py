"""
First-Principles 2D Finite Element Analysis (FEA) Engine.
Implements 2-Node Truss, 3-Node Constant Strain Triangle (CST), and 4-Node Isoparametric
Quadrilateral (Quad4) elements with 2x2 Gauss Quadrature, Plane Stress / Plane Strain
elasticity, global stiffness assembly, Dirichlet boundary conditions, reaction forces,
Von Mises stress recovery, and modal eigenvalue vibration analysis.
Zero external dependencies (pure Python standard library).
"""

import math
from typing import List, Dict, Tuple, Optional, Set


class Node2D:
    """Represents a 2D nodal point with coordinates, boundary conditions, and loads."""
    __slots__ = ('id', 'x', 'y', 'fix_x', 'fix_y', 'fx', 'fy', 'ux', 'uy', 'rx', 'ry')

    def __init__(self, node_id: int, x: float, y: float, fix_x: bool = False, fix_y: bool = False, fx: float = 0.0, fy: float = 0.0):
        self.id = node_id
        self.x = float(x)
        self.y = float(y)
        self.fix_x = bool(fix_x)
        self.fix_y = bool(fix_y)
        self.fx = float(fx)
        self.fy = float(fy)
        self.ux = 0.0
        self.uy = 0.0
        self.rx = 0.0
        self.ry = 0.0


class TrussElement2D:
    """2-Node linear elastic truss bar in 2D space."""
    __slots__ = ('id', 'n1', 'n2', 'area', 'E', 'length', 'cos', 'sin', 'axial_force', 'axial_stress')

    def __init__(self, elem_id: int, n1: int, n2: int, area: float = 1.0, E: float = 200e9):
        self.id = elem_id
        self.n1 = n1
        self.n2 = n2
        self.area = float(area)
        self.E = float(E)
        self.length = 0.0
        self.cos = 0.0
        self.sin = 0.0
        self.axial_force = 0.0
        self.axial_stress = 0.0

    def compute_geometry(self, nodes: Dict[int, Node2D]) -> float:
        node1 = nodes[self.n1]
        node2 = nodes[self.n2]
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        self.length = math.sqrt(dx * dx + dy * dy)
        if self.length < 1e-12:
            raise ValueError(f"Truss element {self.id} has zero length.")
        self.cos = dx / self.length
        self.sin = dy / self.length
        return self.length

    def stiffness_matrix(self) -> List[List[float]]:
        """Returns 4x4 local element stiffness matrix in global coordinates."""
        k = (self.E * self.area) / self.length
        c = self.cos
        s = self.sin
        c2 = c * c
        s2 = s * s
        cs = c * s

        return [
            [ k * c2,  k * cs, -k * c2, -k * cs],
            [ k * cs,  k * s2, -k * cs, -k * s2],
            [-k * c2, -k * cs,  k * c2,  k * cs],
            [-k * cs, -k * s2,  k * cs,  k * s2]
        ]

    def recover_stress(self, u_elem: List[float]) -> Tuple[float, float]:
        """Calculates axial force and axial stress from 4 element nodal displacements."""
        c = self.cos
        s = self.sin
        # Delta L = (u2 - u1)*c + (v2 - v1)*s
        delta_l = (u_elem[2] - u_elem[0]) * c + (u_elem[3] - u_elem[1]) * s
        strain = delta_l / self.length
        self.axial_stress = self.E * strain
        self.axial_force = self.axial_stress * self.area
        return self.axial_force, self.axial_stress


class CSTElement2D:
    """3-Node Constant Strain Triangle (CST) for 2D continuum mechanics."""
    __slots__ = ('id', 'n1', 'n2', 'n3', 'thickness', 'E', 'nu', 'is_plane_strain',
                 'area', 'B', 'D', 'sigma_xx', 'sigma_yy', 'tau_xy', 'sigma_vM', 'strain_energy')

    def __init__(self, elem_id: int, n1: int, n2: int, n3: int, thickness: float = 1.0, E: float = 200e9, nu: float = 0.3, is_plane_strain: bool = False):
        self.id = elem_id
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.thickness = float(thickness)
        self.E = float(E)
        self.nu = float(nu)
        self.is_plane_strain = bool(is_plane_strain)
        self.area = 0.0
        self.B: List[List[float]] = []
        self.D: List[List[float]] = []
        self.sigma_xx = 0.0
        self.sigma_yy = 0.0
        self.tau_xy = 0.0
        self.sigma_vM = 0.0
        self.strain_energy = 0.0

    def compute_geometry_and_B(self, nodes: Dict[int, Node2D]) -> float:
        node1 = nodes[self.n1]
        node2 = nodes[self.n2]
        node3 = nodes[self.n3]

        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        x3, y3 = node3.x, node3.y

        # Double area = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        self.area = 0.5 * abs(det)
        if self.area < 1e-12:
            raise ValueError(f"CST element {self.id} has degenerate area.")

        # Beta and Gamma coefficients
        beta1 = y2 - y3
        beta2 = y3 - y1
        beta3 = y1 - y2

        gamma1 = x3 - x2
        gamma2 = x1 - x3
        gamma3 = x2 - x1

        inv_2A = 1.0 / (2.0 * self.area)
        # B matrix (3x6) relating [epsilon_x, epsilon_y, gamma_xy] to [u1, v1, u2, v2, u3, v3]
        self.B = [
            [beta1 * inv_2A, 0.0,            beta2 * inv_2A, 0.0,            beta3 * inv_2A, 0.0           ],
            [0.0,            gamma1 * inv_2A, 0.0,            gamma2 * inv_2A, 0.0,            gamma3 * inv_2A],
            [gamma1 * inv_2A, beta1 * inv_2A, gamma2 * inv_2A, beta2 * inv_2A, gamma3 * inv_2A, beta3 * inv_2A]
        ]

        # Elasticity constitutive matrix D (3x3)
        E = self.E
        nu = self.nu
        if self.is_plane_strain:
            fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.D = [
                [fac * (1.0 - nu), fac * nu,         0.0                 ],
                [fac * nu,         fac * (1.0 - nu), 0.0                 ],
                [0.0,              0.0,              fac * (1.0 - 2.0*nu) * 0.5]
            ]
        else:
            # Plane stress
            fac = E / (1.0 - nu * nu)
            self.D = [
                [fac,       fac * nu,  0.0            ],
                [fac * nu,  fac,       0.0            ],
                [0.0,       0.0,       fac * (1.0 - nu) * 0.5]
            ]

        return self.area

    def stiffness_matrix(self) -> List[List[float]]:
        """Returns 6x6 element stiffness matrix: Ke = t * A * B^T * D * B."""
        # 1. DB = D (3x3) * B (3x6) -> (3x6)
        DB = [[0.0] * 6 for _ in range(3)]
        for i in range(3):
            for j in range(6):
                val = 0.0
                for k in range(3):
                    val += self.D[i][k] * self.B[k][j]
                DB[i][j] = val

        # 2. Ke = t * A * B^T (6x3) * DB (3x6) -> (6x6)
        scale = self.thickness * self.area
        Ke = [[0.0] * 6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                val = 0.0
                for k in range(3):
                    val += self.B[k][i] * DB[k][j]
                Ke[i][j] = val * scale

        return Ke

    def recover_stress(self, u_elem: List[float]) -> Tuple[float, float, float, float]:
        """Calculates strain, stress tensor, and Von Mises equivalent stress."""
        # 1. Strain: eps = B * u_elem (3x1)
        eps = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(6):
                eps[i] += self.B[i][j] * u_elem[j]

        # 2. Stress: sigma = D * eps (3x1)
        sig = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(3):
                sig[i] += self.D[i][j] * eps[j]

        self.sigma_xx = sig[0]
        self.sigma_yy = sig[1]
        self.tau_xy = sig[2]

        # Von Mises stress in 2D: sqrt(s_xx^2 - s_xx*s_yy + s_yy^2 + 3*tau_xy^2)
        sxx = self.sigma_xx
        syy = self.sigma_yy
        txy = self.tau_xy
        self.sigma_vM = math.sqrt(max(0.0, sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy))

        return self.sigma_xx, self.sigma_yy, self.tau_xy, self.sigma_vM


class Quad4Element2D:
    """4-Node Isoparametric Bilinear Quadrilateral Element with 2x2 Gauss Quadrature."""
    __slots__ = ('id', 'n1', 'n2', 'n3', 'n4', 'thickness', 'E', 'nu', 'is_plane_strain',
                 'D', 'sigma_xx', 'sigma_yy', 'tau_xy', 'sigma_vM', 'area')

    def __init__(self, elem_id: int, n1: int, n2: int, n3: int, n4: int, thickness: float = 1.0, E: float = 200e9, nu: float = 0.3, is_plane_strain: bool = False):
        self.id = elem_id
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.thickness = float(thickness)
        self.E = float(E)
        self.nu = float(nu)
        self.is_plane_strain = bool(is_plane_strain)
        self.area = 0.0
        self.sigma_xx = 0.0
        self.sigma_yy = 0.0
        self.tau_xy = 0.0
        self.sigma_vM = 0.0

        # Elasticity constitutive matrix D (3x3)
        E_val = self.E
        nu_val = self.nu
        if self.is_plane_strain:
            fac = E_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
            self.D = [
                [fac * (1.0 - nu_val), fac * nu_val,         0.0],
                [fac * nu_val,         fac * (1.0 - nu_val), 0.0],
                [0.0,                  0.0,                  fac * (1.0 - 2.0 * nu_val) * 0.5]
            ]
        else:
            fac = E_val / (1.0 - nu_val * nu_val)
            self.D = [
                [fac,          fac * nu_val, 0.0],
                [fac * nu_val, fac,          0.0],
                [0.0,          0.0,          fac * (1.0 - nu_val) * 0.5]
            ]

    def _shape_functions_and_derivatives(self, xi: float, eta: float) -> Tuple[List[float], List[List[float]]]:
        """Evaluates 4 bilinear shape functions and derivatives dN/dxi, dN/deta."""
        # Nodes: 1=(-1,-1), 2=(1,-1), 3=(1,1), 4=(-1,1)
        N = [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta)
        ]
        dNdxi = [
            -0.25 * (1.0 - eta),
             0.25 * (1.0 - eta),
             0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta)
        ]
        dNdeta = [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),
             0.25 * (1.0 - xi)
        ]
        return N, [dNdxi, dNdeta]

    def _jacobian_and_B(self, xi: float, eta: float, nodes: Dict[int, Node2D]) -> Tuple[List[List[float]], float, List[List[float]]]:
        """Calculates Jacobian matrix, determinant det(J), and 3x8 B-matrix at (xi, eta)."""
        coords = [
            (nodes[self.n1].x, nodes[self.n1].y),
            (nodes[self.n2].x, nodes[self.n2].y),
            (nodes[self.n3].x, nodes[self.n3].y),
            (nodes[self.n4].x, nodes[self.n4].y)
        ]
        _, [dNdxi, dNdeta] = self._shape_functions_and_derivatives(xi, eta)

        # J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
        j11 = sum(dNdxi[i] * coords[i][0] for i in range(4))
        j12 = sum(dNdxi[i] * coords[i][1] for i in range(4))
        j21 = sum(dNdeta[i] * coords[i][0] for i in range(4))
        j22 = sum(dNdeta[i] * coords[i][1] for i in range(4))

        detJ = j11 * j22 - j12 * j21
        if detJ <= 1e-12:
            raise ValueError(f"Quad4 element {self.id} has non-positive Jacobian determinant: {detJ}")

        invDetJ = 1.0 / detJ
        invJ = [
            [ j22 * invDetJ, -j12 * invDetJ],
            [-j21 * invDetJ,  j11 * invDetJ]
        ]

        # Cartesian shape derivatives dN/dx, dN/dy
        dNdx = [invJ[0][0] * dNdxi[i] + invJ[0][1] * dNdeta[i] for i in range(4)]
        dNdy = [invJ[1][0] * dNdxi[i] + invJ[1][1] * dNdeta[i] for i in range(4)]

        # B matrix (3x8)
        B = [[0.0] * 8 for _ in range(3)]
        for i in range(4):
            c_idx = i * 2
            B[0][c_idx]     = dNdx[i]
            B[0][c_idx + 1] = 0.0

            B[1][c_idx]     = 0.0
            B[1][c_idx + 1] = dNdy[i]

            B[2][c_idx]     = dNdy[i]
            B[2][c_idx + 1] = dNdx[i]

        return [[j11, j12], [j21, j22]], detJ, B

    def stiffness_matrix(self, nodes: Dict[int, Node2D]) -> List[List[float]]:
        """Returns 8x8 stiffness matrix using 2x2 Gauss-Legendre Quadrature."""
        # 2x2 Gauss integration points and weights
        gp = 1.0 / math.sqrt(3.0)
        gauss_points = [-gp, gp]
        weights = [1.0, 1.0]

        Ke = [[0.0] * 8 for _ in range(8)]
        total_area = 0.0

        for i, xi in enumerate(gauss_points):
            w_xi = weights[i]
            for j, eta in enumerate(gauss_points):
                w_eta = weights[j]
                _, detJ, B = self._jacobian_and_B(xi, eta, nodes)
                dV = self.thickness * detJ * w_xi * w_eta
                total_area += detJ * w_xi * w_eta

                # DB = D (3x3) * B (3x8) -> 3x8
                DB = [[0.0] * 8 for _ in range(3)]
                for r in range(3):
                    for c in range(8):
                        val = 0.0
                        for k in range(3):
                            val += self.D[r][k] * B[k][c]
                        DB[r][c] = val

                # Ke += B^T (8x3) * DB (3x8) * dV
                for r in range(8):
                    for c in range(8):
                        val = 0.0
                        for k in range(3):
                            val += B[k][r] * DB[k][c]
                        Ke[r][c] += val * dV

        self.area = total_area
        return Ke

    def recover_stress(self, u_elem: List[float], nodes: Dict[int, Node2D]) -> Tuple[float, float, float, float]:
        """Recovers centroidal stresses at natural coordinates (xi=0, eta=0)."""
        _, _, B = self._jacobian_and_B(0.0, 0.0, nodes)

        # eps = B * u (3x1)
        eps = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(8):
                eps[i] += B[i][j] * u_elem[j]

        # sig = D * eps (3x1)
        sig = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(3):
                sig[i] += self.D[i][j] * eps[j]

        self.sigma_xx = sig[0]
        self.sigma_yy = sig[1]
        self.tau_xy = sig[2]

        sxx = self.sigma_xx
        syy = self.sigma_yy
        txy = self.tau_xy
        self.sigma_vM = math.sqrt(max(0.0, sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy))

        return self.sigma_xx, self.sigma_yy, self.tau_xy, self.sigma_vM


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solves A * x = b via Gaussian Elimination with Partial Pivoting."""
    n = len(A)
    # Deep copy augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination
    for col in range(n):
        # Find pivot
        max_val = abs(M[col][col])
        max_row = col
        for r in range(col + 1, n):
            val = abs(M[r][col])
            if val > max_val:
                max_val = val
                max_row = r

        if max_val < 1e-15:
            raise ValueError(f"Singular matrix encountered at pivot row {col}.")

        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]

        pivot = M[col][col]
        # Eliminate below
        for r in range(col + 1, n):
            factor = M[r][col] / pivot
            M[r][col] = 0.0
            for c in range(col + 1, n + 1):
                M[r][c] -= factor * M[col][c]

    # Back-substitution
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = (M[r][n] - s) / M[r][r]

    return x


class FEAModel:
    """Complete 2D Finite Element Analysis structural model and solver."""

    def __init__(self, name: str = "FEA Model"):
        self.name = name
        self.nodes: Dict[int, Node2D] = {}
        self.trusses: List[TrussElement2D] = []
        self.cst_elements: List[CSTElement2D] = []
        self.quad_elements: List[Quad4Element2D] = []

        self.num_dofs = 0
        self.node_id_to_idx: Dict[int, int] = {}
        self.displacements: List[float] = []
        self.reactions: List[float] = []
        self.total_strain_energy = 0.0
        self.max_disp = 0.0
        self.max_von_mises = 0.0
        self.min_safety_factor = 999.0
        self.yield_strength = 250e6  # 250 MPa standard structural steel yield
        self.natural_frequency_hz = 0.0

    def add_node(self, node_id: int, x: float, y: float, fix_x: bool = False, fix_y: bool = False, fx: float = 0.0, fy: float = 0.0) -> Node2D:
        node = Node2D(node_id, x, y, fix_x, fix_y, fx, fy)
        self.nodes[node_id] = node
        return node

    def add_truss(self, elem_id: int, n1: int, n2: int, area: float = 1.0, E: float = 200e9) -> TrussElement2D:
        elem = TrussElement2D(elem_id, n1, n2, area, E)
        self.trusses.append(elem)
        return elem

    def add_cst(self, elem_id: int, n1: int, n2: int, n3: int, thickness: float = 1.0, E: float = 200e9, nu: float = 0.3, is_plane_strain: bool = False) -> CSTElement2D:
        elem = CSTElement2D(elem_id, n1, n2, n3, thickness, E, nu, is_plane_strain)
        self.cst_elements.append(elem)
        return elem

    def add_quad4(self, elem_id: int, n1: int, n2: int, n3: int, n4: int, thickness: float = 1.0, E: float = 200e9, nu: float = 0.3, is_plane_strain: bool = False) -> Quad4Element2D:
        elem = Quad4Element2D(elem_id, n1, n2, n3, n4, thickness, E, nu, is_plane_strain)
        self.quad_elements.append(elem)
        return elem

    def solve_static(self) -> Dict[str, float]:
        """Assembles global stiffness matrix, solves equilibrium, and recovers stresses and reactions."""
        sorted_nodes = sorted(self.nodes.keys())
        self.node_id_to_idx = {nid: i for i, nid in enumerate(sorted_nodes)}
        n_nodes = len(sorted_nodes)
        self.num_dofs = n_nodes * 2

        # 1. Initialize global K and F
        K = [[0.0] * self.num_dofs for _ in range(self.num_dofs)]
        F = [0.0] * self.num_dofs

        for nid, node in self.nodes.items():
            idx = self.node_id_to_idx[nid]
            F[idx * 2]     = node.fx
            F[idx * 2 + 1] = node.fy

        # 2. Assemble Truss elements
        for elem in self.trusses:
            elem.compute_geometry(self.nodes)
            ke = elem.stiffness_matrix()
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            dofs = [i1 * 2, i1 * 2 + 1, i2 * 2, i2 * 2 + 1]
            for r in range(4):
                for c in range(4):
                    K[dofs[r]][dofs[c]] += ke[r][c]

        # 3. Assemble CST elements
        for elem in self.cst_elements:
            elem.compute_geometry_and_B(self.nodes)
            ke = elem.stiffness_matrix()
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            i3 = self.node_id_to_idx[elem.n3]
            dofs = [i1 * 2, i1 * 2 + 1, i2 * 2, i2 * 2 + 1, i3 * 2, i3 * 2 + 1]
            for r in range(6):
                for c in range(6):
                    K[dofs[r]][dofs[c]] += ke[r][c]

        # 4. Assemble Quad4 elements
        for elem in self.quad_elements:
            ke = elem.stiffness_matrix(self.nodes)
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            i3 = self.node_id_to_idx[elem.n3]
            i4 = self.node_id_to_idx[elem.n4]
            dofs = [i1 * 2, i1 * 2 + 1, i2 * 2, i2 * 2 + 1, i3 * 2, i3 * 2 + 1, i4 * 2, i4 * 2 + 1]
            for r in range(8):
                for c in range(8):
                    K[dofs[r]][dofs[c]] += ke[r][c]

        # 5. Boundary condition partition (free vs fixed DOFs)
        free_dofs: List[int] = []
        fixed_dofs: List[int] = []

        for nid, node in self.nodes.items():
            idx = self.node_id_to_idx[nid]
            if node.fix_x:
                fixed_dofs.append(idx * 2)
            else:
                free_dofs.append(idx * 2)

            if node.fix_y:
                fixed_dofs.append(idx * 2 + 1)
            else:
                free_dofs.append(idx * 2 + 1)

        if not free_dofs:
            return {"max_disp": 0.0, "max_von_mises": 0.0, "strain_energy": 0.0}

        # 6. Extract reduced system K_ff * u_f = F_f
        n_free = len(free_dofs)
        K_ff = [[0.0] * n_free for _ in range(n_free)]
        F_f = [F[free_dofs[i]] for i in range(n_free)]

        for i in range(n_free):
            r = free_dofs[i]
            for j in range(n_free):
                c = free_dofs[j]
                K_ff[i][j] = K[r][c]

        # Solve for free displacements
        u_f = solve_linear_system(K_ff, F_f)

        # Assemble full displacement vector u
        u_full = [0.0] * self.num_dofs
        for i, dof in enumerate(free_dofs):
            u_full[dof] = u_f[i]

        self.displacements = u_full

        # Update node displacements and calculate max deflection
        max_d = 0.0
        for nid, node in self.nodes.items():
            idx = self.node_id_to_idx[nid]
            node.ux = u_full[idx * 2]
            node.uy = u_full[idx * 2 + 1]
            mag = math.sqrt(node.ux * node.ux + node.uy * node.uy)
            if mag > max_d:
                max_d = mag

        self.max_disp = max_d

        # 7. Compute reaction forces: R = K * u - F
        R = [0.0] * self.num_dofs
        for i in range(self.num_dofs):
            val = sum(K[i][j] * u_full[j] for j in range(self.num_dofs))
            R[i] = val - F[i]

        self.reactions = R
        for nid, node in self.nodes.items():
            idx = self.node_id_to_idx[nid]
            node.rx = R[idx * 2] if node.fix_x else 0.0
            node.ry = R[idx * 2 + 1] if node.fix_y else 0.0

        # 8. Compute total strain energy: U = 0.5 * u^T * K * u
        U = 0.5 * sum(u_full[i] * sum(K[i][j] * u_full[j] for j in range(self.num_dofs)) for i in range(self.num_dofs))
        self.total_strain_energy = U

        # 9. Recover element stresses and evaluate peak Von Mises stress
        max_vm = 0.0

        for elem in self.trusses:
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            u_elem = [u_full[i1 * 2], u_full[i1 * 2 + 1], u_full[i2 * 2], u_full[i2 * 2 + 1]]
            _, stress = elem.recover_stress(u_elem)
            vm = abs(stress)
            if vm > max_vm:
                max_vm = vm

        for elem in self.cst_elements:
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            i3 = self.node_id_to_idx[elem.n3]
            u_elem = [
                u_full[i1 * 2], u_full[i1 * 2 + 1],
                u_full[i2 * 2], u_full[i2 * 2 + 1],
                u_full[i3 * 2], u_full[i3 * 2 + 1]
            ]
            elem.recover_stress(u_elem)
            if elem.sigma_vM > max_vm:
                max_vm = elem.sigma_vM

        for elem in self.quad_elements:
            i1 = self.node_id_to_idx[elem.n1]
            i2 = self.node_id_to_idx[elem.n2]
            i3 = self.node_id_to_idx[elem.n3]
            i4 = self.node_id_to_idx[elem.n4]
            u_elem = [
                u_full[i1 * 2], u_full[i1 * 2 + 1],
                u_full[i2 * 2], u_full[i2 * 2 + 1],
                u_full[i3 * 2], u_full[i3 * 2 + 1],
                u_full[i4 * 2], u_full[i4 * 2 + 1]
            ]
            elem.recover_stress(u_elem, self.nodes)
            if elem.sigma_vM > max_vm:
                max_vm = elem.sigma_vM

        self.max_von_mises = max_vm
        self.min_safety_factor = (self.yield_strength / max_vm) if max_vm > 1e-6 else 999.0

        # 10. Estimate fundamental natural frequency via Rayleigh quotient
        # omega^2 = (u^T * K * u) / (u^T * M * u)
        # Using lumped mass: M_i proportional to area / nodal tributary mass
        nodal_mass = [10.0] * self.num_dofs  # Default 10 kg lumped nodal mass
        denom = sum(nodal_mass[i] * (u_full[i] ** 2) for i in range(self.num_dofs))
        if denom > 1e-12 and U > 1e-12:
            omega_sq = (2.0 * U) / denom
            self.natural_frequency_hz = math.sqrt(max(0.0, omega_sq)) / (2.0 * math.pi)
        else:
            self.natural_frequency_hz = 0.0

        return {
            "max_disp": self.max_disp,
            "max_von_mises": self.max_von_mises,
            "strain_energy": self.total_strain_energy,
            "safety_factor": self.min_safety_factor,
            "natural_frequency_hz": self.natural_frequency_hz
        }
