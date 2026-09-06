"""
Structura: Global Finite Element Analysis Solver & Stress Recovery.
Implements:
  - Global stiffness matrix assembly
  - Load vector construction
  - Exact Dirichlet boundary condition partitioning
  - Preconditioned Conjugate Gradient displacement solution
  - Reaction forces evaluation
  - Element Cauchy stress, Von Mises equivalent stress, and strain recovery
  - Nodal stress averaging and factor of safety calculation
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

from structura.types import (
    Node2D,
    Material,
    StressTensor2D,
    StrainTensor2D,
)
from structura.elements import Element, Beam2D
from structura.mesh import Mesh2D
from structura.sparse import DOKMatrix, CSRMatrix, solve_pcg


@dataclass
class FEAResult:
    """
    Comprehensive static structural analysis solution.
    """
    displacements: List[float]                  # Full displacement vector (size N_dofs)
    reactions: Dict[int, Tuple[float, float, float]] # node_id -> (Rx, Ry, R_moment)
    element_stresses: List[StressTensor2D]      # Stresses per element
    element_strains: List[StrainTensor2D]       # Strains per element
    nodal_von_mises: Dict[int, float]           # Smoothed Von Mises stress at each node
    max_displacement: float                     # Peak resultant displacement
    max_von_mises_stress: float                 # Peak Von Mises equivalent stress (Pa)
    min_factor_of_safety: float                 # Minimum material yield factor of safety
    strain_energy: float                        # Total internal compliance strain energy (Joules)
    solver_iterations: int                      # Iterations taken by PCG
    residual_norm: float                        # Relative convergence residual


class FEASolver:
    """
    Global Finite Element Analysis Solver.
    """
    def __init__(self, mesh: Mesh2D, plane_strain: bool = False):
        self.mesh = mesh
        self.plane_strain = plane_strain
        self.dofs_per_node = 3 if any(isinstance(e, Beam2D) for e in mesh.elements) else 2

    def _get_global_dof(self, node_id: int, local_dof: int) -> int:
        return self.dofs_per_node * node_id + local_dof

    def assemble_global_stiffness(self) -> DOKMatrix:
        """
        Assemble unconstrained global stiffness matrix K_global of size N_dofs x N_dofs.
        """
        n_dofs = self.mesh.num_dofs
        k_global = DOKMatrix(n_dofs, n_dofs)
        nodes_dict = self.mesh.nodes

        for elem in self.mesh.elements:
            k_e = elem.compute_stiffness_matrix(nodes_dict, self.plane_strain)
            elem_dofs = elem.get_dofs(nodes_dict)
            n_elem_dofs = len(elem_dofs)

            for i in range(n_elem_dofs):
                gi = elem_dofs[i]
                for j in range(n_elem_dofs):
                    gj = elem_dofs[j]
                    val = k_e[i][j]
                    if val != 0.0:
                        k_global.add(gi, gj, val)

        return k_global

    def assemble_load_vector(self) -> List[float]:
        """
        Assemble external nodal force vector F_global of size N_dofs.
        """
        f_global = [0.0] * self.mesh.num_dofs
        for load in self.mesh.nodal_loads:
            base = self.dofs_per_node * load.node_id
            f_global[base] += load.fx
            f_global[base + 1] += load.fy
            if self.dofs_per_node == 3:
                f_global[base + 2] += load.moment
        return f_global

    def solve(self, tolerance: float = 1e-7, max_iterations: int = 4000) -> FEAResult:
        """
        Solve linear static finite element system K * u = F subject to boundary conditions.
        Uses exact degree-of-freedom partitioning and Preconditioned Conjugate Gradient (PCG).
        """
        n_dofs = self.mesh.num_dofs
        k_global_dok = self.assemble_global_stiffness()
        f_global = self.assemble_load_vector()

        # 1. Identify constrained (prescribed) and free DOFs
        prescribed_dict: Dict[int, float] = {}
        for bc in self.mesh.boundary_conditions:
            gdof = self._get_global_dof(bc.node_id, bc.dof)
            if gdof < n_dofs:
                prescribed_dict[gdof] = bc.prescribed_value

        fixed_dofs = set(prescribed_dict.keys())
        free_dofs = [d for d in range(n_dofs) if d not in fixed_dofs]

        if not free_dofs:
            raise ValueError("All degrees of freedom are fixed; nothing to solve.")

        # Mapping from global DOF to reduced free DOF
        global_to_free = {gdof: i for i, gdof in enumerate(free_dofs)}
        n_free = len(free_dofs)

        # 2. Partition reduced stiffness matrix K_ff and RHS vector b_f
        # K_ff * u_f = F_f - K_fp * u_p
        k_ff_dok = DOKMatrix(n_free, n_free)
        b_free = [f_global[gdof] for gdof in free_dofs]

        for (r, c), val in k_global_dok.data.items():
            r_is_free = r in global_to_free
            c_is_free = c in global_to_free

            if r_is_free and c_is_free:
                k_ff_dok.set(global_to_free[r], global_to_free[c], val)
            elif r_is_free and not c_is_free:
                # Contribution to RHS from non-zero prescribed displacement
                u_p = prescribed_dict.get(c, 0.0)
                if u_p != 0.0:
                    b_free[global_to_free[r]] -= val * u_p

        # 3. Solve reduced system via Preconditioned Conjugate Gradient
        k_ff_csr = k_ff_dok.to_csr()
        u_free, iters, rel_res = solve_pcg(
            k_ff_csr,
            b_free,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        # 4. Reconstruct full displacement vector
        u_full = [0.0] * n_dofs
        for free_idx, gdof in enumerate(free_dofs):
            u_full[gdof] = u_free[free_idx]
        for gdof, u_p in prescribed_dict.items():
            u_full[gdof] = u_p

        # 5. Evaluate reaction forces: R = K_global * u - F
        k_global_csr = k_global_dok.to_csr()
        ku = k_global_csr.matvec(u_full)
        r_full = [kui - fi for kui, fi in zip(ku, f_global)]

        reactions: Dict[int, Tuple[float, float, float]] = {}
        for nid in self.mesh.nodes.keys():
            base = self.dofs_per_node * nid
            rx = r_full[base]
            ry = r_full[base + 1]
            rm = r_full[base + 2] if self.dofs_per_node == 3 else 0.0
            if abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rm) > 1e-6:
                reactions[nid] = (rx, ry, rm)

        # 6. Compute element stresses, strains, and peak values
        element_stresses: List[StressTensor2D] = []
        element_strains: List[StrainTensor2D] = []
        max_vm = 0.0
        min_fos = float("inf")
        nodes_dict = self.mesh.nodes

        # Data for nodal stress smoothing
        nodal_vm_sum: Dict[int, float] = {nid: 0.0 for nid in nodes_dict.keys()}
        nodal_vm_count: Dict[int, int] = {nid: 0 for nid in nodes_dict.keys()}

        for elem in self.mesh.elements:
            elem_dofs = elem.get_dofs(nodes_dict)
            u_elem = [u_full[d] for d in elem_dofs]
            stress, strain = elem.compute_stress_strain(nodes_dict, u_elem, self.plane_strain)
            element_stresses.append(stress)
            element_strains.append(strain)

            vm = stress.von_mises()
            if vm > max_vm:
                max_vm = vm

            # Factor of safety
            if vm > 1e-6:
                fos = elem.material.yield_strength / vm
                if fos < min_fos:
                    min_fos = fos

            # Accumulate for nodal smoothing
            for nid in elem.node_ids:
                nodal_vm_sum[nid] += vm
                nodal_vm_count[nid] += 1

        nodal_von_mises = {
            nid: (nodal_vm_sum[nid] / nodal_vm_count[nid]) if nodal_vm_count[nid] > 0 else 0.0
            for nid in nodes_dict.keys()
        }

        # 7. Max displacement magnitude
        max_disp = 0.0
        for nid in nodes_dict.keys():
            base = self.dofs_per_node * nid
            ux = u_full[base]
            uy = u_full[base + 1]
            disp_mag = math.sqrt(ux * ux + uy * uy)
            if disp_mag > max_disp:
                max_disp = disp_mag

        # 8. Total internal strain energy U = 0.5 * u^T * K * u
        strain_energy = 0.5 * sum(ui * kui for ui, kui in zip(u_full, ku))

        return FEAResult(
            displacements=u_full,
            reactions=reactions,
            element_stresses=element_stresses,
            element_strains=element_strains,
            nodal_von_mises=nodal_von_mises,
            max_displacement=max_disp,
            max_von_mises_stress=max_vm,
            min_factor_of_safety=min_fos,
            strain_energy=strain_energy,
            solver_iterations=iters,
            residual_norm=rel_res,
        )
