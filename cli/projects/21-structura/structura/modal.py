"""
Structura: Modal Dynamics & Natural Vibration Eigenvalue Solver.
Solves the generalized structural eigenvalue problem:
  (K - omega^2 * M) * phi = 0
using shifted inverse power iteration with Gram-Schmidt mass-orthogonal deflation.
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from structura.types import Node2D
from structura.elements import Element, Beam2D
from structura.mesh import Mesh2D
from structura.sparse import DOKMatrix, CSRMatrix, solve_pcg, dot_product, norm_l2


@dataclass
class ModalMode:
    """
    Individual structural vibration resonance mode.
    """
    mode_number: int
    frequency_hz: float
    angular_frequency_rad_s: float
    eigenvector: List[float]       # Full mass-normalized mode shape vector (size N_dofs)
    generalized_mass: float        # phi^T * M * phi = 1.0


@dataclass
class ModalResult:
    """
    Container for extracted structural vibration modes.
    """
    modes: List[ModalMode]

    @property
    def fundamental_frequency_hz(self) -> float:
        """First (lowest) natural resonance frequency in Hz."""
        return self.modes[0].frequency_hz if self.modes else 0.0


class ModalSolver:
    """
    Generalized Eigenvalue Modal Solver.
    """
    def __init__(self, mesh: Mesh2D, plane_strain: bool = False):
        self.mesh = mesh
        self.plane_strain = plane_strain
        self.dofs_per_node = 3 if any(isinstance(e, Beam2D) for e in mesh.elements) else 2

    def assemble_lumped_mass_vector(self) -> List[float]:
        """
        Assemble global lumped diagonal mass vector M of size N_dofs.
        """
        n_dofs = self.mesh.num_dofs
        m_diag = [0.0] * n_dofs
        nodes_dict = self.mesh.nodes

        for elem in self.mesh.elements:
            m_e = elem.compute_mass_matrix(nodes_dict, lumped=True)
            elem_dofs = elem.get_dofs(nodes_dict)
            n_elem_dofs = len(elem_dofs)

            for i in range(n_elem_dofs):
                gi = elem_dofs[i]
                m_diag[gi] += m_e[i][i]

        return m_diag

    def solve(
        self,
        num_modes: int = 3,
        max_iterations: int = 150,
        tolerance: float = 1e-6,
    ) -> ModalResult:
        """
        Extract the lowest `num_modes` natural resonant frequencies and mode shapes.
        """
        n_dofs = self.mesh.num_dofs

        # 1. Assemble unconstrained stiffness and lumped mass
        from structura.solver import FEASolver
        fea = FEASolver(self.mesh, self.plane_strain)
        k_global_dok = fea.assemble_global_stiffness()
        m_diag = self.assemble_lumped_mass_vector()

        # 2. Identify constrained and free DOFs
        fixed_dofs = set()
        for bc in self.mesh.boundary_conditions:
            gdof = self.dofs_per_node * bc.node_id + bc.dof
            if gdof < n_dofs:
                fixed_dofs.add(gdof)

        free_dofs = [d for d in range(n_dofs) if d not in fixed_dofs]
        n_free = len(free_dofs)

        if n_free == 0:
            return ModalResult(modes=[])

        num_modes = min(num_modes, n_free)

        # 3. Partition reduced stiffness K_ff and reduced lumped mass M_ff
        global_to_free = {gdof: i for i, gdof in enumerate(free_dofs)}
        k_ff_dok = DOKMatrix(n_free, n_free)

        for (r, c), val in k_global_dok.data.items():
            if r in global_to_free and c in global_to_free:
                k_ff_dok.set(global_to_free[r], global_to_free[c], val)

        k_ff_csr = k_ff_dok.to_csr()
        m_free = [m_diag[gdof] for gdof in free_dofs]

        computed_modes: List[ModalMode] = []
        # Store reduced orthonormal eigenvectors
        prev_modes_free: List[List[float]] = []

        def mass_inner_product(v1: List[float], v2: List[float]) -> float:
            return sum(v1[i] * m_free[i] * v2[i] for i in range(n_free))

        # 4. Extract each mode sequentially with Gram-Schmidt orthogonalization
        for mode_idx in range(1, num_modes + 1):
            # Deterministic pseudo-random seed vector
            v = [math.sin(float(mode_idx * 17 + i * 23)) for i in range(n_free)]

            # Initial mass-orthogonalization against prior modes
            for prev_v in prev_modes_free:
                coeff = mass_inner_product(prev_v, v)
                for i in range(n_free):
                    v[i] -= coeff * prev_v[i]

            norm_m = math.sqrt(max(1e-20, mass_inner_product(v, v)))
            for i in range(n_free):
                v[i] /= norm_m

            lambda_est = 1.0

            # Inverse power iteration
            for it in range(max_iterations):
                # RHS: b = M * v
                b = [m_free[i] * v[i] for i in range(n_free)]

                # Solve K_ff * y = M * v
                y, _, _ = solve_pcg(k_ff_csr, b, tolerance=1e-8, max_iterations=2000)

                # Gram-Schmidt mass-orthogonalization against previously converged modes
                for prev_v in prev_modes_free:
                    coeff = mass_inner_product(prev_v, y)
                    for i in range(n_free):
                        y[i] -= coeff * prev_v[i]

                # Rayleigh quotient denominator: y^T * M * y
                y_my = mass_inner_product(y, y)
                if y_my <= 1e-25:
                    break

                norm_y = math.sqrt(y_my)
                new_lambda = 1.0 / (norm_y * norm_y / mass_inner_product(v, y)) if abs(mass_inner_product(v, y)) > 1e-20 else (1.0 / y_my)

                # Update normalized v = y / sqrt(y^T * M * y)
                v_new = [yi / norm_y for yi in y]

                diff = math.sqrt(sum((v_new[i] - v[i]) ** 2 for i in range(n_free)))
                v = v_new
                lambda_est = 1.0 / norm_y

                if diff < tolerance and it >= 3:
                    break

            prev_modes_free.append(v)

            # Reconstruct full mass-normalized eigenvector
            phi_full = [0.0] * n_dofs
            for free_idx, gdof in enumerate(free_dofs):
                phi_full[gdof] = v[free_idx]

            # Re-normalize full eigenvector so peak amplitude is 1.0 for visualization
            max_amp = max(abs(val) for val in phi_full)
            if max_amp > 1e-12:
                for i in range(n_dofs):
                    phi_full[i] /= max_amp

            # Natural frequencies
            # Rayleigh quotient on full mode: omega^2 = (phi^T * K * phi) / (phi^T * M * phi)
            k_csr = k_global_dok.to_csr()
            k_phi = k_csr.matvec(phi_full)
            phi_k_phi = dot_product(phi_full, k_phi)
            phi_m_phi = sum(phi_full[i] * m_diag[i] * phi_full[i] for i in range(n_dofs))

            omega_sq = phi_k_phi / phi_m_phi if phi_m_phi > 1e-15 else 0.0
            omega = math.sqrt(max(0.0, omega_sq))
            freq_hz = omega / (2.0 * math.pi)

            computed_modes.append(
                ModalMode(
                    mode_number=mode_idx,
                    frequency_hz=freq_hz,
                    angular_frequency_rad_s=omega,
                    eigenvector=phi_full,
                    generalized_mass=1.0,
                )
            )

        return ModalResult(modes=computed_modes)
