"""
Structura: Sparse Linear Algebra & Preconditioned Conjugate Gradient (PCG) Solver.
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Optional


class DOKMatrix:
    """
    Dictionary-of-Keys (DOK) sparse matrix for efficient assembly.
    Supports O(1) entry accumulation.
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data: Dict[Tuple[int, int], float] = {}

    def add(self, r: int, c: int, value: float) -> None:
        """Accumulate value at (r, c): A[r, c] += value."""
        if value == 0.0:
            return
        key = (r, c)
        self.data[key] = self.data.get(key, 0.0) + value

    def set(self, r: int, c: int, value: float) -> None:
        """Set value at (r, c)."""
        key = (r, c)
        if value == 0.0:
            self.data.pop(key, None)
        else:
            self.data[key] = value

    def get(self, r: int, c: int) -> float:
        """Get value at (r, c)."""
        return self.data.get((r, c), 0.0)

    def to_csr(self) -> CSRMatrix:
        """
        Convert DOK matrix to Compressed Sparse Row (CSR) format for fast matrix-vector products.
        """
        # Group entries by row and sort columns within each row
        row_entries: List[List[Tuple[int, float]]] = [[] for _ in range(self.rows)]
        for (r, c), val in self.data.items():
            if abs(val) > 1e-18:
                row_entries[r].append((c, val))

        values: List[float] = []
        col_indices: List[int] = []
        row_ptr: List[int] = [0]

        for r in range(self.rows):
            # Sort by column index for sequential memory access
            row_entries[r].sort(key=lambda item: item[0])
            for c, val in row_entries[r]:
                values.append(val)
                col_indices.append(c)
            row_ptr.append(len(values))

        return CSRMatrix(self.rows, self.cols, values, col_indices, row_ptr)


class CSRMatrix:
    """
    Compressed Sparse Row (CSR) matrix representation.
    Optimized for matrix-vector multiplication y = A * x.
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        values: List[float],
        col_indices: List[int],
        row_ptr: List[int],
    ):
        self.rows = rows
        self.cols = cols
        self.values = values
        self.col_indices = col_indices
        self.row_ptr = row_ptr

    @property
    def nnz(self) -> int:
        """Number of non-zero stored entries."""
        return len(self.values)

    def matvec(self, x: List[float], y_out: Optional[List[float]] = None) -> List[float]:
        """
        Perform sparse matrix-vector multiplication: y = A * x.
        Runs in O(nnz) operations.
        """
        if y_out is None:
            y_out = [0.0] * self.rows
        else:
            for i in range(self.rows):
                y_out[i] = 0.0

        vals = self.values
        cols = self.col_indices
        r_ptr = self.row_ptr

        for r in range(self.rows):
            start = r_ptr[r]
            end = r_ptr[r + 1]
            acc = 0.0
            for idx in range(start, end):
                acc += vals[idx] * x[cols[idx]]
            y_out[r] = acc

        return y_out

    def get_diagonal(self) -> List[float]:
        """Extract main diagonal elements A[i, i]."""
        diag = [0.0] * min(self.rows, self.cols)
        vals = self.values
        cols = self.col_indices
        r_ptr = self.row_ptr

        for r in range(len(diag)):
            start = r_ptr[r]
            end = r_ptr[r + 1]
            for idx in range(start, end):
                if cols[idx] == r:
                    diag[r] = vals[idx]
                    break

        return diag


def dot_product(u: List[float], v: List[float]) -> float:
    """Vector dot product u . v."""
    return sum(ui * vi for ui, vi in zip(u, v))


def norm_l2(v: List[float]) -> float:
    """Euclidean L2 vector norm."""
    return math.sqrt(sum(x * x for x in v))


def solve_pcg(
    matrix: CSRMatrix,
    b: List[float],
    x_init: Optional[List[float]] = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-7,
) -> Tuple[List[float], int, float]:
    """
    Solve symmetric positive-definite linear system A * x = b using
    Preconditioned Conjugate Gradient (PCG) with Jacobi (diagonal) preconditioning.

    Returns:
      (solution_vector x, iterations_performed, final_relative_residual)
    """
    n = matrix.rows
    if x_init is not None:
        x = list(x_init)
    else:
        x = [0.0] * n

    # Extract diagonal for Jacobi preconditioner M^-1 = diag(1 / A_ii)
    diag = matrix.get_diagonal()
    inv_diag = [1.0 / d if abs(d) > 1e-15 else 1.0 for d in diag]

    # Initial residual r0 = b - A * x0
    ax = matrix.matvec(x)
    r = [bi - axi for bi, axi in zip(b, ax)]
    norm_b = norm_l2(b)
    if norm_b < 1e-15:
        return [0.0] * n, 0, 0.0

    # Initial preconditioned residual z0 = M^-1 * r0
    z = [ri * inv_d for ri, inv_d in zip(r, inv_diag)]
    # Initial search direction p0 = z0
    p = list(z)

    rz_old = dot_product(r, z)
    rel_res = norm_l2(r) / norm_b

    if rel_res < tolerance:
        return x, 0, rel_res

    w = [0.0] * n  # Buffer for A * p

    for it in range(1, max_iterations + 1):
        # w = A * p
        matrix.matvec(p, y_out=w)
        pw = dot_product(p, w)
        if abs(pw) < 1e-20:
            break

        alpha = rz_old / pw

        # Update solution: x = x + alpha * p
        # Update residual: r = r - alpha * w
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * w[i]

        rel_res = norm_l2(r) / norm_b
        if rel_res < tolerance:
            return x, it, rel_res

        # Solve M * z = r -> z = inv_diag * r
        for i in range(n):
            z[i] = r[i] * inv_diag[i]

        rz_new = dot_product(r, z)
        if abs(rz_old) < 1e-25:
            break

        beta = rz_new / rz_old
        rz_old = rz_new

        # p = z + beta * p
        for i in range(n):
            p[i] = z[i] + beta * p[i]

    return x, max_iterations, rel_res
