"""
Unit tests for Structura Sparse Linear Algebra & PCG Solver.
"""

import unittest
import math
from structura.sparse import DOKMatrix, CSRMatrix, solve_pcg


class TestSparse(unittest.TestCase):
    def test_dok_to_csr_and_matvec(self):
        dok = DOKMatrix(3, 3)
        dok.set(0, 0, 4.0)
        dok.set(0, 1, 1.0)
        dok.set(1, 0, 1.0)
        dok.set(1, 1, 5.0)
        dok.set(1, 2, 2.0)
        dok.set(2, 1, 2.0)
        dok.set(2, 2, 6.0)

        csr = dok.to_csr()
        self.assertEqual(csr.nnz, 7)
        self.assertEqual(csr.row_ptr, [0, 2, 5, 7])

        # Test matvec y = A * [1, 2, 3]^T
        # y[0] = 4(1) + 1(2) = 6
        # y[1] = 1(1) + 5(2) + 2(3) = 17
        # y[2] = 2(2) + 6(3) = 22
        x = [1.0, 2.0, 3.0]
        y = csr.matvec(x)
        self.assertAlmostEqual(y[0], 6.0)
        self.assertAlmostEqual(y[1], 17.0)
        self.assertAlmostEqual(y[2], 22.0)

    def test_pcg_solver_convergence(self):
        # Build 1D discrete Laplacian SPD matrix: diag=2, offdiag=-1
        n = 20
        dok = DOKMatrix(n, n)
        for i in range(n):
            dok.set(i, i, 2.0)
            if i > 0:
                dok.set(i, i - 1, -1.0)
            if i < n - 1:
                dok.set(i, i + 1, -1.0)

        csr = dok.to_csr()
        # Known solution x = [1.0, 1.0, ..., 1.0]
        x_exact = [float(i + 1) for i in range(n)]
        b = csr.matvec(x_exact)

        x_sol, iters, rel_res = solve_pcg(csr, b, tolerance=1e-8, max_iterations=100)
        self.assertLess(rel_res, 1e-7)
        self.assertLess(iters, n + 5)

        for computed, exact in zip(x_sol, x_exact):
            self.assertAlmostEqual(computed, exact, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
