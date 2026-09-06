"""
VeloSLAM: High-Performance Pure Python Linear Algebra & Covariance Matrix Engine.
Provides Matrix and Vector data structures, matrix multiplication, inversion,
determinants, angle normalization, and Mahalanobis distance gating.
"""

import math
from typing import List, Tuple, Union, Optional


def normalize_angle(angle: float) -> float:
    """Normalizes an angle in radians to the range [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Vector:
    """Mathematical 1D vector."""

    def __init__(self, data: List[float]) -> None:
        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __setitem__(self, idx: int, value: float) -> None:
        self.data[idx] = float(value)

    def __repr__(self) -> str:
        return f"Vector({[round(x, 4) for x in self.data]})"

    def to_list(self) -> List[float]:
        return list(self.data)

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError(f"Vector dimension mismatch: {len(self)} != {len(other)}")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError(f"Vector dimension mismatch: {len(self)} != {len(other)}")
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar: float) -> "Vector":
        return Vector([x * scalar for x in self.data])

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def dot(self, other: "Vector") -> float:
        if len(self) != len(other):
            raise ValueError(f"Vector dimension mismatch: {len(self)} != {len(other)}")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def outer(self, other: "Vector") -> "Matrix":
        """Computes outer product: self @ other.T -> Matrix."""
        rows = len(self)
        cols = len(other)
        res = [[self.data[r] * other.data[c] for c in range(cols)] for r in range(rows)]
        return Matrix(res)


class Matrix:
    """Mathematical 2D matrix with linear algebra operations."""

    def __init__(self, data: List[List[float]]) -> None:
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.data = [[float(val) for val in row] for row in data]

    def __getitem__(self, idx: int) -> List[float]:
        return self.data[idx]

    def __setitem__(self, idx: int, value: List[float]) -> None:
        self.data[idx] = [float(v) for v in value]

    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"

    def copy(self) -> "Matrix":
        return Matrix([[x for x in row] for row in self.data])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        return cls([[0.0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        mat = cls.zeros(n, n)
        for i in range(n):
            mat.data[i][i] = 1.0
        return mat

    @classmethod
    def diag(cls, values: List[float]) -> "Matrix":
        n = len(values)
        mat = cls.zeros(n, n)
        for i in range(n):
            mat.data[i][i] = float(values[i])
        return mat

    def transpose(self) -> "Matrix":
        trans = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        return Matrix(trans)

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Matrix dimension mismatch for addition: ({self.rows}x{self.cols}) and ({other.rows}x{other.cols})")
        return Matrix([[self.data[r][c] + other.data[r][c] for c in range(self.cols)] for r in range(self.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Matrix dimension mismatch for subtraction: ({self.rows}x{self.cols}) and ({other.rows}x{other.cols})")
        return Matrix([[self.data[r][c] - other.data[r][c] for c in range(self.cols)] for r in range(self.rows)])

    def __mul__(self, scalar: float) -> "Matrix":
        return Matrix([[self.data[r][c] * scalar for c in range(self.cols)] for r in range(self.rows)])

    def __rmul__(self, scalar: float) -> "Matrix":
        return self.__mul__(scalar)

    def __matmul__(self, other: Union["Matrix", Vector]) -> Union["Matrix", Vector]:
        if isinstance(other, Vector):
            if self.cols != len(other):
                raise ValueError(f"Matrix-Vector size mismatch: ({self.rows}x{self.cols}) @ {len(other)}")
            res: List[float] = []
            for r in range(self.rows):
                val = sum(self.data[r][c] * other.data[c] for c in range(self.cols))
                res.append(val)
            return Vector(res)
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Matrix multiplication size mismatch: ({self.rows}x{self.cols}) @ ({other.rows}x{other.cols})")
            res_mat = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]
            for r in range(self.rows):
                for c in range(other.cols):
                    res_mat[r][c] = sum(self.data[r][k] * other.data[k][c] for k in range(self.cols))
            return Matrix(res_mat)
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def det(self) -> float:
        """Computes determinant of square matrix."""
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square matrix")
        n = self.rows
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        if n == 3:
            return (
                self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) -
                self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0]) +
                self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0])
            )

        # General LU/Gaussian elimination with partial pivoting
        a = [list(row) for row in self.data]
        det_val = 1.0
        for i in range(n):
            # Pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(a[k][i]) > abs(a[max_row][i]):
                    max_row = k
            if abs(a[max_row][i]) < 1e-12:
                return 0.0
            if max_row != i:
                a[i], a[max_row] = a[max_row], a[i]
                det_val = -det_val
            det_val *= a[i][i]
            for j in range(i + 1, n):
                factor = a[j][i] / a[i][i]
                for k in range(i, n):
                    a[j][k] -= factor * a[i][k]
        return det_val

    def inv(self) -> "Matrix":
        """Computes matrix inverse via analytical 2x2 formula or Gauss-Jordan elimination."""
        if self.rows != self.cols:
            raise ValueError("Inverse requires a square matrix")
        n = self.rows

        # Fast analytical 2x2 inverse
        if n == 2:
            d = self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
            if abs(d) < 1e-14:
                raise ValueError("Matrix is singular and cannot be inverted")
            inv_det = 1.0 / d
            return Matrix([
                [self.data[1][1] * inv_det, -self.data[0][1] * inv_det],
                [-self.data[1][0] * inv_det, self.data[0][0] * inv_det]
            ])

        # Gauss-Jordan elimination with partial pivoting for N x N
        # Augment with identity matrix
        aug = [self.data[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            # Partial pivot search
            pivot_row = i
            max_val = abs(aug[i][i])
            for r in range(i + 1, n):
                if abs(aug[r][i]) > max_val:
                    max_val = abs(aug[r][i])
                    pivot_row = r
            if max_val < 1e-14:
                raise ValueError("Matrix is singular or ill-conditioned (pivot too small)")
            if pivot_row != i:
                aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

            pivot = aug[i][i]
            inv_pivot = 1.0 / pivot
            for c in range(i, 2 * n):
                aug[i][c] *= inv_pivot

            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    if factor != 0.0:
                        for c in range(i, 2 * n):
                            aug[r][c] -= factor * aug[i][c]

        inv_data = [row[n:] for row in aug]
        return Matrix(inv_data)


def mahalanobis_distance(diff: Vector, s_inv: Matrix) -> float:
    """
    Computes Mahalanobis distance squared: D_M^2 = diff.T @ S^-1 @ diff.
    """
    if len(diff) != s_inv.rows or s_inv.rows != s_inv.cols:
        raise ValueError("Dimension mismatch in Mahalanobis distance")
    prod = s_inv @ diff  # Vector
    return diff.dot(prod)


def covariance_ellipse_2d(cov_2x2: Matrix, chi2_val: float = 5.991) -> Tuple[float, float, float]:
    """
    Calculates 2D covariance ellipse parameters (semi_major, semi_minor, angle_radians)
    for a given 2x2 covariance matrix.
    Default chi2_val = 5.991 corresponds to 95% confidence contour for 2 degrees of freedom.
    """
    a = cov_2x2[0][0]
    b = cov_2x2[0][1]
    c = cov_2x2[1][1]

    # Eigenvalues of symmetric 2x2 matrix: [a, b; b, c]
    # lambda = (a + c)/2 +/- sqrt(((a - c)/2)^2 + b^2)
    trace = a + c
    diff = a - c
    disc = math.sqrt(max(0.0, (diff * 0.5) ** 2 + b * b))

    lambda1 = max(0.0, trace * 0.5 + disc)
    lambda2 = max(0.0, trace * 0.5 - disc)

    semi_major = math.sqrt(lambda1 * chi2_val)
    semi_minor = math.sqrt(lambda2 * chi2_val)

    # Orientation angle
    if abs(b) < 1e-12:
        angle = 0.0 if a >= c else math.pi * 0.5
    else:
        angle = math.atan2(lambda1 - a, b)

    return semi_major, semi_minor, angle
