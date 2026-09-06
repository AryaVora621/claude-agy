"""
ZetaProof: Polynomial Calculus over Finite Field F_p[X].
Provides:
- Polynomial class with coefficient list representation.
- Addition, subtraction, multiplication (convolution), scalar scaling.
- Polynomial long division with remainder (divmod, /, %).
- Horner's method evaluation P(x).
- Vanishing polynomial construction from roots: Z(X) = prod(X - r_i).
- Lagrange polynomial interpolation across arbitrary point sets.
"""

from typing import List, Sequence, Tuple, Union, Optional
from .field import FieldElement, BN254_SCALAR_FIELD


class Polynomial:
    """
    Univariate polynomial P(X) = c_0 + c_1 X + c_2 X^2 + ... + c_d X^d over F_p.
    Coefficients are stored in ascending order of degree (index i is coefficient of X^i).
    """
    __slots__ = ("coeffs", "p")

    def __init__(self, coeffs: Sequence[Union[int, FieldElement]], p: int = BN254_SCALAR_FIELD) -> None:
        self.p = p
        # Normalize coefficients to FieldElements
        fe_coeffs = [
            c if isinstance(c, FieldElement) else FieldElement(c, p)
            for c in coeffs
        ]
        # Strip trailing zero coefficients to maintain canonical form
        while len(fe_coeffs) > 1 and fe_coeffs[-1].is_zero():
            fe_coeffs.pop()

        if not fe_coeffs:
            fe_coeffs = [FieldElement.zero(p)]

        self.coeffs: List[FieldElement] = fe_coeffs

    @classmethod
    def zero(cls, p: int = BN254_SCALAR_FIELD) -> "Polynomial":
        return cls([FieldElement.zero(p)], p)

    @classmethod
    def one(cls, p: int = BN254_SCALAR_FIELD) -> "Polynomial":
        return cls([FieldElement.one(p)], p)

    @classmethod
    def from_roots(cls, roots: Sequence[Union[int, FieldElement]], p: int = BN254_SCALAR_FIELD) -> "Polynomial":
        """
        Construct monic vanishing polynomial Z(X) = prod_{r in roots} (X - r).
        """
        result = cls.one(p)
        for r in roots:
            # (X - r)
            term = cls([-r, 1], p)
            result = result * term
        return result

    @property
    def degree(self) -> int:
        """
        Degree of the polynomial. Returns -1 for the zero polynomial.
        """
        if len(self.coeffs) == 1 and self.coeffs[0].is_zero():
            return -1
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return len(self.coeffs) == 1 and self.coeffs[0].is_zero()

    def evaluate(self, x: Union[int, FieldElement]) -> FieldElement:
        """
        Evaluate P(x) using Horner's method in O(d) field operations.
        """
        if not isinstance(x, FieldElement):
            x = FieldElement(x, self.p)

        result = FieldElement.zero(self.p)
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __call__(self, x: Union[int, FieldElement]) -> FieldElement:
        return self.evaluate(x)

    def __add__(self, other: Union[int, FieldElement, "Polynomial"]) -> "Polynomial":
        if not isinstance(other, Polynomial):
            other = Polynomial([other], self.p)
        max_len = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = []
        for i in range(max_len):
            c1 = self.coeffs[i] if i < len(self.coeffs) else FieldElement.zero(self.p)
            c2 = other.coeffs[i] if i < len(other.coeffs) else FieldElement.zero(self.p)
            new_coeffs.append(c1 + c2)
        return Polynomial(new_coeffs, self.p)

    def __radd__(self, other: Union[int, FieldElement]) -> "Polynomial":
        return self.__add__(other)

    def __sub__(self, other: Union[int, FieldElement, "Polynomial"]) -> "Polynomial":
        if not isinstance(other, Polynomial):
            other = Polynomial([other], self.p)
        max_len = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = []
        for i in range(max_len):
            c1 = self.coeffs[i] if i < len(self.coeffs) else FieldElement.zero(self.p)
            c2 = other.coeffs[i] if i < len(other.coeffs) else FieldElement.zero(self.p)
            new_coeffs.append(c1 - c2)
        return Polynomial(new_coeffs, self.p)

    def __rsub__(self, other: Union[int, FieldElement]) -> "Polynomial":
        return Polynomial([other], self.p) - self

    def __neg__(self) -> "Polynomial":
        return Polynomial([-c for c in self.coeffs], self.p)

    def __mul__(self, other: Union[int, FieldElement, "Polynomial"]) -> "Polynomial":
        if isinstance(other, (int, FieldElement)):
            fe = other if isinstance(other, FieldElement) else FieldElement(other, self.p)
            if fe.is_zero():
                return Polynomial.zero(self.p)
            return Polynomial([c * fe for c in self.coeffs], self.p)

        # Polynomial multiplication via convolution in O(d1 * d2)
        if self.is_zero() or other.is_zero():
            return Polynomial.zero(self.p)

        n1 = len(self.coeffs)
        n2 = len(other.coeffs)
        out_len = n1 + n2 - 1
        res = [FieldElement.zero(self.p) for _ in range(out_len)]

        for i, a in enumerate(self.coeffs):
            if a.is_zero():
                continue
            for j, b in enumerate(other.coeffs):
                res[i + j] += a * b

        return Polynomial(res, self.p)

    def __rmul__(self, other: Union[int, FieldElement]) -> "Polynomial":
        return self.__mul__(other)

    def __divmod__(self, other: "Polynomial") -> Tuple["Polynomial", "Polynomial"]:
        """
        Polynomial long division: computes (Q, R) such that self = other * Q + R
        with deg(R) < deg(other).
        """
        if other.is_zero():
            raise ZeroDivisionError("Polynomial division by zero polynomial")

        if self.degree < other.degree:
            return Polynomial.zero(self.p), self

        # Remainder initialized to copy of dividend coefficients
        rem = list(self.coeffs)
        d_divisor = other.degree
        lead_divisor_inv = other.coeffs[-1].inverse()

        quot_len = len(rem) - d_divisor
        quot = [FieldElement.zero(self.p) for _ in range(quot_len)]

        for i in range(len(rem) - 1, d_divisor - 1, -1):
            if rem[i].is_zero():
                continue
            lead_coeff = rem[i] * lead_divisor_inv
            shift = i - d_divisor
            quot[shift] = lead_coeff

            for j in range(len(other.coeffs)):
                rem[shift + j] -= lead_coeff * other.coeffs[j]

        # Clean up remainder
        return Polynomial(quot, self.p), Polynomial(rem[:d_divisor], self.p)

    def __floordiv__(self, other: "Polynomial") -> "Polynomial":
        q, _ = divmod(self, other)
        return q

    def __truediv__(self, other: "Polynomial") -> "Polynomial":
        q, r = divmod(self, other)
        if not r.is_zero():
            raise ValueError("Polynomial division does not divide evenly (non-zero remainder)")
        return q

    def __mod__(self, other: "Polynomial") -> "Polynomial":
        _, r = divmod(self, other)
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            if isinstance(other, (int, FieldElement)):
                other = Polynomial([other], self.p)
            else:
                return False
        if len(self.coeffs) != len(other.coeffs):
            return False
        return all(c1 == c2 for c1, c2 in zip(self.coeffs, other.coeffs))

    def __repr__(self) -> str:
        terms = []
        for deg, c in enumerate(self.coeffs):
            if c.is_zero() and len(self.coeffs) > 1:
                continue
            if deg == 0:
                terms.append(str(c.val))
            elif deg == 1:
                terms.append(f"{c.val}*X" if c.val != 1 else "X")
            else:
                terms.append(f"{c.val}*X^{deg}" if c.val != 1 else f"X^{deg}")
        return " + ".join(terms) if terms else "0"


def lagrange_interpolation(
    points: Sequence[Tuple[Union[int, FieldElement], Union[int, FieldElement]]],
    p: int = BN254_SCALAR_FIELD
) -> Polynomial:
    """
    Compute unique interpolating polynomial L(X) such that L(x_i) = y_i
    for distinct evaluation roots x_i over F_p.
    Formula: L(X) = sum_j ( y_j * prod_{k != j} (X - x_k) / (x_j - x_k) )
    """
    if not points:
        return Polynomial.zero(p)

    fe_points = [
        (
            x if isinstance(x, FieldElement) else FieldElement(x, p),
            y if isinstance(y, FieldElement) else FieldElement(y, p)
        )
        for x, y in points
    ]

    # Verify uniqueness of x coordinates
    x_set = set(pt[0].val for pt in fe_points)
    if len(x_set) != len(fe_points):
        raise ValueError("Lagrange interpolation requires distinct x-coordinates")

    result = Polynomial.zero(p)
    n = len(fe_points)

    for j in range(n):
        xj, yj = fe_points[j]
        if yj.is_zero():
            continue

        # Compute basis polynomial l_j(X)
        basis = Polynomial.one(p)
        denom = FieldElement.one(p)

        for k in range(n):
            if k == j:
                continue
            xk, _ = fe_points[k]
            # (X - x_k)
            basis = basis * Polynomial([-xk, 1], p)
            denom = denom * (xj - xk)

        coeff = yj * denom.inverse()
        result = result + (basis * coeff)

    return result
