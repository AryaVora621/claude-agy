"""
ZetaProof: Quadratic Arithmetic Program (QAP) Reduction.
Provides:
- QAP construction from an R1CS instance via Lagrange polynomial interpolation.
- Vanishing polynomial Z(X) = prod_{k=1}^m (X - r_k).
- Computation of aggregate polynomials A(X), B(X), C(X) from witness vector s.
- Exact polynomial quotient H(X) = (A(X)*B(X) - C(X)) / Z(X) verification.
"""

from typing import List, Tuple, Sequence, Optional, Union
from .field import FieldElement, BN254_SCALAR_FIELD
from .polynomial import Polynomial, lagrange_interpolation
from .r1cs import R1CS


class QAP:
    """
    Quadratic Arithmetic Program representation of an R1CS over F_p.
    Consists of:
    - Evaluation domain roots: r_1, ..., r_m
    - Vanishing polynomial: Z(X) = prod_{k=1}^m (X - r_k)
    - Variable polynomials: A_j(X), B_j(X), C_j(X) for j in [0, n-1]
      such that A_j(r_k) = A_{k, j}, B_j(r_k) = B_{k, j}, C_j(r_k) = C_{k, j}.
    """
    __slots__ = (
        "p",
        "r1cs",
        "roots",
        "z_poly",
        "a_polys",
        "b_polys",
        "c_polys",
        "num_variables",
        "num_constraints"
    )

    def __init__(
        self,
        r1cs: R1CS,
        roots: Sequence[Union[int, FieldElement]],
        z_poly: Polynomial,
        a_polys: Sequence[Polynomial],
        b_polys: Sequence[Polynomial],
        c_polys: Sequence[Polynomial]
    ) -> None:
        self.p = r1cs.p
        self.r1cs = r1cs
        self.roots = [
            r if isinstance(r, FieldElement) else FieldElement(r, self.p)
            for r in roots
        ]
        self.z_poly = z_poly
        self.a_polys = list(a_polys)
        self.b_polys = list(b_polys)
        self.c_polys = list(c_polys)
        self.num_variables = r1cs.num_variables
        self.num_constraints = r1cs.num_constraints

    @classmethod
    def from_r1cs(cls, r1cs: R1CS, roots: Optional[Sequence[Union[int, FieldElement]]] = None) -> "QAP":
        """
        Convert R1CS matrices A, B, C into QAP polynomials via Lagrange interpolation.
        If roots are not provided, uses roots r_k = 1, 2, ..., m.
        """
        p = r1cs.p
        m = r1cs.num_constraints
        n = r1cs.num_variables

        if m == 0:
            raise ValueError("Cannot construct QAP from empty R1CS")

        if roots is None:
            eval_roots = [FieldElement(k + 1, p) for k in range(m)]
        else:
            if len(roots) != m:
                raise ValueError(f"Number of roots ({len(roots)}) must match constraint count ({m})")
            eval_roots = [r if isinstance(r, FieldElement) else FieldElement(r, p) for r in roots]

        # Compute vanishing polynomial Z(X) = prod (X - r_k)
        z_poly = Polynomial.from_roots(eval_roots, p)

        # Get dense matrix representations of A, B, C
        mat_a, mat_b, mat_c = r1cs.dense_matrices()

        # For each variable j in 0..n-1, interpolate polynomials across the m roots
        a_polys: List[Polynomial] = []
        b_polys: List[Polynomial] = []
        c_polys: List[Polynomial] = []

        for j in range(n):
            # Points for variable j: (r_k, mat_{k, j})
            pts_a = [(eval_roots[k], mat_a[k][j]) for k in range(m)]
            pts_b = [(eval_roots[k], mat_b[k][j]) for k in range(m)]
            pts_c = [(eval_roots[k], mat_c[k][j]) for k in range(m)]

            a_polys.append(lagrange_interpolation(pts_a, p))
            b_polys.append(lagrange_interpolation(pts_b, p))
            c_polys.append(lagrange_interpolation(pts_c, p))

        return cls(r1cs, eval_roots, z_poly, a_polys, b_polys, c_polys)

    def compute_polynomials(
        self,
        witness: Sequence[Union[int, FieldElement]]
    ) -> Tuple[Polynomial, Polynomial, Polynomial, Polynomial]:
        """
        Given full witness vector s = (s_0, ..., s_{n-1}), compute:
        A(X) = sum_j s_j * A_j(X)
        B(X) = sum_j s_j * B_j(X)
        C(X) = sum_j s_j * C_j(X)
        H(X) = (A(X) * B(X) - C(X)) / Z(X)
        Returns (A, B, C, H).
        Raises ValueError if A(X)*B(X) - C(X) does not divide evenly by Z(X).
        """
        fe_witness = [
            w if isinstance(w, FieldElement) else FieldElement(w, self.p)
            for w in witness
        ]
        if len(fe_witness) != self.num_variables:
            raise ValueError(f"Witness length {len(fe_witness)} != num_variables {self.num_variables}")

        poly_a = Polynomial.zero(self.p)
        poly_b = Polynomial.zero(self.p)
        poly_c = Polynomial.zero(self.p)

        for j, s_j in enumerate(fe_witness):
            if not s_j.is_zero():
                poly_a = poly_a + (self.a_polys[j] * s_j)
                poly_b = poly_b + (self.b_polys[j] * s_j)
                poly_c = poly_c + (self.c_polys[j] * s_j)

        # Compute P(X) = A(X) * B(X) - C(X)
        poly_p = (poly_a * poly_b) - poly_c

        # Divide by vanishing polynomial Z(X)
        quot_h, rem_r = divmod(poly_p, self.z_poly)

        if not rem_r.is_zero():
            raise ValueError("Witness does not satisfy QAP: non-zero remainder in P(X) / Z(X)")

        return poly_a, poly_b, poly_c, quot_h

    def is_satisfied(self, witness: Sequence[Union[int, FieldElement]]) -> bool:
        """
        Returns True if witness satisfies QAP polynomial divisibility condition.
        """
        try:
            self.compute_polynomials(witness)
            return True
        except ValueError:
            return False
