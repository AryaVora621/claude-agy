"""
ZetaProof: Rank-1 Constraint System (R1CS) Representation.
Provides:
- Constraint representation: (a · s) * (b · s) = (c · s).
- R1CS container storing constraint matrices A, B, C over finite field F_p.
- Verification and satisfiability checking for candidate witness vectors.
"""

from typing import List, Dict, Tuple, Sequence, Optional, Union
from .field import FieldElement, BN254_SCALAR_FIELD, inner_product


class Constraint:
    """
    A single Rank-1 constraint: (a · s) * (b · s) = (c · s)
    where a, b, c are sparse linear combinations mapping var_id -> FieldElement.
    """
    __slots__ = ("a", "b", "c")

    def __init__(
        self,
        a: Dict[int, FieldElement],
        b: Dict[int, FieldElement],
        c: Dict[int, FieldElement]
    ) -> None:
        self.a = dict(a)
        self.b = dict(b)
        self.c = dict(c)

    def evaluate_side(self, coeffs: Dict[int, FieldElement], witness: Sequence[FieldElement]) -> FieldElement:
        p = witness[0].p
        total = FieldElement.zero(p)
        for var_id, coeff in coeffs.items():
            if var_id < len(witness):
                total += coeff * witness[var_id]
        return total

    def is_satisfied(self, witness: Sequence[FieldElement]) -> bool:
        """
        Check if (a · s) * (b · s) == (c · s).
        """
        val_a = self.evaluate_side(self.a, witness)
        val_b = self.evaluate_side(self.b, witness)
        val_c = self.evaluate_side(self.c, witness)
        return (val_a * val_b) == val_c

    def __repr__(self) -> str:
        return f"Constraint(a={self.a}, b={self.b}, c={self.c})"


class R1CS:
    """
    Rank-1 Constraint System with m constraints and n variables over F_p.
    Variable 0 is always the constant 1.
    """
    __slots__ = (
        "p",
        "num_variables",
        "num_public_inputs",
        "constraints",
        "variable_names",
        "public_var_ids"
    )

    def __init__(
        self,
        constraints: Sequence[Constraint],
        num_variables: int,
        num_public_inputs: int,
        variable_names: Sequence[str],
        public_var_ids: Sequence[int],
        p: int = BN254_SCALAR_FIELD
    ) -> None:
        self.p = p
        self.constraints = list(constraints)
        self.num_variables = num_variables
        self.num_public_inputs = num_public_inputs
        self.variable_names = list(variable_names)
        self.public_var_ids = list(public_var_ids)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    def is_satisfied(self, witness: Sequence[Union[int, FieldElement]]) -> bool:
        """
        Verify that the witness vector s satisfies all R1CS constraints.
        """
        if len(witness) != self.num_variables:
            return False

        fe_witness = [
            w if isinstance(w, FieldElement) else FieldElement(w, self.p)
            for w in witness
        ]

        # Variable 0 must be 1
        if not fe_witness[0].is_one():
            return False

        for c in self.constraints:
            if not c.is_satisfied(fe_witness):
                return False
        return True

    def dense_matrices(self) -> Tuple[List[List[FieldElement]], List[List[FieldElement]], List[List[FieldElement]]]:
        """
        Convert sparse constraints to dense matrices A, B, C of shape (m, n).
        """
        m = self.num_constraints
        n = self.num_variables

        mat_a = [[FieldElement.zero(self.p) for _ in range(n)] for _ in range(m)]
        mat_b = [[FieldElement.zero(self.p) for _ in range(n)] for _ in range(m)]
        mat_c = [[FieldElement.zero(self.p) for _ in range(n)] for _ in range(m)]

        for row, c in enumerate(self.constraints):
            for col, coeff in c.a.items():
                mat_a[row][col] = coeff
            for col, coeff in c.b.items():
                mat_b[row][col] = coeff
            for col, coeff in c.c.items():
                mat_c[row][col] = coeff

        return mat_a, mat_b, mat_c
