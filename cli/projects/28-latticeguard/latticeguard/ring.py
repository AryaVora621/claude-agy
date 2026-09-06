"""Finite Field Z_q and Polynomial Ring R_q Arithmetic for Lattice-Based Cryptography.

Implements:
1. Ring modulus q = 3329 and polynomial degree n = 256
2. Montgomery reduction (R = 2^16) and Barrett reduction
3. Polynomial ring R_q = Z_q[X] / (X^256 + 1)
4. Vector of polynomials PolyVec in R_q^k (k in {2, 3, 4})
5. Canonical reduction (freeze) to [0, q-1]
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Union

# FIPS 203 Parameters
KYBER_N: int = 256
KYBER_Q: int = 3329

# Montgomery arithmetic constants
# R = 2^16 = 65536
# QINV = -q^(-1) mod 2^16 = 3327 (or q^(-1) mod 2^16 = 62209)
MONTGOMERY_R: int = 65536
MONTGOMERY_R_MOD_Q: int = 2285       # 2^16 mod 3329
MONTGOMERY_R2_MOD_Q: int = 1353      # (2^16)^2 mod 3329
QINV: int = 62209                    # 3329 * 62209 = 207093761 = 1 mod 65536
BARRETT_V: int = 20159               # round(2^26 / 3329)


def freeze(val: int) -> int:
    """Reduce integer val to canonical representative in [0, q - 1]."""
    return val % KYBER_Q


def barrett_reduce(a: int) -> int:
    """Barrett reduction of 16-bit integer a to range [-q/2, q/2]."""
    val = (BARRETT_V * a + (1 << 25)) >> 26
    val = a - val * KYBER_Q
    return val


def montgomery_reduce(a: int) -> int:
    """Montgomery reduction computing a * R^(-1) mod q where R = 2^16."""
    u = (a * QINV) & 0xFFFF
    t = (a - u * KYBER_Q) >> 16
    return t


def fqmul(a: int, b: int) -> int:
    """Montgomery multiplication of two field elements in Montgomery form."""
    return montgomery_reduce(a * b)


class Polynomial:
    """Element of the polynomial quotient ring R_q = Z_q[X] / (X^256 + 1)."""

    def __init__(self, coeffs: Optional[Sequence[int]] = None) -> None:
        """Initialize polynomial with 256 coefficients."""
        if coeffs is None:
            self.coeffs: List[int] = [0] * KYBER_N
        else:
            if len(coeffs) != KYBER_N:
                raise ValueError(f"Polynomial must have exactly {KYBER_N} coefficients, got {len(coeffs)}")
            self.coeffs = [freeze(c) for c in coeffs]

    @classmethod
    def zero(cls) -> Polynomial:
        """Create zero polynomial."""
        return cls([0] * KYBER_N)

    def copy(self) -> Polynomial:
        """Create a deep copy of this polynomial."""
        return Polynomial(list(self.coeffs))

    def __len__(self) -> int:
        return KYBER_N

    def __getitem__(self, idx: int) -> int:
        return self.coeffs[idx]

    def __setitem__(self, idx: int, value: int) -> None:
        self.coeffs[idx] = freeze(value)

    def __add__(self, other: Polynomial) -> Polynomial:
        return Polynomial([(a + b) % KYBER_Q for a, b in zip(self.coeffs, other.coeffs)])

    def __sub__(self, other: Polynomial) -> Polynomial:
        return Polynomial([(a - b) % KYBER_Q for a, b in zip(self.coeffs, other.coeffs)])

    def __neg__(self) -> Polynomial:
        return Polynomial([(KYBER_Q - c) % KYBER_Q for c in self.coeffs])

    def __mul__(self, other: Union[Polynomial, int]) -> Polynomial:
        if isinstance(other, int):
            # Scalar multiplication
            s = freeze(other)
            return Polynomial([(c * s) % KYBER_Q for c in self.coeffs])
        elif isinstance(other, Polynomial):
            # Direct classical polynomial multiplication mod (X^256 + 1)
            # (Used for verification against NTT multiplication)
            result = [0] * KYBER_N
            for i in range(KYBER_N):
                c1 = self.coeffs[i]
                if c1 == 0:
                    continue
                for j in range(KYBER_N):
                    c2 = other.coeffs[j]
                    if c2 == 0:
                        continue
                    prod = c1 * c2
                    idx = i + j
                    if idx < KYBER_N:
                        result[idx] = (result[idx] + prod) % KYBER_Q
                    else:
                        # X^256 = -1 mod (X^256 + 1)
                        result[idx - KYBER_N] = (result[idx - KYBER_N] - prod) % KYBER_Q
            return Polynomial(result)
        return NotImplemented

    def __rmul__(self, other: int) -> Polynomial:
        return self.__mul__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return False
        return self.coeffs == other.coeffs

    @property
    def infinity_norm(self) -> int:
        """Infinity norm: maximum absolute coefficient centered around 0."""
        max_v = 0
        for c in self.coeffs:
            centered = c if c <= KYBER_Q // 2 else c - KYBER_Q
            max_v = max(max_v, abs(centered))
        return max_v

    @property
    def l1_norm(self) -> int:
        """L1 norm: sum of absolute centered coefficients."""
        total = 0
        for c in self.coeffs:
            centered = c if c <= KYBER_Q // 2 else c - KYBER_Q
            total += abs(centered)
        return total

    @property
    def energy(self) -> float:
        """Root-mean-square energy of polynomial coefficients."""
        return math.sqrt(sum(c * c for c in self.coeffs) / KYBER_N)

    def reduce(self) -> None:
        """Canonicalize all coefficients to [0, q-1] in place."""
        self.coeffs = [freeze(c) for c in self.coeffs]

    def to_centered_list(self) -> List[int]:
        """Return coefficients centered in range [-floor(q/2), floor(q/2)]."""
        half = KYBER_Q // 2
        return [c if c <= half else c - KYBER_Q for c in self.coeffs]


class PolyVec:
    """Vector of polynomials in R_q^k."""

    def __init__(self, elements: Sequence[Polynomial]) -> None:
        """Initialize polynomial vector."""
        self.k: int = len(elements)
        self.vec: List[Polynomial] = [p.copy() for p in elements]

    @classmethod
    def zero(cls, k: int) -> PolyVec:
        """Create zero vector of rank k."""
        return cls([Polynomial.zero() for _ in range(k)])

    def copy(self) -> PolyVec:
        """Deep copy of the vector."""
        return PolyVec([p.copy() for p in self.vec])

    def __len__(self) -> int:
        return self.k

    def __getitem__(self, idx: int) -> Polynomial:
        return self.vec[idx]

    def __setitem__(self, idx: int, value: Polynomial) -> None:
        self.vec[idx] = value.copy()

    def __add__(self, other: PolyVec) -> PolyVec:
        if self.k != other.k:
            raise ValueError(f"Vector rank mismatch: {self.k} vs {other.k}")
        return PolyVec([p1 + p2 for p1, p2 in zip(self.vec, other.vec)])

    def __sub__(self, other: PolyVec) -> PolyVec:
        if self.k != other.k:
            raise ValueError(f"Vector rank mismatch: {self.k} vs {other.k}")
        return PolyVec([p1 - p2 for p1, p2 in zip(self.vec, other.vec)])

    def __neg__(self) -> PolyVec:
        return PolyVec([-p for p in self.vec])

    def dot(self, other: PolyVec) -> Polynomial:
        """Compute inner dot product sum(u[i] * v[i])."""
        if self.k != other.k:
            raise ValueError(f"Vector rank mismatch: {self.k} vs {other.k}")
        res = Polynomial.zero()
        for p1, p2 in zip(self.vec, other.vec):
            res = res + (p1 * p2)
        return res

    def reduce(self) -> None:
        """Reduce all polynomials in the vector in place."""
        for p in self.vec:
            p.reduce()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyVec):
            return False
        if self.k != other.k:
            return False
        return all(p1 == p2 for p1, p2 in zip(self.vec, other.vec))
