"""
ZetaProof: Finite Field Arithmetic over Prime Moduli F_p.
Provides:
- FieldElement with operator overloading (+, -, *, /, **, ==).
- Extended Euclidean algorithm and Fermat modular inverse.
- Tonelli-Shanks modular square root algorithm.
- Standard cryptographic prime fields (BN254 scalar field, BLS12-381 scalar field, test primes).
"""

from typing import Union, List, Optional


# Standard BN254 (alt_bn128) scalar field prime (used in Ethereum zk-SNARKs)
BN254_SCALAR_FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Standard prime for fast testing and human-readable debug prints
TEST_PRIME = 1000000007


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean Algorithm: returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    """
    if a == 0:
        return b, 0, 1
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd_val, x, y


class FieldElement:
    """
    Element of the prime finite field F_p.
    Guarantees all values are strictly normalized into [0, p - 1].
    """
    __slots__ = ("val", "p")

    def __init__(self, val: Union[int, "FieldElement"], p: int = BN254_SCALAR_FIELD) -> None:
        if isinstance(val, FieldElement):
            self.val = val.val % p
            self.p = p
        else:
            self.val = int(val) % p
            self.p = p

    @classmethod
    def zero(cls, p: int = BN254_SCALAR_FIELD) -> "FieldElement":
        return cls(0, p)

    @classmethod
    def one(cls, p: int = BN254_SCALAR_FIELD) -> "FieldElement":
        return cls(1, p)

    def is_zero(self) -> bool:
        return self.val == 0

    def is_one(self) -> bool:
        return self.val == 1

    def inverse(self) -> "FieldElement":
        """
        Compute modular multiplicative inverse x^-1 mod p.
        Raises ZeroDivisionError if self is zero.
        """
        if self.val == 0:
            raise ZeroDivisionError("Division by zero in finite field")
        # Fermat's Little Theorem: a^(p-2) = a^(-1) mod p for prime p
        inv = pow(self.val, self.p - 2, self.p)
        return FieldElement(inv, self.p)

    def sqrt(self) -> "FieldElement":
        """
        Compute modular square root using the Tonelli-Shanks algorithm.
        Raises ValueError if element is not a quadratic residue.
        """
        if self.val == 0:
            return FieldElement(0, self.p)

        # Euler's criterion: a^((p-1)/2) == 1 mod p for quadratic residues
        if pow(self.val, (self.p - 1) // 2, self.p) != 1:
            raise ValueError(f"{self.val} is not a quadratic residue modulo {self.p}")

        p = self.p
        # Case 1: p = 3 mod 4 -> sqrt = a^((p+1)/4)
        if p % 4 == 3:
            return FieldElement(pow(self.val, (p + 1) // 4, p), p)

        # Factor p - 1 as Q * 2^S with Q odd
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1

        # Find quadratic non-residue z
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        m = s
        c = pow(z, q, p)
        t = pow(self.val, q, p)
        r = pow(self.val, (q + 1) // 2, p)

        while t != 1:
            # Find lowest i such that t^(2^i) = 1
            i = 0
            temp = t
            while temp != 1 and i < m:
                temp = pow(temp, 2, p)
                i += 1

            if i == m:
                raise ValueError("Modular square root computation failed")

            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p

        return FieldElement(r, p)

    def __add__(self, other: Union[int, "FieldElement"]) -> "FieldElement":
        if isinstance(other, FieldElement):
            if self.p != other.p:
                raise ValueError("Cannot add elements of different finite fields")
            return FieldElement(self.val + other.val, self.p)
        return FieldElement(self.val + other, self.p)

    def __radd__(self, other: int) -> "FieldElement":
        return self.__add__(other)

    def __sub__(self, other: Union[int, "FieldElement"]) -> "FieldElement":
        if isinstance(other, FieldElement):
            if self.p != other.p:
                raise ValueError("Cannot subtract elements of different finite fields")
            return FieldElement(self.val - other.val, self.p)
        return FieldElement(self.val - other, self.p)

    def __rsub__(self, other: int) -> "FieldElement":
        return FieldElement(other - self.val, self.p)

    def __mul__(self, other: Union[int, "FieldElement"]) -> "FieldElement":
        if isinstance(other, FieldElement):
            if self.p != other.p:
                raise ValueError("Cannot multiply elements of different finite fields")
            return FieldElement(self.val * other.val, self.p)
        return FieldElement(self.val * other, self.p)

    def __rmul__(self, other: int) -> "FieldElement":
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, "FieldElement"]) -> "FieldElement":
        if isinstance(other, FieldElement):
            return self * other.inverse()
        if other == 0:
            raise ZeroDivisionError("Division by zero in finite field")
        return self * FieldElement(other, self.p).inverse()

    def __rtruediv__(self, other: int) -> "FieldElement":
        return FieldElement(other, self.p) * self.inverse()

    def __neg__(self) -> "FieldElement":
        return FieldElement(-self.val, self.p)

    def __pow__(self, exp: int) -> "FieldElement":
        if exp < 0:
            return self.inverse() ** (-exp)
        return FieldElement(pow(self.val, exp, self.p), self.p)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FieldElement):
            return self.val == other.val and self.p == other.p
        if isinstance(other, int):
            return self.val == (other % self.p)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"FieldElement({self.val}, p={self.p})"

    def __str__(self) -> str:
        return str(self.val)

    def __hash__(self) -> int:
        return hash((self.val, self.p))


def inner_product(vec_a: List[FieldElement], vec_b: List[FieldElement]) -> FieldElement:
    """
    Compute dot product sum(a_i * b_i) over finite field F_p.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} != {len(vec_b)}")
    if not vec_a:
        raise ValueError("Cannot take inner product of empty vectors")
    p = vec_a[0].p
    total = FieldElement.zero(p)
    for a, b in zip(vec_a, vec_b):
        total += a * b
    return total
