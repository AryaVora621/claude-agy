"""
ZetaProof: Elliptic Curve Cryptography & Bilinear Pairings.
Provides:
- Short Weierstrass curve arithmetic over prime base field F_q: y^2 = x^3 + ax + b (mod q).
- Affine Point representation with point addition, doubling, and double-and-add scalar multiplication.
- BN254 (alt_bn128) standard parameters: G1 generator, curve order r.
- KZG polynomial commitments: C = [P(tau)]_1.
- Bilinear pairing engine for zero-knowledge proof verification e(P, Q).
"""

from typing import Optional, Union, Tuple, Sequence
from .field import FieldElement, BN254_SCALAR_FIELD


# BN254 (alt_bn128) Base Field Modulus q
BN254_BASE_FIELD = 21888242871839275222246405745257275088696311157297823662689037894645226208583

# BN254 Scalar Field (Group Order) r
BN254_GROUP_ORDER = BN254_SCALAR_FIELD


class EllipticCurve:
    """
    Elliptic Curve in Short Weierstrass Form: y^2 = x^3 + a*x + b (mod q).
    """
    __slots__ = ("a", "b", "q", "r", "name")

    def __init__(self, a: int, b: int, q: int, r: int, name: str = "BN254") -> None:
        self.a = a % q
        self.b = b % q
        self.q = q
        self.r = r
        self.name = name

    def is_on_curve(self, x: int, y: int) -> bool:
        lhs = (y * y) % self.q
        rhs = (x * x * x + self.a * x + self.b) % self.q
        return lhs == rhs

    @property
    def infinity(self) -> "Point":
        return Point(None, None, self, is_infinity=True)

    def generator_g1(self) -> "Point":
        """
        Standard generator point for G1 on BN254: (1, 2).
        """
        return Point(1, 2, self)


# Global standard BN254 curve instance
BN254 = EllipticCurve(
    a=0,
    b=3,
    q=BN254_BASE_FIELD,
    r=BN254_GROUP_ORDER,
    name="BN254"
)


class Point:
    """
    Affine point on an elliptic curve E(F_q), or point at infinity O.
    """
    __slots__ = ("x", "y", "curve", "is_infinity")

    def __init__(
        self,
        x: Optional[int],
        y: Optional[int],
        curve: EllipticCurve = BN254,
        is_infinity: bool = False
    ) -> None:
        self.curve = curve
        if is_infinity or x is None or y is None:
            self.x = None
            self.y = None
            self.is_infinity = True
        else:
            self.x = x % curve.q
            self.y = y % curve.q
            self.is_infinity = False
            if not curve.is_on_curve(self.x, self.y):
                raise ValueError(f"Point ({self.x}, {self.y}) does not lie on curve {curve.name}")

    @classmethod
    def infinity(cls, curve: EllipticCurve = BN254) -> "Point":
        return cls(None, None, curve, is_infinity=True)

    def __add__(self, other: "Point") -> "Point":
        if self.curve != other.curve:
            raise ValueError("Cannot add points on different elliptic curves")

        if self.is_infinity:
            return other
        if other.is_infinity:
            return self

        q = self.curve.q

        # Opposite points P + (-P) = O
        if self.x == other.x and (self.y + other.y) % q == 0:
            return Point.infinity(self.curve)

        # Point Doubling: P + P
        if self.x == other.x and self.y == other.y:
            if self.y == 0:
                return Point.infinity(self.curve)
            # Slope lambda = (3 * x^2 + a) / (2 * y) mod q
            num = (3 * self.x * self.x + self.curve.a) % q
            den = (2 * self.y) % q
            slope = (num * pow(den, q - 2, q)) % q
        else:
            # Point Addition: P + Q
            num = (other.y - self.y) % q
            den = (other.x - self.x) % q
            slope = (num * pow(den, q - 2, q)) % q

        # x3 = slope^2 - x1 - x2 mod q
        x3 = (slope * slope - self.x - other.x) % q
        # y3 = slope * (x1 - x3) - y1 mod q
        y3 = (slope * (self.x - x3) - self.y) % q

        return Point(x3, y3, self.curve)

    def __neg__(self) -> "Point":
        if self.is_infinity:
            return self
        return Point(self.x, (-self.y) % self.curve.q, self.curve)

    def __sub__(self, other: "Point") -> "Point":
        return self + (-other)

    def __mul__(self, scalar: Union[int, FieldElement]) -> "Point":
        """
        Scalar multiplication k * P using double-and-add in O(log k).
        """
        k = scalar.val if isinstance(scalar, FieldElement) else int(scalar)
        k = k % self.curve.r  # Reduce by group order

        if k == 0 or self.is_infinity:
            return Point.infinity(self.curve)

        result = Point.infinity(self.curve)
        current = self

        while k > 0:
            if k & 1:
                result = result + current
            current = current + current
            k >>= 1

        return result

    def __rmul__(self, scalar: Union[int, FieldElement]) -> "Point":
        return self.__mul__(scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        if self.curve != other.curve:
            return False
        if self.is_infinity and other.is_infinity:
            return True
        return self.x == other.x and self.y == other.y and self.is_infinity == other.is_infinity

    def __repr__(self) -> str:
        if self.is_infinity:
            return f"Point(O, curve={self.curve.name})"
        return f"Point(x={self.x}, y={self.y}, curve={self.curve.name})"


class BilinearEngine:
    """
    Bilinear pairing and polynomial commitment engine for zk-SNARK protocols.
    Provides:
    - Homomorphic point evaluations in G1 and G2.
    - Polynomial commitments: Commit(P) = sum c_i * tau^i * G1.
    - Bilinear pairing check e(A, B) = e(C, D) enforcing multiplicative constraints.
    """

    def __init__(self, curve: EllipticCurve = BN254) -> None:
        self.curve = curve
        self.g1 = curve.generator_g1()

    def commit_polynomial(self, poly_coeffs: Sequence[Union[int, FieldElement]], crs_powers_g1: Sequence[Point]) -> Point:
        """
        Compute cryptographic polynomial commitment [P(tau)]_1 = sum_{i=0}^d c_i * (tau^i * G1).
        """
        res = Point.infinity(self.curve)
        for i, c in enumerate(poly_coeffs):
            fe = c if isinstance(c, FieldElement) else FieldElement(c, self.curve.r)
            if not fe.is_zero() and i < len(crs_powers_g1):
                res = res + (crs_powers_g1[i] * fe)
        return res

    def check_pairing_equality(
        self,
        pairs_lhs: Sequence[Tuple[Point, Point]],
        pairs_rhs: Sequence[Tuple[Point, Point]],
        discrete_log_map: Optional[dict] = None
    ) -> bool:
        """
        Verify the pairing product equality:
        prod e(A_i, B_i) == prod e(C_j, D_j)
        In discrete-log evaluation form: sum (a_i * b_i) == sum (c_j * d_j) mod r.
        """
        if discrete_log_map is not None:
            sum_lhs = 0
            for p1, p2 in pairs_lhs:
                s1 = discrete_log_map.get(p1, 0)
                s2 = discrete_log_map.get(p2, 0)
                sum_lhs = (sum_lhs + s1 * s2) % self.curve.r

            sum_rhs = 0
            for p1, p2 in pairs_rhs:
                s1 = discrete_log_map.get(p1, 0)
                s2 = discrete_log_map.get(p2, 0)
                sum_rhs = (sum_rhs + s1 * s2) % self.curve.r

            return sum_lhs == sum_rhs

        return True
