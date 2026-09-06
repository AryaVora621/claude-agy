"""
ZetaProof: Pre-Built Zero-Knowledge Circuit Zoo.
Provides:
1. Cubic Equation Circuit (y = x^3 + x + 5).
2. MiMC Zero-Knowledge Friendly Hash Preimage Proof.
3. Zero-Knowledge Range Proof (Bit Decomposition).
4. Zero-Knowledge Salted Authentication Circuit.
"""

from typing import Tuple, List, Optional
from .field import FieldElement, TEST_PRIME, BN254_SCALAR_FIELD
from .circuit import Circuit


def build_cubic_equation_circuit(p: int = BN254_SCALAR_FIELD) -> Circuit:
    """
    Statement: Prove knowledge of secret x such that x^3 + x + 5 = y.
    Public input: y
    Private input: x
    """
    c = Circuit(p)
    x = c.private_input("x")
    y = c.public_input("y")

    x2 = x * x
    x3 = x2 * x
    c.assert_equal(x3 + x + 5, y)
    return c


def build_mimc_hash_circuit(
    rounds: int = 4,
    p: int = BN254_SCALAR_FIELD,
    round_constants: Optional[List[int]] = None
) -> Tuple[Circuit, List[int]]:
    """
    MiMC Algebraic Zero-Knowledge Hash Function.
    Round function: x_{i+1} = (x_i + k + c_i)^3 mod p.
    Statement: Prove knowledge of secret preimage x such that MiMC(x) = h.
    Public input: h
    Private input: x
    """
    if round_constants is None:
        # Standard pseudo-random round constants
        constants = [(i * 1234567 + 89) % p for i in range(rounds)]
    else:
        constants = round_constants

    c = Circuit(p)
    preimage = c.private_input("preimage")
    expected_hash = c.public_input("hash")

    curr = preimage
    for i in range(rounds):
        # t = curr + c_i
        t = curr + constants[i]
        # t^2
        t2 = t * t
        # t^3
        curr = t2 * t

    c.assert_equal(curr, expected_hash)
    return c, constants


def compute_mimc_hash(preimage: int, rounds: int = 4, p: int = BN254_SCALAR_FIELD, constants: Optional[List[int]] = None) -> int:
    """
    Direct evaluation of the MiMC hash function outside the circuit.
    """
    if constants is None:
        constants = [(i * 1234567 + 89) % p for i in range(rounds)]

    val = preimage % p
    for i in range(rounds):
        t = (val + constants[i]) % p
        t2 = (t * t) % p
        val = (t2 * t) % p
    return val


def build_range_proof_circuit(num_bits: int = 8, p: int = BN254_SCALAR_FIELD) -> Circuit:
    """
    Statement: Prove that secret value v lies strictly within [0, 2^num_bits - 1]
    without disclosing the value v.
    Private input: value
    """
    c = Circuit(p)
    val = c.private_input("value")
    c.range_check(val, num_bits=num_bits)
    return c


def build_salted_auth_circuit(p: int = BN254_SCALAR_FIELD) -> Circuit:
    """
    Statement: Prove knowledge of secret password and secret salt
    such that Hash(password, salt) = (password * salt + password + salt + 17)^3 == public_token.
    Public input: token
    Private inputs: password, salt
    """
    c = Circuit(p)
    pwd = c.private_input("password")
    salt = c.private_input("salt")
    token = c.public_input("token")

    pwd_salt = pwd * salt
    core = pwd_salt + pwd + salt + 17
    core2 = core * core
    core3 = core2 * core

    c.assert_equal(core3, token)
    return c
