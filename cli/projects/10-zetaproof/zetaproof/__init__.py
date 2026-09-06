"""
ZetaProof: Zero-Knowledge Proof Engine & zk-SNARK Architecture from First Principles.
Zero external dependencies (pure Python standard library).
"""

from .field import FieldElement, BN254_SCALAR_FIELD, TEST_PRIME, inner_product
from .polynomial import Polynomial, lagrange_interpolation
from .circuit import Circuit, Variable, LinearExpression
from .r1cs import Constraint, R1CS
from .qap import QAP
from .curve import EllipticCurve, Point, BN254, BilinearEngine
from .snark import Snark, ProvingKey, VerifyingKey, Proof
from .circuits_zoo import (
    build_cubic_equation_circuit,
    build_mimc_hash_circuit,
    compute_mimc_hash,
    build_range_proof_circuit,
    build_salted_auth_circuit
)
from .visualizer import CircuitVisualizer, R1CSVisualizer, ProofVisualizer

__all__ = [
    "FieldElement",
    "BN254_SCALAR_FIELD",
    "TEST_PRIME",
    "inner_product",
    "Polynomial",
    "lagrange_interpolation",
    "Circuit",
    "Variable",
    "LinearExpression",
    "Constraint",
    "R1CS",
    "QAP",
    "EllipticCurve",
    "Point",
    "BN254",
    "BilinearEngine",
    "Snark",
    "ProvingKey",
    "VerifyingKey",
    "Proof",
    "build_cubic_equation_circuit",
    "build_mimc_hash_circuit",
    "compute_mimc_hash",
    "build_range_proof_circuit",
    "build_salted_auth_circuit",
    "CircuitVisualizer",
    "R1CSVisualizer",
    "ProofVisualizer",
]
