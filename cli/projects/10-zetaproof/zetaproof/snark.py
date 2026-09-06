"""
ZetaProof: Groth16 / Succinct Zero-Knowledge SNARK Prover & Verifier.
Provides:
- Trusted setup generating ProvingKey (PK) and VerifyingKey (VK).
- Zero-knowledge Prover adding random blinding factors (r, s).
- Constant-time O(1) Verifier checking the pairing equation.
- Complete serialization to JSON-compatible dictionaries.
"""

import random
from typing import Dict, List, Tuple, Sequence, Optional, Union, Any
from .field import FieldElement, BN254_SCALAR_FIELD
from .polynomial import Polynomial
from .circuit import Circuit
from .r1cs import R1CS
from .qap import QAP


class ProvingKey:
    """
    Proving Key (PK) for the zk-SNARK protocol.
    Contains evaluation bases at secret trapdoor x.
    """
    __slots__ = (
        "p",
        "qap",
        "alpha",
        "beta",
        "delta",
        "a_bases",       # A_j(x) for all j
        "b_bases",       # B_j(x) for all j
        "k_bases",       # Evaluation bases for private witness variables (j >= l): (beta*A_j + alpha*B_j + C_j) / delta
        "h_bases",       # Evaluation bases for quotient polynomial H(X): x^i * Z(x) / delta
        "z_tau",         # Z(x)
        "num_public_inputs"
    )

    def __init__(
        self,
        qap: QAP,
        alpha: FieldElement,
        beta: FieldElement,
        delta: FieldElement,
        a_bases: Sequence[FieldElement],
        b_bases: Sequence[FieldElement],
        k_bases: Sequence[FieldElement],
        h_bases: Sequence[FieldElement],
        z_tau: FieldElement,
        num_public_inputs: int
    ) -> None:
        self.p = qap.p
        self.qap = qap
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.a_bases = list(a_bases)
        self.b_bases = list(b_bases)
        self.k_bases = list(k_bases)
        self.h_bases = list(h_bases)
        self.z_tau = z_tau
        self.num_public_inputs = num_public_inputs


class VerifyingKey:
    """
    Verification Key (VK) for the zk-SNARK protocol.
    Succinct size O(l) proportional to number of public inputs, independent of circuit depth.
    """
    __slots__ = (
        "p",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "l_bases",       # Public input bases (j < l): (beta*A_j + alpha*B_j + C_j) / gamma
        "num_public_inputs",
        "public_var_ids"
    )

    def __init__(
        self,
        alpha: FieldElement,
        beta: FieldElement,
        gamma: FieldElement,
        delta: FieldElement,
        l_bases: Sequence[FieldElement],
        num_public_inputs: int,
        public_var_ids: Sequence[int],
        p: int = BN254_SCALAR_FIELD
    ) -> None:
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.l_bases = list(l_bases)
        self.num_public_inputs = num_public_inputs
        self.public_var_ids = list(public_var_ids)


class Proof:
    """
    Succinct Zero-Knowledge Proof consisting of 3 group elements (pi_a, pi_b, pi_c).
    """
    __slots__ = ("pi_a", "pi_b", "pi_c", "p")

    def __init__(
        self,
        pi_a: Union[int, FieldElement],
        pi_b: Union[int, FieldElement],
        pi_c: Union[int, FieldElement],
        p: int = BN254_SCALAR_FIELD
    ) -> None:
        self.p = p
        self.pi_a = pi_a if isinstance(pi_a, FieldElement) else FieldElement(pi_a, p)
        self.pi_b = pi_b if isinstance(pi_b, FieldElement) else FieldElement(pi_b, p)
        self.pi_c = pi_c if isinstance(pi_c, FieldElement) else FieldElement(pi_c, p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pi_a": str(self.pi_a.val),
            "pi_b": str(self.pi_b.val),
            "pi_c": str(self.pi_c.val)
        }

    def __repr__(self) -> str:
        return f"Proof(pi_a={self.pi_a.val}, pi_b={self.pi_b.val}, pi_c={self.pi_c.val})"


class Snark:
    """
    Non-interactive zero-knowledge SNARK engine.
    """

    @staticmethod
    def setup(
        circuit: Circuit,
        toxic_waste: Optional[Tuple[int, int, int, int, int]] = None
    ) -> Tuple[ProvingKey, VerifyingKey]:
        """
        Generate Common Reference String (CRS): Proving Key and Verifying Key.
        Uses toxic waste (alpha, beta, gamma, delta, x).
        """
        p = circuit.p
        r1cs = circuit.to_r1cs()
        qap = QAP.from_r1cs(r1cs)

        # Generate or unpack secret trapdoors
        if toxic_waste is None:
            alpha_val = random.randint(2, p - 2)
            beta_val = random.randint(2, p - 2)
            gamma_val = random.randint(2, p - 2)
            delta_val = random.randint(2, p - 2)
            # x must not be any evaluation root
            x_val = random.randint(qap.num_constraints + 10, p - 2)
        else:
            alpha_val, beta_val, gamma_val, delta_val, x_val = toxic_waste

        alpha = FieldElement(alpha_val, p)
        beta = FieldElement(beta_val, p)
        gamma = FieldElement(gamma_val, p)
        delta = FieldElement(delta_val, p)
        x = FieldElement(x_val, p)

        # Evaluate QAP polynomials at secret point x
        n = qap.num_variables
        l = r1cs.num_public_inputs  # Variables 0, ..., l-1 are public

        a_x = [poly(x) for poly in qap.a_polys]
        b_x = [poly(x) for poly in qap.b_polys]
        c_x = [poly(x) for poly in qap.c_polys]
        z_x = qap.z_poly(x)

        # Compute public input bases L_j(x) = (beta * A_j(x) + alpha * B_j(x) + C_j(x)) / gamma
        gamma_inv = gamma.inverse()
        l_bases: List[FieldElement] = []
        for j in range(l):
            term = (beta * a_x[j]) + (alpha * b_x[j]) + c_x[j]
            l_bases.append(term * gamma_inv)

        # Compute private witness bases K_j(x) = (beta * A_j(x) + alpha * B_j(x) + C_j(x)) / delta
        delta_inv = delta.inverse()
        k_bases: List[FieldElement] = []
        for j in range(l, n):
            term = (beta * a_x[j]) + (alpha * b_x[j]) + c_x[j]
            k_bases.append(term * delta_inv)

        # Compute quotient bases: (x^i * Z(x)) / delta for deg(H)
        max_h_deg = max(0, 2 * qap.num_constraints - qap.num_constraints + 2)
        h_bases: List[FieldElement] = []
        curr_power = FieldElement.one(p)
        for _ in range(max_h_deg + 1):
            h_bases.append((curr_power * z_x) * delta_inv)
            curr_power = curr_power * x

        pk = ProvingKey(
            qap=qap,
            alpha=alpha,
            beta=beta,
            delta=delta,
            a_bases=a_x,
            b_bases=b_x,
            k_bases=k_bases,
            h_bases=h_bases,
            z_tau=z_x,
            num_public_inputs=l
        )

        vk = VerifyingKey(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            l_bases=l_bases,
            num_public_inputs=l,
            public_var_ids=r1cs.public_var_ids,
            p=p
        )

        return pk, vk

    @staticmethod
    def prove(
        pk: ProvingKey,
        witness: Sequence[Union[int, FieldElement]],
        random_nonces: Optional[Tuple[int, int]] = None
    ) -> Proof:
        """
        Generate a zero-knowledge proof for witness s.
        Incorporates random blinding factors (r, s) to ensure zero-knowledge privacy.
        """
        p = pk.p
        fe_witness = [
            w if isinstance(w, FieldElement) else FieldElement(w, p)
            for w in witness
        ]

        # Compute QAP polynomials A(X), B(X), C(X), H(X)
        poly_a, poly_b, poly_c, poly_h = pk.qap.compute_polynomials(fe_witness)

        # Blinding factors for zero-knowledge
        if random_nonces is None:
            r_val = random.randint(1, p - 1)
            s_val = random.randint(1, p - 1)
        else:
            r_val, s_val = random_nonces

        r = FieldElement(r_val, p)
        s = FieldElement(s_val, p)

        l = pk.num_public_inputs
        n = pk.qap.num_variables

        # Evaluate A(x) and B(x) from witness
        # A(x) = sum_j s_j * A_j(x)
        # B(x) = sum_j s_j * B_j(x)
        sum_a = FieldElement.zero(p)
        sum_b = FieldElement.zero(p)
        for j in range(n):
            sj = fe_witness[j]
            if not sj.is_zero():
                sum_a = sum_a + (sj * pk.a_bases[j])
                sum_b = sum_b + (sj * pk.b_bases[j])

        # pi_a = alpha + A(x) + r * delta
        pi_a = pk.alpha + sum_a + (r * pk.delta)

        # pi_b = beta + B(x) + s * delta
        pi_b = pk.beta + sum_b + (s * pk.delta)

        # Evaluate sum_{j >= l} s_j * K_j(x)
        sum_k = FieldElement.zero(p)
        for idx, j in enumerate(range(l, n)):
            sj = fe_witness[j]
            if not sj.is_zero() and idx < len(pk.k_bases):
                sum_k = sum_k + (sj * pk.k_bases[idx])

        # Evaluate H(x)*Z(x)/delta from h_bases and H(X) coefficients
        h_part = FieldElement.zero(p)
        for i, coeff in enumerate(poly_h.coeffs):
            if i < len(pk.h_bases) and not coeff.is_zero():
                h_part = h_part + (coeff * pk.h_bases[i])

        # pi_c = sum_{j >= l} s_j * K_j(x) + H(x)*Z(x)/delta + s * pi_a + r * pi_b - r * s * delta
        pi_c = sum_k + h_part + (s * pi_a) + (r * pi_b) - (r * s * pk.delta)

        return Proof(pi_a, pi_b, pi_c, p)

    @staticmethod
    def verify(
        vk: VerifyingKey,
        public_inputs: Sequence[Union[int, FieldElement]],
        proof: Proof
    ) -> bool:
        """
        Verify proof pi against public inputs in O(l) time:
        e(pi_a, pi_b) == e(alpha, beta) * e(V_pub, gamma) * e(pi_c, delta)
        In discrete log / pairing evaluation:
        pi_a * pi_b == alpha * beta + V_pub * gamma + pi_c * delta (mod p)
        """
        p = vk.p
        fe_pub = [
            x if isinstance(x, FieldElement) else FieldElement(x, p)
            for x in public_inputs
        ]

        if len(fe_pub) != vk.num_public_inputs:
            return False

        # Compute V_pub = sum_{j=0}^{l-1} s_j * L_j(x)
        v_pub = FieldElement.zero(p)
        for j, s_j in enumerate(fe_pub):
            if j < len(vk.l_bases):
                v_pub = v_pub + (s_j * vk.l_bases[j])

        # LHS = pi_a * pi_b
        lhs = proof.pi_a * proof.pi_b

        # RHS = alpha * beta + (v_pub * gamma) + (proof.pi_c * delta)
        rhs = (vk.alpha * vk.beta) + (v_pub * vk.gamma) + (proof.pi_c * vk.delta)

        return lhs == rhs
