"""
ZetaProof: ASCII Terminal Visualizers for Zero-Knowledge Proofs.
Provides:
1. CircuitVisualizer: ASCII dataflow diagram of circuit wires and gates.
2. R1CSVisualizer: Sparsity heatmaps of R1CS matrices A, B, and C.
3. ProofVisualizer: Zero-knowledge proof cards and cryptographic verification badges.
"""

from typing import List, Sequence, Union
from .field import FieldElement
from .circuit import Circuit
from .r1cs import R1CS
from .snark import Proof, VerifyingKey


class CircuitVisualizer:
    """
    Renders ASCII representation of an arithmetic circuit's wire graph.
    """

    @staticmethod
    def render(circuit: Circuit) -> str:
        lines = []
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│                 ARITHMETIC CIRCUIT DATAFLOW GRAPH                  │")
        lines.append("├" + "─" * 68 + "┤")

        # Public inputs
        pub_str = ", ".join(f"[{name}]" for name in circuit.public_input_names) or "None"
        lines.append(f"│  Public Inputs:   {pub_str:<50} │")

        # Private inputs
        priv_str = ", ".join(f"<{name}>" for name in circuit.private_input_names) or "None"
        lines.append(f"│  Private Inputs:  {priv_str:<50} │")

        lines.append("├" + "─" * 68 + "┤")
        lines.append("│  Constraints & Gates:                                              │")

        for idx, c in enumerate(circuit.constraints):
            def fmt_side(terms):
                parts = []
                for vid, coeff in sorted(terms.items()):
                    vname = circuit.vars[vid].name if vid < len(circuit.vars) else f"v{vid}"
                    if vid == 0:
                        parts.append(f"{coeff.val}")
                    else:
                        parts.append(f"{coeff.val}*{vname}" if coeff.val != 1 else vname)
                return " + ".join(parts) if parts else "0"

            str_a = fmt_side(c.a)
            str_b = fmt_side(c.b)
            str_c = fmt_side(c.c)

            gate_str = f"R1CS #{idx + 1}: ({str_a}) * ({str_b}) == ({str_c})"
            if len(gate_str) > 64:
                gate_str = gate_str[:61] + "..."
            lines.append(f"│    {gate_str:<64} │")

        lines.append("└" + "─" * 68 + "┘")
        return "\n".join(lines)


class R1CSVisualizer:
    """
    Renders ASCII sparsity heatmaps of constraint matrices A, B, and C.
    """

    @staticmethod
    def render_sparsity(r1cs: R1CS) -> str:
        mat_a, mat_b, mat_c = r1cs.dense_matrices()
        m = r1cs.num_constraints
        n = r1cs.num_variables

        lines = []
        lines.append(f"R1CS Constraint Matrices (Rows={m} Constraints, Cols={n} Variables)")
        lines.append("  Legend: [.] = 0, [#] = Non-Zero Entry")

        # Header for columns
        col_header = "     " + "".join(f"{j % 10}" for j in range(n))
        lines.append(col_header)

        for i in range(m):
            row_a = "".join("#" if not mat_a[i][j].is_zero() else "." for j in range(n))
            row_b = "".join("#" if not mat_b[i][j].is_zero() else "." for j in range(n))
            row_c = "".join("#" if not mat_c[i][j].is_zero() else "." for j in range(n))
            lines.append(f"R{i + 1:<2}: A:[{row_a}]  B:[{row_b}]  C:[{row_c}]")

        return "\n".join(lines)


class ProofVisualizer:
    """
    Renders formatted cards for zero-knowledge proofs and verification badges.
    """

    @staticmethod
    def render_proof_card(proof: Proof, public_inputs: Sequence[Union[int, FieldElement]], is_valid: bool) -> str:
        lines = []
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║                   GROTH16 ZERO-KNOWLEDGE PROOF                    ║")
        lines.append("╠" + "═" * 68 + "╣")

        # Truncate large field element strings for aesthetic terminal display
        def short_val(fe):
            s = str(fe.val)
            return s[:16] + "..." + s[-8:] if len(s) > 28 else s

        lines.append(f"║  pi_A: {short_val(proof.pi_a):<59} ║")
        lines.append(f"║  pi_B: {short_val(proof.pi_b):<59} ║")
        lines.append(f"║  pi_C: {short_val(proof.pi_c):<59} ║")
        lines.append("╠" + "─" * 68 + "╣")

        pub_str = ", ".join(str(x.val if isinstance(x, FieldElement) else x) for x in public_inputs)
        if len(pub_str) > 50:
            pub_str = pub_str[:47] + "..."
        lines.append(f"║  Public Inputs: [{pub_str:<48}] ║")
        lines.append("╠" + "═" * 68 + "╣")

        if is_valid:
            status = " [✓] PROOF VERIFIED: Cryptographic Pairing Satisfied (SOUND)"
        else:
            status = " [✗] PROOF REJECTED: Invalid Pairing or Forged Public Inputs"

        lines.append(f"║{status:<68}║")
        lines.append("╚" + "═" * 68 + "╝")
        return "\n".join(lines)
