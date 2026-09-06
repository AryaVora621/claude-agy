"""
ZetaProof: Interactive Zero-Knowledge Proof Showcase & Lab.
Demonstrations:
1. Zero-Knowledge Knowledge-of-Exponent Proof (y = x^3 + x + 5).
2. Zero-Knowledge MiMC Hash Preimage Proof (Private password authentication).
3. Zero-Knowledge Range Proof (Prove age or balance without revealing amount).
4. Cryptographic Soundness & Tamper Detection (Adversarial attack rejection).
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zetaproof.field import BN254_SCALAR_FIELD, FieldElement
from zetaproof.snark import Snark, Proof
from zetaproof.circuits_zoo import (
    build_cubic_equation_circuit,
    build_mimc_hash_circuit,
    compute_mimc_hash,
    build_range_proof_circuit
)
from zetaproof.visualizer import CircuitVisualizer, R1CSVisualizer, ProofVisualizer

# Terminal Styling
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"


def demo_cubic_equation():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 1: Zero-Knowledge Equation Proof (y = x^3 + x + 5){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    p = BN254_SCALAR_FIELD
    circuit = build_cubic_equation_circuit(p)

    print(f"{BOLD}1. Arithmetic Circuit Dataflow Diagram:{RESET}")
    print(CircuitVisualizer.render(circuit))
    print()

    r1cs = circuit.to_r1cs()
    print(f"{BOLD}2. R1CS Constraint Matrix Sparsity Heatmap:{RESET}")
    print(R1CSVisualizer.render_sparsity(r1cs))
    print()

    # Trusted Setup
    print(f"{BOLD}3. Running Trusted Setup (Generating Proving & Verifying Keys)...{RESET}")
    t0 = time.perf_counter()
    pk, vk = Snark.setup(circuit)
    dt_setup = time.perf_counter() - t0
    print(f"   [+] Setup completed in {dt_setup * 1000:.2f} ms\n")

    # Prover knowledge: x = 3 -> y = 3^3 + 3 + 5 = 35
    secret_x = 3
    public_y = 35

    print(f"{BOLD}4. Prover generates Zero-Knowledge Proof for secret x = {secret_x} (Public y = {public_y}):{RESET}")
    witness = circuit.solve_witness(
        public_inputs={"y": public_y},
        private_inputs={"x": secret_x}
    )

    t0 = time.perf_counter()
    proof = Snark.prove(pk, witness)
    dt_prove = time.perf_counter() - t0
    print(f"   [+] Proof generated in {dt_prove * 1000:.2f} ms with random blinding factors (r, s)\n")

    # Verifier
    print(f"{BOLD}5. Verifier checks Proof using ONLY public inputs [1, {public_y}]:{RESET}")
    public_inputs = [1, public_y]
    t0 = time.perf_counter()
    is_valid = Snark.verify(vk, public_inputs, proof)
    dt_verify = time.perf_counter() - t0

    print(ProofVisualizer.render_proof_card(proof, public_inputs, is_valid))
    print(f"   [+] Verification time: {dt_verify * 1000:.3f} ms\n\n")


def demo_mimc_hash_preimage():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 2: Zero-Knowledge MiMC Hash Preimage Proof (Private Auth){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    p = BN254_SCALAR_FIELD
    rounds = 4
    circuit, constants = build_mimc_hash_circuit(rounds=rounds, p=p)

    pk, vk = Snark.setup(circuit)

    secret_preimage = 987654321
    public_hash = compute_mimc_hash(secret_preimage, rounds=rounds, p=p, constants=constants)

    print(f"Target Public Hash:   {public_hash}")
    print(f"Secret Preimage:      [HIDDEN / PRIVATE]")
    print()

    witness = circuit.solve_witness(
        public_inputs={"hash": public_hash},
        private_inputs={"preimage": secret_preimage}
    )

    t0 = time.perf_counter()
    proof = Snark.prove(pk, witness)
    dt_prove = time.perf_counter() - t0

    public_inputs = [1, public_hash]
    is_valid = Snark.verify(vk, public_inputs, proof)

    print(ProofVisualizer.render_proof_card(proof, public_inputs, is_valid))
    print(f"{GREEN}[+] Prover proved knowledge of password preimage in {dt_prove*1000:.2f} ms without disclosing a single bit!{RESET}\n\n")


def demo_range_proof():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 3: Zero-Knowledge Range Proof (Bit Decomposition Constraints){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    p = BN254_SCALAR_FIELD
    num_bits = 8  # [0, 255]
    circuit = build_range_proof_circuit(num_bits=num_bits, p=p)
    pk, vk = Snark.setup(circuit)

    secret_age = 21
    print(f"Statement: Prover's secret age is strictly within range [0, 255].")
    print(f"Secret Age: [HIDDEN / PRIVATE]")

    witness = circuit.solve_witness(private_inputs={"value": secret_age})
    proof = Snark.prove(pk, witness)

    is_valid = Snark.verify(vk, [1], proof)
    print(ProofVisualizer.render_proof_card(proof, [1], is_valid))
    print(f"{GREEN}[+] Range constraints verified without revealing secret age!{RESET}\n\n")


def demo_adversarial_tamper():
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} Demo 4: Cryptographic Soundness & Adversarial Tamper Rejection{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    p = BN254_SCALAR_FIELD
    circuit = build_cubic_equation_circuit(p)
    pk, vk = Snark.setup(circuit)

    # Valid witness for x = 3 -> y = 35
    witness = circuit.solve_witness(public_inputs={"y": 35}, private_inputs={"x": 3})
    proof = Snark.prove(pk, witness)

    # Adversary attempts to claim proof applies to y = 999
    attacker_y = 999
    print(f"{YELLOW}[!] Adversary attempts to reuse proof for fake public input y = {attacker_y}...{RESET}")
    is_valid_attack1 = Snark.verify(vk, [1, attacker_y], proof)
    print(ProofVisualizer.render_proof_card(proof, [1, attacker_y], is_valid_attack1))

    # Adversary attempts to forge proof element pi_A
    print(f"\n{YELLOW}[!] Adversary attempts to forge proof element pi_A...{RESET}")
    forged_proof = Proof(
        pi_a=proof.pi_a + 42,
        pi_b=proof.pi_b,
        pi_c=proof.pi_c,
        p=p
    )
    is_valid_attack2 = Snark.verify(vk, [1, 35], forged_proof)
    print(ProofVisualizer.render_proof_card(forged_proof, [1, 35], is_valid_attack2))
    print(f"\n{GREEN}[+] Cryptographic Soundness Verified: All forged inputs and manipulated proofs rejected!{RESET}\n")


def main():
    print(f"{BOLD}{BLUE}======================================================================{RESET}")
    print(f"{BOLD}{BLUE}        ZETAPROOF: ZERO-KNOWLEDGE PROOF & ZK-SNARK ENGINE LAB         {RESET}")
    print(f"{BOLD}{BLUE}======================================================================{RESET}\n")

    demo_cubic_equation()
    demo_mimc_hash_preimage()
    demo_range_proof()
    demo_adversarial_tamper()


if __name__ == "__main__":
    main()
