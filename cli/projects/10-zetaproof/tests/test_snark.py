"""
Unit tests for ZetaProof Groth16 / Succinct ZK-SNARK Prover & Verifier.
"""

import unittest
from zetaproof.field import FieldElement, TEST_PRIME, BN254_SCALAR_FIELD
from zetaproof.snark import Snark, Proof
from zetaproof.circuits_zoo import (
    build_cubic_equation_circuit,
    build_mimc_hash_circuit,
    compute_mimc_hash,
    build_range_proof_circuit,
    build_salted_auth_circuit
)


class TestSnark(unittest.TestCase):
    def test_end_to_end_cubic_snark(self):
        p = TEST_PRIME
        circuit = build_cubic_equation_circuit(p)

        # Trusted Setup
        pk, vk = Snark.setup(circuit)

        # Private secret: x = 3 -> Public output y = 3^3 + 3 + 5 = 35
        secret_x = 3
        public_y = 35

        witness = circuit.solve_witness(
            public_inputs={"y": public_y},
            private_inputs={"x": secret_x}
        )

        # Prover generates zero-knowledge proof
        proof1 = Snark.prove(pk, witness)

        # Verifier checks proof with only public inputs [1, y]
        public_inputs = [1, public_y]
        self.assertTrue(Snark.verify(vk, public_inputs, proof1))

        # Test Soundness: Tampering with public input MUST fail
        bad_public_inputs = [1, 36]
        self.assertFalse(Snark.verify(vk, bad_public_inputs, proof1))

        # Test Soundness: Corrupting proof element MUST fail
        corrupted_proof = Proof(
            pi_a=proof1.pi_a + 1,
            pi_b=proof1.pi_b,
            pi_c=proof1.pi_c,
            p=p
        )
        self.assertFalse(Snark.verify(vk, public_inputs, corrupted_proof))

        # Test Zero-Knowledge: Re-proving with different random blinding factors
        proof2 = Snark.prove(pk, witness)
        self.assertNotEqual(proof1.pi_a, proof2.pi_a)
        self.assertTrue(Snark.verify(vk, public_inputs, proof2))

    def test_mimc_hash_preimage_proof(self):
        p = TEST_PRIME
        circuit, constants = build_mimc_hash_circuit(rounds=4, p=p)
        pk, vk = Snark.setup(circuit)

        secret_preimage = 12345
        public_hash = compute_mimc_hash(secret_preimage, rounds=4, p=p, constants=constants)

        witness = circuit.solve_witness(
            public_inputs={"hash": public_hash},
            private_inputs={"preimage": secret_preimage}
        )

        proof = Snark.prove(pk, witness)
        public_inputs = [1, public_hash]
        self.assertTrue(Snark.verify(vk, public_inputs, proof))

        # False preimage test
        wrong_hash = (public_hash + 1) % p
        self.assertFalse(Snark.verify(vk, [1, wrong_hash], proof))

    def test_range_proof(self):
        p = TEST_PRIME
        # 8-bit range proof: 0 <= value < 256
        circuit = build_range_proof_circuit(num_bits=8, p=p)
        pk, vk = Snark.setup(circuit)

        for secret_val in (0, 42, 255):
            witness = circuit.solve_witness(private_inputs={"value": secret_val})
            proof = Snark.prove(pk, witness)
            # Only public input is constant 1
            self.assertTrue(Snark.verify(vk, [1], proof))

    def test_salted_auth_circuit(self):
        p = TEST_PRIME
        circuit = build_salted_auth_circuit(p)
        pk, vk = Snark.setup(circuit)

        pwd = 987654
        salt = 321
        # Token = (pwd * salt + pwd + salt + 17)^3 mod p
        core = (pwd * salt + pwd + salt + 17) % p
        token = pow(core, 3, p)

        witness = circuit.solve_witness(
            public_inputs={"token": token},
            private_inputs={"password": pwd, "salt": salt}
        )

        proof = Snark.prove(pk, witness)
        self.assertTrue(Snark.verify(vk, [1, token], proof))
        self.assertFalse(Snark.verify(vk, [1, (token + 1) % p], proof))


if __name__ == "__main__":
    unittest.main()
