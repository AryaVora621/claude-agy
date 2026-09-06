"""Unit Tests for Module-LWE IND-CPA Public Key Encryption."""

import os
import unittest

from latticeguard.ind_cpa import (
    PARAMS_512,
    PARAMS_768,
    PARAMS_1024,
    cpa_keygen,
    cpa_encrypt,
    cpa_decrypt,
)


class TestINDCPA(unittest.TestCase):
    """Test suite for K-PKE encryption scheme under ML-KEM-512, 768, and 1024."""

    def test_cpa_roundtrip_all_parameter_sets(self) -> None:
        """Verify keygen -> encrypt -> decrypt roundtrip across all 3 parameter sets."""
        for params in [PARAMS_512, PARAMS_768, PARAMS_1024]:
            kp = cpa_keygen(params)
            self.assertEqual(len(kp.public_key), params.pk_bytes_len)
            self.assertEqual(len(kp.secret_key), params.sk_cpa_bytes_len)

            msg = os.urandom(32)
            ct = cpa_encrypt(params, kp.public_key, msg)
            self.assertEqual(len(ct), params.ct_bytes_len)

            decrypted = cpa_decrypt(params, kp.secret_key, ct)
            self.assertEqual(decrypted, msg, f"Decryption failed for {params.name}")

    def test_cpa_edge_case_plaintexts(self) -> None:
        """Verify encryption and decryption of edge case messages (zeros, ones, alternating)."""
        params = PARAMS_768
        kp = cpa_keygen(params)

        edge_cases = [
            b"\x00" * 32,
            b"\xFF" * 32,
            b"\x55" * 32,
            b"\xAA" * 32,
            bytes(range(32)),
        ]

        for msg in edge_cases:
            ct = cpa_encrypt(params, kp.public_key, msg)
            decrypted = cpa_decrypt(params, kp.secret_key, ct)
            self.assertEqual(decrypted, msg)

    def test_cpa_deterministic_encryption_with_coins(self) -> None:
        """Verify that fixed coins produce identical ciphertexts."""
        params = PARAMS_768
        kp = cpa_keygen(params)
        msg = os.urandom(32)
        coins = os.urandom(32)

        ct1 = cpa_encrypt(params, kp.public_key, msg, coins_32bytes=coins)
        ct2 = cpa_encrypt(params, kp.public_key, msg, coins_32bytes=coins)
        self.assertEqual(ct1, ct2)


if __name__ == "__main__":
    unittest.main()
