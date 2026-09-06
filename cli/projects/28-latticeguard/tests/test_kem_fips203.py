"""Unit Tests for FIPS 203 ML-KEM Key-Encapsulation Mechanism."""

import hashlib
import os
import unittest

from latticeguard.kem import MLKEM512, MLKEM768, MLKEM1024


class TestKEMFIPS203(unittest.TestCase):
    """Test suite for ML-KEM-512, ML-KEM-768, and ML-KEM-1024."""

    def test_kem_roundtrip_all_suites(self) -> None:
        """Verify keygen -> encaps -> decaps yields identical 32-byte shared secret."""
        for kem in [MLKEM512, MLKEM768, MLKEM1024]:
            kp = kem.keygen()
            self.assertEqual(len(kp.ek), kem.ek_len)
            self.assertEqual(len(kp.dk), kem.dk_len)

            enc_res = kem.encaps(kp.ek)
            self.assertEqual(len(enc_res.shared_secret), 32)
            self.assertEqual(len(enc_res.ciphertext), kem.ct_len)

            dec_ss = kem.decaps(kp.dk, enc_res.ciphertext)
            self.assertEqual(dec_ss, enc_res.shared_secret, f"Mismatch in {kem.name}")

    def test_kem_implicit_rejection_on_ciphertext_tamper(self) -> None:
        """Verify FO transform implicit rejection prevents chosen-ciphertext attacks.

        When ciphertext is tampered, decapsulation must not return the true key,
        must not throw an exception, and must deterministically return K_bar = J(z || c).
        """
        for kem in [MLKEM512, MLKEM768, MLKEM1024]:
            kp = kem.keygen()
            enc_res = kem.encaps(kp.ek)

            # Corrupt single bit
            corrupted = bytearray(enc_res.ciphertext)
            corrupted[10] ^= 0x01
            tampered_ct = bytes(corrupted)

            fallback_ss = kem.decaps(kp.dk, tampered_ct)
            # Must not match honest secret
            self.assertNotEqual(fallback_ss, enc_res.shared_secret)
            self.assertEqual(len(fallback_ss), 32)

            # Repeat decapsulation with same tampered ciphertext: must be deterministic
            fallback_ss_again = kem.decaps(kp.dk, tampered_ct)
            self.assertEqual(fallback_ss, fallback_ss_again)

    def test_deterministic_keygen_and_encaps(self) -> None:
        """Verify deterministic execution with pinned seeds."""
        seed_d = b"\x01" * 32
        seed_z = b"\x02" * 32
        kp1 = MLKEM768.keygen(seed_d=seed_d, seed_z=seed_z)
        kp2 = MLKEM768.keygen(seed_d=seed_d, seed_z=seed_z)
        self.assertEqual(kp1.ek, kp2.ek)
        self.assertEqual(kp1.dk, kp2.dk)

        msg_m = b"\x03" * 32
        enc1 = MLKEM768.encaps(kp1.ek, msg_m=msg_m)
        enc2 = MLKEM768.encaps(kp2.ek, msg_m=msg_m)
        self.assertEqual(enc1.shared_secret, enc2.shared_secret)
        self.assertEqual(enc1.ciphertext, enc2.ciphertext)

    def test_invalid_length_guards(self) -> None:
        """Verify defensive guards against malformed key/ciphertext buffers."""
        with self.assertRaises(ValueError):
            MLKEM768.keygen(seed_d=b"\x00" * 31)  # Too short

        with self.assertRaises(ValueError):
            MLKEM768.encaps(ek=b"\x00" * 100)     # Invalid ek length

        kp = MLKEM768.keygen()
        with self.assertRaises(ValueError):
            MLKEM768.decaps(dk=kp.dk, ciphertext=b"\x00" * 50)  # Invalid ct length


if __name__ == "__main__":
    unittest.main()
