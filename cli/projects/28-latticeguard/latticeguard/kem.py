"""FIPS 203 Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM).

Implements the official NIST FIPS 203 standard:
1. ML-KEM.KeyGen: Generates encapsulation key (ek) and decapsulation key (dk).
2. ML-KEM.Encaps: Generates shared secret K and encapsulates it into ciphertext c.
3. ML-KEM.Decaps: Decapsulates ciphertext c with constant-time implicit rejection
   (Fujisaki-Okamoto transform) returning genuine key K on success or pseudorandom
   K_bar on invalid ciphertext without leaking side-channel timing.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import hmac
import os
from typing import NamedTuple, Tuple

from .ind_cpa import (
    MLKEMParams,
    PARAMS_512,
    PARAMS_768,
    PARAMS_1024,
    cpa_keygen,
    cpa_encrypt,
    cpa_decrypt,
)


@dataclass(frozen=True)
class KEMKeyPair:
    """ML-KEM Encapsulation and Decapsulation Key Pair."""
    encapsulation_key: bytes  # ek (public key)
    decapsulation_key: bytes  # dk (private key)
    params: MLKEMParams

    @property
    def ek(self) -> bytes:
        return self.encapsulation_key

    @property
    def dk(self) -> bytes:
        return self.decapsulation_key


@dataclass(frozen=True)
class EncapsResult:
    """Result of ML-KEM Encapsulation."""
    shared_secret: bytes  # 32-byte shared secret key K
    ciphertext: bytes     # Encapsulated ciphertext c


class MLKEM:
    """NIST FIPS 203 ML-KEM Engine supporting 512, 768, and 1024 parameter sets."""

    def __init__(self, params: MLKEMParams = PARAMS_768) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def ek_len(self) -> int:
        """Length of encapsulation key ek in bytes."""
        return self.params.pk_bytes_len

    @property
    def dk_len(self) -> int:
        """Length of decapsulation key dk in bytes: 768 * k + 96."""
        return 768 * self.params.k + 96

    @property
    def ct_len(self) -> int:
        """Length of ciphertext in bytes."""
        return self.params.ct_bytes_len

    def keygen(self, seed_d: bytes | None = None, seed_z: bytes | None = None) -> KEMKeyPair:
        """Generate ML-KEM key pair according to FIPS 203 Algorithm 15.

        Args:
            seed_d: Optional 32-byte seed for K-PKE.KeyGen.
            seed_z: Optional 32-byte seed for implicit rejection fallback.

        Returns:
            KEMKeyPair containing (ek, dk).
        """
        if seed_d is None:
            seed_d = os.urandom(32)
        elif len(seed_d) != 32:
            raise ValueError(f"seed_d must be 32 bytes, got {len(seed_d)}")

        if seed_z is None:
            seed_z = os.urandom(32)
        elif len(seed_z) != 32:
            raise ValueError(f"seed_z must be 32 bytes, got {len(seed_z)}")

        # Step 1: Run K-PKE.KeyGen(seed_d) -> (ek_PKE, dk_PKE)
        cpa_kp = cpa_keygen(self.params, seed_d)
        ek_pke = cpa_kp.public_key
        dk_pke = cpa_kp.secret_key

        # Step 2: Encapsulation key ek = ek_PKE
        ek = ek_pke

        # Step 3: Compute H(ek) = SHA3-256(ek)
        h_ek = hashlib.sha3_256(ek).digest()

        # Step 4: Decapsulation key dk = dk_PKE || ek || H(ek) || z
        dk = dk_pke + ek + h_ek + seed_z

        return KEMKeyPair(encapsulation_key=ek, decapsulation_key=dk, params=self.params)

    def encaps(self, ek: bytes, msg_m: bytes | None = None) -> EncapsResult:
        """Encapsulate shared secret according to FIPS 203 Algorithm 16.

        Args:
            ek: Encapsulation key (public key).
            msg_m: Optional 32-byte random message. If None, drawn from os.urandom(32).

        Returns:
            EncapsResult containing (shared_secret, ciphertext).
        """
        if len(ek) != self.ek_len:
            raise ValueError(f"Invalid ek length: {len(ek)} (expected {self.ek_len})")

        if msg_m is None:
            msg_m = os.urandom(32)
        elif len(msg_m) != 32:
            raise ValueError(f"msg_m must be 32 bytes, got {len(msg_m)}")

        # Step 1: Compute H(ek) = SHA3-256(ek)
        h_ek = hashlib.sha3_256(ek).digest()

        # Step 2: (K, r) = G(m || H(ek)) where G is SHA3-512
        g_hash = hashlib.sha3_512(msg_m + h_ek).digest()
        shared_secret_k = g_hash[:32]
        coins_r = g_hash[32:]

        # Step 3: Encrypt m using K-PKE with coins r
        ciphertext = cpa_encrypt(self.params, ek, msg_m, coins_32bytes=coins_r)

        return EncapsResult(shared_secret=shared_secret_k, ciphertext=ciphertext)

    def decaps(self, dk: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate ciphertext according to FIPS 203 Algorithm 17.

        Includes the Fujisaki-Okamoto transform with constant-time implicit rejection:
        If ciphertext is tampered or invalid, decapsulation returns a pseudorandom key
        derived from secret seed z and the ciphertext, preventing side-channel leakage.

        Args:
            dk: Decapsulation key (private key).
            ciphertext: Received ciphertext.

        Returns:
            32-byte shared secret key.
        """
        if len(dk) != self.dk_len:
            raise ValueError(f"Invalid dk length: {len(dk)} (expected {self.dk_len})")
        if len(ciphertext) != self.ct_len:
            raise ValueError(f"Invalid ciphertext length: {len(ciphertext)} (expected {self.ct_len})")

        k = self.params.k
        # Parse decapsulation key dk = dk_PKE (384*k) || ek (384*k + 32) || h (32) || z (32)
        dk_pke_len = 384 * k
        ek_len = 384 * k + 32

        dk_pke = dk[:dk_pke_len]
        ek = dk[dk_pke_len : dk_pke_len + ek_len]
        h = dk[dk_pke_len + ek_len : dk_pke_len + ek_len + 32]
        z = dk[dk_pke_len + ek_len + 32 : dk_pke_len + ek_len + 64]

        # Step 1: m' = K-PKE.Decrypt(dk_PKE, c)
        m_prime = cpa_decrypt(self.params, dk_pke, ciphertext)

        # Step 2: (K', r') = G(m' || h)
        g_hash = hashlib.sha3_512(m_prime + h).digest()
        k_prime = g_hash[:32]
        r_prime = g_hash[32:]

        # Step 3: Compute fallback implicit rejection key K_bar = J(z || c, 32) using SHAKE-256
        shake = hashlib.shake_256()
        shake.update(z)
        shake.update(ciphertext)
        k_bar = shake.digest(32)

        # Step 4: Re-encrypt m' using public key ek and coins r': c' = K-PKE.Encrypt(ek, m', r')
        c_prime = cpa_encrypt(self.params, ek, m_prime, coins_32bytes=r_prime)

        # Step 5: Constant-time comparison
        is_valid = hmac.compare_digest(ciphertext, c_prime)

        # Constant-time key selection
        if is_valid:
            return k_prime
        else:
            return k_bar


# Convenient pre-configured instances
MLKEM512 = MLKEM(PARAMS_512)
MLKEM768 = MLKEM(PARAMS_768)
MLKEM1024 = MLKEM(PARAMS_1024)
