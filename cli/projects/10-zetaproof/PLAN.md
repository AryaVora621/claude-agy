# Architecture Plan: Project 10 - ZetaProof
## Zero-Knowledge Proof Engine & zk-SNARK Architecture from First Principles

ZetaProof is an educational and production-grade zero-knowledge proof engine, Rank-1 Constraint System (R1CS) compiler, Quadratic Arithmetic Program (QAP) reduction pipeline, and non-interactive zk-SNARK prover and verifier implemented entirely in pure Python standard library.

---

## 1. System Architecture

```
+-------------------------------------------------------------------------------+
|                       HIGH-LEVEL ARITHMETIC CIRCUIT (DSL)                     |
|   circuit = Circuit()                                                         |
|   x = circuit.private_input("x")                                              |
|   y = circuit.public_input("y")                                               |
|   circuit.assert_equal(x * x * x + x + 5, y)                                  |
+-------------------------------------------------------------------------------+
                                       |
                   Flattening & Gate Decomposition
                                       |
+-------------------------------------------------------------------------------+
|                      RANK-1 CONSTRAINT SYSTEM (R1CS)                          |
|   Constraints of form:  (A · s) * (B · s) = (C · s)                           |
|   Witness vector: s = [1, public_inputs..., private_witness...]               |
+-------------------------------------------------------------------------------+
                                       |
                   Lagrange Polynomial Interpolation
                                       |
+-------------------------------------------------------------------------------+
|                    QUADRATIC ARITHMETIC PROGRAM (QAP)                         |
|   Polynomials: A(X), B(X), C(X) such that A(r_k) = A_k, etc.                  |
|   Vanishing polynomial: Z(X) = Π (X - r_k)                                    |
|   Divisibility condition: A(X)*B(X) - C(X) = H(X) * Z(X)                      |
+-------------------------------------------------------------------------------+
                                       |
                    Trusted Setup / CRS & Commitments
                                       |
+-------------------------------------------------------------------------------+
|                   GROTH16 / SUCCINCT ZK-SNARK PROTOCOL                        |
|   - Setup(Circuit) -> (ProvingKey, VerifyingKey)                              |
|   - Prove(PK, public_inputs, private_witness) -> Proof(π_A, π_B, π_C)         |
|   - Verify(VK, public_inputs, Proof) -> True / False (O(1) Verification)      |
+-------------------------------------------------------------------------------+
                                       |
             +-------------------------+-------------------------+
             |                                                   |
+----------------------------+              +-----------------------------------+
|      ZK APPLICATIONS       |              |        TERMINAL VISUALIZERS       |
| - Private Hash Preimage    |              | - ASCII Circuit DAG               |
| - Zero-Knowledge Range     |              | - R1CS Matrix Sparsity Heatmaps   |
| - Secret Password Proof    |              | - Interactive Proof Verification  |
+----------------------------+              +-----------------------------------+
```

---

## 2. Directory Structure

```
projects/10-zetaproof/
├── zetaproof/
│   ├── __init__.py           # Package exports
│   ├── field.py              # Prime finite field F_p arithmetic, modular inverse, Tonelli-Shanks
│   ├── polynomial.py         # Polynomial ring F_p[X], multiplication, division, Lagrange interpolation
│   ├── circuit.py            # High-level arithmetic circuit DSL, variable tracking, DAG compiler
│   ├── r1cs.py               # Rank-1 Constraint System representation, witness solving, matrix validator
│   ├── qap.py                # Quadratic Arithmetic Program reduction, vanishing polynomial, quotient H(X)
│   ├── curve.py              # Elliptic curve cryptography, point addition/doubling, pairings / commitments
│   ├── snark.py              # Groth16 zk-SNARK setup, prover with blinding factors, verifier
│   ├── circuits_zoo.py       # Pre-built circuits: MiMC hash preimage, range proof, secret auth
│   └── visualizer.py         # ASCII circuit diagrams, R1CS matrix heatmaps, verification badges
├── examples/
│   └── zk_demo.py            # Interactive zero-knowledge lab and showcase
├── benchmarks/
│   └── bench_zk.py           # Field ops, polynomial division, R1CS generation, proof times
├── tests/
│   ├── test_field.py         # F_p modular arithmetic, inverse, Fermat, Tonelli-Shanks
│   ├── test_polynomial.py    # Polynomial add/mul/div, roots, Lagrange interpolation
│   ├── test_circuit.py       # Circuit building, flattening, witness generation
│   ├── test_r1cs.py          # R1CS matrix satisfaction, invalid witness detection
│   ├── test_qap.py           # QAP reduction, exact divisibility by Z(X), quotient degree
│   ├── test_curve.py         # Elliptic curve point operations, scalar multiplication
│   └── test_snark.py         # End-to-end zk-SNARK setup, proof generation, sound verification
└── README.md
```

---

## 3. Implementation Steps

1. **Step 1: Finite Field & Polynomial Calculus (`field.py`, `polynomial.py`)**
   - Implement `FieldElement` over prime modulus $p$ with full operator overloading (`+`, `-`, `*`, `/`, `**`).
   - Extended Euclidean algorithm for modular inverse.
   - Implement `Polynomial` with addition, multiplication (convolution), polynomial long division with remainder, Lagrange interpolation across roots, and vanishing polynomial construction.
2. **Step 2: Arithmetic Circuit DSL & R1CS Compiler (`circuit.py`, `r1cs.py`)**
   - Implement `Circuit`, `Variable`, `Expression` (linear combinations).
   - Multiplication gates emitting R1CS constraint triples $(a, b, c)$ where $(a \cdot s) \times (b \cdot s) = (c \cdot s)$.
   - Witness solver calculating intermediate wire values.
   - R1CS sparse matrix representation and validator.
3. **Step 3: QAP Reduction & Elliptic Curve Cryptography (`qap.py`, `curve.py`)**
   - Interpolate polynomials $A_i(X), B_i(X), C_i(X)$ for each wire $i$.
   - Compute aggregate polynomials $A(X), B(X), C(X)$.
   - Verify exact polynomial divisibility $H(X) = (A(X) \cdot B(X) - C(X)) / Z(X)$.
   - Implement elliptic curve point arithmetic for cryptographic commitments / evaluation trapdoors.
4. **Step 4: Groth16 / Succinct ZK-SNARK Prover & Verifier (`snark.py`, `circuits_zoo.py`)**
   - Trusted setup generating evaluation keys at secret toxic waste $(\alpha, \beta, \gamma, \delta, x)$.
   - Prover generating proof $\pi = (A, B, C)$ with random blinding factors $r, s$ for zero knowledge.
   - Verifier checking pairing / commitment identity with public inputs in $O(1)$ time.
   - Pre-built circuits: MiMC hash preimage, range proof (bit decomposition), secret equation.
5. **Step 5: Visualizers, Benchmarks, Interactive Demo & Tests**
   - ASCII visualizer for circuit DAG and R1CS matrix heatmaps.
   - Unit tests covering all subsystems.
   - Benchmark suite measuring ops/second and proof times.
   - Interactive CLI lab in `examples/zk_demo.py`.
6. **Step 6: Master Integration & Documentation**
   - Integrate into `showcase.py` as Option 10.
   - Update `projects.md`, `TASK_QUEUE.md`, `CHECKPOINT_LAST.md`, and `tracker/data.json`.
