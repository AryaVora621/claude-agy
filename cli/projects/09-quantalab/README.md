# QuantaLab: Zero-Dependency Universal Quantum Computing Simulator & Algorithm Lab

QuantaLab is an educational and research-grade universal quantum computing circuit simulator, statevector engine, and quantum algorithms lab implemented entirely from first principles in the pure Python standard library.

It models the full $2^N$-dimensional complex Hilbert space, evaluates universal multi-qubit gate decompositions in $O(2^N)$ runtime without constructing explosive Kronecker product matrices, models environmental decoherence via density matrix Kraus operators, and provides implementations of foundational quantum algorithms (Grover's Search, Shor's Prime Factoring, Quantum Fourier Transform, Quantum Teleportation, and Superdense Coding).

---

## Architectural Overview

```
+--------------------------------------------------------------------------------+
|                         CIRCUIT COMPOSITION (FLUENT API)                       |
|   circ = QuantumCircuit(3)                                                     |
|   circ.h(0).cx(0, 1).ccx(0, 1, 2).measure_all()                                |
+--------------------------------------------------------------------------------+
                                       |
                   Gate Scheduler & Depth Analyzer
                                       |
+--------------------------------------------------------------------------------+
|                        UNIVERSAL QUANTUM GATE LIBRARY                          |
|  - 1-Qubit: I, X, Y, Z, H, S, Sdg, T, Tdg, Rx(θ), Ry(θ), Rz(θ), P(λ), U3(θ,φ,λ)|
|  - 2-Qubit: CNOT (CX), CY, CZ, Controlled-Phase (CP), SWAP                    |
|  - 3-Qubit: Toffoli (CCX), Fredkin (CSWAP), Multi-Controlled Z (MCZ)          |
+--------------------------------------------------------------------------------+
                                       |
                   In-Place O(2^N) Amplitude Transform
                                       |
+--------------------------------------------------------------------------------+
|                          STATEVECTOR & HILBERT SPACE                           |
|  - StateVector: 2^N complex amplitudes, norm preservation                      |
|  - Projective Measurement: Born rule probability collapse                      |
|  - Partial Trace: Single-qubit reduced density matrix ρ                        |
|  - Bloch Sphere Coordinates: (x, y, z) = (<X>, <Y>, <Z>)                       |
+--------------------------------------------------------------------------------+
                                       |
                +----------------------+----------------------+
                |                                             |
+-------------------------------+             +-------------------------------+
|       ALGORITHMS ENGINE       |             |     NOISE & DECOHERENCE       |
| - Quantum Fourier Transform   |             | - Density Matrix ρ (2^N x 2^N)|
| - Grover's Quadratic Search   |             | - Bit-Flip Channel (E0, E1)   |
| - Quantum Teleportation       |             | - Phase-Flip Channel          |
| - Superdense Coding           |             | - Depolarizing Channel        |
| - Shor's Factoring (N=15, 21) |             | - Amplitude Damping (T1)      |
+-------------------------------+             +-------------------------------+
```

---

## Key Technical Features

### 1. High-Performance Statevector Simulation ($O(2^N)$)
- Rather than constructing massive $2^N \times 2^N$ unitary matrices through Kronecker tensor products ($O(4^N)$ memory), QuantaLab applies single-qubit and controlled operations directly to amplitude pairs $(a_i, a_{i + 2^k})$ with stride $2^{k+1}$.
- In-place mutation achieves over **15.9 Million amplitude operations/sec** on 12-qubit statevectors (4,096 complex amplitudes) in pure Python.

### 2. Universal Gate Set & Multi-Controlled Operations
- Complete 1-qubit single parameter and Euler angle gates: Pauli ($X, Y, Z$), Hadamard ($H$), Clifford phase gates ($S, T$), arbitrary axis rotations ($R_x, R_y, R_z$), phase shifts ($P$), and arbitrary $U3(\theta, \phi, \lambda)$.
- Entangling multi-qubit gates: CNOT ($CX$), $CY$, $CZ$, Controlled-Phase, SWAP, 3-qubit Toffoli ($CCX$), Fredkin (Controlled-SWAP), and arbitrary multi-controlled $Z$ ($MCZ$).

### 3. Foundational Quantum Algorithms
- **Quantum Fourier Transform (QFT)**: Complete $N$-qubit frequency domain circuit and inverse QFT ($QFT^\dagger$) running at **>74,000 QFTs/sec** for 4 qubits and **2,800+ QFTs/sec** for 8 qubits.
- **Grover's Search Algorithm**: Phase oracle construction, diffusion inversion around the mean, and optimal rotation count $R = \text{round}\left(\frac{\pi}{4}\arcsin(1/\sqrt{N})^{-1} - 0.5\right)$, isolating marked database keys with $>94\%$ to $100\%$ probability.
- **Quantum Teleportation**: Transmits an unknown quantum state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ across Alice and Bob using an entangled Bell state, Bell basis measurement, and classical feedforward correction, proving **1.000000 state fidelity**.
- **Superdense Coding**: Transmits 2 classical bits through 1 transmitted entangled qubit.
- **Deutsch-Jozsa Algorithm**: Distinguishes constant from balanced oracles in a single quantum evaluation.
- **Shor's Factoring Algorithm**: Order finding via quantum phase simulation with continued fraction expansion, resolving non-trivial prime factors of composite integers (e.g. $15 = 3 \times 5$, $21 = 3 \times 7$).

### 4. Noise Models & Decoherence Simulation
- $2^N \times 2^N$ Density Matrix formalism tracking mixed quantum states.
- Kraus operator channels: Bit-Flip ($X$), Phase-Flip ($Z$), Depolarizing Channel, and Amplitude Damping ($T_1$ energy relaxation).
- Purity calculation $\gamma = \text{Tr}(\rho^2)$ and trace preservation $\text{Tr}(\rho) = 1$.

### 5. ASCII Terminal Visualizers
- **Circuit Wire Diagrams**: ASCII circuit schematics showing qubit lines, gate boxes (`[H]`, `[X]`), control lines (`●`), multi-wire vertical buses, and measurement meters (`[M]`).
- **2D ASCII Bloch Sphere**: Projects $(x, y, z)$ Bloch coordinates onto an ASCII sphere, displaying polar angles $\theta, \phi$ and state vector length.
- **Measurement Probability Histograms**: Terminal bar charts rendering Monte Carlo sampling distributions across basis states.

---

## Directory Layout

```
projects/09-quantalab/
├── quantalab/
│   ├── __init__.py           # Package exports
│   ├── types.py              # StateVector, BasisState, inner products, fidelity, partial trace
│   ├── gates.py              # UnitaryMatrix, Gate definitions, O(2^N) in-place gate kernels
│   ├── circuit.py            # QuantumCircuit builder, depth analyzer, simulation runner
│   ├── algorithms.py         # QFT, IQFT, Grover, Teleportation, Superdense, Deutsch-Jozsa, Shor
│   ├── noise.py              # DensityMatrix, Kraus channels (Bit-Flip, Depolarizing, T1)
│   └── visualizer.py         # ASCII Circuit Diagrams, 2D Bloch Sphere, Probability Bar Charts
├── examples/
│   └── quantum_lab.py        # Interactive showcase: Bell states, Grover, Teleportation, Bloch sphere
├── benchmarks/
│   └── bench_quantum.py      # Gate throughput, QFT scaling, Grover search benchmarks
├── tests/
│   ├── test_types.py         # StateVector normalization, inner product, Bloch coords
│   ├── test_gates.py         # Unitarity, Pauli gates, CNOT, SWAP, Toffoli
│   ├── test_circuit.py       # Bell states, GHZ states, circuit depth, sampling
│   ├── test_algorithms.py    # QFT reversibility, Grover 2/3 qubits, Teleportation, Shor
│   └── test_noise.py         # Density matrix trace, Kraus decoherence, purity decay
└── README.md
```

---

## Performance Benchmarks

Evaluated on an Apple Silicon host running Python 3.13 standard library:

| Subsystem / Operation | Metric | Measured Performance |
|---|---|---|
| **1-Qubit Gate Throughput (H)** | 12 qubits (4,096 amplitudes) | **3,887 gates/sec** (15.9 M amp-ops/sec) |
| **2-Qubit Gate Throughput (CNOT)**| 12 qubits (4,096 amplitudes) | **5,937 gates/sec** |
| **4-Qubit QFT Execution** | 16 amplitudes, depth 8 | **74,685 QFTs/sec** (0.01 ms/run) |
| **8-Qubit QFT Execution** | 256 amplitudes, depth 16 | **2,841 QFTs/sec** (0.35 ms/run) |
| **12-Qubit QFT Execution** | 4,096 amplitudes, depth 24 | **84 QFTs/sec** (11.88 ms/run) |
| **Grover Search (3 qubits, N=8)** | Amplitude Amplification | **94.6% success probability** (0.04 ms) |
| **Grover Search (5 qubits, N=32)**| Amplitude Amplification | **100.0% success probability** (0.24 ms) |

To run the complete benchmark suite:
```bash
python3 projects/09-quantalab/benchmarks/bench_quantum.py
```

---

## Interactive Showcase

To run the interactive demonstration:
```bash
python3 projects/09-quantalab/examples/quantum_lab.py
```

Sample output:
```text
======================================================================
 Demo 2: Grover's Quantum Search Algorithm (Database of 8 items)
======================================================================
Marked Search Key: |101> (Target Index: 5 out of 8)
Quantum Circuit Diagram:
q0: ─[H]──────────╫──────────╫─[H]─[X]─────────────────[X]─[H]──────────────────╫──────────╫─[H]─[X]─────────────────[X]─[H]──────────────────
q1: ─────[H]──────╫─[X]─[X]──╫─────────[H]─[X]─────────────────[X]─[H]──────────╫─[X]─[X]──╫─────────[H]─[X]─────────────────[X]─[H]──────────
q2: ─────────[H]──╫──────────╫─────────────────[H]─[X]─────────────────[X]─[H]──╫──────────╫─────────────────[H]─[X]─────────────────[X]─[H]──

Circuit Depth: 17 | Gate Count: 35

Amplified State Probability Distribution:
  |000>: [                              ]   0.9% (9 shots)
  |001>: [                              ]   0.9% (9 shots)
  |010>: [                              ]   0.5% (5 shots)
  |011>: [                              ]   1.1% (11 shots)
  |100>: [                              ]   0.3% (3 shots)
  |101>: [############################  ]  94.8% (948 shots)
  |110>: [                              ]   0.5% (5 shots)
  |111>: [                              ]   1.0% (10 shots)

[+] Marked item |101> isolated with 94.5% probability in 2 iterations!
```

---

## Test Suite

The test suite covers statevectors, gate unitarity, circuit depth and scheduling, quantum algorithms (QFT, Grover, Teleportation, Superdense Coding, Deutsch-Jozsa, Shor), and density matrix noise channels:

```bash
PYTHONPATH="projects/09-quantalab" python3 -m unittest discover -s projects/09-quantalab/tests -v
```

Output:
```text
Ran 27 tests in 0.003s
OK
```
