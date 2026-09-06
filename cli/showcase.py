#!/usr/bin/env python3
"""
AGY Engineering Showcase Lab - Master Launcher & Interactive Runner.
Unifies all 23 computer science & software systems built from first principles:
1. NanoTensor   - Zero-Dependency Autograd & Modern LLM Architecture (RoPE, RMSNorm, SwiGLU, KV-Cache)
2. ChronoDB     - Embeddable LSM-Tree Storage Engine with Bloom Filters & HNSW Vector Index
3. AetherVM     - Register-Based SSA Optimizing Compiler & Bytecode VM (>4.4 MIPS)
4. SwarmRaft    - Fault-Tolerant Distributed Consensus & Network Partition Simulator (>10k TPS)
5. NexusOS      - Capability-Based Microkernel & Virtual Memory Operating System Simulator (>5.6M context switches/s)
6. PhotonPBR    - Physically-Based Monte Carlo Path Tracer with SAH BVH & TrueColor Terminal Output (>1M rays/s)
7. HydraNet     - User-Space TCP/IP Protocol Stack with RFC 793 FSM, Reno Congestion Control, & POSIX Sockets
8. SynapseDB    - Zero-Dependency Vectorized Columnar Analytics Engine & SQL Processor (>44M rows/s)
9. QuantaLab    - Universal Quantum Computing Simulator, Grover Search, Shor Factoring & Bloch Sphere
10. ZetaProof   - Zero-Knowledge SNARK Engine, Finite Fields, R1CS Compiler, QAP Reduction, Groth16 Prover & Verifier
11. WasmCore    - WebAssembly MVP Virtual Machine, 64KB Paged Memory, LEB128 Codec, Bytecode Emitter & Stack Debugger
12. NovaPhysics - 2D Rigid Body Physics Engine, Symplectic Integrator, GJK/EPA Collision & PBD Cloth
13. HelixGit    - Git Content-Addressable Storage Engine, Merkle Trees, DIRC v2, Packfiles & Myers Diff
14. GeoPrism    - Spatial Indexing & Geospatial Engine, R*-Tree, H3 Discrete Global Grid System & Voronoi
15. AuraDSP     - Digital Signal Processing & Spectral Audio Synthesis, FFT, Biquad IIR & Vocoder
16. VeloSLAM    - Autonomous Robotics, Extended Kalman Filter SLAM, Hybrid A* & Dynamic Window Avoidance
17. ApexMatch   - Ultra-Low Latency Limit Order Book, Price-Time Priority FIFO Matching & ITCH/OUCH
18. NucleoCore  - Computational Genomics & Sequence Assembly, FM-Index, Gotoh Affine Gaps & De Bruijn Graphs
19. OrbitMech   - Orbital Mechanics, Danby Kepler Solver, Lambert Targeting, J2/SRP & Störmer-Verlet
20. AeroFlow    - Computational Fluid Dynamics, Lattice Boltzmann D2Q9 BGK & Chorin Projection Navier-Stokes
21. Structura   - Finite Element Analysis, 2D Continuum Mechanics, QuadQ4 Gauss Quadrature & Modal Dynamics
22. Atomix      - Molecular Dynamics, Statistical Mechanics, 3D Linked Cells, Velocity Verlet, SHAKE & NVT/NPT
23. Solida      - 3D B-Rep Solid Modeling CAD Kernel, NURBS Surfaces, CSG BSP Booleans & STL/OBJ Codecs
24. OptiRoute   - Vehicle Routing Problem (VRP/CVRP/VRPTW), Clarke-Wright, 2-Opt/3-Opt, Genetic & Tabu Search
25. BioSim      - Ecological Evolution Simulator, Neural Automata, Predator-Prey Dynamics, Speciation & Braille Canvas
26. SiliconRISC - Cycle-Accurate 5-Stage Pipelined RV32I Processor, Hazard Detection, Data Forwarding, Branch Predictor & ELF Loader
27. WaveForge   - 2D/3D Time-Domain Wave Simulation, Acoustic & Maxwell FDTD, PML Boundary & Doppler Effect
28. LatticeCrypt- Post-Quantum Cryptography, NIST FIPS 203 ML-KEM (Kyber), Polynomial Rings, Number Theoretic Transform (NTT)
29. LogicCraft  - Electronic Design Automation, ROBDD, And-Inverter Graphs (AIG), Technology Mapping & Static Timing Analysis
30. Relativitas  - General Relativity, Curved Spacetime Geodesics, Kerr Black Hole Accretion Disk, Ray Tracing & Redshift
31. StellarFusion - Magnetohydrodynamics (MHD), Grad-Shafranov Solver, Solovev Equilibrium, Boris Pusher & Banana Orbits
32. ThermoProp  - Compressible Gas Dynamics, 2D Method of Characteristics Supersonic Nozzle & Rocket Propulsion Engine
33. ChromaSplat - 3D Gaussian Splatting, Radiance Fields, EWA Perspective Projection & Tiled Volume Rasterizer

Usage:
  python3 showcase.py               # Interactive menu
  python3 showcase.py --all-tests    # Execute complete test suite across all 33 projects
  python3 showcase.py --all-bench   # Execute benchmarks across all 33 projects
  python3 showcase.py --demo 1|...|33 # Run a specific project demo
"""

import sys
import os
import time
import subprocess
import argparse

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

PROJECTS = [
    {
        "num": 1,
        "name": "NanoTensor",
        "domain": "Deep Learning / Autograd Systems",
        "dir": "projects/01-nanotensor",
        "demo": "examples/train_toy.py",
        "tests": "tests",
        "desc": "Reverse-mode autograd engine, RoPE, RMSNorm, SwiGLU, KV-Cache, BPE Tokenizer, ASCII Attention Heatmaps",
        "stats": "17 tests passing, 0 external dependencies"
    },
    {
        "num": 2,
        "name": "ChronoDB",
        "domain": "Database Systems / Vector Search",
        "dir": "projects/02-chronodb",
        "demo": "examples/tui_dashboard.py",
        "tests": "tests",
        "desc": "LSM-Tree with WAL, SkipList MemTable, Bloom Filter SSTables, Leveled Compaction, HNSW Vector Index",
        "stats": "11 tests passing, ~40,000 write ops/sec, >136k Bloom ops/sec"
    },
    {
        "num": 3,
        "name": "AetherVM",
        "domain": "Compilers & Runtimes",
        "dir": "projects/03-aethervm",
        "demo": "examples/demo_pipeline.py",
        "tests": "tests",
        "desc": "Pratt Parser, Braun et al. SSA IR, Constant Folding, CSE, DCE, Linear Scan Allocator, 16-Reg VM",
        "stats": "17 tests passing, 4.45 Million Instructions/sec (MIPS)"
    },
    {
        "num": 4,
        "name": "SwarmRaft",
        "domain": "Distributed Systems / Consensus",
        "dir": "projects/04-swarm-raft",
        "demo": "examples/cluster_tui.py",
        "tests": "tests",
        "desc": "Raft Consensus Protocol, Replicated State Machine, Snapshotting, Split-Brain Partition Simulator",
        "stats": "9 tests passing, 10,573 commits/sec, 0.09 ms median latency"
    },
    {
        "num": 5,
        "name": "NexusOS",
        "domain": "Operating Systems / Microkernels",
        "dir": "projects/05-nexus-os",
        "demo": "examples/kernel_top.py",
        "tests": "tests",
        "desc": "Hierarchical Paging, TLB, COW fork(), MLFQ Starvation-Free Scheduler, L4 IPC Rendezvous, Unix Pipes",
        "stats": "24 tests passing, >5.6M context switches/sec, 453k IPC ops/sec"
    },
    {
        "num": 6,
        "name": "PhotonPBR",
        "domain": "Computer Graphics / Physically-Based Rendering",
        "dir": "projects/06-photon-pbr",
        "demo": "examples/render_cornell_box.py",
        "tests": "tests",
        "desc": "Monte Carlo Path Tracer, SAH BVH Acceleration, Möller-Trumbore Triangles, Snell Refraction, TrueColor Terminal",
        "stats": "31 tests passing, >1M sphere rays/s, >791k tri rays/s, 3.7x BVH speedup"
    },
    {
        "num": 7,
        "name": "HydraNet",
        "domain": "Computer Networks / TCP/IP Protocol Stack",
        "dir": "projects/07-hydranet",
        "demo": "examples/packet_sniffer.py",
        "tests": "tests",
        "desc": "RFC 793 11-State TCP FSM, Sliding Window, RFC 5681 Reno Congestion Control, ARP, IPv4 LPM Routing, POSIX Sockets",
        "stats": "21 tests passing, 1.25M frames/sec, >82k LPM route lookups/sec"
    },
    {
        "num": 8,
        "name": "SynapseDB",
        "domain": "Database Systems / Vectorized Analytics",
        "dir": "projects/08-synapsedb",
        "demo": "examples/analytics_dashboard.py",
        "tests": "tests",
        "desc": "Vectorized Columnar Storage, RLE/Dict/Bit-Packing Codecs, Zone Maps, SQL Parser, Volcano Hash Joins & Aggregates",
        "stats": "19 tests passing, 44.6M rows/s scan, 261M vals/s RLE decode"
    },
    {
        "num": 9,
        "name": "QuantaLab",
        "domain": "Quantum Computing & Information",
        "dir": "projects/09-quantalab",
        "demo": "examples/quantum_lab.py",
        "tests": "tests",
        "desc": "Universal Quantum Circuit Simulator, O(2^N) Amplitude Transforms, Grover Search, Shor Factoring, Density Matrix, Bloch Sphere",
        "stats": "27 tests passing, 15.9M amplitude ops/sec, >74k QFTs/sec"
    },
    {
        "num": 10,
        "name": "ZetaProof",
        "domain": "Applied Cryptography / Zero-Knowledge",
        "dir": "projects/10-zetaproof",
        "demo": "examples/zk_demo.py",
        "tests": "tests",
        "desc": "Prime Finite Field F_p, Polynomial Ring, Circuit-to-R1CS Compiler, QAP Reduction, Groth16 zk-SNARK Prover & Verifier",
        "stats": "26 tests passing, 0.13 ms prove time, >240k verifs/sec"
    },
    {
        "num": 11,
        "name": "WasmCore",
        "domain": "Virtual Machines / Binary Toolchains",
        "dir": "projects/11-wasmcore",
        "demo": "examples/wasm_lab.py",
        "tests": "tests",
        "desc": "WebAssembly MVP Stack Machine Interpreter, 64KB Paged Memory, LEB128 Codec, Binary Parser/Emitter, Terminal Debugger",
        "stats": "34 tests passing, >973k LEB128 ops/s, >85k module decodes/s, 0.84 MIPS"
    },
    {
        "num": 12,
        "name": "NovaPhysics",
        "domain": "Physics Engines / Computational Mechanics",
        "dir": "projects/12-novaphysics",
        "demo": "examples/physics_lab.py",
        "tests": "tests",
        "desc": "Symplectic Euler, Dynamic AABB BVH, GJK/EPA Collision Solver, Warm Starting, Distance/Revolute/Spring Joints, PBD Cloth, Braille Visualizer",
        "stats": "25 tests passing, 10.5M vec ops/s, 40.6k GJK/EPA checks/s, 165 world steps/s"
    },
    {
        "num": 13,
        "name": "HelixGit",
        "domain": "Version Control Systems / Merkle Storage Engines",
        "dir": "projects/13-helixgit",
        "demo": "examples/git_lab.py",
        "tests": "tests",
        "desc": "Cryptographic Merkle DAG, DIRC v2 Binary Index, Packfile & Delta Compression, Myers O((N+M)D) Diff, 3-Way Merge, ANSI Terminal DAG Graph",
        "stats": "27 tests passing, 1.27M index ops/s, 690k diff lines/s, 580k .idx lookups/s"
    },
    {
        "num": 14,
        "name": "GeoPrism",
        "domain": "Spatial Indexing / Computational Geometry / DGGS",
        "dir": "projects/14-geoprism",
        "demo": "examples/spatial_lab.py",
        "tests": "tests",
        "desc": "Beckmann R*-Tree, Forced Reinsertion, MinDist k-NN, 64-bit H3 DGGS Hexagonal Grid, Convex Hull, Bowyer-Watson Delaunay, Voronoi Dual, Braille Canvas",
        "stats": "21 tests passing, 85k range queries/s, 35k k-NN/s (68x speedup), 648k H3 encodes/s, 1.38M hull pts/s"
    },
    {
        "num": 15,
        "name": "AuraDSP",
        "domain": "Audio DSP / Spectral Analysis / Sound Synthesis",
        "dir": "projects/15-auradsp",
        "demo": "examples/audio_lab.py",
        "tests": "tests",
        "desc": "Cooley-Tukey Radix-2 FFT/IFFT, RBJ Biquad IIR Filters, Polyphonic Oscillators, ADSR, 16-bit WAV Codec, Braille Waterfall Spectrogram",
        "stats": "28 tests passing, 106x FFT speedup (1.0ms), 9.9M biquad samples/s, 3.2M synth samples/s, 59 MB/s WAV decode"
    },
    {
        "num": 16,
        "name": "VeloSLAM",
        "domain": "Autonomous Robotics / SLAM / Kinodynamic Motion Planning",
        "dir": "projects/16-veloslam",
        "demo": "examples/robotics_lab.py",
        "tests": "tests",
        "desc": "Extended Kalman Filter SLAM, Bresenham Occupancy Grid, Dubins Shortest Paths, Hybrid A* 3D Motion Planner, DWA Local Reactive Avoidance, Braille Map Visualizer",
        "stats": "27 tests passing, 24k EKF predicts/s, 76k raycasts/s, 329k Dubins evals/s, 3.3k Hybrid A* plans/s, 67 Hz DWA control"
    },
    {
        "num": 17,
        "name": "ApexMatch",
        "domain": "Financial Exchanges / High-Frequency Matching Engines",
        "dir": "projects/17-apexmatch",
        "demo": "examples/exchange_sim.py",
        "tests": "tests",
        "desc": "Ultra-Low Latency L3 Limit Order Book, Continuous Price-Time FIFO Matching, IOC/FOK/Iceberg Orders, STP, Pre-Trade Risk Gate, Binary ITCH/OUCH Codec, Depth Ladder",
        "stats": "18 tests passing, 5.6M inserts/s, 5.1M cancels/s, 556k trades/s, 1.04 µs median latency, 4.3M ITCH msgs/s"
    },
    {
        "num": 18,
        "name": "NucleoCore",
        "domain": "Computational Genomics / Sequence Assembly Engines",
        "dir": "projects/18-nucleocore",
        "demo": "examples/genomics_lab.py",
        "tests": "tests",
        "desc": "BWT & FM-Index Backward Search, Gotoh 3-Matrix Affine Gap Pairwise Alignment, De Bruijn Graph De Novo Assembly with Tip-Clipping & Bubble-Popping, Profile HMM Viterbi Decoding, Braille Dot-Plot",
        "stats": "29 tests passing, 42k FM-Index searches/s, 2.3M Gotoh DP cells/s, 3.5M kmers/s DBG, 1.7M bases/s Viterbi, 276 dot-plots/s"
    },
    {
        "num": 19,
        "name": "OrbitMech",
        "domain": "Orbital Mechanics / Astrodynamics / Trajectory Optimization",
        "dir": "projects/19-orbitmech",
        "demo": "examples/mission_control.py",
        "tests": "tests",
        "desc": "Danby 3rd-Order Kepler Solver, Universal Variable Conics, Lambert Targeting, J2/Drag/SRP Propagators, Symplectic Störmer-Verlet, Hohmann & Bi-elliptic Transfers, Hyperbolic Gravity Assists, Porkchop Plots, 3D Braille Orbit Visualizer",
        "stats": "26 tests passing, 1.37M Kepler roots/s, 41k Lambert solves/s, 223k Verlet steps/s, 3.7k Braille FPS"
    },
    {
        "num": 20,
        "name": "AeroFlow",
        "domain": "Computational Fluid Dynamics / Lattice Boltzmann / Aerodynamics",
        "dir": "projects/20-aeroflow",
        "demo": "examples/wind_tunnel.py",
        "tests": "tests",
        "desc": "D2Q9 LBM BGK Solver, Chorin Navier-Stokes Projection, Half-Way Bounce-Back, Momentum Exchange Drag/Lift, NACA 4-Digit Airfoils, Von Karman Vortex Shedding, Strouhal Number, 2x4 Sub-Pixel Braille Visualizer",
        "stats": "24 tests passing, 0.370 MLUPS, 61 NS steps/s, 905k advections/s, 164 Braille FPS"
    },
    {
        "num": 21,
        "name": "Structura",
        "domain": "Finite Element Analysis / Continuum Mechanics / Structural Dynamics",
        "dir": "projects/21-structura",
        "demo": "examples/structural_lab.py",
        "tests": "tests",
        "desc": "Truss/Beam/CST/QuadQ4 Elements, 2x2 Gauss Quadrature, Plane Stress/Strain, DOK/CSR Sparse Matrix, Jacobi PCG Solver, Exact Dirichlet Partitioning, Cauchy/Von Mises Stresses, Modal Dynamics Resonance, Sub-Pixel Braille Deformed Visualizer & TrueColor Heatmaps",
        "stats": "25 tests passing, 16.9k QuadQ4/s, 134k CST/s, 2.34M Truss/s, 10.6k DOFs/s PCG, 428 Braille FPS"
    },
    {
        "num": 22,
        "name": "Atomix",
        "domain": "Molecular Dynamics / Statistical Mechanics / Computational Biophysics",
        "dir": "projects/22-atomix",
        "demo": "examples/biophysics_lab.py",
        "tests": "tests",
        "desc": "Lennard-Jones 12-6, Shifted Potential, Coulomb Electrostatics, Harmonic Bond & Angle, Periodic Dihedrals, 3D Linked Cell Lists, Velocity Verlet, SHAKE Holonomic Constraints, Berendsen & Nosé-Hoover Thermostats, Virial Pressure, g(r) RDF, MSD Diffusion, 3D Sub-Pixel Braille Visualizer & CPK Colors",
        "stats": "31 tests passing, 5.7M vector ops/s, 1.67M LJ pairs/s, 83.1 MD steps/s (108 atoms), 2,374 Braille FPS"
    },
    {
        "num": 23,
        "name": "Solida",
        "domain": "3D Solid Modeling CAD / NURBS / Geometric Kernel",
        "dir": "projects/23-solida",
        "demo": "examples/cad_workbench.py",
        "tests": "tests",
        "desc": "2-Manifold Half-Edge B-Rep Topology, Euler Invariants, Divergence Theorem Volume, Cox-de Boor NURBS Curves/Surfaces, CSG BSP-Tree Booleans, Linear Extrude with Draft/Twist, Revolve, Loft, NACA Airfoils, ASCII/Binary STL & OBJ Codecs, 3D Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "30 tests passing, 1.37M vector ops/s, 338k transforms/s, 37.2k NURBS evals/s, 1.5k B-Rep volumes/s, 308 CSG booleans/s, 1,619 Braille FPS"
    },
    {
        "num": 24,
        "name": "NeuroSynapse",
        "domain": "Neuromorphic Computing / Spiking Neural Networks / Event-Based Vision",
        "dir": "projects/24-neurosynapse",
        "demo": "examples/neuromorphic_workbench.py",
        "tests": "tests",
        "desc": "LIF with Adaptive Thresholds, Izhikevich 2D Dynamical Presets, Hodgkin-Huxley RK4, Pair/Triplet STDP Plasticity, Surrogate Gradient BPTT (Fast Sigmoid/ArcTan/Triangular), AER DVS Stream Filtering, Time Surface Normal Optical Flow, Liquid State Machine 3D Reservoir, Sub-Pixel Braille Visualizers",
        "stats": "32 tests passing, 6.7M LIF steps/s, 4.6M Izhikevich steps/s, 288k HH RK4/s, 10.0M STDP syn-steps/s, 35.5k Spike-BPTT steps/s, 883k DVS events/s, 5.6k Braille FPS"
    },
    {
        "num": 25,
        "name": "Avionix",
        "domain": "6-DOF Aerial Robotics / Differential Flatness / SE(3) Control",
        "dir": "projects/25-avionix",
        "demo": "examples/flight_sim.py",
        "tests": "tests",
        "desc": "6-DOF Rigid-Body Dynamics on SE(3), Singularity-Free Quaternion Kinematics & RK4, Differential Flatness, Minimum-Snap Quintic Splines, Lee-Leok-McClamroch (2010) Geometric Tracking Control on SO(3), Motor Mixer with Anti-Saturation, 15-State Error-State EKF (IMU/Baro/Mag/GPS), Disturbance Observer, EFIS Primary Flight Display & Sub-Pixel Braille 3D Orbit Visualizer",
        "stats": "35 tests passing, 27.8k RK4 steps/s, 484k quat-ops/s, 4.2k splines/s, 175k flat-evals/s, 33.7k SE(3) cycles/s, 550k mixes/s, 2.5k EKF/s, 7.2k Braille FPS, 13.8k PFD FPS"
    },
    {
        "num": 26,
        "name": "SiliconRISC",
        "domain": "Computer Architecture / Cycle-Accurate RISC-V RV64GC Processor",
        "dir": "projects/26-siliconrisc",
        "demo": "examples/system_workbench.py",
        "tests": "tests",
        "desc": "RV64GC (RV64IMAFD) ISA, 5-Stage In-Order Pipeline, Forwarding Unit, Load-Use Hazard Stalls, Branch Flushes, SV39 MMU 3-Level Page Table Walking, Fully-Associative TLB, 4-State MESI Cache Hierarchy (L1I, L1D, L2), Tournament/Gshare Branch Predictor, ELF64 Parser & Assembler, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "40 tests passing, 797k insts/s decoder, 332k insts/s interpreter, 142k cycles/s pipeline, 1.44M cache accesses/s, 1.56M TLB translations/s, 99.85% Tournament branch accuracy"
    },
    {
        "num": 27,
        "name": "LuminaWave",
        "domain": "Computational Electromagnetics / 2D FDTD / Silicon Nanophotonics",
        "dir": "projects/27-luminawave",
        "demo": "examples/photonics_workbench.py",
        "tests": "tests",
        "desc": "2D TMz Maxwell FDTD Solver, Yee Staggered Lattice, Berenger Split-Field PML Absorbing Boundaries, Continuous-Wave & Gaussian Optical Sources, Silicon-on-Insulator Waveguides/Bends/Couplers/Ring Resonators, On-the-Fly DFT Poynting Flux Monitors, S-Parameters & Cavity Q-Factors, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "30 tests passing, 5.27 MC/s leapfrog updates, 950k DFT phasor updates/s, 2.65 MC/s PIC simulation, 547 Braille FPS"
    },
    {
        "num": 28,
        "name": "LatticeGuard",
        "domain": "Post-Quantum Cryptography / NIST FIPS 203 ML-KEM / Lattice Systems",
        "dir": "projects/28-latticeguard",
        "demo": "examples/pqc_workbench.py",
        "tests": "tests",
        "desc": "NIST FIPS 203 ML-KEM (Kyber), Ring R_q = Z_q[X]/(X^256+1), q=3329, 7-Stage Cooley-Tukey NTT & Gentleman-Sande INTT, Montgomery & Barrett Reductions, CBD Noise Sampling, SHAKE-128 Rejection Matrix Generation, ML-KEM-512/768/1024, K-PKE & Fujisaki-Okamoto Transform, Constant-Time Implicit Rejection, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "33 tests passing, 11.4M Montgomery reds/s, 9.25k NTTs/s (15.6x over O(n^2)), 400 ML-KEM-768 keygen/s, 187 decaps/s, 1,144 Braille FPS"
    },
    {
        "num": 29,
        "name": "LogicCraft",
        "domain": "Electronic Design Automation / Logic Synthesis & Static Timing Analysis",
        "dir": "projects/29-logiccraft",
        "demo": "examples/eda_workbench.py",
        "tests": "tests",
        "desc": "Reduced Ordered Binary Decision Diagrams (ROBDD), Shannon Decomposition, Unique Table & ITE Operator, And-Inverter Graphs (AIG) with Structural Hashing (Strashing), Liberty NLDM 2D Bilinear Interpolation, DAGON Dynamic Programming Technology Mapping, Gate-Level Mapped Netlists & Verilog Export, Static Timing Analysis (AT, RAT, Slack, WNS, TNS), Critical Path Extraction, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "25 tests passing, 136k ROBDD ops/s, 76.8k AIG nodes/s, 35.2k NLDM lookups/s, 6.46k TechMap gates/s, 4.74k STA passes/s, 1,698 Braille FPS"
    },
    {
        "num": 30,
        "name": "Relativitas",
        "domain": "General Relativity / Curved Spacetime Geodesics & Black Hole Accretion",
        "dir": "projects/30-relativitas",
        "demo": "examples/blackhole_workbench.py",
        "tests": "tests",
        "desc": "Pseudo-Riemannian 4D Manifolds, Schwarzschild & Kerr Metrics in Boyer-Lindquist Coordinates, Christoffel Symbols, 8-State First-Order Geodesic RK4 Integrator, Exact Machine-Precision Killing Invariants (E, L), Keplerian Accretion Disk Kinematics, Relativistic Doppler Beaming (I ~ g^4) & Gravitational Redshift, Backward Curved-Spacetime Ray Tracer, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "29 tests passing, 472k Christoffel ops/s, 43.1k RK4 steps/s, 314k accretion evals/s, 432.9 rays/s, 944.5 Braille FPS"
    },
    {
        "num": 31,
        "name": "StellarFusion",
        "domain": "Magnetohydrodynamics (MHD) / Plasma Equilibrium & Tokamak Confinement",
        "dir": "projects/31-stellarfusion",
        "demo": "examples/tokamak_workbench.py",
        "tests": "tests",
        "desc": "2D Non-Linear Grad-Shafranov Elliptic PDE Solver, Shafranov Operator Delta*, Solovev Exact Analytical Benchmarks, Div(B)=0 Invariance, Safety Factor q(psi) Contour Integrals & Magnetic Shear s(r), Symplectic Volume-Preserving Boris Particle Pusher, Neoclassical Trapped Banana Orbits & Passing Ions, 3D Field Line RK4 Tracer, Poincaré Surface-of-Section Maps & Resonant Magnetic Perturbations (RMP), Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "30 tests passing, 1.30M Solovev evals/s, 2.03k SOR sweeps/s (31x31), 1.41M B-field evals/s, 613k Boris steps/s, 335k Poincaré RK4 steps/s, 796.8 Braille FPS"
    },
    {
        "num": 32,
        "name": "ThermoProp",
        "domain": "Aerothermodynamics / Compressible Gas Dynamics & Rocket Propulsion",
        "dir": "projects/32-thermoprop",
        "demo": "examples/rocket_workbench.py",
        "tests": "tests",
        "desc": "1D/2D Compressible Gas Dynamics, Halley Area-Mach Inversion, Rankine-Hugoniot Normal & Oblique Shocks (Weak/Strong Branches, Detachment), Prandtl-Meyer Expansion Fans, 2D Method of Characteristics (MOC) Minimum-Length Supersonic Bell Nozzle Synthesis, Rocket Propulsion Thermochemistry (c*, CF, Isp, Tsiolkovsky Delta-V), Propellant Library (Methalox, Hydrolox, Kerolox, Hypergolic), Bartz Convective Heat Transfer & Coupled 1D Regenerative Cooling Channels, Supersonic Exhaust Plume Adaptation, Summerfield Flow Separation, Prandtl Periodic Mach Diamond Shock Cells, Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "30 tests passing, 1.08M isentropic evals/s, 434k Halley inversions/s, 49.3k shock pairs/s, 9.20k MOC nozzles/s, 181k engine evals/s, 213k Bartz cooling stations/s, 1,306 Braille FPS"
    },
    {
        "num": 33,
        "name": "ChromaSplat",
        "domain": "Computer Graphics / 3D Gaussian Splatting & Volume Radiance Fields",
        "dir": "projects/33-chromasplat",
        "demo": "examples/splat_workbench.py",
        "tests": "tests",
        "desc": "3D Gaussian Parameterization, Quaternion SO(3) Algebra, Positive Semi-Definite 3D Covariance, Real Spherical Harmonics Degrees 0-3 Directional Radiance & Specular Lobes, Pinhole Camera with EWA Projective Jacobian, 2D Screen Covariance with Low-Pass Filter, Tiled Front-to-Back Volume Rasterization, Early Ray Saturation, Stanford PLY ASCII Codec, Procedural Scenes (Rings, Cornell Box, DNA Helix), Sub-Pixel Braille Visualizer & Telemetry HUD",
        "stats": "30 tests passing, 661k 3D cov/s, 822k SH deg-2/s, 145k 2D EWA/s, 73.0 raster FPS, 426 Braille FPS, 58.5 end-to-end FPS"
    }
]


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                       AGY ENGINEERING SHOWCASE LAB                           ║
║         Zero-Dependency, First-Principles Computer Science Systems           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_project_demo(proj_num: int):
    proj = next((p for p in PROJECTS if p["num"] == proj_num), None)
    if not proj:
        print(f"Error: Unknown project number {proj_num}")
        return

    proj_dir = os.path.join(ROOT_DIR, proj["dir"])
    demo_script = os.path.join(proj_dir, proj["demo"])

    print(f"\n{'=' * 80}")
    print(f" RUNNING DEMO: Project {proj['num']} - {proj['name']}")
    print(f" Domain: {proj['domain']}")
    print(f" Highlights: {proj['desc']}")
    print(f"{'=' * 80}\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = proj_dir

    subprocess.run([sys.executable, demo_script], cwd=proj_dir, env=env)


def run_all_tests():
    print_banner()
    print("EXECUTING COMPREHENSIVE TEST SUITE ACROSS ALL PROJECTS...\n")

    results = []
    total_start = time.perf_counter()

    for proj in PROJECTS:
        proj_dir = os.path.join(ROOT_DIR, proj["dir"])
        print(f"Testing [{proj['name']}] ({proj['domain']})... ", end="", flush=True)

        env = dict(os.environ)
        env["PYTHONPATH"] = proj_dir

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=proj_dir,
            env=env,
            capture_output=True,
            text=True
        )
        elapsed = time.perf_counter() - t0

        passed = (proc.returncode == 0)
        # Parse test count from unittest output
        test_count = "?"
        for line in proc.stderr.splitlines():
            if "Ran " in line and " tests" in line:
                test_count = line.split("Ran ")[1].split(" tests")[0]
                break

        status_str = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status_str} ({test_count} tests in {elapsed * 1000:.1f} ms)")
        results.append({
            "proj": proj["name"],
            "passed": passed,
            "count": test_count,
            "time": elapsed,
            "stderr": proc.stderr
        })

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("                    MASTER TEST SUMMARY MATRIX")
    print("=" * 70)
    print(f"{'Project':<16} | {'Status':<8} | {'Tests':<10} | {'Duration'}")
    print("-" * 70)

    all_passed = True
    total_tests = 0
    for r in results:
        status_tag = "PASSED" if r["passed"] else "FAILED"
        print(f"{r['proj']:<16} | {status_tag:<8} | {r['count']:<10} | {r['time'] * 1000:.1f} ms")
        if not r["passed"]:
            all_passed = False
        try:
            total_tests += int(r["count"])
        except ValueError:
            pass

    print("-" * 70)
    final_verdict = "ALL TESTS PASSED PERFECTLY" if all_passed else "SOME TESTS FAILED"
    print(f"Total: {total_tests} tests across {len(PROJECTS)} projects in {total_time:.2f}s  ->  {final_verdict}")
    print("=" * 70 + "\n")


def run_all_benchmarks():
    print_banner()
    print("EXECUTING PERFORMANCE BENCHMARK SUITE...\n")

    for proj in PROJECTS:
        bench_dir = os.path.join(ROOT_DIR, proj["dir"], "benchmarks")
        if not os.path.exists(bench_dir):
            continue

        bench_files = [f for f in os.listdir(bench_dir) if f.startswith("bench_") and f.endswith(".py")]
        if not bench_files:
            continue

        bench_script = os.path.join(bench_dir, bench_files[0])
        proj_dir = os.path.join(ROOT_DIR, proj["dir"])

        env = dict(os.environ)
        env["PYTHONPATH"] = proj_dir

        print(f"\n>>> Running Benchmark: {proj['name']} ({bench_files[0]})\n")
        subprocess.run([sys.executable, bench_script], cwd=proj_dir, env=env)


def interactive_menu():
    print_banner()
    while True:
        print("Master Projects:")
        for p in PROJECTS:
            print(f"  [{p['num']}] {p['name']:<12} - {p['domain']:<32} ({p['stats']})")
        print("\nGlobal Actions:")
        print("  [T] Run Comprehensive Test Suite (all projects)")
        print("  [B] Run Performance Benchmarks (all projects)")
        print("  [Q] Exit")

        try:
            choice = input(f"\nSelect an option [1-{len(PROJECTS)}, T, B, Q]: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice in [str(i) for i in range(1, len(PROJECTS) + 1)]:
            run_project_demo(int(choice))
            input("\nPress Enter to return to menu...")
            print("\n" * 2)
            print_banner()
        elif choice == "T":
            run_all_tests()
            input("\nPress Enter to return to menu...")
            print("\n" * 2)
            print_banner()
        elif choice == "B":
            run_all_benchmarks()
            input("\nPress Enter to return to menu...")
            print("\n" * 2)
            print_banner()
        elif choice == "Q":
            print("Goodbye.")
            break
        else:
            print(f"Invalid selection. Please choose 1-{len(PROJECTS)}, T, B, or Q.")


def main():
    parser = argparse.ArgumentParser(description="AGY Engineering Showcase Lab Launcher")
    parser.add_argument("--all-tests", action="store_true", help=f"Run full test suite across all {len(PROJECTS)} projects")
    parser.add_argument("--all-bench", action="store_true", help=f"Run benchmarks across all {len(PROJECTS)} projects")
    parser.add_argument("--demo", type=int, choices=list(range(1, len(PROJECTS) + 1)), help=f"Run demo for project 1 through {len(PROJECTS)}")

    args = parser.parse_args()

    if args.all_tests:
        run_all_tests()
    elif args.all_bench:
        run_all_benchmarks()
    elif args.demo:
        run_project_demo(args.demo)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
