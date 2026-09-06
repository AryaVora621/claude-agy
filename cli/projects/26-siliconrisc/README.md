# SiliconRISC: Cycle-Accurate RV64GC Processor Architecture & Hardware Simulator

A pure Python 3.10+ cycle-accurate simulation framework for the RISC-V 64-bit architecture (RV64GC: RV64IMAFD). SiliconRISC models the complete hardware microarchitecture of modern out-of-order and in-order RISC processors with zero external dependencies.

```
       +-------------------------------------------------------------------+
       |                       SILICONRISC RV64GC                         |
       |                   Cycle-Accurate Microarchitecture                |
       +-------------------------------------------------------------------+
                                         |
     +-----------------+-----------------+-----------------+-----------------+
     |                 |                 |                 |                 |
     v                 v                 v                 v                 v
+---------+      +-----------+     +-----------+     +-----------+     +-----------+
| 5-Stage |      | SV39 MMU  |     | Multi-Lvl |     | Dynamic   |     | Sub-Pixel |
| Classic |      | 3-Level   |     | Cache &   |     | Branch    |     | Braille   |
| Pipeline|      | Page Walk |     | MESI Coher|     | Predictor |     | Visualizer|
+---------+      +-----------+     +-----------+     +-----------+     +-----------+
```

---

## Key Technical Features

### 1. RV64GC Instruction Set Architecture (`siliconrisc/isa.py`)
- Complete 64-bit integer registers (`x0` through `x31`) with strict hardwired zero on `x0`.
- 64-bit IEEE 754 double-precision floating-point registers (`f0` through `f31`).
- Support for base integer instructions (RV64I), standard integer multiplication/division (RV64M), atomic memory operations (RV64A), and single/double-precision floating-point (RV64F/D).
- Machine and Supervisor privilege modes with complete Control and Status Register (CSR) support: `mstatus`, `misa`, `mtvec`, `mepc`, `mcause`, `mtval`, `satp`, `cycle`, and `instret`.
- Two-pass canonical RISC-V disassembler converting 32-bit machine words to human-readable assembly.

### 2. 5-Stage In-Order Pipeline with Hazard Resolution (`siliconrisc/pipeline.py`)
- Classic 5-stage pipeline: **IF** (Instruction Fetch), **ID** (Decode & Register Read), **EX** (Execute / ALU), **MEM** (Memory Access), and **WB** (Write Back).
- **Forwarding Unit**: Eliminates Read-After-Write (RAW) data hazards by multiplexing results directly from EX/MEM and MEM/WB stage registers to the ALU inputs, avoiding pipeline stalls on back-to-back register dependencies.
- **Hazard Detection Unit**: Identifies Load-Use data dependencies. Stalls the Program Counter (PC) and IF/ID register for 1 cycle while injecting a pipeline bubble into ID/EX.
- **Branch Resolution & Flushes**: Evaluates conditional branch outcomes and target addresses in the EX stage. Flushes speculatively fetched instructions in IF/ID and ID/EX upon branch mispredictions.

```
Cycle      0    1    2    3    4    5    6    7    8
Inst 0:   [IF] [ID] [EX] [MEM] [WB]
Inst 1:        [IF] [ID] [EX] [MEM] [WB]          (Forwarded EX->EX)
Inst 2 (Load):      [IF] [ID] [EX] [MEM] [WB]
Inst 3 (Use):            [IF] [ID] [---] [EX] [MEM] [WB] (1-cycle Stall Bubble)
```

### 3. SV39 Virtual Memory MMU & TLB (`siliconrisc/memory.py`)
- Sparse physical memory engine using on-demand 4KB page allocations (`PhysicalMemory`).
- Full SV39 3-level page table walking: translates 39-bit virtual addresses decomposed into `VPN[2]` (1GB gigapages), `VPN[1]` (2MB megapages), and `VPN[0]` (4KB standard pages) plus a 12-bit page offset.
- Canonical address verification ensuring bits 63:39 match sign-extended bit 38.
- Comprehensive Page Table Entry (PTE) permission checking: Valid (`V`), Read (`R`), Write (`W`), Execute (`X`), User (`U`), Accessed (`A`), and Dirty (`D`).
- Fully-associative Translation Lookaside Buffers (ITLB and DTLB) with Least-Recently-Used (LRU) eviction policy.

```
       Virtual Address (39 bits):
       +---------------+---------------+---------------+-------------------+
       | VPN[2] (9 b)  | VPN[1] (9 b)  | VPN[0] (9 b)  | Page Offset (12b) |
       +---------------+---------------+---------------+-------------------+
               |               |               |                 |
               v               v               v                 |
          +---------+     +---------+     +---------+            |
  satp -> | Level 2 | --> | Level 1 | --> | Level 0 |            |
  root    | (1 GB)  |     | (2 MB)  |     | (4 KB)  |            |
          +---------+     +---------+     +---------+            |
                                               |                 |
                                               v                 v
                                  +--------------------+-------------------+
                                  | Physical Page (PPN)| Page Offset (12b) |
                                  +--------------------+-------------------+
```

### 4. Multi-Level Cache Hierarchy & MESI Coherence (`siliconrisc/cache.py`)
- Configurable N-way set-associative caches:
  - 8 KB 4-way L1 Instruction Cache (L1I) (1-cycle hit latency)
  - 8 KB 4-way L1 Data Cache (L1D) with write-back and write-allocate (1-cycle hit latency)
  - 64 KB 8-way Unified L2 Cache (6-cycle hit latency)
  - Physical RAM backing store (50-cycle miss latency)
- Full 4-state MESI Coherence Protocol:
  - **Modified (M)**: Line is valid, exclusive to this cache, and dirty.
  - **Exclusive (E)**: Line is valid, exclusive to this cache, and clean.
  - **Shared (S)**: Line is valid, shared across multiple caches, and clean.
  - **Invalid (I)**: Line contains no valid data.
- Bus transaction snooping: `BusRd`, `BusRdX`, `BusUpgr`, and `BusWb` for multi-core configurations.
- Real-time Average Memory Access Time (AMAT) telemetry tracking.

### 5. Advanced Branch Prediction Unit (`siliconrisc/branch.py`)
- **Branch Target Buffer (BTB)**: Direct-mapped 512-entry cache storing predicted target PCs.
- **Return Address Stack (RAS)**: 16-entry LIFO stack tracking call/return pairs (`jal ra` and `jalr`) for zero-stall procedure returns.
- **Bimodal Predictor**: 1024-entry table of 2-bit saturating up/down counters (Strongly Not-Taken, Weakly Not-Taken, Weakly Taken, Strongly Taken).
- **Gshare Predictor**: 10-bit Global History Register (GHR) XORed with PC address bits indexing a 1024-entry Pattern History Table (PHT).
- **Tournament Predictor**: Meta-predictor dynamically arbitrating between local and global predictors based on historical accuracy.

### 6. ELF64 Binary Parser, Loader, and Assembler (`siliconrisc/elf.py`)
- Two-pass RV64GC assembler supporting labels, immediates, and pseudo-instructions (`li`, `mv`, `nop`, `ret`, `j`).
- ELF64 executable generator and validator creating compliant binary images.
- ELF64 binary loader mapping `PT_LOAD` segments directly into memory and configuring SV39 page tables.

### 7. Sub-Pixel Braille Hardware Visualizer (`siliconrisc/visualizer.py`)
- 2x4 dot sub-pixel resolution Unicode Braille plotting canvas (`U+2800` through `U+28FF`).
- Real-time IPC sparkline and execution telemetry waveform.
- Full 5-stage pipeline stage HUD displaying instruction flow, forwarding paths, and hazard flags.
- Complete 32-register formatted architectural state grid.

---

## Architectural Verification & Microbenchmarks

Run the complete benchmark suite to measure subsystem throughput:

```bash
python3 benchmarks/bench_siliconrisc.py
```

### Benchmark Results (Apple Silicon M-Series, Python 3.13)

| Microbenchmark Subsystem | Metric | Measured Throughput |
|:---|:---|:---|
| **Instruction Decoder** | 32-bit RV64GC decoding speed | **797,000 instructions/sec** |
| **Functional Interpreter** | Tight ALU loop execution | **332,000 instructions/sec** |
| **5-Stage Pipeline Core** | Cycle-accurate hardware simulation | **142,000 cycles/sec** |
| **L1D Cache Access** | 1-cycle hit access rate | **1,440,000 accesses/sec** |
| **SV39 DTLB Translation** | Address translation speed | **1,563,000 translations/sec** |
| **Branch Predictor (Gshare)** | Branch pattern recognition | **99.75% accuracy** |
| **Branch Predictor (Tournament)**| Correlated loop prediction | **99.85% accuracy** |

---

## Interactive Terminal Hardware Workbench

SiliconRISC includes an interactive terminal workbench with live pipeline visualizations:

```bash
# Run Vector Dot Product with RAW forwarding in the 5-stage pipeline
python3 examples/system_workbench.py --program dot

# Run Recursive Fibonacci stack execution
python3 examples/system_workbench.py --program fibonacci --mode functional --max-cycles 1000

# Run in-memory 64-bit Bubble Sort
python3 examples/system_workbench.py --program sort --mode functional --max-cycles 1000

# Run live cycle-by-cycle ANSI animation
python3 examples/system_workbench.py --program dot --animate --delay 0.05
```

---

## Unit and Integration Test Suite

The test suite contains 40 tests across 6 modules covering ISA decoding, MMU page table walks, cache hierarchy, branch prediction, pipeline hazard stalls, and algorithmic integration:

```bash
python3 -m unittest discover tests
```

Output:
```
........................................
----------------------------------------------------------------------
Ran 40 tests in 0.012s

OK
```

---

## Project File Layout

```
projects/26-siliconrisc/
├── README.md                      # Architectural documentation and guide
├── PLAN.md                        # Formal engineering specifications
├── siliconrisc/                   # Core microarchitecture package
│   ├── __init__.py                # Package exports
│   ├── isa.py                     # RV64GC instruction definitions, decoder, disassembler
│   ├── memory.py                  # Sparse PhysicalMemory, SV39 MMU, TLB
│   ├── cache.py                   # N-way cache, MESI coherence, MemoryHierarchy
│   ├── branch.py                  # BTB, RAS, Bimodal, Gshare, Tournament BPU
│   ├── pipeline.py                # 5-stage pipeline, ForwardingUnit, HazardDetectionUnit
│   ├── core.py                    # Dual-mode CPU (Functional & Pipelined), CSRs, Traps
│   ├── elf.py                     # ELF64 parser, loader, RV64GC assembler
│   └── visualizer.py              # Braille sub-pixel canvas, pipeline HUD, telemetry
├── tests/                         # Exhaustive unit test suite (40 tests)
│   ├── test_isa_and_decoder.py    # Instruction decoding & bit manipulation tests
│   ├── test_memory_and_mmu.py     # Physical memory, SV39 page walks, TLB tests
│   ├── test_cache_and_coherence.py# Cache hits/misses, evictions, MESI snooping
│   ├── test_branch_predictor.py   # BTB, RAS, Bimodal, Gshare, Tournament tests
│   ├── test_pipeline_hazards.py   # RAW forwarding, load-use stalls, branch flushes
│   ├── test_core.py               # Functional & pipelined execution, CSRs, traps
│   ├── test_elf_and_assembler.py  # ELF parsing, segment loading, assembler tests
│   ├── test_visualizer.py         # Braille rendering, pipeline HUD, dashboards
│   └── test_programs_integration.py # Fibonacci, Bubble Sort, Dot Product, FP
├── benchmarks/
│   └── bench_siliconrisc.py       # Performance microbenchmarks
└── examples/
    └── system_workbench.py        # Terminal Hardware Workbench demo
```

---

## License

MIT License. Designed and built from first principles in pure Python 3.10+ standard library.
