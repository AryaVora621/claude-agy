# SiliconRISC: Architectural Specification & Implementation Plan

A cycle-accurate RV64IMAFD (RV64GC) microarchitectural CPU simulator, 5-stage pipelined processor core, multi-level cache hierarchy with MESI coherence, tournament branch predictor, SV39 virtual memory MMU, ELF64 loader, and sub-pixel Unicode Braille hardware telemetry visualizer.

Zero external dependencies. Pure Python 3.10+ standard library exclusively.

---

## 1. Architectural Overview & System Design

```
+---------------------------------------------------------------------------------------------------+
|                                  SILICONRISC HARVARD ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   |                       5-STAGE IN-ORDER PIPELINE (CLASSIC RISC)                            |   |
|   |                                                                                           |   |
|   |   +----------+      +----------+      +----------+      +----------+      +-----------+   |   |
|   |   | IF:Fetch | ===> | ID:Decode| ===> |EX:Execute| ===> |MEM:Memory| ===> |WB:WriteBck|   |   |
|   |   +----------+      +----------+      +----------+      +----------+      +-----------+   |   |
|   |        ^                 |                 ^   |             ^   |              |         |   |
|   |        |                 v                 |   |             |   |              |         |   |
|   |   +----+----+      +-----------+           |   +-------------+---+--------------+         |   |
|   |   | Branch  |      | Register  |           |        FORWARDING & HAZARD UNIT              |   |
|   |   |Predictor|      | File (X/F)|           |     (EX->EX, MEM->EX, Load-Use Stall)        |   |
|   |   +---------+      +-----------+           +----------------------------------------------+   |
|   +-------------------------------------------------------------------------------------------+   |
|              |                                                        |                           |
|              v                                                        v                           |
|   +---------------------+                                  +---------------------+                |
|   |  ITLB (SV39 Paging) |                                  |  DTLB (SV39 Paging) |                |
|   +---------------------+                                  +---------------------+                |
|              |                                                        |                           |
|              v                                                        v                           |
|   +---------------------+                                  +---------------------+                |
|   |   L1-I Cache (8KB)  |                                  |   L1-D Cache (8KB)  |                |
|   |  4-Way Set-Assoc    |                                  |  4-Way, Write-Back  |                |
|   +---------------------+                                  +---------------------+                |
|              |                                                        |                           |
|              +---------------------------+----------------------------+                           |
|                                          | (Coherence Bus / Snooper)                              |
|                                          v                                                        |
|                             +--------------------------+                                          |
|                             |   L2 Unified Cache (64KB)|                                          |
|                             |      8-Way Set-Assoc     |                                          |
|                             |      MESI State Machine  |                                          |
|                             +--------------------------+                                          |
|                                          |                                                        |
|                                          v                                                        |
|                             +--------------------------+                                          |
|                             | Physical RAM Bus / DRAM  |                                          |
|                             +--------------------------+                                          |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Instruction Set Architecture (RV64GC) Specification

### 2.1 Register File
- **Integer Registers**: 32 x 64-bit general-purpose registers (`x0` through `x31`).
  - `x0` (`zero`): Hardwired constant 0 (writes discarded).
  - `x1` (`ra`): Return address.
  - `x2` (`sp`): Stack pointer.
  - `x3` (`gp`): Global pointer.
  - `x4` (`tp`): Thread pointer.
  - `x5`–`x7`, `x28`–`x31` (`t0`–`t6`): Temporary registers.
  - `x8`–`x9`, `x18`–`x27` (`s0`–`s11`): Callee-saved registers (`s0` / `fp` frame pointer).
  - `x10`–`x17` (`a0`–`a7`): Function arguments and return values.
- **Floating-Point Registers**: 32 x 64-bit IEEE 754 registers (`f0` through `f31`).
- **Program Counter**: 64-bit `pc`.
- **Control & Status Registers (CSRs)**: `mstatus`, `misa`, `mie`, `mtvec`, `mscratch`, `mepc`, `mcause`, `mtval`, `mip`, `satp`, `cycle`, `time`, `instret`.

### 2.2 Instruction Formats
- **R-type**: `funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]`
- **I-type**: `imm[31:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]`
- **S-type**: `imm[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[11:7] | opcode[6:0]`
- **B-type**: `imm[31|7|30:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[11:8|12] | opcode[6:0]`
- **U-type**: `imm[31:12] | rd[11:7] | opcode[6:0]`
- **J-type**: `imm[31|19:12|20|30:21] | rd[11:7] | opcode[6:0]`
- **R4-type (Floating Fused)**: `rs3[31:27] | funct2[26:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]`

### 2.3 Instruction Encodings Supported
- **RV64I**: Base integer instructions (32-bit & 64-bit signed/unsigned math, shifts, comparisons, conditional branches, unconditional jumps, byte/half/word/double loads and stores, fence, system/ecall/ebreak).
- **RV64M**: Standard integer multiply and divide extension (signed/unsigned products, upper 64-bit halves, signed/unsigned quotient and remainder, 32-bit word variants `mulw`, `divw`, `remw`).
- **RV64A**: Standard atomic instructions (`lr.w`, `sc.w`, `lr.d`, `sc.d`, `amoswap`, `amoadd`, `amoxor`, `amoand`, `amoor`, `amomin`, `amomax`).
- **RV64F / RV64D**: Single and double precision IEEE 754 floating-point operations (`fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, `fsgnj`, `fmin`, `fmax`, `feq`, `flt`, `fle`, `fcvt`, `fmv`).

---

## 3. Microarchitectural 5-Stage Pipeline Design

The 5-stage classic RISC pipeline:

1. **Instruction Fetch (IF)**:
   - Fetches 32-bit instruction from memory/L1-I cache at virtual `pc`.
   - Consults Branch Predictor (Tournament + BTB + RAS).
   - Computes next `pc`: predicted branch target or sequential `pc + 4`.
   - Passes `(pc, inst, pred_taken, pred_target)` to IF/ID pipeline register.

2. **Instruction Decode & Register Fetch (ID)**:
   - Decodes opcode, funct3, funct7, registers (`rs1`, `rs2`, `rs3`, `rd`), immediate fields.
   - Reads operands from Integer Register File or Floating Register File.
   - Hazard Detection Unit: Checks for Load-Use data hazards. If `ID.rs1` or `ID.rs2` matches `EX.rd` where `EX` is a memory load, stalls pipeline (freezes PC and IF/ID, injects NOP bubble into ID/EX).

3. **Execution & Address Calculation (EX)**:
   - 64-bit ALU: Performs arithmetic, logical, shift, and address calculations.
   - Forwarding Unit: Detects RAW (Read-After-Write) hazards:
     - `EX/MEM.rd` forward to `EX.operand_A / operand_B` (EX-to-EX forwarding).
     - `MEM/WB.rd` forward to `EX.operand_A / operand_B` (MEM-to-EX forwarding).
   - Branch Resolution: Evaluates actual branch condition. If actual outcome != predicted outcome:
     - Flushes IF/ID and ID/EX pipeline registers.
     - Corrects `pc` to true branch target or sequential fall-through.
     - Updates Branch Predictor with true outcome.

4. **Memory Access (MEM)**:
   - Reads or writes data to L1-D Cache / DTLB.
   - Handles byte alignment, sign-extension, and Atomic Memory Operations (AMO).
   - Records cache hit or miss latency cycles.

5. **Write Back (WB)**:
   - Writes computed ALU result or loaded memory data back to destination register `rd`.
   - Updates committed instruction counter (`instret`) and retirement stats.

---

## 4. Branch Prediction Unit

- **Branch Target Buffer (BTB)**: Direct-mapped/associative cache of branch instructions storing predicted target addresses.
- **Return Address Stack (RAS)**: 16-entry LIFO stack tracking call/return pairs (`jal` x1, `jalr` x0, x1) for zero-stall procedure returns.
- **2-Bit Saturating Counter (Bimodal)**: States: Strongly Not Taken (00), Weakly Not Taken (01), Weakly Taken (10), Strongly Taken (11).
- **Gshare Predictor**: Global History Register (GHR) XORed with low bits of PC to index a 1024-entry Pattern History Table (PHT). Captures complex path-dependent branch correlations.
- **Tournament Predictor**: Dual predictors (Bimodal Local + Gshare Global) selected by a 2-bit meta-predictor tracking which predictor has been more accurate historically.

---

## 5. Multi-Level Cache Hierarchy with MESI Coherence

### 5.1 Cache Structure
- **L1-I Cache**: 8 KB, 4-way set associative, 64-byte line size (32 sets).
- **L1-D Cache**: 8 KB, 4-way set associative, 64-byte line size, write-back, write-allocate.
- **L2 Cache**: 64 KB, 8-way set associative, 64-byte line size, inclusive unified cache.
- **Address Breakdown**:
  - `Offset`: $\log_2(64) = 6$ bits (`addr[5:0]`).
  - `Index`: $\log_2(\text{Sets})$ bits.
  - `Tag`: Remaining upper physical address bits.

### 5.2 MESI Coherence State Machine
- **Modified (M)**: Line is present only in this cache, is dirty, and differs from main memory.
- **Exclusive (E)**: Line is present only in this cache, is clean, and matches main memory.
- **Shared (S)**: Line may be present in other caches, is clean, and matches main memory.
- **Invalid (I)**: Line does not contain valid data.
- **Bus Operations**: BusRead, BusReadX (Read with intent to modify), BusUpgrade, Flush/Writeback.

---

## 6. Memory Management Unit (SV39 Virtual Memory)

- 64-bit Virtual Address in SV39:
  - `VPN[2]` (9 bits, [38:30]) -> Page Directory Pointer Table (1GB gigapage).
  - `VPN[1]` (9 bits, [29:21]) -> Page Directory (2MB megapage).
  - `VPN[0]` (9 bits, [20:12]) -> Page Table (4KB page).
  - `Page Offset` (12 bits, [11:0]).
- 56-bit Physical Address in SV39 (`PPN[2]`, `PPN[1]`, `PPN[0]`, `Offset`).
- Page Table Entry (PTE): `PPN[2:0] | RSW | D | A | G | U | X | W | R | V`.
- Translation Lookaside Buffer (TLB):
  - Fully associative or multi-way DTLB (32 entries) and ITLB (16 entries).
  - Software/Hardware Page Table Walker traversing memory on TLB miss.

---

## 7. ELF64 Loader & Assembler

- **ELF64 Parser**:
  - Validates `\x7fELF`, 64-bit architecture, little-endian format, RISC-V machine type (0xF3).
  - Parses Program Headers (`Elf64_Phdr`): identifies `PT_LOAD` segments, maps virtual memory addresses to binary file data, sets permissions (R/W/X).
  - Parses Section Headers and Symbol Tables (`.symtab`, `.strtab`): extracts function names, labels, and entry points.
- **Integrated RISC-V Assembler**:
  - Supports standard RISC-V mnemonics, labels, `.text`, `.data`, `.word`, `.dword`, `.asciiz`.
  - Two-pass assembler resolving forward label references into exact 32-bit machine code instructions.

---

## 8. Sub-Pixel Braille Hardware Visualizer

- 2x4 Sub-Pixel Unicode Braille rendering (`U+2800..U+28FF`):
  - **Pipeline Stage Visualizer**: Displays current instruction in IF, ID, EX, MEM, WB with active hazard markers and forwarding dataflow arrows.
  - **Cache Set Heatmap**: Renders set-associative ways, tags, MESI states, and dirtiness.
  - **Execution Metrics HUD**: IPC, CPI, branch accuracy, cache hit/miss ratio, TLB translation statistics.

---

## 9. Verification & Benchmark Matrix

- Target: 30+ comprehensive unit tests covering:
  - RV64I arithmetic, logic, shifts, branches, jumps, loads, stores.
  - RV64M integer multiply and divide.
  - RV64A atomic operations.
  - RV64F/D floating-point operations.
  - 5-stage pipeline hazards: RAW forwarding, load-use stalls, branch misprediction flushes.
  - Branch predictors: Bimodal, Gshare, Tournament, BTB, RAS.
  - Cache operations: L1/L2 hits/misses, dirty evictions, MESI snooping transitions.
  - MMU operations: SV39 3-level page walks, TLB hits/misses, page faults.
  - Real programs: Recursive Fibonacci, Quicksort, 4x4 Matrix Multiplication.
- Benchmarks:
  - Instructions/sec in functional mode (>100k inst/s).
  - Pipeline cycles/sec in cycle-accurate mode (>20k cycles/s).
  - Cache lookup latency and MMU walk throughput.
  - Visualizer frame generation rate (>500 FPS).
