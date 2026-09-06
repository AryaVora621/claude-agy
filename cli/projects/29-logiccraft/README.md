# LogicCraft: Electronic Design Automation (EDA) Logic Synthesis & Static Timing Analysis Engine

**LogicCraft** is an industrial-grade digital logic synthesis, technology mapping, and Static Timing Analysis (STA) engine implemented in pure Python 3.10+ standard library with **zero external dependencies**.

LogicCraft provides a complete, modern front-to-back digital ASIC physical design flow:
- **Reduced Ordered Binary Decision Diagrams (ROBDD)**: Canonical Boolean function representation with Shannon decomposition, unique table subgraph sharing, memoized ternary If-Then-Else (ITE) operator, and $O(1)$ formal equivalence checking.
- **And-Inverter Graphs (AIG)**: Homogeneous directed acyclic graph (DAG) of 2-input AND gates with 1-bit inverted edge pointers (LSB literal encoding), two-level structural hashing ("strashing") with on-the-fly Boolean reductions, and bit-parallel logic simulation.
- **Liberty Standard Cell Library & NLDM**: Comprehensive standard cell library (INV, BUF, NAND2, NOR2, AND2, OR2, XOR2, AOI21, OAI21, DFF) with physical silicon area, static leakage power, sink pin capacitances, and Non-Linear Delay Model (NLDM) 2D lookup tables evaluated via high-precision 2D bilinear interpolation.
- **Dynamic Programming Technology Mapping**: Implementation of the DAGON tree-covering algorithm partitioning AIGs into single-fanout trees, matching standard cell sub-graphs, and minimizing silicon area or critical path propagation delay.
- **Physical Gate-Level Netlist & Verilog Export**: Netlist container tracking cell instances, pin connections, electrical nets, wire load models, and canonical structural Verilog netlist generation.
- **Static Timing Analysis (STA) Engine**: Directed timing graph, topological sort, forward Arrival Time (AT) and slew propagation, backward Required Arrival Time (RAT) propagation, pin-level slack calculation, setup timing closure (WNS, TNS), and physical critical path extraction.
- **Sub-Pixel Braille Circuit Visualizer & HUD**: High-resolution Unicode Braille (`U+2800..U+28FF`) 2x4 dot matrix canvas plotting critical path delay progression curves and ANSI timing closure telemetry HUD.

---

## Architectural Overview

```
                      +-----------------------------+
                      |   RTL / Boolean Functions   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     ROBDD Canonical Form    |
                      | (ITE, Shannon, Model Count) |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      And-Inverter Graph     |
                      |  (Strashing & Levelization) |
                      +-----------------------------+
                                     |
                                     v
+------------------------+    +-----------------------------+
| Liberty Cell Library   |--->|    Technology Mapping       |
| (NLDM 2D Tables, Area) |    | (DAGON DP Tree Covering)    |
+------------------------+    +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Gate-Level Mapped Netlist |
                      | (Verilog Codec, Wire Loads) |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Static Timing Analysis (STA)|
                      | (AT, RAT, Slack, WNS, TNS)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Braille HUD & Critical Path |
                      +-----------------------------+
```

---

## Mathematical Foundations

### 1. Reduced Ordered Binary Decision Diagrams (ROBDD)

Any Boolean function $f(x_1, \dots, x_n)$ can be decomposed with respect to variable $x_i$ via Shannon expansion:

$$f = x_i \cdot f_{x_i} + \bar{x}_i \cdot f_{\bar{x}_i}$$

where $f_{x_i} = f(x_1, \dots, x_i = 1, \dots, x_n)$ is the positive cofactor and $f_{\bar{x}_i} = f(x_1, \dots, x_i = 0, \dots, x_n)$ is the negative cofactor.

Every Boolean operation is expressed via the ternary **If-Then-Else (ITE)** operator:

$$\text{ITE}(F, G, H) = F \cdot G + \bar{F} \cdot H$$

- $\text{NOT}(F) = \text{ITE}(F, 0, 1)$
- $\text{AND}(F, G) = \text{ITE}(F, G, 0)$
- $\text{OR}(F, G) = \text{ITE}(F, 1, G)$
- $\text{XOR}(F, G) = \text{ITE}(F, \bar{G}, G)$
- $\text{NAND}(F, G) = \text{ITE}(F, \bar{G}, 1)$

Using a global unique table for canonical sub-graph sharing and a computed cache for ITE operations, equivalence checking between two functions $F$ and $G$ is an exact $O(1)$ integer pointer comparison ($F == G$).

### 2. And-Inverter Graph (AIG) Structural Hashing ("Strashing")

An AIG is a directed acyclic graph where every internal vertex is a 2-input logical AND gate, and edges may carry inverter flags.
Literals are encoded in a single integer where the least significant bit (LSB) denotes negation:

$$\text{lit} = (\text{node\_id} \ll 1) \mid (\text{inverted} \ ? \ 1 : 0)$$

On-the-fly Boolean reductions during construction:
- $x \wedge 0 = 0$
- $x \wedge 1 = x$
- $x \wedge x = x$
- $x \wedge \bar{x} = 0$
- Canonical ordering: $(u, v)$ with $u \le v$

### 3. Liberty Non-Linear Delay Model (NLDM)

Propagation delay $t_{pd}$ and output transition time $\tau_{out}$ through a cell timing arc are functions of input slew $\tau_{in}$ and output capacitive load $C_{load}$:

$$t_{pd} = \text{TableLookup}(\tau_{in}, C_{load})$$

$$C_{load} = C_{wire} + \sum_{k \in \text{fanout}} C_{pin, k}$$

Intermediate values are computed via 2D bilinear interpolation over the bounding grid cell:

$$f(x, y) \approx \frac{(x_2 - x)(y_2 - y) f_{11} + (x - x_1)(y_2 - y) f_{21} + (x_2 - x)(y - y_1) f_{12} + (x - x_1)(y - y_1) f_{22}}{(x_2 - x_1)(y_2 - y_1)}$$

### 4. Technology Mapping via DAGON Tree Covering

The unmapped AIG is partitioned into maximum single-fanout trees by cutting at multi-fanout vertices.
At each node $v$, for every valid matching cell pattern $m$:

$$\text{Cost}(v, m) = \text{CellCost}(m) + \sum_{u \in \text{leaves}(m)} \text{BestCost}(u)$$

The optimal match $m^*(v) = \arg\min_m \text{Cost}(v, m)$ is retained, optimizing for minimum silicon area ($\mu\text{m}^2$) or intrinsic propagation delay.

### 5. Static Timing Analysis (STA)

1. **Forward Arrival Time (AT)**:
   Starting at primary inputs ($AT = 0, \tau = \tau_{in}$):
   $$AT(v) = \max_{u \in \text{pred}(v)} (AT(u) + d_{u \to v})$$

2. **Backward Required Arrival Time (RAT)**:
   Starting at primary outputs ($RAT = T_{clk}$):
   $$RAT(v) = \min_{w \in \text{succ}(v)} (RAT(w) - d_{v \to w})$$

3. **Slack Calculation & Setup Timing Closure**:
   $$\text{Slack}(v) = RAT(v) - AT(v)$$
   - Worst Negative Slack (WNS): $\min_v \text{Slack}(v)$
   - Total Negative Slack (TNS): $\sum_{v, \text{Slack}(v) < 0} \text{Slack}(v)$
   - Maximum Operating Frequency: $F_{max} = \frac{1}{\max_v AT(v)}$

---

## Directory Structure

```
projects/29-logiccraft/
├── logiccraft/
│   ├── __init__.py          # Public API exports
│   ├── bdd.py               # ROBDD engine, Shannon expansion, ITE, equivalence checking
│   ├── aig.py               # And-Inverter Graph, structural hashing, multi-level logic
│   ├── liberty.py           # Standard cell library, NLDM models, 2D bilinear interpolation
│   ├── techmap.py           # Pattern matching, DP tree covering technology mapper
│   ├── netlist.py           # Gate-level mapped netlist, pins, wires, cell instances
│   ├── sta.py               # Static Timing Analysis, topological sort, AT/RAT/Slack, WNS
│   └── visualizer.py        # Sub-pixel Braille circuit DAG, slack histogram, timing HUD
├── tests/
│   ├── test_bdd.py          # ROBDD canonicity, ITE operators, Boolean equivalence
│   ├── test_aig.py          # AIG strashing, evaluation, graph minimization
│   ├── test_liberty.py      # Liberty cell parser, NLDM bilinear interpolation
│   ├── test_techmap.py      # Technology mapping correctness, area/delay tradeoffs
│   ├── test_sta.py          # STA arrival/required times, slack, critical path extraction
│   └── test_visualizer.py   # Braille graph plotting, slack histogram, ANSI rendering
├── benchmarks/
│   └── bench_logiccraft.py  # Performance microbenchmarks (ROBDD, AIG, Mapper, STA)
├── examples/
│   └── eda_workbench.py     # Interactive terminal EDA synthesis & timing workbench
├── PLAN.md                  # Comprehensive architectural specification
├── TASK_QUEUE.md            # Milestone tracker
├── CHECKPOINT_LAST.md       # Operational state checkpoint
└── README.md                # Documentation and architectural guide
```

---

## Verification & Test Results

The test suite contains 25 comprehensive unit tests covering all components:

```bash
PYTHONPATH="." python3 -m unittest discover -s tests -v
```

Output:
```text
test_derived_logic_gates (test_aig.TestAIG.test_derived_logic_gates) ... ok
test_literal_manipulation (test_aig.TestAIG.test_literal_manipulation) ... ok
test_logic_simulation (test_aig.TestAIG.test_logic_simulation) ... ok
test_structural_hashing_strashing (test_aig.TestAIG.test_structural_hashing_strashing) ... ok
test_topological_sort_and_depth (test_aig.TestAIG.test_topological_sort_and_depth) ... ok
test_canonicity_and_identities (test_bdd.TestROBDD.test_canonicity_and_identities) ... ok
test_constants_and_variables (test_bdd.TestROBDD.test_constants_and_variables) ... ok
test_de_morgan_laws (test_bdd.TestROBDD.test_de_morgan_laws) ... ok
test_expression_parser (test_bdd.TestROBDD.test_expression_parser) ... ok
test_sat_counting_and_witness (test_bdd.TestROBDD.test_sat_counting_and_witness) ... ok
test_xor_and_mux_equivalence (test_bdd.TestROBDD.test_xor_and_mux_equivalence) ... ok
test_bilinear_interpolation_grid_exact (test_liberty.TestLibertyModels.test_bilinear_interpolation_grid_exact) ... ok
test_bilinear_interpolation_midpoint (test_liberty.TestLibertyModels.test_bilinear_interpolation_midpoint) ... ok
test_default_cells_available (test_liberty.TestLibertyModels.test_default_cells_available) ... ok
test_nldm_delay_scaling_with_load (test_liberty.TestLibertyModels.test_nldm_delay_scaling_with_load) ... ok
test_critical_path_start_and_end (test_sta.TestStaticTimingAnalysis.test_critical_path_start_and_end) ... ok
test_inverter_chain_timing (test_sta.TestStaticTimingAnalysis.test_inverter_chain_timing) ... ok
test_timing_violation_detection (test_sta.TestStaticTimingAnalysis.test_timing_violation_detection) ... ok
test_map_complex_logic (test_techmap.TestTechMapper.test_map_complex_logic) ... ok
test_map_inverter_and_buffer (test_techmap.TestTechMapper.test_map_inverter_and_buffer) ... ok
test_map_nand_and_nor_gates (test_techmap.TestTechMapper.test_map_nand_and_nor_gates) ... ok
test_map_verilog_export (test_techmap.TestTechMapper.test_map_verilog_export) ... ok
test_braille_canvas_set_pixel_and_render (test_visualizer.TestVisualizer.test_braille_canvas_set_pixel_and_render) ... ok
test_braille_delay_curve_render (test_visualizer.TestVisualizer.test_braille_delay_curve_render) ... ok
test_timing_hud_and_waterfall (test_visualizer.TestVisualizer.test_timing_hud_and_waterfall) ... ok

Ran 25 tests in 0.004s
OK
```

---

## Performance Benchmarks

Run microbenchmarks:

```bash
python3 benchmarks/bench_logiccraft.py
```

Results on standard hardware:
- **ROBDD Synthesis & SAT Count (16-var)**: `136,063 ops/sec`
- **AIG Construction & Strashing**: `76,877 ops/sec`
- **Liberty NLDM 2D Bilinear Interpolation**: `35,247 ops/sec` (700,000+ individual delay/slew lookups/sec)
- **DAGON Dynamic Programming Tech Mapping**: `6,467 ops/sec`
- **Static Timing Analysis (AT/RAT/Slack)**: `4,745 ops/sec`
- **Sub-Pixel Braille Canvas Rasterizer**: `1,698 FPS`

---

## Interactive EDA Synthesis Workbench

Run the interactive CLI workbench to synthesize and analyze circuits:

```bash
# Synthesize 4-bit carry-lookahead adder and export structural Verilog
python3 examples/eda_workbench.py --circuit adder4 --clock 800.0 --verilog

# Synthesize 8-bit parity and majority tree
python3 examples/eda_workbench.py --circuit parity --clock 600.0

# Synthesize 4-to-2 priority arbiter
python3 examples/eda_workbench.py --circuit encoder --clock 500.0
```

Sample output:
```text
============================================================================
LOGICCRAFT: AUTOMATED DIGITAL SYNTHESIS & TIMING CLOSURE WORKBENCH
============================================================================

[STEP 1/5] Logic Synthesis & And-Inverter Graph Decomposition
  Target Circuit:        4-Bit Carry-Lookahead Adder (9 Inputs, 5 Outputs)
  Internal AND Nodes:    36
  Graph Logic Depth:     10 levels
  Structural Hash Hits:  Active (On-The-Fly Redundancy Elimination)

[STEP 2/5] Standard Cell Technology Mapping (DAGON Algorithm)
  Target Library:        TSMC45_Simulated (45nm Standard CMOS, VDD=1.1V, Temp=25C)
  Optimization Goal:     Minimum Silicon AREA
  Mapped Cells:          48 standard cells
  Physical Area:         92.40 um^2
  Static Leakage Power:  40.00 nW

[STEP 3/5] Static Timing Analysis (NLDM 2D Bilinear Interpolation)
  Target Clock Period:   800.0 ps (Requested Freq: 1.25 GHz)

[STEP 4/5] Timing Closure & Circuit Telemetry HUD
┌──────────────────────────────────────────────────────────────────────────┐
│ LOGICCRAFT EDA SYNTHESIS & STA TELEMETRY: cla_adder_4bit                 │
├──────────────────────────────────────────────────────────────────────────┤
│  Module Name:        cla_adder_4bit     Physical Area:      92.40 um^2   │
│  Standard Cells:     48                 Leakage Power:      40.00 nW     │
│  Electrical Nets:    57                 I/O Port Count:     9 In / 5 Out │
├──────────────────────────────────────────────────────────────────────────┤
│  Clock Period:       800.0 ps           Max Operating Freq: 4.08 GHz     │
│  Worst Slack (WNS):  554.97 ps          Critical Path Delay: 245.03 ps   │
│  Total Slack (TNS):  0.00 ps            Timing Status:      TIMING MET   │
└──────────────────────────────────────────────────────────────────────────┘
```
