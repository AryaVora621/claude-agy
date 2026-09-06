# PLAN.md: LogicCraft Electronic Design Automation (EDA) Engine

## 1. Executive Summary & Objectives

**LogicCraft** is a zero-dependency, pure Python 3.10+ standard library implementation of an industrial-grade Electronic Design Automation (EDA) logic synthesis, technology mapping, and Static Timing Analysis (STA) toolchain.

The objective of LogicCraft is to model the complete silicon compilation flow from abstract Boolean logic / gate-level netlists to optimized standard-cell physical implementations with timing sign-off:
- **Canonical Boolean Reasoning**: Reduced Ordered Binary Decision Diagrams (ROBDD) with Shannon expansion, computed tables, and constant-time formal equivalence checking.
- **And-Inverter Graphs (AIG)**: Compact homogeneous multi-level logic representation with two-level structural hashing (strashing) and algebraic simplification.
- **Liberty Cell Library**: Standard cell modelling (INV, NAND, NOR, AOI, OAI, DFF) with Non-Linear Delay Model (NLDM) 2D bilinear interpolation of propagation delay and output slew.
- **Technology Mapping**: Dynamic programming tree covering (DAGON algorithm) optimizing for silicon area ($\mu\text{m}^2$) or critical path delay ($ps$).
- **Static Timing Analysis (STA)**: Graph-based timing engine with topological levelization, Arrival Time (AT) forward propagation, Required Arrival Time (RAT) backward propagation, pin slack calculation, setup/hold checks, and Worst Negative Slack (WNS) critical path extraction.
- **Sub-Pixel Braille Visualizer**: ANSI TrueColor terminal visualization of circuit DAG topologies and slack distribution histograms.

---

## 2. Mathematical & Algorithmic Foundations

### 2.1 Reduced Ordered Binary Decision Diagrams (ROBDD)

A Binary Decision Diagram (BDD) represents a Boolean function $f: \{0, 1\}^n \to \{0, 1\}$ as a directed acyclic graph rooted at $f$ with two terminal sinks: $0$ and $1$.

#### Shannon Expansion Theorem
For any variable $x_i$:

$$f = x_i \cdot f_{x_i=1} + \bar{x}_i \cdot f_{x_i=0}$$

where $f_{x_i=1}$ is the positive cofactor (high branch) and $f_{x_i=0}$ is the negative cofactor (low branch).

#### Ordering & Reduction Rules
Given a total ordering of variables $\pi = (x_1 < x_2 < \dots < x_n)$:
1. **Ordered (OBDD)**: Every path from root to terminal visits variables in ascending $\pi$-order.
2. **Reduced (ROBDD)**:
   - **Node Elimination**: No node $v$ has $low(v) == high(v)$. (Redundant test eliminated).
   - **Isomorphism Merging**: No two distinct nodes $u, v$ have $var(u) == var(v)$, $low(u) == low(v)$, and $high(u) == high(v)$.

#### Canonicity
Under a fixed variable ordering $\pi$, the ROBDD representation of any Boolean function is strictly canonical:

$$f \equiv g \iff \text{root}(f) == \text{root}(g)$$

This enables constant-time $O(1)$ formal equivalence checking.

#### If-Then-Else (ITE) Operator
All Boolean operations are computed through the ternary operator:

$$\text{ITE}(F, G, H) = F \cdot G + \bar{F} \cdot H$$

- $F \wedge G = \text{ITE}(F, G, 0)$
- $F \vee G = \text{ITE}(F, 1, G)$
- $F \oplus G = \text{ITE}(F, \bar{G}, G)$
- $\neg F = \text{ITE}(F, 0, 1)$

Using a unique table and memoized computed cache, ITE executes in $O(|F| \cdot |G|)$ time.

---

### 2.2 And-Inverter Graphs (AIG)

An And-Inverter Graph is a directed acyclic graph where:
- Every node has in-degree 2 and represents the logical AND operation.
- Every directed edge carries a 1-bit Boolean attribute indicating whether the signal is inverted.
- Primary inputs (PI) and primary outputs (PO) connect the graph to external ports.

#### Structural Hashing (Strashing)
During graph construction, a two-level lookup table ensures canonical local structures:
- Commutativity: $\text{AND}(a, b) == \text{AND}(b, a)$
- Identity: $a \wedge 1 = a$, $a \wedge 0 = 0$
- Idempotence: $a \wedge a = a$
- Contradiction: $a \wedge \bar{a} = 0$
- Global node deduplication: if an AND node with fanins $(u, v)$ already exists in the graph, the existing node is returned without allocation.

---

### 2.3 Liberty Standard Cell Library & NLDM Timing Models

A standard cell library defines physical building blocks:
- Inverters: `INV_X1`, `INV_X2`, `INV_X4`
- NAND gates: `NAND2_X1`, `NAND3_X1`
- NOR gates: `NOR2_X1`, `NOR3_X1`
- AND gates: `AND2_X1`
- OR gates: `OR2_X1`
- XOR gates: `XOR2_X1`
- Complex gates: `AOI21_X1` ($Y = \overline{(A_1 \cdot A_2) + B}$), `OAI21_X1` ($Y = \overline{(A_1 + A_2) \cdot B}$)
- Sequential elements: `DFF_X1` (Positive edge-triggered D flip-flop)

#### Non-Linear Delay Model (NLDM)
For each input-to-output timing arc, cell propagation delay $t_{pd}$ and output slew $t_{slew}$ are determined by 2D lookup tables indexed by input transition time ($\tau_{in}$) and total capacitive load ($C_{load}$):

$$t_{pd} = \text{TableLookup}(\tau_{in}, C_{load})$$

$$C_{load} = C_{wire} + \sum_{k \in \text{fanout}} C_{pin, k}$$

Values between grid points are evaluated via 2D bilinear interpolation:

$$f(x, y) \approx \frac{(x_2 - x)(y_2 - y) f_{11} + (x - x_1)(y_2 - y) f_{21} + (x_2 - x)(y - y_1) f_{12} + (x - x_1)(y - y_1) f_{22}}{(x_2 - x_1)(y_2 - y_1)}$$

---

### 2.4 Technology Mapping via Dynamic Programming Tree Covering

Technology mapping transforms an unmapped AIG into a network of target standard cells.
1. **Tree Decomposition**: The AIG is partitioned into maximum single-fanout trees by cutting at nodes with out-degree $\ge 2$.
2. **Pattern Matching**: Library cell logic functions are matched against sub-trees rooted at each AIG node.
3. **Dynamic Programming Bottom-Up Pass**:
   At each node $v$, for every valid matching cell pattern $m$:
   $$\text{Cost}(v, m) = \text{CellCost}(m) + \sum_{u \in \text{leaves}(m)} \text{BestCost}(u)$$
   The optimal match $m^*(v) = \arg\min_m \text{Cost}(v, m)$ is stored.
4. **Top-Down Cell Generation**: Standard cell instances are emitted starting from tree roots toward primary inputs.

---

### 2.5 Static Timing Analysis (STA) Engine

The timing engine verifies that all signals propagate fast enough to meet clock setup times and slow enough to satisfy hold times.

#### 1. Graph Levelization & Topological Sort
The circuit netlist is converted into a directed timing graph $G = (V, E)$, where vertices $V$ are cell pins and primary ports, and edges $E$ are cell timing arcs and net connections.

#### 2. Forward Arrival Time (AT) Propagation
Starting at primary inputs and register clock pins ($AT = 0$):

$$AT(v) = \max_{u \in \text{pred}(v)} (AT(u) + d_{u \to v})$$

where $d_{u \to v}$ is the pin-to-pin propagation delay.

#### 3. Backward Required Arrival Time (RAT) Propagation
Starting at primary outputs and register D pins with clock period $T_{clk}$:

$$RAT(v) = \min_{w \in \text{succ}(v)} (RAT(w) - d_{v \to w})$$

For data pins of registers:
$$RAT(D) = T_{clk} - t_{\text{setup}}$$

#### 4. Slack Calculation & Timing Closure
$$\text{Slack}(v) = RAT(v) - AT(v)$$

- $\text{Slack}(v) \ge 0$: Timing is met (slack positive).
- $\text{Slack}(v) < 0$: Timing violation (setup violation).
- **Worst Negative Slack (WNS)**: $\min_{v} \text{Slack}(v)$.
- **Total Negative Slack (TNS)**: $\sum_{v, \text{Slack}(v) < 0} \text{Slack}(v)$.

#### 5. Critical Path Backtracking
Backtracking from the endpoint pin with minimum slack along the worst-slack fanin edges extracts the exact physical timing critical path.

---

## 3. Directory Layout & Module Structure

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
