# AetherVM: Register-Based SSA Optimizing Compiler & Bytecode VM

A high-performance, zero-dependency compiler toolchain and register-based virtual machine built entirely from scratch in pure Python. AetherVM transforms a high-level imperative programming language into Static Single Assignment (SSA) intermediate representation, executes classical optimizing compiler passes, maps infinite virtual registers to 16 physical machine registers via linear scan allocation, and executes 3-address bytecode at over **4.4 Million Instructions Per Second (MIPS)**.

---

## 🌟 Architectural Overview

```
                      +-----------------------------+
                      |     High-Level Source       |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Lexer & Tokenizer      |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Pratt Expression Parser   |
                      +-----------------------------+
                                     |  Abstract Syntax Tree (AST)
                                     v
                      +-----------------------------+
                      |    Braun et al. SSA Builder |
                      |    (Phi-Nodes & CFG Blocks) |
                      +-----------------------------+
                                     |  SSA Control Flow Graph
                                     v
                 +---------------------------------------+
                 |       SSA Optimization Pipeline       |
                 | - Constant Folding & Propagation      |
                 | - Common Subexpression Elimination    |
                 | - Dead Code Elimination (Backward)    |
                 +---------------------------------------+
                                     |  Optimized SSA CFG
                                     v
                 +---------------------------------------+
                 | De-SSA & Linear Scan Register Alloc   |
                 | - Maps SSA variables to R1-R14        |
                 | - Automatic Stack Frame Spilling      |
                 | - Caller/Callee Parameter ABI Preserv |
                 +---------------------------------------+
                                     |  Linear 3-Address Bytecode
                                     v
                 +---------------------------------------+
                 |   Register-Based Virtual Machine      |
                 | - 16 Physical Registers per Frame     |
                 | - Activation Call Stack               |
                 | - Instruction Profiler & Tracer       |
                 +---------------------------------------+
```

---

## 🚀 Key Features

1. **Pratt Operator-Precedence Parser (TDOP)**:
   - Eliminates grammar ambiguity and recursive descent bloat for arithmetic, logical, and relational operators.
   - Dynamic binding power table supporting prefix, infix, and ternary chaining.

2. **SSA Intermediate Representation (Braun et al. 2013)**:
   - Direct SSA construction from AST without prior non-SSA intermediate code or dominance frontiers calculation.
   - On-demand $\phi$-node generation and automatic phi-simplification at join blocks.

3. **Multi-Pass SSA Optimizations**:
   - **Constant Folding & Constant Propagation**: Recursively evaluates compile-time expressions and propagates literal values through dataflow chains.
   - **Common Subexpression Elimination (CSE)**: Global value numbering replaces duplicate expressions with existing definitions.
   - **Dead Code Elimination (DCE)**: Backward liveness analysis removes unused calculations and unreferenced variables.

4. **Linear Scan Register Allocation**:
   - Computes live ranges for all SSA values.
   - Maps infinite SSA variables to 16 physical registers (`R0`–`R15`).
   - Automatically generates `SPILL` and `RELOAD` instructions when live variable density exceeds physical register capacity.
   - Preserves function arguments according to standard ABI calling conventions.

5. **Register-Based 3-Address Bytecode Interpreter**:
   - Executes 3-address instructions (`OP r_dest, r_src1, r_src2`).
   - Significantly reduces VM instruction dispatch overhead compared to stack machines.
   - Measures instruction counts and provides per-opcode profiling.

6. **Full Terminal Pipeline Visualizer**:
   - ASCII visualizer displaying the transformation at each compiler phase: Source $\rightarrow$ AST $\rightarrow$ Raw SSA $\rightarrow$ Optimized SSA $\rightarrow$ Machine Bytecode $\rightarrow$ Execution Profile.

---

## 📊 Performance Benchmarks

Benchmarked on Apple Silicon (Python 3.13 standard library, single-threaded):

| Benchmark Test | Workload Description | Bytecode Instructions | Execution Time | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Recursive Fibonacci** | `fib(20)` recursive call tree | 229,850 ops | 71.18 ms | **3.23 MIPS** |
| **Tight Summation Loop** | While loop summing 1 to 50,000 | 450,010 ops | 106.41 ms | **4.23 MIPS** |
| **Collatz Hailstone** | Nested loops search for $n=1..100$ | 97,554 ops | 21.94 ms | **4.45 MIPS** |

---

## 🛠 Quickstart

### 1. Run Complete Compiler Pipeline Visualizer
```bash
python3 examples/demo_pipeline.py
```

### 2. Run Benchmark Suite
```bash
python3 benchmarks/bench_vm.py
```

### 3. Run Unit Tests
```bash
python3 -m unittest discover -s tests
```

---

## 💻 Python API Usage

```python
from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer
from aethervm.codegen import BytecodeEmitter
from aethervm.vm import VirtualMachine

source_code = """
fn factorial(n) {
    let acc = 1;
    let i = n;
    while (i > 1) {
        acc = acc * i;
        i = i - 1;
    }
    return acc;
}

fn main() {
    return factorial(6);
}
"""

# 1. Lexing & Parsing
tokens = Lexer(source_code).tokenize()
ast = Parser(tokens).parse_program()

# 2. SSA Translation & Optimization
ir = SSABuilder().build_program(ast)
Optimizer().optimize_program(ir)

# 3. Register Allocation & Code Generation
emitter = BytecodeEmitter()
compiled_funcs = [emitter.compile_function(fn) for fn in ir.functions.values()]

# 4. Virtual Machine Execution
vm = VirtualMachine()
vm.load_program(compiled_funcs)
result = vm.run(entry_function="main")

print("Result:", result) # Outputs 720
print("Instructions Executed:", vm.instruction_count)
```

---

## 📜 Bytecode Opcode Specification

| Opcode | Operands | Description |
| :--- | :--- | :--- |
| `LOAD_CONST` | `R_dest, imm` | Load literal constant into register |
| `MOV` | `R_dest, R_src` | Copy register value |
| `ADD` | `R_dest, R_src1, R_src2` | `R_dest = R_src1 + R_src2` |
| `SUB` | `R_dest, R_src1, R_src2` | `R_dest = R_src1 - R_src2` |
| `MUL` | `R_dest, R_src1, R_src2` | `R_dest = R_src1 * R_src2` |
| `DIV` | `R_dest, R_src1, R_src2` | `R_dest = R_src1 / R_src2` |
| `MOD` | `R_dest, R_src1, R_src2` | `R_dest = R_src1 % R_src2` |
| `CMP_<OP>` | `R_dest, R_src1, R_src2` | Relational compare (`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`) |
| `JMP` | `target_pc` | Unconditional jump |
| `JMP_IF` | `R_src, target_pc` | Jump if `R_src` is truthy (non-zero) |
| `JMP_IF_NOT` | `R_src, target_pc` | Jump if `R_src` is falsy (zero) |
| `CALL` | `fn_name, [arg_regs] -> R_dest` | Push call frame and transfer arguments |
| `RET` | `R_src` | Pop call frame and return value |
| `SPILL` | `[Stack#slot], R_src` | Save register to spilled stack frame slot |
| `RELOAD` | `R_dest, [Stack#slot]` | Load value from spilled stack frame slot |
| `PRINT` | `R_src` | Output register value |

---

## 🔬 Directory Structure

```
projects/03-aethervm/
├── aethervm/
│   ├── __init__.py           # Public module interface
│   ├── ast_nodes.py          # AST node classes
│   ├── codegen.py            # Linear scan register allocation & bytecode emitter
│   ├── ir.py                 # SSA IR definitions (BasicBlock, Instruction, Value)
│   ├── lexer.py              # Tokenizer with positional line/col tracking
│   ├── opt.py                # Optimization passes (Constant folding, CSE, DCE)
│   ├── parser.py             # Pratt top-down operator precedence parser
│   ├── ssa_builder.py        # Braun et al. SSA construction
│   ├── visualizer.py         # Terminal pipeline visualizer
│   └── vm.py                 # Register-based bytecode virtual machine
├── benchmarks/
│   └── bench_vm.py           # Throughput & MIPS benchmarks
├── examples/
│   └── demo_pipeline.py      # End-to-end multi-stage compiler demonstration
├── tests/
│   ├── test_lexer.py         # Lexer unit tests
│   ├── test_parser.py        # Pratt parser unit tests
│   ├── test_ssa.py           # SSA construction & optimization tests
│   └── test_vm.py            # VM execution & spilling tests
└── README.md
```
