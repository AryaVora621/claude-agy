# WasmCore: WebAssembly (WASM) Binary Virtual Machine & Stack Architecture

## Architectural Vision & Goals

WasmCore is a research-grade, spec-adherent WebAssembly (WASM) virtual machine, binary bytecode decoder, program synthesis assembler, and stack machine interpreter implemented from first principles in the pure Python standard library with zero external dependencies.

WebAssembly is an open standard binary code format designed for fast, efficient, low-level execution across computing architectures. WasmCore implements the WebAssembly MVP core specification, decoding the binary format directly from raw bytes, managing 64KB paged linear memory, handling structured control flow with label-unwinding branch semantics, and supporting host function imports and indirect table dispatch.

---

## Technical Architecture & Subsystem Layout

```
+-------------------------------------------------------------------------------+
|                           WASM BINARY MODULE STREAM                           |
|       Magic Header: b"\x00asm" (0x6d736100), Version: 0x00000001              |
+-------------------------------------------------------------------------------+
                                       |
                   LEB128 & Binary Section Decoding
                                       |
+-------------------------------------------------------------------------------+
|                         WASM MODULE STRUCTURE                                 |
|  Section 1:  Type Section      (Function Signatures: params -> results)       |
|  Section 2:  Import Section    (Host Functions, Memories, Tables, Globals)    |
|  Section 3:  Function Section  (Signature Type Indices)                       |
|  Section 4:  Table Section     (Function Reference Tables)                    |
|  Section 5:  Memory Section    (Linear Memory Page Limits: min, max)          |
|  Section 6:  Global Section    (Typed Mutable / Immutable Globals)            |
|  Section 7:  Export Section    (Names for Functions, Memories, Globals)       |
|  Section 8:  Start Section     (Module Initialization Function Index)         |
|  Section 9:  Element Section   (Table Initializer Segments)                   |
|  Section 10: Code Section      (Locals Declarations + Bytecode Instructions)  |
|  Section 11: Data Section      (Passive/Active Linear Memory Segments)        |
+-------------------------------------------------------------------------------+
                                       |
                       Module Instantiation & Linking
                                       |
+-------------------------------------------------------------------------------+
|                      WASM INSTANCE RUNTIME ENVIRONMENT                        |
|  - Linear Memory: Bytearray of 64KB pages with bounds and alignment checks    |
|  - Table: Vector of function references (funcref) for call_indirect           |
|  - Globals: Typed storage cells (i32, i64, f32, f64)                          |
|  - Host Environment: Pre-linked native Python callbacks (e.g. print/trace)    |
+-------------------------------------------------------------------------------+
                                       |
                        Stack Machine Execution Engine
                                       |
+-------------------------------------------------------------------------------+
|                   STACK MACHINE INTERPRETER & CONTROL FLOW                    |
|  - Operand Value Stack: Strongly typed values (I32, I64, F32, F64)            |
|  - Call Stack: CallFrame (Locals, Return Arity, Instruction Pointer)          |
|  - Control Stack (Labels): Structured blocks (block, loop, if/else)           |
|  - Branch Resolution: Relative depth unwinding (br, br_if, br_table)          |
|  - Trap Subsystem: Out-of-bounds, Integer division by zero, Unreachable       |
+-------------------------------------------------------------------------------+
                                       |
             +-------------------------+-------------------------+
             |                                                   |
+----------------------------+              +-----------------------------------+
|   IN-MEMORY WASM EMITTER   |              |        TERMINAL VISUALIZERS       |
| - Programmatic Assembler   |              | - Interactive Disassembler        |
| - Fluent Instruction Chain |              | - Live Stack Machine Step-Tracer  |
| - Standard .wasm Compiler  |              | - ASCII Memory Hex Dump Viewer    |
+----------------------------+              +-----------------------------------+
```

---

## Detailed Component Specifications

### 1. Types & Value Representation (`wasmcore/types.py`)
- Value types:
  - `I32` (0x7F): 32-bit signed/unsigned integer.
  - `I64` (0x7E): 64-bit signed/unsigned integer.
  - `F32` (0x7D): 32-bit IEEE 754 single-precision float.
  - `F64` (0x7C): 64-bit IEEE 754 double-precision float.
- `Value`: Container storing typed value with bitcast conversions and 32-bit/64-bit wrap-around semantics.
- `FuncType`: Parameter types tuple and result types tuple `([t1, t2] -> [r1])`.
- `Limits`: Minimum and optional maximum limits for memories and tables.
- `GlobalType`: Value type and mutability flag (const vs var).

### 2. Variable-Length Integer Codecs (`wasmcore/leb128.py`)
- Standard Little-Endian Base 128 (LEB128) encoding and decoding:
  - `decode_u32(stream)` / `decode_u64(stream)`: Unsigned LEB128 up to 5/10 bytes.
  - `decode_i32(stream)` / `decode_i64(stream)`: Signed two's-complement LEB128 with sign-extension.
  - `encode_u32(val)` / `encode_u64(val)`: Unsigned LEB128 encoder.
  - `encode_i32(val)` / `encode_i64(val)`: Signed LEB128 encoder.

### 3. Opcode Matrix (`wasmcore/opcodes.py`)
- Control flow: `unreachable` (0x00), `nop` (0x01), `block` (0x02), `loop` (0x03), `if` (0x04), `else` (0x05), `end` (0x0B), `br` (0x0C), `br_if` (0x0D), `br_table` (0x0E), `return` (0x0F), `call` (0x10), `call_indirect` (0x11).
- Parametric: `drop` (0x1A), `select` (0x1B).
- Variable access: `local.get` (0x20), `local.set` (0x21), `local.tee` (0x22), `global.get` (0x23), `global.set` (0x24).
- Memory operations:
  - `i32.load` (0x28), `i64.load` (0x29), `f32.load` (0x2A), `f64.load` (0x2B).
  - `i32.load8_s` (0x2C), `i32.load8_u` (0x2D), `i32.load16_s` (0x2E), `i32.load16_u` (0x2F).
  - `i32.store` (0x36), `i64.store` (0x37), `f32.store` (0x38), `f64.store` (0x39).
  - `i32.store8` (0x3A), `i32.store16` (0x3B).
  - `memory.size` (0x3F), `memory.grow` (0x40).
- Constants: `i32.const` (0x41), `i64.const` (0x42), `f32.const` (0x43), `f64.const` (0x44).
- Integer arithmetic & comparisons:
  - Comparison: `eqz`, `eq`, `ne`, `lt_s`, `lt_u`, `gt_s`, `gt_u`, `le_s`, `le_u`, `ge_s`, `ge_u`.
  - Arithmetic: `clz`, `ctz`, `popcnt`, `add`, `sub`, `mul`, `div_s`, `div_u`, `rem_s`, `rem_u`, `and`, `or`, `xor`, `shl`, `shr_s`, `shr_u`, `rotl`, `rotr`.
- Type conversions & reinterpretations: `wrap`, `extend`, `trunc`, `convert`, `reinterpret`.

### 4. Binary Module Parser (`wasmcore/parser.py`)
- Reads raw binary `.wasm` format:
  - Checks magic bytes `\x00asm` and version `1`.
  - Iterates through binary section IDs (1-12) with LEB128 payload sizes.
  - Decodes types, imports, functions, tables, memories, globals, exports, start, elements, codes, and data segments.
  - Disassembles bytecode instructions into strongly-typed `Instruction` tokens.

### 5. In-Memory Assembler & Emitter (`wasmcore/emitter.py`)
- Fluent Python DSL for constructing valid `.wasm` binary files in memory:
  ```python
  module = WasmModuleBuilder()
  # Define type
  ftype = module.add_type([ValType.I32, ValType.I32], [ValType.I32])
  # Define function
  fn = module.add_function("add", ftype)
  fn.emit(Opcode.LOCAL_GET, 0)
  fn.emit(Opcode.LOCAL_GET, 1)
  fn.emit(Opcode.I32_ADD)
  fn.emit(Opcode.END)
  module.add_export("add", ExportDesc.FUNC, fn.index)
  raw_bytes = module.build()
  ```

### 6. Linear Memory & Paging Subsystem (`wasmcore/memory.py`)
- WebAssembly page size: 65,536 bytes (64 KB).
- Dynamic page growth: `memory.grow` with max page cap validation.
- Endianness: Little-endian load and store operations with byte-level addressing.
- Out-of-bounds trap detection and alignment checks.

### 7. Stack Machine Execution Engine (`wasmcore/interpreter.py`)
- Structured control flow handling:
  - `block`: Label with continuation target at the matching `end`.
  - `loop`: Label with continuation target at the start of the loop.
  - `if`/`else`: Branching based on top-of-stack boolean condition.
  - `br $depth`: Unwinds $depth$ labels, restoring operand stack state.
  - `br_if $depth`: Conditional branch.
  - `br_table`: Multi-way jump table with default fallback.
  - `call` and `call_indirect`: Activation frame setup, parameter binding, result return.
- Trap safety: Captures integer division by zero, floating point invalid operations, memory out-of-bounds, stack exhaustion, and `unreachable` instructions.

### 8. Interactive Debugger & Visualizer (`wasmcore/visualizer.py`)
- Disassembler output formatting with indentation levels matching block nesting.
- Step-by-step debugger tracking instruction pointer, stack changes, and local variables.
- ASCII linear memory hex dump viewer with ASCII character panel.

---

## Verification & Testing Plan

1. **Unit Tests (`tests/`)**:
   - `test_leb128.py`: Validates signed and unsigned variable-length integer codecs against edge cases (0, -1, max uint32, max int64).
   - `test_types.py`: Value bitcasts, IEEE-754 floats, wrapping arithmetic.
   - `test_parser_emitter.py`: Round-trip module construction, encoding, and parsing.
   - `test_memory.py`: Loads, stores, signed/unsigned conversions, page growing, out-of-bounds traps.
   - `test_control_flow.py`: Blocks, loops, if-else, break depth, break tables, recursion.
   - `test_algorithms.py`: Fibonacci, factorial, prime sieve in linear memory, array bubble sort.
2. **Benchmark Suite (`benchmarks/bench_wasm.py`)**:
   - Instruction execution rate (MIPS).
   - Function call overhead (calls/sec).
   - Memory read/write throughput (MB/sec).
3. **Interactive Lab Showcase (`examples/wasm_lab.py`)**:
   - Compiling and executing Fibonacci sequence.
   - Bubble sort on linear memory buffers.
   - String formatting and printing via linear memory.
   - Disassembly and step-by-step stack visualization.
