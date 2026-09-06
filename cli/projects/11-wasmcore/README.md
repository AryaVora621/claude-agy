# WasmCore: WebAssembly MVP Virtual Machine and Binary Toolchain

A production-grade, spec-compliant WebAssembly (WASM) MVP virtual machine, binary bytecode parser, fluent assembler/emitter, and interactive stack debugger implemented from first principles in the pure Python standard library with zero external dependencies.

---

## 1. Architectural Blueprint

```
                      +------------------------------------------+
                      |         WebAssembly Binary (.wasm)       |
                      |   Magic: \x00asm | Version: 0x00000001   |
                      +--------------------+---------------------+
                                           |
                                           v
                             +---------------------------+
                             |   LEB128 VarInt Codec     |
                             |   u32, i32, i64, Names    |
                             +-------------+-------------+
                                           |
                                           v
                             +---------------------------+
                             |     WasmParser (AST)      |
                             | 11 Standard WASM Sections |
                             +-------------+-------------+
                                           |
                                           v
                             +---------------------------+
                             |      WasmInstance         |
                             |  Functions, Tables, Mems, |
                             |  Globals & Host Callbacks |
                             +-------------+-------------+
                                           |
                                           v
                    +----------------------------------------------+
                    |           WasmInterpreter                    |
                    |                                              |
                    |  +------------------+  +------------------+  |
                    |  |  Operand Stack   |  |   Control Stack  |  |
                    |  |  (Value Box)     |  | (Blocks & Loops) |  |
                    |  +------------------+  +------------------+  |
                    |  +------------------+  +------------------+  |
                    |  |    Call Stack    |  |  Linear Memory   |  |
                    |  |  (CallFrames)    |  |  (64KB Paged)    |  |
                    |  +------------------+  +------------------+  |
                    +----------------------------------------------+
```

---

## 2. Core Capabilities & Specification Compliance

1. **W3C Standard WebAssembly MVP Binary Serialization**:
   - Magic header validation (`\x00asm`) and versioning (`0x00000001`).
   - Binary section multiplexing: Type (1), Import (2), Function (3), Table (4), Memory (5), Global (6), Export (7), Start (8), Element (9), Code (10), Data (11).
   - Custom section support for debug names and embedded metadata.

2. **LEB128 Variable-Length Integer Codecs**:
   - High-throughput unsigned (`uleb128`) and signed two's complement (`sleb128`) 32-bit and 64-bit integer encoding/decoding.
   - Stream safety bounds checks with integer overflow trapping past maximum bit widths.

3. **Strongly Typed Value System & Bitcasting**:
   - First-class boxing for all 4 WebAssembly numeric primitives: `i32`, `i64`, `f32`, and `f64`.
   - IEEE 754 bitcast reinterpretations between integers and floating point values (`i32.reinterpret_f32`, `f64.reinterpret_i64`).
   - Two's complement wraparound arithmetic matching hardware CPU registers.

4. **64KB Paged Linear Memory & Protection**:
   - Address space managed in WebAssembly standard 64KB (65,536 bytes) page chunks.
   - Dynamic memory growth via `memory.grow` and capacity inspection via `memory.size`.
   - Little-endian sub-word accessors: `i32.load8_s/u`, `i32.load16_s/u`, `i64.load32_s/u`, and corresponding stores with alignment offsets.
   - Trap detection on out-of-bounds memory accesses.

5. **Stack Machine Interpreter & Control Flow**:
   - Call stack activation records (`CallFrame`) tracking local variables and instruction pointers.
   - Structured label scopes (`block`, `loop`, `if`/`else`, `end`) supporting arbitrary nested relative depth branching (`br`, `br_if`, `br_table`).
   - Operand stack unwinding preserving block result signatures.
   - Two-way host function calling and parameter marshalling.

---

## 3. Directory Layout

```
projects/11-wasmcore/
|-- wasmcore/
|   |-- __init__.py          # Public package API exports
|   |-- types.py             # ValType, FuncType, Limits, Value, and WasmTrap
|   |-- leb128.py            # Variable-length integer codecs
|   |-- opcodes.py           # Opcode definitions, mnemonics, and Instruction
|   |-- module.py            # Abstract syntax tree for WebAssembly modules
|   |-- parser.py            # Binary WASM module decoder
|   |-- emitter.py           # Fluent programmatic bytecode assembler
|   |-- memory.py            # 64KB paged linear memory buffer
|   |-- instance.py          # Module instantiation and host linking
|   |-- interpreter.py       # Stack machine execution engine
|   `-- visualizer.py        # Terminal disassembler, hexdump, and debugger
|-- tests/
|   |-- test_types.py        # Value boxing and bitcast tests
|   |-- test_leb128.py       # LEB128 codec round-trip tests
|   |-- test_parser_emitter.py # Binary serialization tests
|   |-- test_memory.py       # Linear memory and paging tests
|   |-- test_instance.py     # Host linking and export tests
|   |-- test_interpreter.py  # Bytecode execution, loops, and traps
|   `-- test_visualizer.py   # Disassembly and debugger tests
|-- benchmarks/
|   `-- bench_wasm.py        # Performance benchmark suite
|-- examples/
|   `-- wasm_lab.py          # Sieve of Eratosthenes demo in linear memory
|-- PLAN.md                  # System architecture specification
`-- README.md                # System documentation
```

---

## 4. Benchmark Performance

Measured on Apple Silicon (macOS Darwin, Python 3.13 stdlib):

| Benchmark Stage | Workload | Latency / Throughput |
|---|---|---|
| **LEB128 Codec** | 100,000 u32 encodes and decodes | 973,619 ops/sec |
| **Binary Parser** | 2,000 complete module decodes | 85,400 modules/sec |
| **Recursive Fibonacci** | `fib(20) = 6765` recursion | 121,494 calls/sec |
| **Interpreter Loop** | 500,000 loop iterations | 0.84 MIPS |
| **Linear Memory Access** | 1024 KB sequential 32-bit stores | 0.49 MB/sec |

---

## 5. Verification & Test Suite

All 34 unit tests pass with zero external dependencies:

```bash
cd projects/11-wasmcore
python3 -m unittest discover tests
# Ran 34 tests in 0.005s -> OK
```

To run the interactive demonstration:

```bash
python3 examples/wasm_lab.py
```
