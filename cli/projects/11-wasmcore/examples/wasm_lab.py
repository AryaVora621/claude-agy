"""
WasmCore: Interactive Terminal Demonstration & WebAssembly Laboratory.
Demonstrates bytecode assembly, Sieve of Eratosthenes prime generation in linear memory,
host callback linking, visual disassembly, and live execution tracing.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wasmcore.types import ValType, ExportDesc, Limits
from wasmcore.opcodes import Opcode
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser
from wasmcore.instance import WasmInstance
from wasmcore.interpreter import WasmInterpreter
from wasmcore.visualizer import WasmDisassembler, MemoryViewer, WasmDebugger


def build_sieve_module() -> bytes:
    """
    Builds a WebAssembly module that computes the Sieve of Eratosthenes
    for primes up to max_n in linear memory.
    Memory layout:
      data[i] = 1 if i is prime, 0 if composite
    Returns total prime count.
    """
    builder = WasmModuleBuilder()
    builder.add_memory(min_pages=1, max_pages=1)

    t_sieve = builder.add_type([ValType.I32], [ValType.I32])
    fn = builder.add_function(t_sieve, name="sieve")
    # locals:
    # 0: max_n (param)
    # 1: i (loop counter)
    # 2: j (composite multiple counter)
    # 3: prime_count
    fn.add_locals(3, ValType.I32)

    # 1. Initialize memory [2..max_n] to 1 (mark as prime candidates)
    # i = 2
    fn.i32_const(2).local_set(1)

    # loop init:
    fn.block()
    fn.loop()
    # if i > max_n break
    fn.local_get(1).local_get(0).i32_gt_s().br_if(1)
    # store 1 at addr i
    fn.local_get(1).i32_const(1).emit(Opcode.I32_STORE8)
    # i += 1
    fn.local_get(1).i32_const(1).i32_add().local_set(1)
    fn.br(0)
    fn.end()  # end loop
    fn.end()  # end block

    # 2. Sieve loop: for p = 2; p * p <= max_n; p++
    fn.i32_const(2).local_set(1)  # p = 2
    fn.block()
    fn.loop()
    # if p * p > max_n break
    fn.local_get(1).local_get(1).i32_mul().local_get(0).i32_gt_s().br_if(1)

    # if mem[p] == 1: mark multiples
    fn.local_get(1).emit(Opcode.I32_LOAD8_U)
    fn.i32_const(1).i32_eq()
    fn.if_()
    # j = p * p
    fn.local_get(1).local_get(1).i32_mul().local_set(2)
    fn.block()
    fn.loop()
    # if j > max_n break
    fn.local_get(2).local_get(0).i32_gt_s().br_if(1)
    # mem[j] = 0
    fn.local_get(2).i32_const(0).emit(Opcode.I32_STORE8)
    # j += p
    fn.local_get(2).local_get(1).i32_add().local_set(2)
    fn.br(0)
    fn.end()  # end loop
    fn.end()  # end block
    fn.end()  # end if

    # p += 1
    fn.local_get(1).i32_const(1).i32_add().local_set(1)
    fn.br(0)
    fn.end()  # end loop
    fn.end()  # end block

    # 3. Count primes: for i = 2; i <= max_n; i++ if mem[i] == 1: count++
    fn.i32_const(2).local_set(1)
    fn.i32_const(0).local_set(3)  # prime_count = 0
    fn.block()
    fn.loop()
    # if i > max_n break
    fn.local_get(1).local_get(0).i32_gt_s().br_if(1)
    # if mem[i] == 1: count++
    fn.local_get(1).emit(Opcode.I32_LOAD8_U)
    fn.i32_const(1).i32_eq()
    fn.if_()
    fn.local_get(3).i32_const(1).i32_add().local_set(3)
    fn.end()
    # i += 1
    fn.local_get(1).i32_const(1).i32_add().local_set(1)
    fn.br(0)
    fn.end()  # end loop
    fn.end()  # end block

    # Return prime count
    fn.local_get(3)
    fn.end()  # end func

    builder.add_export("sieve", ExportDesc.FUNC, fn.func_idx)
    builder.add_export("memory", ExportDesc.MEM, 0)
    return builder.build()


def run_demo():
    print("==================================================================")
    print("   WasmCore: First-Principles WebAssembly MVP Runtime Lab        ")
    print("==================================================================")

    # 1. Assembling Sieve module
    print("\n[1] Assembling Sieve of Eratosthenes WASM Module...")
    wasm_bytes = build_sieve_module()
    print(f"    Compiled binary size: {len(wasm_bytes)} bytes")
    print(f"    Header magic: {wasm_bytes[:4]} (version: {list(wasm_bytes[4:8])})")

    # 2. Decoding with parser
    print("\n[2] Parsing and Validating WASM Module Structure...")
    parser = WasmParser(wasm_bytes)
    module = parser.parse()
    print(f"    Types: {len(module.types)}, Functions: {len(module.functions)}, Memories: {len(module.memories)}")
    print(f"    Exports: {[e.name for e in module.exports]}")

    # 3. Disassembly View
    print("\n[3] WebAssembly Disassembly Output:")
    disasm = WasmDisassembler.disassemble_module(module)
    # Print first 25 lines of disassembly
    disasm_preview = "\n".join(disasm.splitlines()[:28])
    print(disasm_preview)
    print("    ... [remaining instructions omitted for brevity] ...")

    # 4. Instantiation & Execution
    print("\n[4] Instantiating and Executing in Virtual Machine...")
    instance = WasmInstance(module)
    interp = WasmInterpreter(instance)

    max_n = 100
    t0 = time.perf_counter()
    prime_count = interp.invoke("sieve", max_n)
    elapsed = time.perf_counter() - t0

    print(f"    sieve({max_n}) completed in {elapsed * 1000:.2f} ms")
    print(f"    Total primes found in [2..{max_n}]: {prime_count} (Expected: 25)")

    # 5. Extract Primes from Linear Memory
    mem = instance.get_export("memory")
    primes = [i for i in range(2, max_n + 1) if mem.data[i] == 1]
    print(f"    Primes in linear memory: {primes[:15]}... ({len(primes)} total)")

    # 6. Memory Hexdump
    print("\n[5] Linear Memory Buffer Hexdump (offsets 0x00 to 0x70):")
    dump = MemoryViewer.hexdump(mem, offset=0, length=112)
    print(dump)

    # 7. Host Function Callback Demonstration
    print("\n[6] Host Import & Two-Way Python Interop Demo:")
    host_builder = WasmModuleBuilder()
    t_fn = host_builder.add_type([ValType.I32], [ValType.I32])
    imp_fn = host_builder.add_import_func("host", "on_event", t_fn)
    caller_fn = host_builder.add_function(t_fn, name="trigger_event")
    caller_fn.local_get(0).call(imp_fn).i32_const(10).i32_add().end()
    host_builder.add_export("trigger_event", ExportDesc.FUNC, caller_fn.func_idx)

    host_module = WasmParser(host_builder.build()).parse()
    events_logged = []

    def python_callback(val: int) -> int:
        events_logged.append(val)
        return val * 2

    host_instance = WasmInstance(host_module, imports={"host": {"on_event": python_callback}})
    host_interp = WasmInterpreter(host_instance)

    res = host_interp.invoke("trigger_event", 21)
    print(f"    WASM called Python host callback with 21 -> returned {res - 10}")
    print(f"    WASM completed execution with result: {res} (21 * 2 + 10 = 52)")
    print(f"    Host received event log: {events_logged}")

    # 8. Interactive Debugger State Snapshot
    print("\n[7] Debugger Runtime State Snapshot:")
    dbg = WasmDebugger(instance)
    print(dbg.snapshot_state())

    print("\n==================================================================")
    print("   WasmCore Showcase Laboratory Completed Successfully!          ")
    print("==================================================================")


if __name__ == "__main__":
    run_demo()
