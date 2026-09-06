"""
WasmCore: Comprehensive Performance Benchmark Suite.
Measures LEB128 codecs, binary parsing, bytecode execution, and memory throughput.
"""

import time
import io
import os
import sys

# Ensure wasmcore is importable from benchmarks directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wasmcore.types import ValType, ExportDesc, Limits
from wasmcore.opcodes import Opcode
from wasmcore.leb128 import encode_u32, decode_u32, encode_i32, decode_i32
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser
from wasmcore.instance import WasmInstance
from wasmcore.interpreter import WasmInterpreter


def bench_leb128(iterations: int = 100000) -> float:
    t0 = time.perf_counter()
    for i in range(iterations):
        val = (i * 31337) & 0xFFFFFFFF
        enc = encode_u32(val)
        dec = decode_u32(io.BytesIO(enc))
    elapsed = time.perf_counter() - t0
    ops_per_sec = iterations / elapsed
    print(f"  LEB128 u32 Encode/Decode: {iterations:,} ops in {elapsed:.4f}s ({ops_per_sec:,.0f} ops/sec)")
    return ops_per_sec


def bench_parser_emitter(iterations: int = 2000) -> float:
    builder = WasmModuleBuilder()
    builder.add_memory(min_pages=1, max_pages=2)
    t_idx = builder.add_type([ValType.I32, ValType.I32], [ValType.I32])
    fn = builder.add_function(t_idx, name="add")
    fn.local_get(0).local_get(1).i32_add().end()
    builder.add_export("add", ExportDesc.FUNC, fn.func_idx)
    builder.add_data(0, 16, b"Benchmark Data Buffer")
    wasm_bytes = builder.build()

    t0 = time.perf_counter()
    for _ in range(iterations):
        module = WasmParser(wasm_bytes).parse()
    elapsed = time.perf_counter() - t0
    ops_per_sec = iterations / elapsed
    print(f"  Binary Parser Throughput: {iterations:,} modules in {elapsed:.4f}s ({ops_per_sec:,.0f} modules/sec)")
    return ops_per_sec


def bench_recursive_fibonacci() -> float:
    builder = WasmModuleBuilder()
    type_idx = builder.add_type([ValType.I32], [ValType.I32])

    fn = builder.add_function(type_idx, name="fib")
    fn.local_get(0)
    fn.i32_const(1)
    fn.i32_le_s()
    fn.if_(ValType.I32)
    fn.local_get(0)
    fn.else_()
    fn.local_get(0)
    fn.i32_const(1)
    fn.i32_sub()
    fn.call(fn.func_idx)
    fn.local_get(0)
    fn.i32_const(2)
    fn.i32_sub()
    fn.call(fn.func_idx)
    fn.i32_add()
    fn.end()
    fn.end()
    builder.add_export("fib", ExportDesc.FUNC, fn.func_idx)

    instance = WasmInstance(WasmParser(builder.build()).parse())
    interp = WasmInterpreter(instance)

    # Warmup
    interp.invoke("fib", 10)

    n = 20
    t0 = time.perf_counter()
    res = interp.invoke("fib", n)
    elapsed = time.perf_counter() - t0
    print(f"  Recursive fib({n}) = {res}: computed in {elapsed:.4f}s ({21891 / elapsed:,.0f} calls/sec)")
    return elapsed


def bench_loop_throughput(iterations: int = 500000) -> float:
    builder = WasmModuleBuilder()
    type_idx = builder.add_type([ValType.I32], [ValType.I32])

    fn = builder.add_function(type_idx, name="count_loop")
    fn.add_locals(1, ValType.I32)  # counter
    fn.block()
    fn.loop()
    # if counter >= n break
    fn.local_get(1)
    fn.local_get(0)
    fn.i32_ge_s()
    fn.br_if(1)
    # counter += 1
    fn.local_get(1)
    fn.i32_const(1)
    fn.i32_add()
    fn.local_set(1)
    fn.br(0)
    fn.end()
    fn.end()
    fn.local_get(1)
    fn.end()
    builder.add_export("count_loop", ExportDesc.FUNC, fn.func_idx)

    instance = WasmInstance(WasmParser(builder.build()).parse())
    interp = WasmInterpreter(instance)

    t0 = time.perf_counter()
    res = interp.invoke("count_loop", iterations)
    elapsed = time.perf_counter() - t0
    instrs = iterations * 5  # ~5 instructions per iteration
    mips = (instrs / elapsed) / 1_000_000
    print(f"  Interpreter Loop ({iterations:,} iterations): {elapsed:.4f}s ({mips:.2f} MIPS)")
    return mips


def bench_memory_throughput(total_bytes: int = 1024 * 1024) -> float:
    # 1MB through linear memory
    builder = WasmModuleBuilder()
    builder.add_memory(min_pages=16, max_pages=16)  # 1MB
    type_idx = builder.add_type([ValType.I32], [ValType.I32])

    # Fill memory with sequential 32-bit ints
    fn = builder.add_function(type_idx, name="mem_fill")
    fn.add_locals(1, ValType.I32)  # offset
    fn.block()
    fn.loop()
    # if offset >= limit break
    fn.local_get(1)
    fn.local_get(0)
    fn.i32_ge_s()
    fn.br_if(1)
    # store offset at offset
    fn.local_get(1)
    fn.local_get(1)
    fn.i32_store(offset=0)
    # offset += 4
    fn.local_get(1)
    fn.i32_const(4)
    fn.i32_add()
    fn.local_set(1)
    fn.br(0)
    fn.end()
    fn.end()
    fn.local_get(1)
    fn.end()
    builder.add_export("mem_fill", ExportDesc.FUNC, fn.func_idx)

    instance = WasmInstance(WasmParser(builder.build()).parse())
    interp = WasmInterpreter(instance)

    t0 = time.perf_counter()
    interp.invoke("mem_fill", total_bytes)
    elapsed = time.perf_counter() - t0
    mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed
    print(f"  Linear Memory Fill ({total_bytes // 1024} KB): {elapsed:.4f}s ({mb_per_sec:.2f} MB/s)")
    return mb_per_sec


def main():
    print("\n=======================================================")
    print("   WasmCore: WebAssembly MVP Benchmark Suite          ")
    print("=======================================================\n")

    bench_leb128()
    bench_parser_emitter()
    bench_recursive_fibonacci()
    bench_loop_throughput()
    bench_memory_throughput()

    print("\n=======================================================")
    print("   All Benchmarks Completed Successfully!              ")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
