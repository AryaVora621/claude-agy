#!/usr/bin/env python3
"""
AetherVM Throughput & Execution Benchmarks:
Evaluates instruction execution rate (MIPS), recursive function dispatch overhead,
and loop processing performance.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer
from aethervm.codegen import BytecodeEmitter
from aethervm.vm import VirtualMachine


def compile_and_run(source: str, entry_fn: str = "main", args=None):
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse_program()
    ir = SSABuilder().build_program(ast)
    Optimizer().optimize_program(ir)
    emitter = BytecodeEmitter()
    compiled = [emitter.compile_function(fn) for fn in ir.functions.values()]
    vm = VirtualMachine(max_instructions=50_000_000)
    vm.load_program(compiled)

    start_time = time.perf_counter()
    result = vm.run(entry_function=entry_fn, args=args)
    elapsed = time.perf_counter() - start_time

    return result, vm.instruction_count, elapsed, vm


def main():
    print("=" * 70)
    print("           AETHERVM INSTRUCTION & EXECUTION BENCHMARKS")
    print("=" * 70)

    # Benchmark 1: Recursive Fibonacci (Deep activation frames and call dispatch)
    fib_source = """
    fn fib(n) {
        if (n <= 1) {
            return n;
        }
        return fib(n - 1) + fib(n - 2);
    }
    """
    print("\n[Benchmark 1] Recursive Fibonacci fib(20):")
    res, inst_count, elapsed, vm = compile_and_run(fib_source, entry_fn="fib", args=[20])
    mips = (inst_count / elapsed) / 1_000_000 if elapsed > 0 else 0
    print(f"  Result                : {res} (Expected: 6765)")
    print(f"  Execution Time        : {elapsed * 1000:.2f} ms")
    print(f"  Total Instructions    : {inst_count:,}")
    print(f"  Instruction Rate      : {mips:.2f} MIPS ({inst_count / elapsed:,.0f} ops/sec)")

    # Benchmark 2: Tight While Loop (Summation 1..100,000)
    loop_source = """
    fn sum_loop(limit) {
        let sum = 0;
        let i = 0;
        while (i < limit) {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
    """
    limit = 50_000
    print(f"\n[Benchmark 2] Tight Summation Loop (limit = {limit:,}):")
    res, inst_count, elapsed, vm = compile_and_run(loop_source, entry_fn="sum_loop", args=[limit])
    mips = (inst_count / elapsed) / 1_000_000 if elapsed > 0 else 0
    expected_sum = (limit - 1) * limit // 2
    print(f"  Result                : {res:,} (Expected: {expected_sum:,})")
    print(f"  Execution Time        : {elapsed * 1000:.2f} ms")
    print(f"  Total Instructions    : {inst_count:,}")
    print(f"  Instruction Rate      : {mips:.2f} MIPS ({inst_count / elapsed:,.0f} ops/sec)")

    # Benchmark 3: Collatz Hailstone Search (Iterate starting points 1..200)
    collatz_source = """
    fn collatz_max(start_limit) {
        let max_steps = 0;
        let n = 1;
        while (n < start_limit) {
            let val = n;
            let steps = 0;
            while (val > 1) {
                if (val % 2 == 0) {
                    val = val / 2;
                } else {
                    val = 3 * val + 1;
                }
                steps = steps + 1;
            }
            if (steps > max_steps) {
                max_steps = steps;
            }
            n = n + 1;
        }
        return max_steps;
    }
    """
    collatz_limit = 100
    print(f"\n[Benchmark 3] Collatz Hailstone Search (n = 1..{collatz_limit}):")
    res, inst_count, elapsed, vm = compile_and_run(collatz_source, entry_fn="collatz_max", args=[collatz_limit])
    mips = (inst_count / elapsed) / 1_000_000 if elapsed > 0 else 0
    print(f"  Max Steps Found       : {res}")
    print(f"  Execution Time        : {elapsed * 1000:.2f} ms")
    print(f"  Total Instructions    : {inst_count:,}")
    print(f"  Instruction Rate      : {mips:.2f} MIPS ({inst_count / elapsed:,.0f} ops/sec)")

    print("\n" + "=" * 70)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
