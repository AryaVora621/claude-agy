#!/usr/bin/env python3
"""
AetherVM Pipeline Demo:
Demonstrates the full end-to-end compilation and execution flow:
Source Code -> Lexical Tokens -> AST -> Raw SSA -> SSA Optimizations -> 16-Reg Machine Bytecode -> VM Execution & Profiling.
"""

import sys
import os

# Ensure local module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aethervm.visualizer import render_pipeline_view


SAMPLE_PROGRAM_1 = """
fn fib(n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

fn main() {
    return fib(12);
}
"""

SAMPLE_PROGRAM_2 = """
fn optimize_me() {
    // Constant folding test
    let a = 10 + 20 * 2;
    // Common subexpression test
    let x = a + 5;
    let y = a + 5;
    // Dead variable test
    let unused = 999 * 123;
    return x + y;
}
"""

SAMPLE_PROGRAM_3 = """
fn collatz(n) {
    let steps = 0;
    let x = n;
    while (x > 1) {
        if (x % 2 == 0) {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        steps = steps + 1;
    }
    return steps;
}
"""


def main():
    print("=" * 80)
    print(" DEMO 1: Recursive Fibonacci Execution & SSA Translation")
    print("=" * 80)
    print(render_pipeline_view(SAMPLE_PROGRAM_1, entry_fn="main"))

    print("\n" + "=" * 80)
    print(" DEMO 2: Optimization Passes (Constant Folding + CSE + DCE)")
    print("=" * 80)
    print(render_pipeline_view(SAMPLE_PROGRAM_2, entry_fn="optimize_me"))

    print("\n" + "=" * 80)
    print(" DEMO 3: While Loop & Conditionals (Collatz Sequence for n = 27)")
    print("=" * 80)
    print(render_pipeline_view(SAMPLE_PROGRAM_3, entry_fn="collatz", args=[27]))


if __name__ == "__main__":
    main()
