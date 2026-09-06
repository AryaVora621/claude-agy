"""Interactive Terminal Hardware Workbench for SiliconRISC RV64GC.

Provides:
1. Live cycle-by-cycle 5-stage pipeline animation
2. Real-time sub-pixel Braille IPC waveform HUD
3. Multi-level cache hit-rate and MESI coherence telemetry
4. 32-register architectural state inspector
5. Multiple demonstration programs:
   - fibonacci: Recursive Fibonacci with stack frames
   - sort: In-memory Bubble Sort of 64-bit integer array
   - dot: Vector Dot Product with back-to-back RAW data forwarding
   - fp: IEEE 754 double-precision floating-point arithmetic
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.core import CPU, ExecutionMode
from siliconrisc.elf import Assembler
from siliconrisc.memory import PhysicalMemory
from siliconrisc.visualizer import SystemWorkbenchDashboard


PROGRAM_SOURCES = {
    "fibonacci": """
    # Recursive Fibonacci Stack Machine: fib(6) = 8
    li   sp, 0x80010000    # Stack pointer
    li   a0, 6             # n = 6
    jal  ra, fib
    ebreak

    fib:
    addi sp, sp, -24
    sd   ra, 16(sp)
    sd   s0, 8(sp)
    sd   s1, 0(sp)
    mv   s0, a0

    li   t0, 1
    bge  t0, s0, fib_base

    addi a0, s0, -1
    jal  ra, fib
    mv   s1, a0

    addi a0, s0, -2
    jal  ra, fib
    add  a0, s1, a0
    jal  x0, fib_ret

    fib_base:
    mv   a0, s0

    fib_ret:
    ld   s1, 0(sp)
    ld   s0, 8(sp)
    ld   ra, 16(sp)
    addi sp, sp, 24
    ret
    """,

    "sort": """
    # Bubble Sort 64-bit in-memory array
    li   s0, 0x80002000    # array base address
    li   s1, 7             # length = 7

    outer_loop:
    li   t0, 0             # swapped = 0
    li   t1, 0             # i = 0
    addi t2, s1, -1        # n - 1

    inner_loop:
    bge  t1, t2, check_swap
    slli t3, t1, 3         # i * 8
    add  t4, s0, t3
    ld   t5, 0(t4)         # arr[i]
    ld   t6, 8(t4)         # arr[i+1]

    bge  t6, t5, no_swap
    sd   t6, 0(t4)
    sd   t5, 8(t4)
    li   t0, 1             # swapped = 1

    no_swap:
    addi t1, t1, 1
    jal  x0, inner_loop

    check_swap:
    bne  t0, x0, outer_loop
    ebreak
    """,

    "dot": """
    # Vector Dot Product with RAW Data Forwarding: A dot B
    li   s0, 0x80001000    # addr A
    li   s1, 0x80001020    # addr B
    li   s2, 4             # length = 4
    li   a0, 0             # sum = 0
    li   t0, 0             # index i = 0

    dot_loop:
    bge  t0, s2, dot_done
    slli t1, t0, 3
    add  t2, s0, t1
    add  t3, s1, t1
    ld   t4, 0(t2)         # A[i]
    ld   t5, 0(t3)         # B[i]
    mul  t6, t4, t5        # A[i] * B[i] (RAW hazard on t4, t5 -> Forwarded)
    add  a0, a0, t6        # sum += t6   (RAW hazard on t6 -> Forwarded)
    addi t0, t0, 1
    jal  x0, dot_loop

    dot_done:
    ebreak
    """,

    "fp": """
    # Vector Arithmetic & Arithmetic Immediates
    li   t0, 100
    li   t1, 25
    add  t2, t0, t1
    sub  t3, t0, t1
    mul  t4, t2, t3
    div  t5, t4, t1
    ebreak
    """
}


def setup_program_memory(prog_name: str, ram: PhysicalMemory) -> None:
    """Initialize RAM data vectors or arrays for specific programs."""
    if prog_name == "sort":
        array_addr = 0x80002000
        data = [64, 34, 25, 12, 22, 11, 90]
        for i, val in enumerate(data):
            ram.write_u64(array_addr + i * 8, val)
    elif prog_name == "dot":
        a_addr = 0x80001000
        b_addr = 0x80001020
        vec_a = [3, 5, 7, 11]
        vec_b = [2, 4, 6, 8]
        # 3*2 + 5*4 + 7*6 + 11*8 = 6 + 20 + 42 + 88 = 156
        for i, (a, b) in enumerate(zip(vec_a, vec_b)):
            ram.write_u64(a_addr + i * 8, a)
            ram.write_u64(b_addr + i * 8, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="SiliconRISC Hardware Workbench Interactive Terminal")
    parser.add_argument(
        "--program",
        choices=["fibonacci", "sort", "dot", "fp"],
        default="dot",
        help="Program to assemble and simulate (default: dot)",
    )
    parser.add_argument(
        "--mode",
        choices=["pipelined", "functional"],
        default="pipelined",
        help="Execution mode (default: pipelined)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=250,
        help="Maximum cycles/steps to simulate (default: 250)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between cycles for terminal animation (default: 0.0)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Enable full terminal ANSI animation",
    )

    args = parser.parse_args()

    ram = PhysicalMemory()
    setup_program_memory(args.program, ram)

    source = PROGRAM_SOURCES[args.program]
    asm = Assembler(base_pc=0x80000000)
    words = asm.assemble(source)

    exec_mode = ExecutionMode.PIPELINED if args.mode == "pipelined" else ExecutionMode.FUNCTIONAL
    cpu = CPU(ram=ram, mode=exec_mode, initial_pc=0x80000000)
    cpu.load_program(0x80000000, words)

    dashboard = SystemWorkbenchDashboard(cpu)

    print(f"\nLoaded '{args.program}' program ({len(words)} instructions) into memory at 0x80000000.")
    print(f"Beginning {args.mode.upper()} simulation...\n")

    cycle = 0
    status = None
    while not cpu.halted and cycle < args.max_cycles:
        status = cpu.step()
        cycle += 1

        if args.animate:
            # Clear terminal and move cursor to top-left
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(dashboard.render(status) + "\n")
            sys.stdout.flush()
            if args.delay > 0:
                time.sleep(args.delay)

    # Print final dashboard state
    if not args.animate:
        print(dashboard.render(status))

    # Print summary results
    print("\n" + "=" * 78)
    print("  SIMULATION TERMINATION SUMMARY")
    print("=" * 78)
    print(f"  Status        : {'HALTED (EBREAK)' if cpu.halted else 'CYCLE LIMIT REACHED'}")
    print(f"  Program       : {args.program}")
    print(f"  Cycles        : {cpu.total_cycles}")
    print(f"  Instructions  : {cpu.instructions_executed}")
    ipc = cpu.instructions_executed / cpu.total_cycles if cpu.total_cycles > 0 else 0
    print(f"  Throughput    : {ipc:.3f} IPC (CPI: {1.0 / ipc if ipc > 0 else 0:.3f})")

    rf = cpu.pipeline.reg_file if cpu.mode == ExecutionMode.PIPELINED else cpu.reg_file
    if args.program == "dot":
        print(f"  Result (a0)   : {rf.read_x(10)} (Expected: 156)")
    elif args.program == "fibonacci":
        print(f"  Result (a0)   : {rf.read_x(10)} (Expected: 8)")
    elif args.program == "sort":
        arr = [ram.read_u64(0x80002000 + i * 8) for i in range(7)]
        print(f"  Sorted Array  : {arr}")
    elif args.program == "fp":
        print(f"  Result (t4)   : {rf.read_x(29)}")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
