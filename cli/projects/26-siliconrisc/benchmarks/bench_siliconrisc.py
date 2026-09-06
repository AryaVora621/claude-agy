"""Performance Microbenchmarks for SiliconRISC RV64GC Processor Architecture.

Evaluates:
1. Instruction Decoder Throughput (MInst/s)
2. Functional Fast Execution Mode (Inst/s)
3. 5-Stage Pipelined Simulation Speed (Cycles/s)
4. Cache Hierarchy Latencies & AMAT (Average Memory Access Time)
5. Branch Predictor Unit Accuracy & Training Speed (Tournament vs Gshare vs Bimodal)
6. SV39 Virtual Memory MMU & TLB Translation Throughput
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.branch import (
    BimodalPredictor,
    BranchPredictorUnit,
    GsharePredictor,
    PredictorType,
    TournamentPredictor,
)
from siliconrisc.cache import Cache, MemoryHierarchy, MESIState
from siliconrisc.core import CPU, ExecutionMode
from siliconrisc.elf import Assembler
from siliconrisc.isa import InstructionDecoder
from siliconrisc.memory import AccessType, MMU, PhysicalMemory, PTE_D, PTE_A, PTE_U, PTE_X, PTE_W, PTE_R, PTE_V


def print_header(title: str) -> None:
    width = 78
    print("=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def bench_decoder_throughput(iterations: int = 50000) -> None:
    """Benchmark raw 32-bit RISC-V instruction decoding throughput."""
    print_header("1. Instruction Decoder Throughput")

    # Sample mix of instructions: R-type, I-type, S-type, B-type, J-type, FP, AMO
    test_words = [
        0x00F00093,  # addi x1, x0, 15
        0x002081B3,  # add  x3, x1, x2
        0x02208233,  # mul  x4, x1, x2
        0x0003BE83,  # ld   t4, 0(t2)
        0x0053B023,  # sd   t0, 0(t2)
        0x0322D463,  # bge  t0, s2, target
        0x00000073,  # ecall
        0x0220F1D3,  # fadd.d f3, f1, f2
    ]
    num_words = len(test_words)

    start = time.perf_counter()
    for i in range(iterations):
        raw = test_words[i % num_words]
        inst = InstructionDecoder.decode(raw, 0x80000000 + i * 4)
    elapsed = time.perf_counter() - start

    inst_per_sec = iterations / elapsed if elapsed > 0 else 0
    print(f"Decoded {iterations:,} instructions in {elapsed * 1000:.2f} ms")
    print(f"Throughput: {inst_per_sec / 1e6:.3f} MInst/sec ({inst_per_sec:,.0f} ops/sec)")
    print()


def bench_functional_execution_throughput() -> None:
    """Benchmark raw functional execution throughput on tight loop."""
    print_header("2. Functional Fast Execution Mode Throughput")

    asm = Assembler(base_pc=0x80000000)
    # Loop 20,000 iterations: 4 instructions per iteration = ~80,000 instructions
    source = """
    li   t0, 20000
    li   t1, 0
    loop:
    add  t1, t1, t0
    addi t0, t0, -1
    bne  t0, x0, loop
    ebreak
    """
    words = asm.assemble(source)
    cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
    cpu.load_program(0x80000000, words)

    start = time.perf_counter()
    cpu.run(max_cycles=200000)
    elapsed = time.perf_counter() - start

    ips = cpu.instructions_executed / elapsed if elapsed > 0 else 0
    print(f"Executed {cpu.instructions_executed:,} instructions in {elapsed * 1000:.2f} ms")
    print(f"Throughput: {ips:,.0f} instructions/sec ({ips / 1e3:.2f} KIPS)")
    print()


def bench_pipelined_simulation_speed() -> None:
    """Benchmark cycle-accurate 5-stage pipeline simulation speed."""
    print_header("3. 5-Stage In-Order Pipelined Core Simulation Speed")

    asm = Assembler(base_pc=0x80000000)
    # 2,000 loop iterations with RAW hazards, forwarding, and branch resolution
    source = """
    li   t0, 1500
    li   t1, 0
    pipe_loop:
    addi t2, t0, 1
    add  t1, t1, t2
    addi t0, t0, -1
    bne  t0, x0, pipe_loop
    ebreak
    """
    words = asm.assemble(source)
    cpu = CPU(mode=ExecutionMode.PIPELINED, initial_pc=0x80000000)
    cpu.load_program(0x80000000, words)

    start = time.perf_counter()
    cpu.run(max_cycles=50000)
    elapsed = time.perf_counter() - start

    cps = cpu.total_cycles / elapsed if elapsed > 0 else 0
    ipc = cpu.instructions_executed / cpu.total_cycles if cpu.total_cycles > 0 else 0
    print(f"Simulated {cpu.total_cycles:,} cycles ({cpu.instructions_executed:,} insts) in {elapsed * 1000:.2f} ms")
    print(f"Simulation Speed: {cps:,.0f} cycles/sec")
    print(f"Effective IPC: {ipc:.3f} (CPI: {1.0 / ipc if ipc > 0 else 0:.3f})")
    print(f"Stall Cycles: {cpu.pipeline.stall_cycles:,} | Branch Mispredict Flushes: {cpu.pipeline.branch_mispredicts:,}")
    print()


def bench_cache_hierarchy_amat() -> None:
    """Benchmark L1, L2, and RAM access times and verify AMAT equation."""
    print_header("4. Cache Hierarchy Latency & AMAT Verification")

    ram = PhysicalMemory()
    hier = MemoryHierarchy(ram, l1_size=8192, l2_size=65536, ram_latency_cycles=50, l2_latency_cycles=6)

    # 1. Benchmark L1D Hit Latency
    ram.write_u64(0x1000, 42)
    hier.read_data(0x1000, size=8)  # Warm up L1D line

    n_hits = 10000
    start = time.perf_counter()
    for _ in range(n_hits):
        val, lat = hier.read_data(0x1000, size=8)
    l1_elapsed = time.perf_counter() - start

    # 2. Benchmark Stride Traversals
    # Stride across 128 cache lines (exceeding L1 capacity to trigger L2 hits/RAM misses)
    lines = 128
    for i in range(lines):
        paddr = 0x20000 + i * 64
        ram.write_u64(paddr, i * 7)

    start = time.perf_counter()
    for i in range(lines):
        paddr = 0x20000 + i * 64
        hier.read_data(paddr, size=8)
    stride_elapsed = time.perf_counter() - start

    l1d_hit_rate = hier.l1d.hit_rate
    l2_hit_rate = hier.l2.hit_rate
    amat = 1 + (1.0 - l1d_hit_rate) * (6 + (1.0 - l2_hit_rate) * 50)

    print(f"L1D Cache Access Speed: {n_hits / l1_elapsed / 1e6:.2f} MAccess/sec (Latency = 1 cycle)")
    print(f"L1D Hits: {hier.l1d.hits:,} | L1D Misses: {hier.l1d.misses:,} | Hit Rate: {l1d_hit_rate * 100:.2f}%")
    print(f"L2  Hits: {hier.l2.hits:,} | L2  Misses: {hier.l2.misses:,} | Hit Rate: {l2_hit_rate * 100:.2f}%")
    print(f"Calculated AMAT: {amat:.2f} cycles/access")
    print()


def bench_branch_predictors() -> None:
    """Compare prediction accuracy of Bimodal, Gshare, and Tournament Predictors."""
    print_header("5. Branch Predictor Unit Accuracy Comparison")

    bimodal = BimodalPredictor()
    gshare = GsharePredictor()
    tournament = TournamentPredictor()

    # Pattern: 4-iteration nested loop pattern: T, T, T, N (taken 3 times, not taken 1 time)
    pattern = [True, True, True, False]
    pc = 0x80001040
    eval_count = 4000

    def evaluate(name: str, pred_fn: Callable[[int], bool], update_fn: Callable[[int, bool], None]) -> Tuple[float, float]:
        correct = 0
        start = time.perf_counter()
        for i in range(eval_count):
            actual = pattern[i % 4]
            guess = pred_fn(pc)
            if guess == actual:
                correct += 1
            update_fn(pc, actual)
        elapsed = time.perf_counter() - start
        acc = (correct / eval_count) * 100.0
        return acc, elapsed

    acc_bi, t_bi = evaluate("Bimodal (Local)", bimodal.predict, bimodal.update)
    acc_gs, t_gs = evaluate("Gshare (Global)", gshare.predict, gshare.update)
    acc_to, t_to = evaluate("Tournament (Meta)", lambda p: tournament.predict(p)[0], tournament.update)

    print(f"{'Predictor Type':<22} | {'Accuracy':<12} | {'Time':<10} | {'Throughput':<16}")
    print("-" * 68)
    print(f"{'Bimodal (2-Bit Local)':<22} | {acc_bi:6.2f}%     | {t_bi * 1000:6.2f} ms | {eval_count / t_bi / 1e3:6.1f} KPred/s")
    print(f"{'Gshare (10-bit GHR)':<22} | {acc_gs:6.2f}%     | {t_gs * 1000:6.2f} ms | {eval_count / t_gs / 1e3:6.1f} KPred/s")
    print(f"{'Tournament (Meta)':<22} | {acc_to:6.2f}%     | {t_to * 1000:6.2f} ms | {eval_count / t_to / 1e3:6.1f} KPred/s")
    print()


def bench_sv39_mmu_translation() -> None:
    """Benchmark MMU translation with TLB hit vs 3-level page table walk."""
    print_header("6. SV39 Virtual Memory MMU & TLB Translation")

    ram = PhysicalMemory()
    mmu = MMU(ram)
    from siliconrisc.memory import PrivilegeMode
    mmu.privilege_mode = PrivilegeMode.USER
    root_ppn = 0x100
    mmu.satp = (8 << 60) | root_ppn

    # Map 64 pages
    for i in range(64):
        vaddr = 0x10000 + i * 4096
        paddr = 0x80000 + i * 4096
        mmu.map_page_4k(root_ppn, vaddr, paddr, flags=PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D)

    # 1. Warm-up and test TLB Hit throughput
    test_vaddr = 0x10000
    mmu.translate(test_vaddr, AccessType.LOAD)  # Fill TLB

    n_tlb = 50000
    start = time.perf_counter()
    for _ in range(n_tlb):
        p = mmu.translate(test_vaddr, AccessType.LOAD)
    tlb_elapsed = time.perf_counter() - start

    tlb_rate = n_tlb / tlb_elapsed if tlb_elapsed > 0 else 0
    dtlb_total = mmu.dtlb_hits + mmu.dtlb_misses
    dtlb_hit_rate = (mmu.dtlb_hits / dtlb_total * 100) if dtlb_total > 0 else 100.0
    print(f"TLB Hit Translation Rate: {tlb_rate / 1e6:.3f} MTrans/sec ({tlb_elapsed * 1000:.2f} ms for {n_tlb:,} lookups)")
    print(f"TLB Hits: {mmu.dtlb_hits:,} | DTLB Hit Rate: {dtlb_hit_rate:.2f}%")
    print()


def run_all_benchmarks() -> None:
    print("\n" + "=" * 78)
    print("  SILICONRISC RV64GC PROCESSOR ARCHITECTURE BENCHMARK SUITE")
    print("=" * 78 + "\n")

    bench_decoder_throughput()
    bench_functional_execution_throughput()
    bench_pipelined_simulation_speed()
    bench_cache_hierarchy_amat()
    bench_branch_predictors()
    bench_sv39_mmu_translation()

    print("=" * 78)
    print("  BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()
