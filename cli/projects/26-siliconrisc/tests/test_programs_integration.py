"""Integration tests executing complete algorithmic programs on SiliconRISC."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.core import CPU, ExecutionMode
from siliconrisc.elf import Assembler
from siliconrisc.memory import PhysicalMemory


class TestProgramsIntegration(unittest.TestCase):
    """Test full algorithmic execution (Fibonacci, Quicksort, Vector Math)."""

    def test_recursive_fibonacci_functional(self) -> None:
        """Compute fib(7) = 13 recursively using stack frames."""
        asm = Assembler(base_pc=0x80000000)
        source = """
        li   sp, 0x80010000    # Initialize stack pointer
        li   a0, 7             # n = 7
        jal  ra, fib
        ebreak

        # fib(n):
        fib:
        addi sp, sp, -24
        sd   ra, 16(sp)
        sd   s0, 8(sp)
        sd   s1, 0(sp)
        mv   s0, a0            # s0 = n

        # Base case: if n <= 1 return n
        li   t0, 1
        bge  t0, s0, fib_base

        # Recursive case: fib(n-1) + fib(n-2)
        addi a0, s0, -1
        jal  ra, fib
        mv   s1, a0            # s1 = fib(n-1)

        addi a0, s0, -2
        jal  ra, fib           # a0 = fib(n-2)
        add  a0, s1, a0        # a0 = fib(n-1) + fib(n-2)
        jal  x0, fib_return

        fib_base:
        mv   a0, s0

        fib_return:
        ld   s1, 0(sp)
        ld   s0, 8(sp)
        ld   ra, 16(sp)
        addi sp, sp, 24
        ret
        """
        words = asm.assemble(source)
        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        cpu.load_program(0x80000000, words)
        cpu.run(max_cycles=50000)

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.reg_file.read_x(10), 13)  # a0 = fib(7) = 13

    def test_bubble_sort_memory_array(self) -> None:
        """Sort array [64, 34, 25, 12, 22, 11, 90] in memory."""
        ram = PhysicalMemory()
        array_addr = 0x80002000
        initial_array = [64, 34, 25, 12, 22, 11, 90]
        for i, val in enumerate(initial_array):
            ram.write_u64(array_addr + i * 8, val)

        asm = Assembler(base_pc=0x80000000)
        source = """
        li   s0, 0x80002000    # array base address
        li   s1, 7             # n = 7

        outer_loop:
        li   t0, 0             # swapped = 0
        li   t1, 0             # i = 0
        addi t2, s1, -1        # n - 1

        inner_loop:
        bge  t1, t2, check_swap
        slli t3, t1, 3         # i * 8
        add  t4, s0, t3        # addr of arr[i]
        ld   t5, 0(t4)         # arr[i]
        ld   t6, 8(t4)         # arr[i+1]

        # if arr[i] <= arr[i+1], skip swap
        bge  t6, t5, no_swap
        # swap arr[i] and arr[i+1]
        sd   t6, 0(t4)
        sd   t5, 8(t4)
        li   t0, 1             # swapped = 1

        no_swap:
        addi t1, t1, 1         # i++
        jal  x0, inner_loop

        check_swap:
        bne  t0, x0, outer_loop
        ebreak
        """
        words = asm.assemble(source)
        cpu = CPU(ram=ram, mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        cpu.load_program(0x80000000, words)
        cpu.run(max_cycles=20000)

        self.assertTrue(cpu.halted)
        sorted_array = [ram.read_u64(array_addr + i * 8) for i in range(7)]
        self.assertEqual(sorted_array, sorted(initial_array))

    def test_pipelined_matrix_dot_product(self) -> None:
        """Compute vector dot product A dot B = [2, 3, 5] dot [7, 11, 13] in 5-stage pipeline."""
        ram = PhysicalMemory()
        vec_a_addr = 0x80001000
        vec_b_addr = 0x80001020

        # Vector A = [2, 3, 5], Vector B = [7, 11, 13]
        # Dot product = 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        for i, (a, b) in enumerate(zip([2, 3, 5], [7, 11, 13])):
            ram.write_u64(vec_a_addr + i * 8, a)
            ram.write_u64(vec_b_addr + i * 8, b)

        asm = Assembler(base_pc=0x80000000)
        source = """
        li   s0, 0x80001000    # addr A
        li   s1, 0x80001020    # addr B
        li   s2, 3             # length = 3
        li   a0, 0             # sum = 0
        li   t0, 0             # index i = 0

        dot_loop:
        bge  t0, s2, dot_done
        slli t1, t0, 3
        add  t2, s0, t1
        add  t3, s1, t1
        ld   t4, 0(t2)         # A[i]
        ld   t5, 0(t3)         # B[i]
        mul  t6, t4, t5        # A[i] * B[i]
        add  a0, a0, t6        # sum += A[i] * B[i]
        addi t0, t0, 1         # i++
        jal  x0, dot_loop

        dot_done:
        ebreak
        """
        words = asm.assemble(source)
        cpu = CPU(ram=ram, mode=ExecutionMode.PIPELINED, initial_pc=0x80000000)
        cpu.load_program(0x80000000, words)
        cpu.run(max_cycles=300)

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.pipeline.reg_file.read_x(10), 112)

    def test_pipelined_fp_double_precision(self) -> None:
        """Test IEEE 754 double precision floating point in functional and pipelined execution."""
        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        # f1 = 3.5, f2 = 2.0
        cpu.reg_file.write_f(1, 3.5)
        cpu.reg_file.write_f(2, 2.0)

        # fadd.d f3, f1, f2 (5.5) -> raw: 0x0220F1D3
        # fmul.d f4, f1, f2 (7.0) -> raw: 0x1220F253
        # ebreak
        cpu.ram.write_u32(0x80000000, 0x0220F1D3)  # fadd.d f3, f1, f2
        cpu.ram.write_u32(0x80000004, 0x1220F253)  # fmul.d f4, f1, f2
        cpu.ram.write_u32(0x80000008, 0x00100073)  # ebreak
        cpu.pc = 0x80000000
        cpu.run()

        self.assertTrue(cpu.halted)
        self.assertAlmostEqual(cpu.reg_file.read_f(3), 5.5)
        self.assertAlmostEqual(cpu.reg_file.read_f(4), 7.0)

    def test_mesi_bus_coherence_multicore_transition(self) -> None:
        """Test MESI coherence state transitions across simulated multi-core bus."""
        from siliconrisc.cache import Cache, MESIState, BusTransaction

        core0_cache = Cache(name="core0_l1", size_bytes=1024, associativity=4)
        core1_cache = Cache(name="core1_l1", size_bytes=1024, associativity=4)

        paddr = 0x80004000
        # Core 0 reads address -> allocates in EXCLUSIVE state
        core0_cache.allocate_line(paddr, bytearray(64), MESIState.EXCLUSIVE)
        line0 = core0_cache.find_line(paddr)
        self.assertIsNotNone(line0)
        self.assertEqual(line0.state, MESIState.EXCLUSIVE)

        # Core 1 snoops with BusRd -> Core 0 transitions from EXCLUSIVE to SHARED
        hit, dirty_data = core0_cache.snoop(BusTransaction.BUS_RD, paddr)
        self.assertTrue(hit)
        self.assertEqual(line0.state, MESIState.SHARED)

        # Core 1 writes to line -> snoops with BusUpgr -> Core 0 transitions to INVALID
        hit, dirty_data = core0_cache.snoop(BusTransaction.BUS_UPGR, paddr)
        self.assertTrue(hit)
        self.assertEqual(line0.state, MESIState.INVALID)

    def test_sv39_user_supervisor_privilege_fault(self) -> None:
        """Test SV39 MMU user mode permission enforcement."""
        from siliconrisc.memory import MMU, AccessType, PrivilegeMode, StorePageFault, PTE_V, PTE_R, PTE_W

        ram = PhysicalMemory()
        mmu = MMU(ram)
        root_ppn = 0x10
        # Map page with Supervisor-only permissions (PTE_U bit NOT set)
        mmu.map_page_4k(root_ppn=root_ppn, vaddr=0x1000, paddr=0x5000, flags=PTE_V | PTE_R | PTE_W)
        mmu.satp = (8 << 60) | root_ppn  # Enable SV39 mode with root_ppn

        # In Supervisor mode -> write succeeds
        mmu.privilege_mode = PrivilegeMode.SUPERVISOR
        paddr = mmu.translate(0x1000, AccessType.STORE)
        self.assertEqual(paddr, 0x5000)

        # In User mode -> accessing supervisor page raises StorePageFault
        mmu.privilege_mode = PrivilegeMode.USER
        with self.assertRaises(StorePageFault):
            mmu.translate(0x1000, AccessType.STORE)


if __name__ == "__main__":
    unittest.main()
