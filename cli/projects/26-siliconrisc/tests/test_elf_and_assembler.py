"""Unit tests for ELF64 Parser, Loader, and RV64GC Assembler."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.core import CPU, ExecutionMode
from siliconrisc.elf import Assembler, Elf64Parser, ElfLoader, EM_RISCV, PT_LOAD
from siliconrisc.memory import MMU, PhysicalMemory


class TestElfAndAssembler(unittest.TestCase):
    """Test RV64GC Assembler, ELF64 generation, parsing, and execution."""

    def test_assembler_arithmetic_and_pseudos(self) -> None:
        asm = Assembler(base_pc=0x80000000)
        source = """
        li   x1, 42
        li   x2, 10
        add  x3, x1, x2
        sub  x4, x1, x2
        mul  x5, x1, x2
        ebreak
        """
        words = asm.assemble(source)
        self.assertGreater(len(words), 0)

        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        cpu.load_program(0x80000000, words)
        cpu.run()

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.reg_file.read_x(1), 42)
        self.assertEqual(cpu.reg_file.read_x(2), 10)
        self.assertEqual(cpu.reg_file.read_x(3), 52)
        self.assertEqual(cpu.reg_file.read_x(4), 32)
        self.assertEqual(cpu.reg_file.read_x(5), 420)

    def test_assembler_loop_and_labels(self) -> None:
        asm = Assembler(base_pc=0x80000000)
        # Compute sum from 1 to 5 = 15
        source = """
        li   x1, 5      # count = 5
        li   x2, 0      # sum = 0
        loop:
        beq  x1, x0, done
        add  x2, x2, x1
        addi x1, x1, -1
        jal  x0, loop
        done:
        ebreak
        """
        words = asm.assemble(source)

        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        cpu.load_program(0x80000000, words)
        cpu.run(max_cycles=100)

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.reg_file.read_x(1), 0)
        self.assertEqual(cpu.reg_file.read_x(2), 15)

    def test_elf64_binary_creation_and_parsing(self) -> None:
        asm = Assembler(base_pc=0x80000000)
        source = """
        li   a0, 100
        addi a0, a0, 50
        ebreak
        """
        words = asm.assemble(source)
        elf_bytes = Assembler.create_elf_binary(0x80000000, words)

        # Parse ELF binary
        elf = Elf64Parser.parse(elf_bytes)
        self.assertEqual(elf.entry, 0x80000000)
        self.assertEqual(len(elf.ph_headers), 1)
        self.assertEqual(elf.ph_headers[0].p_type, PT_LOAD)
        self.assertEqual(elf.ph_headers[0].p_vaddr, 0x80000000)
        self.assertTrue(elf.ph_headers[0].is_executable)
        self.assertTrue(elf.ph_headers[0].is_readable)

    def test_elf_loader_and_cpu_execution(self) -> None:
        asm = Assembler(base_pc=0x80000000)
        source = """
        li   t0, 7
        li   t1, 8
        mul  a0, t0, t1
        ebreak
        """
        words = asm.assemble(source)
        elf_bytes = Assembler.create_elf_binary(0x80000000, words)
        elf = Elf64Parser.parse(elf_bytes)

        ram = PhysicalMemory()
        entry, segments = ElfLoader.load(elf, ram)
        self.assertEqual(entry, 0x80000000)
        self.assertEqual(len(segments), 1)

        cpu = CPU(ram=ram, mode=ExecutionMode.FUNCTIONAL, initial_pc=entry)
        cpu.run()

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.reg_file.read_x(10), 56)  # a0 = 7 * 8


if __name__ == "__main__":
    unittest.main()
