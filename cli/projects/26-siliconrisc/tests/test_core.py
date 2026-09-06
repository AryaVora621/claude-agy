"""Unit tests for SiliconRISC CPU Core (Functional and Pipelined execution, CSRs, Traps)."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.core import CPU, ExecutionMode, EXC_ECALL_M_MODE
from siliconrisc.isa import CSR_MTVEC, CSR_MEPC, CSR_MCAUSE, CSR_SATP
from siliconrisc.memory import PhysicalMemory


class TestCPUCore(unittest.TestCase):
    """Test CPU core execution in both Functional and Pipelined modes."""

    def test_functional_execution_arithmetic(self) -> None:
        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        # Sequence:
        # addi x1, x0, 15
        # addi x2, x0, 25
        # add  x3, x1, x2
        # mul  x4, x1, x2
        # ebreak
        insts = [
            0x00F00093,  # addi x1, x0, 15
            0x01900113,  # addi x2, x0, 25
            0x002081B3,  # add  x3, x1, x2
            0x02208233,  # mul  x4, x1, x2
            0x00100073,  # ebreak
        ]
        cpu.load_program(0x80000000, insts)
        cycles = cpu.run()

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.reg_file.read_x(1), 15)
        self.assertEqual(cpu.reg_file.read_x(2), 25)
        self.assertEqual(cpu.reg_file.read_x(3), 40)
        self.assertEqual(cpu.reg_file.read_x(4), 375)
        self.assertEqual(cpu.instructions_executed, 5)

    def test_functional_csr_access(self) -> None:
        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        # Test CSR read / write:
        # csrw mtvec, x1 (using csrrw x0, mtvec, x1)
        # addi x1, x0, 0x100
        # csrrw x0, 0x305, x1  (mtvec = 0x100)
        # csrrs x2, 0x305, x0  (read mtvec into x2)
        # ebreak
        insts = [
            0x10000093,  # addi x1, x0, 0x100
            # csrrw rd=0, csr=0x305, rs1=1 -> 0x30509073
            0x30509073,  # csrrw x0, mtvec, x1
            # csrrs rd=2, csr=0x305, rs1=0 -> 0x30502173
            0x30502173,  # csrrs x2, mtvec, x0
            0x00100073,  # ebreak
        ]
        cpu.load_program(0x80000000, insts)
        cpu.run()

        self.assertEqual(cpu.read_csr(CSR_MTVEC), 0x100)
        self.assertEqual(cpu.reg_file.read_x(2), 0x100)

    def test_functional_trap_handling(self) -> None:
        cpu = CPU(mode=ExecutionMode.FUNCTIONAL, initial_pc=0x80000000)
        cpu.write_csr(CSR_MTVEC, 0x80000100)  # Trap vector handler

        # Program:
        # 0x80000000: ecall
        # Trap vector at 0x80000100:
        # 0x80000100: addi x5, x0, 99
        # 0x80000104: ebreak
        cpu.ram.write_u32(0x80000000, 0x00000073)  # ecall
        cpu.ram.write_u32(0x80000100, 0x06300293)  # addi x5, x0, 99
        cpu.ram.write_u32(0x80000104, 0x00100073)  # ebreak

        cpu.pc = 0x80000000
        cpu.run(max_cycles=10)

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.read_csr(CSR_MEPC), 0x80000000)
        self.assertEqual(cpu.read_csr(CSR_MCAUSE), EXC_ECALL_M_MODE)
        self.assertEqual(cpu.reg_file.read_x(5), 99)

    def test_pipelined_execution_and_metrics(self) -> None:
        cpu = CPU(mode=ExecutionMode.PIPELINED, initial_pc=0x80000000)
        insts = [
            0x00A00093,  # addi x1, x0, 10
            0x01400113,  # addi x2, x0, 20
            0x002081B3,  # add  x3, x1, x2
            0x00100073,  # ebreak
        ]
        cpu.load_program(0x80000000, insts)
        cpu.run(max_cycles=50)

        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.pipeline.reg_file.read_x(1), 10)
        self.assertEqual(cpu.pipeline.reg_file.read_x(2), 20)
        self.assertEqual(cpu.pipeline.reg_file.read_x(3), 30)

        metrics = cpu.get_metrics()
        self.assertEqual(metrics["mode"], "PIPELINED")
        self.assertGreater(metrics["cycles"], 0)
        self.assertGreater(metrics["ipc"], 0.0)


if __name__ == "__main__":
    unittest.main()
