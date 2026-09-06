"""Unit tests for 5-Stage In-Order Pipeline data forwarding and hazard detection."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.branch import BranchPredictorUnit, PredictorType
from siliconrisc.cache import MemoryHierarchy
from siliconrisc.memory import MMU, PhysicalMemory
from siliconrisc.pipeline import PipelinedCore


class TestPipelineHazards(unittest.TestCase):
    """Test RAW forwarding, load-use pipeline stalls, and branch misprediction flushes."""

    def test_raw_hazard_forwarding(self) -> None:
        ram = PhysicalMemory()
        mmu = MMU(ram)
        cache = MemoryHierarchy(ram)
        bpu = BranchPredictorUnit()

        # Sequence with back-to-back RAW data dependencies:
        # 0x80000000: addi x1, x0, 10    (writes x1)
        # 0x80000004: addi x2, x1, 5     (reads x1 -> EX-to-EX forwarding!)
        # 0x80000008: add  x3, x1, x2    (reads x1, x2 -> MEM-to-EX and EX-to-EX forwarding!)
        # 0x8000000C: ebreak
        ram.write_u32(0x80000000, 0x00A00093)  # addi x1, x0, 10
        ram.write_u32(0x80000004, 0x00508113)  # addi x2, x1, 5
        ram.write_u32(0x80000008, 0x002081B3)  # add x3, x1, x2
        ram.write_u32(0x8000000C, 0x00100073)  # ebreak

        core = PipelinedCore(mmu, cache, bpu, initial_pc=0x80000000)

        # Run pipeline steps until halted
        saw_forward_a = False
        saw_forward_b = False
        for _ in range(15):
            status = core.step()
            if status.forward_a > 0:
                saw_forward_a = True
            if status.forward_b > 0:
                saw_forward_b = True
            if core.halted:
                break

        self.assertTrue(saw_forward_a)
        self.assertTrue(saw_forward_b)
        self.assertEqual(core.reg_file.read_x(1), 10)
        self.assertEqual(core.reg_file.read_x(2), 15)
        self.assertEqual(core.reg_file.read_x(3), 25)

    def test_load_use_hazard_stall(self) -> None:
        ram = PhysicalMemory()
        mmu = MMU(ram)
        cache = MemoryHierarchy(ram)
        bpu = BranchPredictorUnit()

        # Pre-seed memory
        ram.write_u64(0x80001000, 100)

        # Load-Use hazard:
        # 0x80000000: ld   x5, 0(x6)     (x6=0x80001000)
        # 0x80000004: addi x7, x5, 20    (requires 1-cycle stall bubble because x5 is from load!)
        # 0x80000008: ebreak
        core = PipelinedCore(mmu, cache, bpu, initial_pc=0x80000000)
        core.reg_file.write_x(6, 0x80001000)

        ram.write_u32(0x80000000, 0x00033283)  # ld x5, 0(x6)
        ram.write_u32(0x80000004, 0x01428393)  # addi x7, x5, 20
        ram.write_u32(0x80000008, 0x00100073)  # ebreak

        stalled_observed = False
        for _ in range(12):
            status = core.step()
            if status.stalled:
                stalled_observed = True
            if core.halted:
                break

        self.assertTrue(stalled_observed)
        self.assertEqual(core.reg_file.read_x(5), 100)
        self.assertEqual(core.reg_file.read_x(7), 120)

    def test_branch_misprediction_flush(self) -> None:
        ram = PhysicalMemory()
        mmu = MMU(ram)
        cache = MemoryHierarchy(ram)
        # Force predictor to ALWAYS_NOT_TAKEN so taken branch will cause mispredict flush
        bpu = BranchPredictorUnit(predictor_type=PredictorType.ALWAYS_NOT_TAKEN)

        # Branch taken:
        # 0x80000000: addi x1, x0, 1
        # 0x80000004: bne  x1, x0, +12   (taken to 0x80000010!)
        # 0x80000008: addi x2, x0, 999   (should be flushed!)
        # 0x8000000C: addi x3, x0, 888   (should be flushed!)
        # 0x80000010: addi x4, x0, 42    (target of branch)
        # 0x80000014: ebreak
        ram.write_u32(0x80000000, 0x00100093)  # addi x1, x0, 1
        # bne x1, x0, +12 -> imm=12 -> imm[12]=0, imm[11]=0, imm[10:5]=0, imm[4:1]=6 (0110b)
        # raw: imm[12]=0, imm[10:5]=0, rs2=0, rs1=1, funct3=1, imm[4:1]=6, imm[11]=0, opcode=0x63
        ram.write_u32(0x80000004, (0 << 25) | (0 << 20) | (1 << 15) | (1 << 12) | (6 << 8) | 0x63)
        ram.write_u32(0x80000008, 0x3E700113)  # addi x2, x0, 999
        ram.write_u32(0x8000000C, 0x37800193)  # addi x3, x0, 888
        ram.write_u32(0x80000010, 0x02A00213)  # addi x4, x0, 42
        ram.write_u32(0x80000014, 0x00100073)  # ebreak

        core = PipelinedCore(mmu, cache, bpu, initial_pc=0x80000000)

        flushed_observed = False
        for _ in range(20):
            status = core.step()
            if status.flushed:
                flushed_observed = True
            if core.halted:
                break

        self.assertTrue(flushed_observed)
        self.assertEqual(core.reg_file.read_x(1), 1)
        self.assertEqual(core.reg_file.read_x(2), 0)   # Flushed!
        self.assertEqual(core.reg_file.read_x(3), 0)   # Flushed!
        self.assertEqual(core.reg_file.read_x(4), 42)  # Executed after branch!


if __name__ == "__main__":
    unittest.main()
