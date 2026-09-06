"""Unit tests for RV64GC ISA decoder, register files, and disassembler."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.isa import (
    ABI_REG_NAMES,
    DecodedInstruction,
    Disassembler,
    InstCategory,
    InstFormat,
    InstructionDecoder,
    RegisterFile,
    sign_extend,
    to_signed_32,
    to_signed_64,
    to_unsigned_32,
    to_unsigned_64,
)


class TestISAAndDecoder(unittest.TestCase):
    """Test register operations, bitwise conversions, decoding, and disassembly."""

    def test_bitwise_helpers(self) -> None:
        self.assertEqual(to_unsigned_64(-1), 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(to_signed_64(0xFFFF_FFFF_FFFF_FFFF), -1)
        self.assertEqual(to_unsigned_32(-1), 0xFFFF_FFFF)
        self.assertEqual(to_signed_32(0xFFFF_FFFF), -1)
        self.assertEqual(sign_extend(0xFFF, 12), -1)
        self.assertEqual(sign_extend(0x7FF, 12), 2047)

    def test_register_file(self) -> None:
        rf = RegisterFile()
        self.assertEqual(rf.read_x(0), 0)
        rf.write_x(0, 12345)
        self.assertEqual(rf.read_x(0), 0)  # x0 is hardwired to 0

        rf.write_x(1, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(rf.read_x(1), 0x1234_5678_9ABC_DEF0)

        # Test floating-point registers
        rf.write_f(1, 3.141592653589793)
        self.assertAlmostEqual(rf.read_f(1), 3.141592653589793, places=10)

    def test_decode_addi(self) -> None:
        # addi x1, x2, 42
        # imm=42 (0x02a), rs1=2, funct3=0, rd=1, opcode=0x13
        # 000000101010 00010 000 00001 0010011 -> 0x02A10093
        raw = 0x02A10093
        inst = InstructionDecoder.decode(raw)
        self.assertEqual(inst.mnemonic, "addi")
        self.assertEqual(inst.rd, 1)
        self.assertEqual(inst.rs1, 2)
        self.assertEqual(inst.imm, 42)
        self.assertTrue(inst.reads_rs1)
        self.assertTrue(inst.writes_rd)
        self.assertFalse(inst.reads_rs2)
        dis = Disassembler.disassemble(inst)
        self.assertIn("addi", dis)
        self.assertIn("ra", dis)
        self.assertIn("sp", dis)
        self.assertIn("42", dis)

    def test_decode_add_sub_mul(self) -> None:
        # add x3, x1, x2 -> funct7=0x00, rs2=2, rs1=1, funct3=0x0, rd=3, opcode=0x33
        raw_add = 0x002081B3
        inst_add = InstructionDecoder.decode(raw_add)
        self.assertEqual(inst_add.mnemonic, "add")
        self.assertEqual(inst_add.rd, 3)
        self.assertEqual(inst_add.rs1, 1)
        self.assertEqual(inst_add.rs2, 2)

        # sub x3, x1, x2 -> funct7=0x20
        raw_sub = 0x402081B3
        inst_sub = InstructionDecoder.decode(raw_sub)
        self.assertEqual(inst_sub.mnemonic, "sub")

        # mul x3, x1, x2 (RV64M) -> funct7=0x01, funct3=0x0
        raw_mul = 0x022081B3
        inst_mul = InstructionDecoder.decode(raw_mul)
        self.assertEqual(inst_mul.mnemonic, "mul")

    def test_decode_branch(self) -> None:
        # beq x1, x2, offset +16
        # opcode=0x63, funct3=0x0
        # imm = 16 -> imm[12]=0, imm[11]=0, imm[10:5]=0, imm[4:1]=8 (1000b) -> raw
        # Let's test with bne x5, x6, -8
        # imm = -8 -> 0b1111111111000
        # imm[12]=1, imm[11]=1, imm[10:5]=0b111111, imm[4:1]=0b1100
        # raw: imm[12]=1 (bit31), imm[10:5]=0x3f (bits30:25), rs2=6 (bits24:20), rs1=5 (bits19:15), funct3=1, imm[4:1]=0xC (bits11:8), imm[11]=1 (bit7), opcode=0x63
        raw_bne = (1 << 31) | (0x3F << 25) | (6 << 20) | (5 << 15) | (1 << 12) | (0xC << 8) | (1 << 7) | 0x63
        inst = InstructionDecoder.decode(raw_bne, pc=0x1000)
        self.assertEqual(inst.mnemonic, "bne")
        self.assertEqual(inst.rs1, 5)
        self.assertEqual(inst.rs2, 6)
        self.assertEqual(inst.imm, -8)
        self.assertTrue(inst.is_branch)

    def test_decode_load_store(self) -> None:
        # ld x5, 16(x10) -> opcode=0x03, funct3=0x3, rd=5, rs1=10, imm=16
        raw_ld = (16 << 20) | (10 << 15) | (3 << 12) | (5 << 7) | 0x03
        inst_ld = InstructionDecoder.decode(raw_ld)
        self.assertEqual(inst_ld.mnemonic, "ld")
        self.assertEqual(inst_ld.rd, 5)
        self.assertEqual(inst_ld.rs1, 10)
        self.assertEqual(inst_ld.imm, 16)
        self.assertTrue(inst_ld.is_load)

        # sd x5, -8(x2) -> opcode=0x23, funct3=0x3, rs1=2, rs2=5, imm=-8 (0xFF8)
        # imm[11:5] = 0x7F, imm[4:0] = 0x18
        raw_sd = (0x7F << 25) | (5 << 20) | (2 << 15) | (3 << 12) | (0x18 << 7) | 0x23
        inst_sd = InstructionDecoder.decode(raw_sd)
        self.assertEqual(inst_sd.mnemonic, "sd")
        self.assertEqual(inst_sd.rs1, 2)
        self.assertEqual(inst_sd.rs2, 5)
        self.assertEqual(inst_sd.imm, -8)
        self.assertTrue(inst_sd.is_store)

    def test_decode_fadd_d(self) -> None:
        # fadd.d f3, f1, f2 -> opcode=0x53, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x01
        raw = (0x01 << 25) | (2 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x53
        inst = InstructionDecoder.decode(raw)
        self.assertEqual(inst.mnemonic, "fadd.d")
        self.assertTrue(inst.reads_fp_rs1)
        self.assertTrue(inst.reads_fp_rs2)
        self.assertTrue(inst.writes_fp_rd)


if __name__ == "__main__":
    unittest.main()
