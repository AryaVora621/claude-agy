"""RISC-V RV64IMAFD (RV64GC) Instruction Set Architecture, Decoder, and Disassembler.

Provides:
1. 64-bit integer and floating-point register files with standard ABI names
2. Instruction formats (R, I, S, B, U, J, R4) and immediate decoders
3. Complete decoding for RV64I, RV64M, RV64A, and RV64F/D extensions
4. Bitwise arithmetic helpers for 64-bit and 32-bit signed/unsigned conversions
5. Standard Control and Status Register (CSR) definitions
6. Human-readable RISC-V disassembler
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import Dict, List, Optional, Tuple


# =====================================================================
# Bitwise and Arithmetic Helpers
# =====================================================================

MASK_64 = 0xFFFF_FFFF_FFFF_FFFF
MASK_32 = 0xFFFF_FFFF
SIGN_BIT_64 = 1 << 63
SIGN_BIT_32 = 1 << 31


def to_unsigned_64(val: int) -> int:
    """Mask integer to 64-bit unsigned representation."""
    return val & MASK_64


def to_signed_64(val: int) -> int:
    """Convert 64-bit unsigned integer to signed Python integer."""
    v = val & MASK_64
    if v & SIGN_BIT_64:
        return v - (1 << 64)
    return v


def to_unsigned_32(val: int) -> int:
    """Mask integer to 32-bit unsigned representation."""
    return val & MASK_32


def to_signed_32(val: int) -> int:
    """Convert 32-bit unsigned integer to signed Python integer."""
    v = val & MASK_32
    if v & SIGN_BIT_32:
        return v - (1 << 32)
    return v


def sign_extend(val: int, bits: int) -> int:
    """Sign-extend a value of given bit length to a signed Python integer."""
    sign_mask = 1 << (bits - 1)
    if val & sign_mask:
        return val - (1 << bits)
    return val


def float_to_bits_64(f: float) -> int:
    """Convert IEEE 754 64-bit double float to 64-bit unsigned integer bits."""
    return struct.unpack(">Q", struct.pack(">d", f))[0]


def bits_to_float_64(b: int) -> float:
    """Convert 64-bit unsigned integer bits to IEEE 754 64-bit double float."""
    return struct.unpack(">d", struct.pack(">Q", b & MASK_64))[0]


def float_to_bits_32(f: float) -> int:
    """Convert IEEE 754 32-bit single float to 32-bit unsigned integer bits."""
    return struct.unpack(">I", struct.pack(">f", f))[0]


def bits_to_float_32(b: int) -> float:
    """Convert 32-bit unsigned integer bits to IEEE 754 32-bit single float."""
    return struct.unpack(">f", struct.pack(">I", b & MASK_32))[0]


# =====================================================================
# RISC-V Primary 7-bit Opcodes
# =====================================================================

OP_LOAD       = 0x03  # 0000011: LB, LH, LW, LD, LBU, LHU, LWU
OP_MISC_MEM   = 0x0F  # 0001111: FENCE, FENCE.I
OP_OP_IMM     = 0x13  # 0010011: ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
OP_AUIPC      = 0x17  # 0010111: AUIPC
OP_OP_IMM_32  = 0x1B  # 0011011: ADDIW, SLLIW, SRLIW, SRAIW
OP_STORE      = 0x23  # 0100011: SB, SH, SW, SD
OP_AMO        = 0x2F  # 0101111: LR, SC, AMOSWAP, AMOADD, AMOXOR, AMOAND, AMOOR, AMOMIN, AMOMAX
OP_OP         = 0x33  # 0110011: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND, MUL, DIV, REM
OP_LUI        = 0x37  # 0110111: LUI
OP_OP_32      = 0x3B  # 0111011: ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVW, REMW
OP_BRANCH     = 0x63  # 1100011: BEQ, BNE, BLT, BGE, BLTU, BGEU
OP_JALR       = 0x67  # 1100111: JALR
OP_JAL        = 0x6F  # 1101111: JAL
OP_SYSTEM     = 0x73  # 1110011: ECALL, EBREAK, CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI, MRET, SRET
OP_FP_LOAD    = 0x07  # 0000111: FLW, FLD
OP_FP_STORE   = 0x27  # 0100111: FSW, FSD
OP_FP_OP      = 0x53  # 1010011: FADD, FSUB, FMUL, FDIV, FSQRT, FSGNJ, FMIN, FMAX, FCVT, FEQ, FLT, FLE, FMV


# =====================================================================
# Registers and ABI Names
# =====================================================================

ABI_REG_NAMES = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
]

ABI_FP_REG_NAMES = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
]

REG_ALIAS_TO_NUM: Dict[str, int] = {}
for i, name in enumerate(ABI_REG_NAMES):
    REG_ALIAS_TO_NUM[f"x{i}"] = i
    REG_ALIAS_TO_NUM[name] = i
REG_ALIAS_TO_NUM["fp"] = 8  # s0 is frame pointer

FP_REG_ALIAS_TO_NUM: Dict[str, int] = {}
for i, name in enumerate(ABI_FP_REG_NAMES):
    FP_REG_ALIAS_TO_NUM[f"f{i}"] = i
    FP_REG_ALIAS_TO_NUM[name] = i


@dataclass
class RegisterFile:
    """64-bit Integer and IEEE 754 Floating-Point Register File."""
    x: List[int] = field(default_factory=lambda: [0] * 32)
    f: List[int] = field(default_factory=lambda: [0] * 32)  # Raw 64-bit IEEE bits

    def read_x(self, idx: int) -> int:
        """Read 64-bit unsigned integer from register x[idx]. x0 always returns 0."""
        if idx == 0:
            return 0
        return self.x[idx & 31] & MASK_64

    def read_x_signed(self, idx: int) -> int:
        """Read 64-bit signed integer from register x[idx]."""
        return to_signed_64(self.read_x(idx))

    def write_x(self, idx: int, val: int) -> None:
        """Write 64-bit integer to register x[idx]. Writes to x0 are discarded."""
        if idx != 0:
            self.x[idx & 31] = val & MASK_64

    def read_f(self, idx: int) -> float:
        """Read 64-bit double float from register f[idx]."""
        return bits_to_float_64(self.f[idx & 31])

    def read_f_bits(self, idx: int) -> int:
        """Read raw 64-bit float bits from register f[idx]."""
        return self.f[idx & 31] & MASK_64

    def write_f(self, idx: int, val: float) -> None:
        """Write 64-bit double float to register f[idx]."""
        self.f[idx & 31] = float_to_bits_64(val)

    def write_f_bits(self, idx: int, bits: int) -> None:
        """Write raw 64-bit float bits to register f[idx]."""
        self.f[idx & 31] = bits & MASK_64

    def reset(self) -> None:
        """Reset all registers to zero."""
        for i in range(32):
            self.x[i] = 0
            self.f[i] = 0


# =====================================================================
# Standard Control and Status Registers (CSRs)
# =====================================================================

CSR_USTATUS   = 0x000
CSR_FFLAGS    = 0x001
CSR_FRM       = 0x002
CSR_FCSR      = 0x003

CSR_SSTATUS   = 0x100
CSR_SIE       = 0x104
CSR_STVEC     = 0x105
CSR_SSCRATCH  = 0x140
CSR_SEPC      = 0x141
CSR_SCAUSE    = 0x142
CSR_STVAL     = 0x143
CSR_SIP       = 0x144
CSR_SATP      = 0x180

CSR_MSTATUS   = 0x300
CSR_MISA      = 0x301
CSR_MEDELEG   = 0x302
CSR_MIDELEG   = 0x303
CSR_MIE       = 0x304
CSR_MTVEC     = 0x305
CSR_MSCRATCH  = 0x340
CSR_MEPC      = 0x341
CSR_MCAUSE    = 0x342
CSR_MTVAL     = 0x343
CSR_MIP       = 0x344

CSR_CYCLE     = 0xC00
CSR_TIME      = 0xC01
CSR_INSTRET   = 0xC02
CSR_CYCLEH    = 0xC80
CSR_TIMEH     = 0xC81
CSR_INSTRETH  = 0xC82


# =====================================================================
# Instruction Formats and Classification
# =====================================================================

class InstFormat(Enum):
    R = auto()
    I = auto()
    S = auto()
    B = auto()
    U = auto()
    J = auto()
    R4 = auto()
    UNKNOWN = auto()


class InstCategory(Enum):
    ALU = auto()
    ALU_32 = auto()
    BRANCH = auto()
    JUMP = auto()
    LOAD = auto()
    STORE = auto()
    AMO = auto()
    FP = auto()
    SYSTEM = auto()
    CSR = auto()
    NOP = auto()
    UNKNOWN = auto()


@dataclass
class DecodedInstruction:
    """Decoded RISC-V 32-bit instruction with all control signals."""
    raw: int
    pc: int = 0
    opcode: int = 0
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    rs3: int = 0
    funct3: int = 0
    funct7: int = 0
    imm: int = 0
    shamt: int = 0
    mnemonic: str = "unknown"
    fmt: InstFormat = InstFormat.UNKNOWN
    category: InstCategory = InstCategory.UNKNOWN

    # Pipeline hazard and dataflow indicators
    reads_rs1: bool = False
    reads_rs2: bool = False
    reads_rs3: bool = False
    reads_fp_rs1: bool = False
    reads_fp_rs2: bool = False
    writes_rd: bool = False
    writes_fp_rd: bool = False
    is_branch: bool = False
    is_jump: bool = False
    is_load: bool = False
    is_store: bool = False
    is_csr: bool = False
    is_system: bool = False
    is_amo: bool = False

    def __repr__(self) -> str:
        return f"{self.mnemonic} (0x{self.raw:08x})"


# =====================================================================
# RISC-V Instruction Decoder
# =====================================================================

class InstructionDecoder:
    """High-performance single-pass decoder for RV64GC instructions."""

    @staticmethod
    def decode(raw: int, pc: int = 0) -> DecodedInstruction:
        raw = raw & MASK_32
        opcode = raw & 0x7F
        rd = (raw >> 7) & 0x1F
        funct3 = (raw >> 12) & 0x07
        rs1 = (raw >> 15) & 0x1F
        rs2 = (raw >> 20) & 0x1F
        funct7 = (raw >> 25) & 0x7F
        rs3 = (raw >> 27) & 0x1F

        inst = DecodedInstruction(
            raw=raw,
            pc=pc,
            opcode=opcode,
            rd=rd,
            rs1=rs1,
            rs2=rs2,
            rs3=rs3,
            funct3=funct3,
            funct7=funct7,
        )

        # NOP detection (ADDI x0, x0, 0)
        if raw == 0x00000013:
            inst.mnemonic = "nop"
            inst.fmt = InstFormat.I
            inst.category = InstCategory.NOP
            inst.writes_rd = False
            return inst

        # --- LUI (Load Upper Immediate) ---
        if opcode == OP_LUI:
            inst.fmt = InstFormat.U
            inst.category = InstCategory.ALU
            inst.imm = sign_extend((raw >> 12) << 12, 32)
            inst.writes_rd = True
            inst.mnemonic = "lui"
            return inst

        # --- AUIPC (Add Upper Immediate to PC) ---
        if opcode == OP_AUIPC:
            inst.fmt = InstFormat.U
            inst.category = InstCategory.ALU
            inst.imm = sign_extend((raw >> 12) << 12, 32)
            inst.writes_rd = True
            inst.mnemonic = "auipc"
            return inst

        # --- JAL (Jump and Link) ---
        if opcode == OP_JAL:
            inst.fmt = InstFormat.J
            inst.category = InstCategory.JUMP
            # imm[20|10:1|11|19:12]
            imm_20 = (raw >> 31) & 1
            imm_10_1 = (raw >> 21) & 0x3FF
            imm_11 = (raw >> 20) & 1
            imm_19_12 = (raw >> 12) & 0xFF
            imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
            inst.imm = sign_extend(imm, 21)
            inst.writes_rd = True
            inst.is_jump = True
            inst.mnemonic = "jal"
            return inst

        # --- JALR (Jump and Link Register) ---
        if opcode == OP_JALR:
            inst.fmt = InstFormat.I
            inst.category = InstCategory.JUMP
            inst.imm = sign_extend(raw >> 20, 12)
            inst.reads_rs1 = True
            inst.writes_rd = True
            inst.is_jump = True
            inst.mnemonic = "jalr"
            return inst

        # --- BRANCH ---
        if opcode == OP_BRANCH:
            inst.fmt = InstFormat.B
            inst.category = InstCategory.BRANCH
            # imm[12|10:5|4:1|11]
            imm_12 = (raw >> 31) & 1
            imm_10_5 = (raw >> 25) & 0x3F
            imm_4_1 = (raw >> 8) & 0xF
            imm_11 = (raw >> 7) & 1
            imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
            inst.imm = sign_extend(imm, 13)
            inst.reads_rs1 = True
            inst.reads_rs2 = True
            inst.is_branch = True

            branch_map = {
                0x0: "beq",
                0x1: "bne",
                0x4: "blt",
                0x5: "bge",
                0x6: "bltu",
                0x7: "bgeu",
            }
            inst.mnemonic = branch_map.get(funct3, "unknown_branch")
            return inst

        # --- LOAD ---
        if opcode == OP_LOAD:
            inst.fmt = InstFormat.I
            inst.category = InstCategory.LOAD
            inst.imm = sign_extend(raw >> 20, 12)
            inst.reads_rs1 = True
            inst.writes_rd = True
            inst.is_load = True

            load_map = {
                0x0: "lb",
                0x1: "lh",
                0x2: "lw",
                0x3: "ld",
                0x4: "lbu",
                0x5: "lhu",
                0x6: "lwu",
            }
            inst.mnemonic = load_map.get(funct3, "unknown_load")
            return inst

        # --- STORE ---
        if opcode == OP_STORE:
            inst.fmt = InstFormat.S
            inst.category = InstCategory.STORE
            # imm[11:5] | imm[4:0]
            imm_11_5 = (raw >> 25) & 0x7F
            imm_4_0 = (raw >> 7) & 0x1F
            inst.imm = sign_extend((imm_11_5 << 5) | imm_4_0, 12)
            inst.reads_rs1 = True
            inst.reads_rs2 = True
            inst.is_store = True

            store_map = {
                0x0: "sb",
                0x1: "sh",
                0x2: "sw",
                0x3: "sd",
            }
            inst.mnemonic = store_map.get(funct3, "unknown_store")
            return inst

        # --- OP_IMM (64-bit Integer ALU Immediate) ---
        if opcode == OP_OP_IMM:
            inst.fmt = InstFormat.I
            inst.category = InstCategory.ALU
            inst.imm = sign_extend(raw >> 20, 12)
            inst.shamt = (raw >> 20) & 0x3F  # 6-bit shift amount for RV64
            inst.reads_rs1 = True
            inst.writes_rd = True

            if funct3 == 0x0:
                inst.mnemonic = "addi"
            elif funct3 == 0x2:
                inst.mnemonic = "slti"
            elif funct3 == 0x3:
                inst.mnemonic = "sltiu"
            elif funct3 == 0x4:
                inst.mnemonic = "xori"
            elif funct3 == 0x6:
                inst.mnemonic = "ori"
            elif funct3 == 0x7:
                inst.mnemonic = "andi"
            elif funct3 == 0x1:
                inst.mnemonic = "slli"
            elif funct3 == 0x5:
                if (funct7 >> 1) == 0x00:
                    inst.mnemonic = "srli"
                elif (funct7 >> 1) == 0x10:
                    inst.mnemonic = "srai"
            return inst

        # --- OP_IMM_32 (32-bit Word ALU Immediate for RV64) ---
        if opcode == OP_OP_IMM_32:
            inst.fmt = InstFormat.I
            inst.category = InstCategory.ALU_32
            inst.imm = sign_extend(raw >> 20, 12)
            inst.shamt = (raw >> 20) & 0x1F  # 5-bit shift amount for 32-bit words
            inst.reads_rs1 = True
            inst.writes_rd = True

            if funct3 == 0x0:
                inst.mnemonic = "addiw"
            elif funct3 == 0x1:
                inst.mnemonic = "slliw"
            elif funct3 == 0x5:
                if funct7 == 0x00:
                    inst.mnemonic = "srliw"
                elif funct7 == 0x20:
                    inst.mnemonic = "sraiw"
            return inst

        # --- OP (64-bit Integer Register-Register ALU and RV64M Multiply/Divide) ---
        if opcode == OP_OP:
            inst.fmt = InstFormat.R
            inst.reads_rs1 = True
            inst.reads_rs2 = True
            inst.writes_rd = True

            if funct7 == 0x00:
                inst.category = InstCategory.ALU
                op_map = {
                    0x0: "add",
                    0x1: "sll",
                    0x2: "slt",
                    0x3: "sltu",
                    0x4: "xor",
                    0x5: "srl",
                    0x6: "or",
                    0x7: "and",
                }
                inst.mnemonic = op_map.get(funct3, "unknown_op")
            elif funct7 == 0x20:
                inst.category = InstCategory.ALU
                if funct3 == 0x0:
                    inst.mnemonic = "sub"
                elif funct3 == 0x5:
                    inst.mnemonic = "sra"
            elif funct7 == 0x01:  # RV64M Standard Extension
                inst.category = InstCategory.ALU
                m_map = {
                    0x0: "mul",
                    0x1: "mulh",
                    0x2: "mulhsu",
                    0x3: "mulhu",
                    0x4: "div",
                    0x5: "divu",
                    0x6: "rem",
                    0x7: "remu",
                }
                inst.mnemonic = m_map.get(funct3, "unknown_m")
            return inst

        # --- OP_32 (32-bit Register-Register ALU and RV64M Word Extensions) ---
        if opcode == OP_OP_32:
            inst.fmt = InstFormat.R
            inst.category = InstCategory.ALU_32
            inst.reads_rs1 = True
            inst.reads_rs2 = True
            inst.writes_rd = True

            if funct7 == 0x00:
                if funct3 == 0x0:
                    inst.mnemonic = "addw"
                elif funct3 == 0x1:
                    inst.mnemonic = "sllw"
                elif funct3 == 0x5:
                    inst.mnemonic = "srlw"
            elif funct7 == 0x20:
                if funct3 == 0x0:
                    inst.mnemonic = "subw"
                elif funct3 == 0x5:
                    inst.mnemonic = "sraw"
            elif funct7 == 0x01:  # RV64M Word variants
                m32_map = {
                    0x0: "mulw",
                    0x4: "divw",
                    0x5: "divuw",
                    0x6: "remw",
                    0x7: "remuw",
                }
                inst.mnemonic = m32_map.get(funct3, "unknown_m32")
            return inst

        # --- SYSTEM & CSR Instructions ---
        if opcode == OP_SYSTEM:
            inst.category = InstCategory.SYSTEM
            csr_addr = (raw >> 20) & 0xFFF
            inst.imm = csr_addr

            if funct3 == 0x0:
                if raw == 0x00000073:
                    inst.mnemonic = "ecall"
                    inst.is_system = True
                elif raw == 0x00100073:
                    inst.mnemonic = "ebreak"
                    inst.is_system = True
                elif raw == 0x30200073:
                    inst.mnemonic = "mret"
                    inst.is_system = True
                elif raw == 0x10200073:
                    inst.mnemonic = "sret"
                    inst.is_system = True
                elif raw == 0x10500073:
                    inst.mnemonic = "wfi"
                    inst.is_system = True
            else:
                inst.is_csr = True
                inst.category = InstCategory.CSR
                inst.writes_rd = True
                csr_map = {
                    0x1: "csrrw",
                    0x2: "csrrs",
                    0x3: "csrrc",
                    0x5: "csrrwi",
                    0x6: "csrrsi",
                    0x7: "csrrci",
                }
                inst.mnemonic = csr_map.get(funct3, "unknown_csr")
                if funct3 in (0x1, 0x2, 0x3):
                    inst.reads_rs1 = True
            return inst

        # --- MISC_MEM (FENCE, FENCE.I) ---
        if opcode == OP_MISC_MEM:
            inst.category = InstCategory.SYSTEM
            if funct3 == 0x0:
                inst.mnemonic = "fence"
            elif funct3 == 0x1:
                inst.mnemonic = "fence.i"
            return inst

        # --- AMO (Atomic Memory Operations) ---
        if opcode == OP_AMO:
            inst.category = InstCategory.AMO
            inst.fmt = InstFormat.R
            inst.reads_rs1 = True
            inst.reads_rs2 = True
            inst.writes_rd = True
            inst.is_amo = True

            amo_funct5 = (funct7 >> 2) & 0x1F
            width_str = "w" if funct3 == 0x2 else ("d" if funct3 == 0x3 else "?")

            amo_names = {
                0x02: "lr",
                0x03: "sc",
                0x01: "amoswap",
                0x00: "amoadd",
                0x04: "amoxor",
                0x0C: "amoand",
                0x08: "amoor",
                0x10: "amomin",
                0x14: "amomax",
                0x18: "amominu",
                0x1C: "amomaxu",
            }
            base_name = amo_names.get(amo_funct5, "unknown_amo")
            inst.mnemonic = f"{base_name}.{width_str}"
            return inst

        # --- FP LOAD & STORE ---
        if opcode == OP_FP_LOAD:
            inst.fmt = InstFormat.I
            inst.category = InstCategory.FP
            inst.imm = sign_extend(raw >> 20, 12)
            inst.reads_rs1 = True
            inst.writes_fp_rd = True
            inst.is_load = True
            inst.mnemonic = "fld" if funct3 == 0x3 else ("flw" if funct3 == 0x2 else "unknown_fload")
            return inst

        if opcode == OP_FP_STORE:
            inst.fmt = InstFormat.S
            inst.category = InstCategory.FP
            imm_11_5 = (raw >> 25) & 0x7F
            imm_4_0 = (raw >> 7) & 0x1F
            inst.imm = sign_extend((imm_11_5 << 5) | imm_4_0, 12)
            inst.reads_rs1 = True
            inst.reads_fp_rs2 = True
            inst.is_store = True
            inst.mnemonic = "fsd" if funct3 == 0x3 else ("fsw" if funct3 == 0x2 else "unknown_fstore")
            return inst

        # --- FP COMPUTATIONAL OPERATIONS ---
        if opcode == OP_FP_OP:
            inst.fmt = InstFormat.R
            inst.category = InstCategory.FP
            inst.reads_fp_rs1 = True
            inst.reads_fp_rs2 = True
            inst.writes_fp_rd = True

            fmt_code = (funct7 >> 0) & 0x3  # 0: single, 1: double
            is_double = (fmt_code == 1)
            suffix = ".d" if is_double else ".s"
            fp_funct5 = funct7 >> 2

            if fp_funct5 == 0x00:
                inst.mnemonic = "fadd" + suffix
            elif fp_funct5 == 0x01:
                inst.mnemonic = "fsub" + suffix
            elif fp_funct5 == 0x02:
                inst.mnemonic = "fmul" + suffix
            elif fp_funct5 == 0x03:
                inst.mnemonic = "fdiv" + suffix
            elif fp_funct5 == 0x0B:
                inst.mnemonic = "fsqrt" + suffix
                inst.reads_fp_rs2 = False
            elif fp_funct5 == 0x04:
                inst.mnemonic = "fsgnj" + suffix
            elif fp_funct5 == 0x05:
                inst.mnemonic = "fmin" + suffix if funct3 == 0 else "fmax" + suffix
            elif fp_funct5 == 0x14:  # Comparisons
                inst.writes_fp_rd = False
                inst.writes_rd = True  # Writes integer register
                comp_map = {0: "feq", 1: "flt", 2: "fle"}
                inst.mnemonic = comp_map.get(funct3, "fcmp") + suffix
            elif fp_funct5 == 0x18:  # FCVT.W.D, FCVT.L.D, etc.
                inst.writes_fp_rd = False
                inst.writes_rd = True
                inst.reads_fp_rs2 = False
                inst.mnemonic = "fcvt.l" + suffix if rs2 == 2 else "fcvt.w" + suffix
            elif fp_funct5 == 0x1A:  # FCVT.D.W, FCVT.D.L
                inst.reads_fp_rs1 = False
                inst.reads_rs1 = True  # Reads integer register
                inst.reads_fp_rs2 = False
                inst.mnemonic = "fcvt" + suffix + ".l" if rs2 == 2 else "fcvt" + suffix + ".w"
            elif fp_funct5 == 0x1C:  # FMV.X.D / FMV.X.W
                inst.writes_fp_rd = False
                inst.writes_rd = True
                inst.reads_fp_rs2 = False
                inst.mnemonic = "fmv.x.d" if is_double else "fmv.x.w"
            elif fp_funct5 == 0x1E:  # FMV.D.X / FMV.W.X
                inst.reads_fp_rs1 = False
                inst.reads_rs1 = True
                inst.reads_fp_rs2 = False
                inst.mnemonic = "fmv.d.x" if is_double else "fmv.w.x"
            return inst

        # Fallback for unrecognized instruction
        inst.mnemonic = f"unknown_0x{opcode:02x}"
        return inst


# =====================================================================
# RISC-V Disassembler
# =====================================================================

class Disassembler:
    """Disassembles decoded instructions into canonical assembly text."""

    @staticmethod
    def disassemble(inst: DecodedInstruction) -> str:
        name = inst.mnemonic

        if name == "nop":
            return "nop"

        rd_s = ABI_REG_NAMES[inst.rd]
        rs1_s = ABI_REG_NAMES[inst.rs1]
        rs2_s = ABI_REG_NAMES[inst.rs2]
        frd_s = ABI_FP_REG_NAMES[inst.rd]
        frs1_s = ABI_FP_REG_NAMES[inst.rs1]
        frs2_s = ABI_FP_REG_NAMES[inst.rs2]

        if inst.fmt == InstFormat.U:
            return f"{name:<8} {rd_s}, 0x{inst.imm & 0xFFFFFFFF:x}"

        if inst.fmt == InstFormat.J:
            target = inst.pc + inst.imm
            return f"{name:<8} {rd_s}, 0x{target:x} (offset {inst.imm:+d})"

        if inst.fmt == InstFormat.B:
            target = inst.pc + inst.imm
            return f"{name:<8} {rs1_s}, {rs2_s}, 0x{target:x} (offset {inst.imm:+d})"

        if inst.is_load:
            if inst.writes_fp_rd:
                return f"{name:<8} {frd_s}, {inst.imm}({rs1_s})"
            return f"{name:<8} {rd_s}, {inst.imm}({rs1_s})"

        if inst.is_store:
            if inst.reads_fp_rs2:
                return f"{name:<8} {frs2_s}, {inst.imm}({rs1_s})"
            return f"{name:<8} {rs2_s}, {inst.imm}({rs1_s})"

        if inst.fmt == InstFormat.I:
            if name == "jalr":
                return f"jalr     {rd_s}, {inst.imm}({rs1_s})"
            if inst.is_csr:
                csr_name = f"0x{inst.imm:03x}"
                if name.endswith("i"):
                    return f"{name:<8} {rd_s}, {csr_name}, {inst.rs1}"
                return f"{name:<8} {rd_s}, {csr_name}, {rs1_s}"
            if name in ("slli", "srli", "srai", "slliw", "srliw", "sraiw"):
                return f"{name:<8} {rd_s}, {rs1_s}, {inst.shamt}"
            return f"{name:<8} {rd_s}, {rs1_s}, {inst.imm}"

        if inst.fmt == InstFormat.R:
            if inst.category == InstCategory.FP:
                if name.startswith("feq") or name.startswith("flt") or name.startswith("fle"):
                    return f"{name:<8} {rd_s}, {frs1_s}, {frs2_s}"
                if name.startswith("fmv.x"):
                    return f"{name:<8} {rd_s}, {frs1_s}"
                if name.startswith("fmv.d") or name.startswith("fmv.w"):
                    return f"{name:<8} {frd_s}, {rs1_s}"
                if name.startswith("fcvt") and not inst.writes_fp_rd:
                    return f"{name:<8} {rd_s}, {frs1_s}"
                if name.startswith("fcvt") and inst.writes_fp_rd:
                    return f"{name:<8} {frd_s}, {rs1_s}"
                if name.startswith("fsqrt"):
                    return f"{name:<8} {frd_s}, {frs1_s}"
                return f"{name:<8} {frd_s}, {frs1_s}, {frs2_s}"

            if inst.is_amo:
                return f"{name:<8} {rd_s}, {rs2_s}, ({rs1_s})"

            return f"{name:<8} {rd_s}, {rs1_s}, {rs2_s}"

        if inst.category == InstCategory.SYSTEM:
            return name

        return f"{name:<8} raw=0x{inst.raw:08x}"
