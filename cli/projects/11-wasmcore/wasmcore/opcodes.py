"""
WasmCore: WebAssembly MVP Opcodes and Instruction Representation.
Complete byte mapping and instruction token container.
"""

from enum import IntEnum
from typing import Any, Tuple, Optional


class Opcode(IntEnum):
    # Control flow
    UNREACHABLE = 0x00
    NOP = 0x01
    BLOCK = 0x02
    LOOP = 0x03
    IF = 0x04
    ELSE = 0x05
    END = 0x0B
    BR = 0x0C
    BR_IF = 0x0D
    BR_TABLE = 0x0E
    RETURN = 0x0F
    CALL = 0x10
    CALL_INDIRECT = 0x11

    # Parametric
    DROP = 0x1A
    SELECT = 0x1B

    # Variable access
    LOCAL_GET = 0x20
    LOCAL_SET = 0x21
    LOCAL_TEE = 0x22
    GLOBAL_GET = 0x23
    GLOBAL_SET = 0x24

    # Memory load
    I32_LOAD = 0x28
    I64_LOAD = 0x29
    F32_LOAD = 0x2A
    F64_LOAD = 0x2B
    I32_LOAD8_S = 0x2C
    I32_LOAD8_U = 0x2D
    I32_LOAD16_S = 0x2E
    I32_LOAD16_U = 0x2F
    I64_LOAD8_S = 0x30
    I64_LOAD8_U = 0x31
    I64_LOAD16_S = 0x32
    I64_LOAD16_U = 0x33
    I64_LOAD32_S = 0x34
    I64_LOAD32_U = 0x35

    # Memory store
    I32_STORE = 0x36
    I64_STORE = 0x37
    F32_STORE = 0x38
    F64_STORE = 0x39
    I32_STORE8 = 0x3A
    I32_STORE16 = 0x3B
    I64_STORE8 = 0x3C
    I64_STORE16 = 0x3D
    I64_STORE32 = 0x3E
    MEMORY_SIZE = 0x3F
    MEMORY_GROW = 0x40

    # Constants
    I32_CONST = 0x41
    I64_CONST = 0x42
    F32_CONST = 0x43
    F64_CONST = 0x44

    # i32 Comparisons
    I32_EQZ = 0x45
    I32_EQ = 0x46
    I32_NE = 0x47
    I32_LT_S = 0x48
    I32_LT_U = 0x49
    I32_GT_S = 0x4A
    I32_GT_U = 0x4B
    I32_LE_S = 0x4C
    I32_LE_U = 0x4D
    I32_GE_S = 0x4E
    I32_GE_U = 0x4F

    # i64 Comparisons
    I64_EQZ = 0x50
    I64_EQ = 0x51
    I64_NE = 0x52
    I64_LT_S = 0x53
    I64_LT_U = 0x54
    I64_GT_S = 0x55
    I64_GT_U = 0x56
    I64_LE_S = 0x57
    I64_LE_U = 0x58
    I64_GE_S = 0x59
    I64_GE_U = 0x5A

    # f32 Comparisons
    F32_EQ = 0x5B
    F32_NE = 0x5C
    F32_LT = 0x5D
    F32_GT = 0x5E
    F32_LE = 0x5F
    F32_GE = 0x60

    # f64 Comparisons
    F64_EQ = 0x61
    F64_NE = 0x62
    F64_LT = 0x63
    F64_GT = 0x64
    F64_LE = 0x65
    F64_GE = 0x66

    # i32 Arithmetic & bitwise
    I32_CLZ = 0x67
    I32_CTZ = 0x68
    I32_POPCNT = 0x69
    I32_ADD = 0x6A
    I32_SUB = 0x6B
    I32_MUL = 0x6C
    I32_DIV_S = 0x6D
    I32_DIV_U = 0x6E
    I32_REM_S = 0x6F
    I32_REM_U = 0x70
    I32_AND = 0x71
    I32_OR = 0x72
    I32_XOR = 0x73
    I32_SHL = 0x74
    I32_SHR_S = 0x75
    I32_SHR_U = 0x76
    I32_ROTL = 0x77
    I32_ROTR = 0x78

    # i64 Arithmetic & bitwise
    I64_CLZ = 0x79
    I64_CTZ = 0x7A
    I64_POPCNT = 0x7B
    I64_ADD = 0x7C
    I64_SUB = 0x7D
    I64_MUL = 0x7E
    I64_DIV_S = 0x7F
    I64_DIV_U = 0x80
    I64_REM_S = 0x81
    I64_REM_U = 0x82
    I64_AND = 0x83
    I64_OR = 0x84
    I64_XOR = 0x85
    I64_SHL = 0x86
    I64_SHR_S = 0x87
    I64_SHR_U = 0x88
    I64_ROTL = 0x89
    I64_ROTR = 0x8A

    # f32 Arithmetic
    F32_ABS = 0x8B
    F32_NEG = 0x8C
    F32_CEIL = 0x8D
    F32_FLOOR = 0x8E
    F32_TRUNC = 0x8F
    F32_NEAREST = 0x90
    F32_SQRT = 0x91
    F32_ADD = 0x92
    F32_SUB = 0x93
    F32_MUL = 0x94
    F32_DIV = 0x95
    F32_MIN = 0x96
    F32_MAX = 0x97
    F32_COPYSIGN = 0x98

    # f64 Arithmetic
    F64_ABS = 0x99
    F64_NEG = 0x9A
    F64_CEIL = 0x9B
    F64_FLOOR = 0x9C
    F64_TRUNC = 0x9D
    F64_NEAREST = 0x9E
    F64_SQRT = 0x9F
    F64_ADD = 0xA0
    F64_SUB = 0xA1
    F64_MUL = 0xA2
    F64_DIV = 0xA3
    F64_MIN = 0xA4
    F64_MAX = 0xA5
    F64_COPYSIGN = 0xA6

    # Conversions
    I32_WRAP_I64 = 0xA7
    I32_TRUNC_F32_S = 0xA8
    I32_TRUNC_F32_U = 0xA9
    I32_TRUNC_F64_S = 0xAA
    I32_TRUNC_F64_U = 0xAB

    I64_EXTEND_I32_S = 0xAC
    I64_EXTEND_I32_U = 0xAD
    I64_TRUNC_F32_S = 0xAE
    I64_TRUNC_F32_U = 0xAF
    I64_TRUNC_F64_S = 0xB0
    I64_TRUNC_F64_U = 0xB1

    F32_CONVERT_I32_S = 0xB2
    F32_CONVERT_I32_U = 0xB3
    F32_CONVERT_I64_S = 0xB4
    F32_CONVERT_I64_U = 0xB5
    F32_DEMOTE_F64 = 0xB6

    F64_CONVERT_I32_S = 0xB7
    F64_CONVERT_I32_U = 0xB8
    F64_CONVERT_I64_S = 0xB9
    F64_CONVERT_I64_U = 0xBA
    F64_PROMOTE_F32 = 0xBB

    I32_REINTERPRET_F32 = 0xBC
    I64_REINTERPRET_F64 = 0xBD
    F32_REINTERPRET_I32 = 0xBE
    F64_REINTERPRET_I64 = 0xBF


# Mapping from opcode enum values to mnemonic names
OPCODE_MNEMONICS = {
    Opcode.UNREACHABLE: "unreachable",
    Opcode.NOP: "nop",
    Opcode.BLOCK: "block",
    Opcode.LOOP: "loop",
    Opcode.IF: "if",
    Opcode.ELSE: "else",
    Opcode.END: "end",
    Opcode.BR: "br",
    Opcode.BR_IF: "br_if",
    Opcode.BR_TABLE: "br_table",
    Opcode.RETURN: "return",
    Opcode.CALL: "call",
    Opcode.CALL_INDIRECT: "call_indirect",
    Opcode.DROP: "drop",
    Opcode.SELECT: "select",
    Opcode.LOCAL_GET: "local.get",
    Opcode.LOCAL_SET: "local.set",
    Opcode.LOCAL_TEE: "local.tee",
    Opcode.GLOBAL_GET: "global.get",
    Opcode.GLOBAL_SET: "global.set",
    Opcode.I32_LOAD: "i32.load",
    Opcode.I64_LOAD: "i64.load",
    Opcode.F32_LOAD: "f32.load",
    Opcode.F64_LOAD: "f64.load",
    Opcode.I32_LOAD8_S: "i32.load8_s",
    Opcode.I32_LOAD8_U: "i32.load8_u",
    Opcode.I32_LOAD16_S: "i32.load16_s",
    Opcode.I32_LOAD16_U: "i32.load16_u",
    Opcode.I64_LOAD8_S: "i64.load8_s",
    Opcode.I64_LOAD8_U: "i64.load8_u",
    Opcode.I64_LOAD16_S: "i64.load16_s",
    Opcode.I64_LOAD16_U: "i64.load16_u",
    Opcode.I64_LOAD32_S: "i64.load32_s",
    Opcode.I64_LOAD32_U: "i64.load32_u",
    Opcode.I32_STORE: "i32.store",
    Opcode.I64_STORE: "i64.store",
    Opcode.F32_STORE: "f32.store",
    Opcode.F64_STORE: "f64.store",
    Opcode.I32_STORE8: "i32.store8",
    Opcode.I32_STORE16: "i32.store16",
    Opcode.I64_STORE8: "i64.store8",
    Opcode.I64_STORE16: "i64.store16",
    Opcode.I64_STORE32: "i64.store32",
    Opcode.MEMORY_SIZE: "memory.size",
    Opcode.MEMORY_GROW: "memory.grow",
    Opcode.I32_CONST: "i32.const",
    Opcode.I64_CONST: "i64.const",
    Opcode.F32_CONST: "f32.const",
    Opcode.F64_CONST: "f64.const",
    Opcode.I32_EQZ: "i32.eqz",
    Opcode.I32_EQ: "i32.eq",
    Opcode.I32_NE: "i32.ne",
    Opcode.I32_LT_S: "i32.lt_s",
    Opcode.I32_LT_U: "i32.lt_u",
    Opcode.I32_GT_S: "i32.gt_s",
    Opcode.I32_GT_U: "i32.gt_u",
    Opcode.I32_LE_S: "i32.le_s",
    Opcode.I32_LE_U: "i32.le_u",
    Opcode.I32_GE_S: "i32.ge_s",
    Opcode.I32_GE_U: "i32.ge_u",
    Opcode.I64_EQZ: "i64.eqz",
    Opcode.I64_EQ: "i64.eq",
    Opcode.I64_NE: "i64.ne",
    Opcode.I64_LT_S: "i64.lt_s",
    Opcode.I64_LT_U: "i64.lt_u",
    Opcode.I64_GT_S: "i64.gt_s",
    Opcode.I64_GT_U: "i64.gt_u",
    Opcode.I64_LE_S: "i64.le_s",
    Opcode.I64_LE_U: "i64.le_u",
    Opcode.I64_GE_S: "i64.ge_s",
    Opcode.I64_GE_U: "i64.ge_u",
    Opcode.F32_EQ: "f32.eq",
    Opcode.F32_NE: "f32.ne",
    Opcode.F32_LT: "f32.lt",
    Opcode.F32_GT: "f32.gt",
    Opcode.F32_LE: "f32.le",
    Opcode.F32_GE: "f32.ge",
    Opcode.F64_EQ: "f64.eq",
    Opcode.F64_NE: "f64.ne",
    Opcode.F64_LT: "f64.lt",
    Opcode.F64_GT: "f64.gt",
    Opcode.F64_LE: "f64.le",
    Opcode.F64_GE: "f64.ge",
    Opcode.I32_CLZ: "i32.clz",
    Opcode.I32_CTZ: "i32.ctz",
    Opcode.I32_POPCNT: "i32.popcnt",
    Opcode.I32_ADD: "i32.add",
    Opcode.I32_SUB: "i32.sub",
    Opcode.I32_MUL: "i32.mul",
    Opcode.I32_DIV_S: "i32.div_s",
    Opcode.I32_DIV_U: "i32.div_u",
    Opcode.I32_REM_S: "i32.rem_s",
    Opcode.I32_REM_U: "i32.rem_u",
    Opcode.I32_AND: "i32.and",
    Opcode.I32_OR: "i32.or",
    Opcode.I32_XOR: "i32.xor",
    Opcode.I32_SHL: "i32.shl",
    Opcode.I32_SHR_S: "i32.shr_s",
    Opcode.I32_SHR_U: "i32.shr_u",
    Opcode.I32_ROTL: "i32.rotl",
    Opcode.I32_ROTR: "i32.rotr",
    Opcode.I64_CLZ: "i64.clz",
    Opcode.I64_CTZ: "i64.ctz",
    Opcode.I64_POPCNT: "i64.popcnt",
    Opcode.I64_ADD: "i64.add",
    Opcode.I64_SUB: "i64.sub",
    Opcode.I64_MUL: "i64.mul",
    Opcode.I64_DIV_S: "i64.div_s",
    Opcode.I64_DIV_U: "i64.div_u",
    Opcode.I64_REM_S: "i64.rem_s",
    Opcode.I64_REM_U: "i64.rem_u",
    Opcode.I64_AND: "i64.and",
    Opcode.I64_OR: "i64.or",
    Opcode.I64_XOR: "i64.xor",
    Opcode.I64_SHL: "i64.shl",
    Opcode.I64_SHR_S: "i64.shr_s",
    Opcode.I64_SHR_U: "i64.shr_u",
    Opcode.I64_ROTL: "i64.rotl",
    Opcode.I64_ROTR: "i64.rotr",
    Opcode.F32_ABS: "f32.abs",
    Opcode.F32_NEG: "f32.neg",
    Opcode.F32_CEIL: "f32.ceil",
    Opcode.F32_FLOOR: "f32.floor",
    Opcode.F32_TRUNC: "f32.trunc",
    Opcode.F32_NEAREST: "f32.nearest",
    Opcode.F32_SQRT: "f32.sqrt",
    Opcode.F32_ADD: "f32.add",
    Opcode.F32_SUB: "f32.sub",
    Opcode.F32_MUL: "f32.mul",
    Opcode.F32_DIV: "f32.div",
    Opcode.F32_MIN: "f32.min",
    Opcode.F32_MAX: "f32.max",
    Opcode.F32_COPYSIGN: "f32.copysign",
    Opcode.F64_ABS: "f64.abs",
    Opcode.F64_NEG: "f64.neg",
    Opcode.F64_CEIL: "f64.ceil",
    Opcode.F64_FLOOR: "f64.floor",
    Opcode.F64_TRUNC: "f64.trunc",
    Opcode.F64_NEAREST: "f64.nearest",
    Opcode.F64_SQRT: "f64.sqrt",
    Opcode.F64_ADD: "f64.add",
    Opcode.F64_SUB: "f64.sub",
    Opcode.F64_MUL: "f64.mul",
    Opcode.F64_DIV: "f64.div",
    Opcode.F64_MIN: "f64.min",
    Opcode.F64_MAX: "f64.max",
    Opcode.F64_COPYSIGN: "f64.copysign",
    Opcode.I32_WRAP_I64: "i32.wrap_i64",
    Opcode.I32_TRUNC_F32_S: "i32.trunc_f32_s",
    Opcode.I32_TRUNC_F32_U: "i32.trunc_f32_u",
    Opcode.I32_TRUNC_F64_S: "i32.trunc_f64_s",
    Opcode.I32_TRUNC_F64_U: "i32.trunc_f64_u",
    Opcode.I64_EXTEND_I32_S: "i64.extend_i32_s",
    Opcode.I64_EXTEND_I32_U: "i64.extend_i32_u",
    Opcode.I64_TRUNC_F32_S: "i64.trunc_f32_s",
    Opcode.I64_TRUNC_F32_U: "i64.trunc_f32_u",
    Opcode.I64_TRUNC_F64_S: "i64.trunc_f64_s",
    Opcode.I64_TRUNC_F64_U: "i64.trunc_f64_u",
    Opcode.F32_CONVERT_I32_S: "f32.convert_i32_s",
    Opcode.F32_CONVERT_I32_U: "f32.convert_i32_u",
    Opcode.F32_CONVERT_I64_S: "f32.convert_i64_s",
    Opcode.F32_CONVERT_I64_U: "f32.convert_i64_u",
    Opcode.F32_DEMOTE_F64: "f32.demote_f64",
    Opcode.F64_CONVERT_I32_S: "f64.convert_i32_s",
    Opcode.F64_CONVERT_I32_U: "f64.convert_i32_u",
    Opcode.F64_CONVERT_I64_S: "f64.convert_i64_s",
    Opcode.F64_CONVERT_I64_U: "f64.convert_i64_u",
    Opcode.F64_PROMOTE_F32: "f64.promote_f32",
    Opcode.I32_REINTERPRET_F32: "i32.reinterpret_f32",
    Opcode.I64_REINTERPRET_F64: "i64.reinterpret_f64",
    Opcode.F32_REINTERPRET_I32: "f32.reinterpret_i32",
    Opcode.F64_REINTERPRET_I64: "f64.reinterpret_i64",
}


class MemArg:
    """Memory argument: alignment exponent and byte offset."""
    __slots__ = ("align", "offset")

    def __init__(self, align: int, offset: int) -> None:
        self.align = int(align)
        self.offset = int(offset)

    def __repr__(self) -> str:
        return f"MemArg(align={self.align}, offset={self.offset})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MemArg):
            return False
        return self.align == other.align and self.offset == other.offset


class Instruction:
    """
    Decoded WebAssembly bytecode instruction.
    Contains opcode, immediate operands, and bytecode stream offset.
    """
    __slots__ = ("opcode", "operands", "offset")

    def __init__(self, opcode: Opcode, operands: Any = None, offset: int = 0) -> None:
        self.opcode = opcode
        self.operands = operands
        self.offset = offset

    @property
    def mnemonic(self) -> str:
        return OPCODE_MNEMONICS.get(self.opcode, f"unknown_0x{self.opcode:02X}")

    def __repr__(self) -> str:
        if self.operands is not None:
            return f"{self.mnemonic} {self.operands}"
        return self.mnemonic

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Instruction):
            return False
        return self.opcode == other.opcode and self.operands == other.operands
