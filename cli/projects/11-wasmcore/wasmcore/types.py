"""
WasmCore: Value Types, Function Signatures, and Value Representations.
Spec-compliant WebAssembly MVP type definitions and boxing.
"""

from enum import IntEnum
from typing import Tuple, Optional, Any, Union
import struct


class ValType(IntEnum):
    """WebAssembly Value Types encoded as negative 7-bit integers."""
    I32 = 0x7F
    I64 = 0x7E
    F32 = 0x7D
    F64 = 0x7C
    FUNCREF = 0x70
    EXTERNREF = 0x6F

    def __str__(self) -> str:
        names = {
            ValType.I32: "i32",
            ValType.I64: "i64",
            ValType.F32: "f32",
            ValType.F64: "f64",
            ValType.FUNCREF: "funcref",
            ValType.EXTERNREF: "externref",
        }
        return names.get(self, f"type_{hex(self.value)}")


class ExportDesc(IntEnum):
    """External export description tags."""
    FUNC = 0x00
    TABLE = 0x01
    MEM = 0x02
    GLOBAL = 0x03


class SectionId(IntEnum):
    """WebAssembly binary section IDs."""
    CUSTOM = 0
    TYPE = 1
    IMPORT = 2
    FUNCTION = 3
    TABLE = 4
    MEMORY = 5
    GLOBAL = 6
    EXPORT = 7
    START = 8
    ELEMENT = 9
    CODE = 10
    DATA = 11
    DATA_COUNT = 12


class Limits:
    """Memory and table size limits."""
    __slots__ = ("min", "max")

    def __init__(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        *,
        min: Optional[int] = None,
        max: Optional[int] = None
    ) -> None:
        actual_min = min if min is not None else (min_size if min_size is not None else 0)
        actual_max = max if max is not None else max_size
        self.min = int(actual_min)
        self.max = int(actual_max) if actual_max is not None else None

    def __repr__(self) -> str:
        return f"Limits(min={self.min}, max={self.max})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Limits):
            return False
        return self.min == other.min and self.max == other.max


class FuncType:
    """WebAssembly function type signature: (params) -> (results)."""
    __slots__ = ("params", "results")

    def __init__(
        self,
        params: Union[Tuple[ValType, ...], list],
        results: Union[Tuple[ValType, ...], list]
    ) -> None:
        self.params = tuple(params)
        self.results = tuple(results)

    def __repr__(self) -> str:
        params_str = ", ".join(str(p) for p in self.params)
        results_str = ", ".join(str(r) for r in self.results)
        return f"FuncType([{params_str}] -> [{results_str}])"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FuncType):
            return False
        return self.params == other.params and self.results == other.results

    def __hash__(self) -> int:
        return hash((self.params, self.results))


class TableType:
    """Table type definition."""
    __slots__ = ("element_type", "limits")

    def __init__(self, element_type: ValType, limits: Limits) -> None:
        self.element_type = element_type
        self.limits = limits

    def __repr__(self) -> str:
        return f"TableType({self.element_type}, {self.limits})"


class MemoryType:
    """Memory type definition."""
    __slots__ = ("limits",)

    def __init__(self, limits: Limits) -> None:
        self.limits = limits

    def __repr__(self) -> str:
        return f"MemoryType({self.limits})"


class GlobalType:
    """Global variable type definition."""
    __slots__ = ("val_type", "mutable")

    def __init__(self, val_type: ValType, mutable: bool) -> None:
        self.val_type = val_type
        self.mutable = bool(mutable)

    def __repr__(self) -> str:
        mut_str = "var" if self.mutable else "const"
        return f"GlobalType({mut_str} {self.val_type})"


class WasmTrap(Exception):
    """Runtime execution trap in WebAssembly virtual machine."""
    pass


class WasmValidationError(Exception):
    """Validation or decoding error in WebAssembly binary module."""
    pass


# Bit manipulation & two's complement utilities
MASK_32 = 0xFFFFFFFF
MASK_64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT_32 = 1 << 31
SIGN_BIT_64 = 1 << 63


def to_signed_32(n: int) -> int:
    """Convert unsigned 32-bit integer to signed Python integer."""
    n = n & MASK_32
    return n - (1 << 32) if n & SIGN_BIT_32 else n


def to_unsigned_32(n: int) -> int:
    """Convert signed Python integer to unsigned 32-bit integer."""
    return n & MASK_32


def to_signed_64(n: int) -> int:
    """Convert unsigned 64-bit integer to signed Python integer."""
    n = n & MASK_64
    return n - (1 << 64) if n & SIGN_BIT_64 else n


def to_unsigned_64(n: int) -> int:
    """Convert signed Python integer to unsigned 64-bit integer."""
    return n & MASK_64


def float_to_i32_bits(f: float) -> int:
    """Bitcast IEEE-754 32-bit float to unsigned 32-bit integer."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def i32_bits_to_float(i: int) -> float:
    """Bitcast unsigned 32-bit integer to IEEE-754 32-bit float."""
    return struct.unpack("<f", struct.pack("<I", i & MASK_32))[0]


def double_to_i64_bits(d: float) -> int:
    """Bitcast IEEE-754 64-bit float to unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def i64_bits_to_double(i: int) -> float:
    """Bitcast unsigned 64-bit integer to IEEE-754 64-bit float."""
    return struct.unpack("<d", struct.pack("<Q", i & MASK_64))[0]


class Value:
    """
    Strongly-typed WebAssembly value box.
    Stores raw bits and type tag, with arithmetic normalization.
    """
    __slots__ = ("val_type", "raw")

    def __init__(self, val_type: ValType, raw: Union[int, float]) -> None:
        self.val_type = val_type
        if val_type == ValType.I32:
            self.raw = to_signed_32(int(raw))
        elif val_type == ValType.I64:
            self.raw = to_signed_64(int(raw))
        elif val_type == ValType.F32:
            # Canonicalize single precision float
            bits = float_to_i32_bits(float(raw))
            self.raw = i32_bits_to_float(bits)
        elif val_type == ValType.F64:
            self.raw = float(raw)
        else:
            self.raw = raw

    @classmethod
    def i32(cls, val: int) -> "Value":
        return cls(ValType.I32, val)

    @classmethod
    def i64(cls, val: int) -> "Value":
        return cls(ValType.I64, val)

    @classmethod
    def f32(cls, val: float) -> "Value":
        return cls(ValType.F32, val)

    @classmethod
    def f64(cls, val: float) -> "Value":
        return cls(ValType.F64, val)

    def as_i32(self) -> int:
        if self.val_type != ValType.I32:
            raise WasmTrap(f"Type mismatch: expected i32, got {self.val_type}")
        return self.raw

    def as_u32(self) -> int:
        return to_unsigned_32(self.as_i32())

    def as_i64(self) -> int:
        if self.val_type != ValType.I64:
            raise WasmTrap(f"Type mismatch: expected i64, got {self.val_type}")
        return self.raw

    def as_u64(self) -> int:
        return to_unsigned_64(self.as_i64())

    def as_f32(self) -> float:
        if self.val_type != ValType.F32:
            raise WasmTrap(f"Type mismatch: expected f32, got {self.val_type}")
        return self.raw

    def as_f64(self) -> float:
        if self.val_type != ValType.F64:
            raise WasmTrap(f"Type mismatch: expected f64, got {self.val_type}")
        return self.raw

    def to_py(self) -> Union[int, float]:
        """Extract native Python value."""
        return self.raw

    @classmethod
    def default_for_type(cls, val_type: ValType) -> "Value":
        """Return zero value for given type."""
        if val_type == ValType.I32:
            return cls.i32(0)
        elif val_type == ValType.I64:
            return cls.i64(0)
        elif val_type == ValType.F32:
            return cls.f32(0.0)
        elif val_type == ValType.F64:
            return cls.f64(0.0)
        return cls(val_type, 0)

    def __repr__(self) -> str:
        return f"{self.val_type}:{self.raw}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Value):
            return False
        return self.val_type == other.val_type and self.raw == other.raw
