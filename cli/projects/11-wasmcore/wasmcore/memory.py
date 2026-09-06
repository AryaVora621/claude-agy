"""
WasmCore: Linear Memory Management Subsystem.
WebAssembly 64KB paged linear memory with boundary and alignment protection.
"""

import struct
from typing import Optional
from .types import Limits, WasmTrap

PAGE_SIZE = 65536  # 64 KB
MAX_WASM_PAGES = 65536  # 4 GB address space cap in 32-bit WASM


class LinearMemory:
    """
    WebAssembly Linear Memory buffer.
    Dynamically resizable in units of 64KB pages with bounds-checked access.
    """
    __slots__ = ("limits", "data")

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        initial_bytes = limits.min * PAGE_SIZE
        self.data = bytearray(initial_bytes)

    @property
    def pages(self) -> int:
        """Current number of allocated 64KB pages."""
        return len(self.data) // PAGE_SIZE

    def size(self) -> int:
        """WebAssembly memory.size opcode semantic (returns page count)."""
        return self.pages

    def grow(self, delta: int) -> int:
        """
        WebAssembly memory.grow opcode semantic.
        Allocates delta additional 64KB pages up to the declared maximum limit.
        Returns previous page count on success, or -1 on allocation failure.
        """
        old_pages = self.pages
        if delta < 0:
            return -1
        if delta == 0:
            return old_pages

        new_pages = old_pages + delta
        if new_pages > MAX_WASM_PAGES:
            return -1
        if self.limits.max is not None and new_pages > self.limits.max:
            return -1

        additional_bytes = delta * PAGE_SIZE
        try:
            self.data.extend(b"\x00" * additional_bytes)
        except (MemoryError, OverflowError):
            return -1

        return old_pages

    def _check_bounds(self, addr: int, size: int) -> None:
        if addr < 0 or (addr + size) > len(self.data):
            raise WasmTrap(f"out of bounds memory access: addr={addr}, size={size}, mem_len={len(self.data)}")

    def read_bytes(self, addr: int, length: int) -> bytes:
        self._check_bounds(addr, length)
        return bytes(self.data[addr:addr + length])

    def write_bytes(self, addr: int, src: bytes) -> None:
        self._check_bounds(addr, len(src))
        self.data[addr:addr + len(src)] = src

    # Loads
    def load_i32(self, addr: int) -> int:
        self._check_bounds(addr, 4)
        return struct.unpack("<i", self.data[addr:addr + 4])[0]

    def load_u32(self, addr: int) -> int:
        self._check_bounds(addr, 4)
        return struct.unpack("<I", self.data[addr:addr + 4])[0]

    def load_i8_s(self, addr: int) -> int:
        self._check_bounds(addr, 1)
        return struct.unpack("<b", self.data[addr:addr + 1])[0]

    def load_i8_u(self, addr: int) -> int:
        self._check_bounds(addr, 1)
        return struct.unpack("<B", self.data[addr:addr + 1])[0]

    def load_i16_s(self, addr: int) -> int:
        self._check_bounds(addr, 2)
        return struct.unpack("<h", self.data[addr:addr + 2])[0]

    def load_i16_u(self, addr: int) -> int:
        self._check_bounds(addr, 2)
        return struct.unpack("<H", self.data[addr:addr + 2])[0]

    def load_i64(self, addr: int) -> int:
        self._check_bounds(addr, 8)
        return struct.unpack("<q", self.data[addr:addr + 8])[0]

    def load_u64(self, addr: int) -> int:
        self._check_bounds(addr, 8)
        return struct.unpack("<Q", self.data[addr:addr + 8])[0]

    def load_i64_32_s(self, addr: int) -> int:
        return self.load_i32(addr)

    def load_i64_32_u(self, addr: int) -> int:
        return self.load_u32(addr)

    def load_f32(self, addr: int) -> float:
        self._check_bounds(addr, 4)
        return struct.unpack("<f", self.data[addr:addr + 4])[0]

    def load_f64(self, addr: int) -> float:
        self._check_bounds(addr, 8)
        return struct.unpack("<d", self.data[addr:addr + 8])[0]

    # Stores
    def store_i32(self, addr: int, val: int) -> None:
        self._check_bounds(addr, 4)
        self.data[addr:addr + 4] = struct.pack("<I", int(val) & 0xFFFFFFFF)

    def store_i8(self, addr: int, val: int) -> None:
        self._check_bounds(addr, 1)
        self.data[addr:addr + 1] = struct.pack("<B", int(val) & 0xFF)

    def store_i16(self, addr: int, val: int) -> None:
        self._check_bounds(addr, 2)
        self.data[addr:addr + 2] = struct.pack("<H", int(val) & 0xFFFF)

    def store_i64(self, addr: int, val: int) -> None:
        self._check_bounds(addr, 8)
        self.data[addr:addr + 8] = struct.pack("<Q", int(val) & 0xFFFFFFFFFFFFFFFF)

    def store_f32(self, addr: int, val: float) -> None:
        self._check_bounds(addr, 4)
        self.data[addr:addr + 4] = struct.pack("<f", float(val))

    def store_f64(self, addr: int, val: float) -> None:
        self._check_bounds(addr, 8)
        self.data[addr:addr + 8] = struct.pack("<d", float(val))
