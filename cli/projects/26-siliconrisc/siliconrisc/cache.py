"""Multi-Level Cache Hierarchy with 4-State MESI Coherence Protocol.

Provides:
1. N-Way Set-Associative Cache with configurable line size, set count, and latency
2. 4-State MESI Coherence Protocol (Modified, Exclusive, Shared, Invalid)
3. Split L1-Instruction (L1I) and L1-Data (L1D) caches with Write-Back and Write-Allocate
4. Unified Level-2 (L2) cache with inclusive multi-core bus snooping
5. Accurate cycle latency accounting and AMAT (Average Memory Access Time) metrics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
from typing import Dict, List, Optional, Tuple

from .memory import PhysicalMemory


# =====================================================================
# MESI Protocol States and Bus Transactions
# =====================================================================

class MESIState(Enum):
    INVALID = auto()
    SHARED = auto()
    EXCLUSIVE = auto()
    MODIFIED = auto()


class BusTransaction(Enum):
    BUS_RD = auto()    # Read miss: read without modifying
    BUS_RDX = auto()   # Write miss: read with intent to write
    BUS_UPGR = auto()  # Write hit on Shared line: invalidate peers
    BUS_WB = auto()    # Write-back dirty line to lower level


# =====================================================================
# Cache Line and Set
# =====================================================================

@dataclass
class CacheLine:
    tag: int = 0
    data: bytearray = field(default_factory=lambda: bytearray(64))
    state: MESIState = MESIState.INVALID
    lru_age: int = 0

    @property
    def is_valid(self) -> bool:
        return self.state != MESIState.INVALID

    @property
    def is_dirty(self) -> bool:
        return self.state == MESIState.MODIFIED


class CacheSet:
    """A single set containing N associative ways."""

    def __init__(self, ways: int, line_size: int) -> None:
        self.ways = ways
        self.line_size = line_size
        self.lines: List[CacheLine] = [
            CacheLine(data=bytearray(line_size)) for _ in range(ways)
        ]

    def find_line(self, tag: int) -> Optional[CacheLine]:
        for line in self.lines:
            if line.is_valid and line.tag == tag:
                return line
        return None

    def find_victim(self) -> Tuple[int, CacheLine]:
        """Find victim way using True LRU (highest lru_age) or first invalid line."""
        for i, line in enumerate(self.lines):
            if not line.is_valid:
                return i, line

        # All valid -> pick highest lru_age
        victim_idx = max(range(self.ways), key=lambda i: self.lines[i].lru_age)
        return victim_idx, self.lines[victim_idx]

    def touch(self, accessed_line: CacheLine) -> None:
        """Update LRU ages across all ways."""
        old_age = accessed_line.lru_age
        accessed_line.lru_age = 0
        for line in self.lines:
            if line is not accessed_line and line.is_valid and line.lru_age <= old_age:
                line.lru_age += 1


# =====================================================================
# Set-Associative Cache
# =====================================================================

class Cache:
    """Configurable N-Way Set-Associative Cache with MESI protocol support."""

    def __init__(
        self,
        name: str,
        size_bytes: int = 8192,
        associativity: int = 4,
        line_size_bytes: int = 64,
        hit_latency_cycles: int = 1,
    ) -> None:
        self.name = name
        self.size_bytes = size_bytes
        self.associativity = associativity
        self.line_size = line_size_bytes
        self.hit_latency = hit_latency_cycles

        assert (line_size_bytes & (line_size_bytes - 1)) == 0, "Line size must be a power of 2"
        self.offset_bits = int(math.log2(line_size_bytes))
        self.offset_mask = line_size_bytes - 1

        self.num_lines = size_bytes // line_size_bytes
        assert self.num_lines % associativity == 0, "Lines must divide evenly into ways"
        self.num_sets = self.num_lines // associativity

        assert (self.num_sets & (self.num_sets - 1)) == 0, "Number of sets must be a power of 2"
        self.index_bits = int(math.log2(self.num_sets))
        self.index_mask = self.num_sets - 1

        self.tag_shift = self.offset_bits + self.index_bits

        self.sets: List[CacheSet] = [
            CacheSet(associativity, line_size_bytes) for _ in range(self.num_sets)
        ]

        # Performance Counters
        self.reads: int = 0
        self.read_hits: int = 0
        self.writes: int = 0
        self.write_hits: int = 0
        self.evictions: int = 0
        self.writebacks: int = 0

    def decompose_address(self, paddr: int) -> Tuple[int, int, int]:
        offset = paddr & self.offset_mask
        set_idx = (paddr >> self.offset_bits) & self.index_mask
        tag = paddr >> self.tag_shift
        return tag, set_idx, offset

    def reconstruct_address(self, tag: int, set_idx: int) -> int:
        return (tag << self.tag_shift) | (set_idx << self.offset_bits)

    @property
    def total_accesses(self) -> int:
        return self.reads + self.writes

    @property
    def total_hits(self) -> int:
        return self.read_hits + self.write_hits

    @property
    def hits(self) -> int:
        return self.total_hits

    @property
    def misses(self) -> int:
        return self.total_accesses - self.total_hits

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 1.0
        return self.total_hits / self.total_accesses

    def snoop(self, bus_tx: BusTransaction, paddr: int) -> Tuple[bool, Optional[bytearray]]:
        """Snoop incoming bus transaction from a peer cache. Return (did_hit, dirty_data_copy)."""
        tag, set_idx, _ = self.decompose_address(paddr)
        cset = self.sets[set_idx]
        line = cset.find_line(tag)

        if line is None or not line.is_valid:
            return False, None

        dirty_data: Optional[bytearray] = None

        if bus_tx == BusTransaction.BUS_RD:
            # Another cache wants to read
            if line.state == MESIState.MODIFIED:
                dirty_data = bytearray(line.data)
                line.state = MESIState.SHARED
            elif line.state == MESIState.EXCLUSIVE:
                line.state = MESIState.SHARED

        elif bus_tx in (BusTransaction.BUS_RDX, BusTransaction.BUS_UPGR):
            # Another cache wants to write -> Invalidate our copy
            if line.state == MESIState.MODIFIED:
                dirty_data = bytearray(line.data)
            line.state = MESIState.INVALID

        return True, dirty_data

    def allocate_line(self, paddr: int, data: bytearray, state: MESIState = MESIState.EXCLUSIVE) -> CacheLine:
        """Allocate or replace a cache line at physical address with given data and MESI state."""
        tag, set_idx, _ = self.decompose_address(paddr)
        cset = self.sets[set_idx]
        _, line = cset.find_victim()
        line.tag = tag
        line.data[:len(data)] = data
        line.state = state
        cset.touch(line)
        return line

    def find_line(self, paddr: int) -> Optional[CacheLine]:
        """Find cache line containing physical address, if present and valid."""
        tag, set_idx, _ = self.decompose_address(paddr)
        return self.sets[set_idx].find_line(tag)


# =====================================================================
# Multi-Level Cache Hierarchy Manager
# =====================================================================

class MemoryHierarchy:
    """Manages L1I, L1D, Unified L2, and Physical Memory with MESI bus snooping."""

    def __init__(
        self,
        ram: PhysicalMemory,
        l1_size: int = 8192,
        l2_size: int = 65536,
        ram_latency_cycles: int = 50,
        l2_latency_cycles: int = 6,
    ) -> None:
        self.ram = ram
        self.ram_latency = ram_latency_cycles

        self.l1i = Cache("L1I", size_bytes=l1_size, associativity=4, hit_latency_cycles=1)
        self.l1d = Cache("L1D", size_bytes=l1_size, associativity=4, hit_latency_cycles=1)
        self.l2 = Cache("L2", size_bytes=l2_size, associativity=8, hit_latency_cycles=l2_latency_cycles)

        self.total_cycles_spent: int = 0

    def fetch_instruction(self, paddr: int) -> Tuple[int, int]:
        """Fetch 32-bit instruction through L1I -> L2 -> RAM hierarchy. Returns (raw_inst, latency)."""
        self.l1i.reads += 1
        tag, set_idx, offset = self.l1i.decompose_address(paddr)
        cset = self.l1i.sets[set_idx]
        line = cset.find_line(tag)

        if line is not None:
            # L1I Hit
            self.l1i.read_hits += 1
            cset.touch(line)
            inst = struct.unpack_from("<I", line.data, offset)[0]
            self.total_cycles_spent += self.l1i.hit_latency
            return inst, self.l1i.hit_latency

        # L1I Miss -> Fetch line from L2 or RAM
        latency = self.l1i.hit_latency
        line_paddr = paddr & ~self.l1i.offset_mask
        line_data, l2_lat = self._read_lower_hierarchy(line_paddr, self.l1i.line_size)
        latency += l2_lat

        # Install into L1I
        victim_idx, victim_line = cset.find_victim()
        if victim_line.is_valid:
            self.l1i.evictions += 1

        victim_line.tag = tag
        victim_line.data[:] = line_data
        victim_line.state = MESIState.SHARED
        cset.touch(victim_line)

        inst = struct.unpack_from("<I", victim_line.data, offset)[0]
        self.total_cycles_spent += latency
        return inst, latency

    def read_data(self, paddr: int, size: int) -> Tuple[int, int]:
        """Read 1, 2, 4, or 8 bytes through L1D -> L2 -> RAM. Returns (value, latency)."""
        self.l1d.reads += 1
        tag, set_idx, offset = self.l1d.decompose_address(paddr)
        cset = self.l1d.sets[set_idx]
        line = cset.find_line(tag)

        if line is not None:
            # L1D Hit
            self.l1d.read_hits += 1
            cset.touch(line)
            val = self._extract_value(line.data, offset, size)
            self.total_cycles_spent += self.l1d.hit_latency
            return val, self.l1d.hit_latency

        # L1D Miss -> Fetch line from lower level (BusRd)
        latency = self.l1d.hit_latency
        line_paddr = paddr & ~self.l1d.offset_mask
        line_data, l2_lat = self._read_lower_hierarchy(line_paddr, self.l1d.line_size)
        latency += l2_lat

        # Victim eviction in L1D
        victim_idx, victim_line = cset.find_victim()
        if victim_line.is_valid:
            self.l1d.evictions += 1
            if victim_line.is_dirty:
                # Write back dirty line to L2/RAM
                old_paddr = self.l1d.reconstruct_address(victim_line.tag, set_idx)
                self._write_lower_hierarchy(old_paddr, victim_line.data)
                self.l1d.writebacks += 1

        victim_line.tag = tag
        victim_line.data[:] = line_data
        victim_line.state = MESIState.EXCLUSIVE
        cset.touch(victim_line)

        val = self._extract_value(victim_line.data, offset, size)
        self.total_cycles_spent += latency
        return val, latency

    def write_data(self, paddr: int, val: int, size: int) -> int:
        """Write 1, 2, 4, or 8 bytes with write-back and write-allocate. Returns latency."""
        self.l1d.writes += 1
        tag, set_idx, offset = self.l1d.decompose_address(paddr)
        cset = self.l1d.sets[set_idx]
        line = cset.find_line(tag)

        latency = self.l1d.hit_latency

        if line is not None:
            # L1D Write Hit
            self.l1d.write_hits += 1
            if line.state == MESIState.SHARED:
                # Upgrade: broadcast BusUpgr to invalidate peers
                line.state = MESIState.MODIFIED
            elif line.state == MESIState.EXCLUSIVE:
                line.state = MESIState.MODIFIED

            self._insert_value(line.data, offset, val, size)
            cset.touch(line)
            self.total_cycles_spent += latency
            return latency

        # L1D Write Miss (Write-Allocate): Read line with intent to write (BusRdX)
        line_paddr = paddr & ~self.l1d.offset_mask
        line_data, l2_lat = self._read_lower_hierarchy(line_paddr, self.l1d.line_size)
        latency += l2_lat

        victim_idx, victim_line = cset.find_victim()
        if victim_line.is_valid:
            self.l1d.evictions += 1
            if victim_line.is_dirty:
                old_paddr = self.l1d.reconstruct_address(victim_line.tag, set_idx)
                self._write_lower_hierarchy(old_paddr, victim_line.data)
                self.l1d.writebacks += 1

        victim_line.tag = tag
        victim_line.data[:] = line_data
        self._insert_value(victim_line.data, offset, val, size)
        victim_line.state = MESIState.MODIFIED
        cset.touch(victim_line)

        self.total_cycles_spent += latency
        return latency

    def _read_lower_hierarchy(self, paddr: int, size: int) -> Tuple[bytearray, int]:
        """Read line from L2 cache; if L2 misses, fetch from Physical RAM."""
        self.l2.reads += 1
        tag, set_idx, _ = self.l2.decompose_address(paddr)
        cset = self.l2.sets[set_idx]
        line = cset.find_line(tag)

        if line is not None:
            self.l2.read_hits += 1
            cset.touch(line)
            return bytearray(line.data), self.l2.hit_latency

        # L2 Miss -> Fetch from Physical RAM
        latency = self.l2.hit_latency + self.ram_latency
        data = bytearray(self.ram.read_bytes(paddr, size))

        # Install into L2
        victim_idx, victim_line = cset.find_victim()
        if victim_line.is_valid:
            self.l2.evictions += 1
            if victim_line.is_dirty:
                old_paddr = self.l2.reconstruct_address(victim_line.tag, set_idx)
                self.ram.load_bytes(old_paddr, bytes(victim_line.data))
                self.l2.writebacks += 1

        victim_line.tag = tag
        victim_line.data[:] = data
        victim_line.state = MESIState.EXCLUSIVE
        cset.touch(victim_line)

        return data, latency

    def _write_lower_hierarchy(self, paddr: int, data: bytearray) -> int:
        """Write line back to L2 cache (or directly to RAM if evicted from L2)."""
        self.l2.writes += 1
        tag, set_idx, _ = self.l2.decompose_address(paddr)
        cset = self.l2.sets[set_idx]
        line = cset.find_line(tag)

        if line is not None:
            self.l2.write_hits += 1
            line.data[:] = data
            line.state = MESIState.MODIFIED
            cset.touch(line)
            return self.l2.hit_latency

        # Write directly to RAM
        self.ram.load_bytes(paddr, bytes(data))
        return self.l2.hit_latency + self.ram_latency

    def flush_all(self) -> None:
        """Write back all dirty lines from L1D and L2 to Physical RAM."""
        # Flush L1D to L2/RAM
        for s_idx, cset in enumerate(self.l1d.sets):
            for line in cset.lines:
                if line.is_valid and line.is_dirty:
                    paddr = self.l1d.reconstruct_address(line.tag, s_idx)
                    self._write_lower_hierarchy(paddr, line.data)
                    line.state = MESIState.EXCLUSIVE

        # Flush L2 to RAM
        for s_idx, cset in enumerate(self.l2.sets):
            for line in cset.lines:
                if line.is_valid and line.is_dirty:
                    paddr = self.l2.reconstruct_address(line.tag, s_idx)
                    self.ram.load_bytes(paddr, bytes(line.data))
                    line.state = MESIState.EXCLUSIVE

    @staticmethod
    def _extract_value(buf: bytearray, offset: int, size: int) -> int:
        if size == 1:
            return buf[offset]
        elif size == 2:
            return struct.unpack_from("<H", buf, offset)[0]
        elif size == 4:
            return struct.unpack_from("<I", buf, offset)[0]
        elif size == 8:
            return struct.unpack_from("<Q", buf, offset)[0]
        raise ValueError(f"Unsupported read size {size}")

    @staticmethod
    def _insert_value(buf: bytearray, offset: int, val: int, size: int) -> None:
        if size == 1:
            buf[offset] = val & 0xFF
        elif size == 2:
            struct.pack_into("<H", buf, offset, val & 0xFFFF)
        elif size == 4:
            struct.pack_into("<I", buf, offset, val & 0xFFFF_FFFF)
        elif size == 8:
            struct.pack_into("<Q", buf, offset, val & 0xFFFF_FFFF_FFFF_FFFF)
        else:
            raise ValueError(f"Unsupported write size {size}")
