"""Unit tests for Multi-Level Cache Hierarchy and MESI Coherence Protocol."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.cache import (
    BusTransaction,
    Cache,
    MemoryHierarchy,
    MESIState,
)
from siliconrisc.memory import PhysicalMemory


class TestCacheAndCoherence(unittest.TestCase):
    """Test L1/L2 hits, misses, evictions, write-back, write-allocate, and MESI snooping."""

    def test_cache_indexing_and_hit(self) -> None:
        cache = Cache("TestL1", size_bytes=1024, associativity=2, line_size_bytes=64, hit_latency_cycles=1)
        self.assertEqual(cache.num_lines, 16)
        self.assertEqual(cache.num_sets, 8)
        self.assertEqual(cache.offset_bits, 6)
        self.assertEqual(cache.index_bits, 3)

        tag, set_idx, offset = cache.decompose_address(0x1040)
        self.assertEqual(offset, 0)
        self.assertEqual(set_idx, 1)

    def test_memory_hierarchy_read_write(self) -> None:
        ram = PhysicalMemory()
        hier = MemoryHierarchy(ram, l1_size=1024, l2_size=4096, ram_latency_cycles=50, l2_latency_cycles=5)

        # Pre-seed RAM
        ram.write_u64(0x1000, 0x1122334455667788)

        # First read -> miss in L1D, miss in L2 -> fetches from RAM
        val, lat1 = hier.read_data(0x1000, size=8)
        self.assertEqual(val, 0x1122334455667788)
        self.assertGreaterEqual(lat1, 50)  # L1 + L2 + RAM latency

        # Second read from same line -> L1D hit (1 cycle latency)
        val2, lat2 = hier.read_data(0x1008, size=8)
        self.assertEqual(lat2, 1)

        # Write to L1D -> hit, modifies line in L1D
        lat_w = hier.write_data(0x1000, 0xDEADBEEFCAFEBABE, size=8)
        self.assertEqual(lat_w, 1)

        # Verify read back from L1D
        val_w, _ = hier.read_data(0x1000, size=8)
        self.assertEqual(val_w, 0xDEADBEEFCAFEBABE)

        # Flush hierarchy -> RAM should now have updated data
        hier.flush_all()
        self.assertEqual(ram.read_u64(0x1000), 0xDEADBEEFCAFEBABE)

    def test_instruction_fetch_hierarchy(self) -> None:
        ram = PhysicalMemory()
        hier = MemoryHierarchy(ram, l1_size=1024, l2_size=4096)

        # Write RISC-V instruction into RAM (e.g. ADDI x1, x0, 42 -> 0x02A10093)
        ram.write_u32(0x2000, 0x02A10093)

        inst1, lat1 = hier.fetch_instruction(0x2000)
        self.assertEqual(inst1, 0x02A10093)
        self.assertGreater(lat1, 1)

        # Second fetch in same cache line -> L1I hit
        inst2, lat2 = hier.fetch_instruction(0x2004)
        self.assertEqual(lat2, 1)

    def test_mesi_bus_snooping(self) -> None:
        c1 = Cache("Core1_L1", size_bytes=1024, associativity=2, line_size_bytes=64)
        c2 = Cache("Core2_L1", size_bytes=1024, associativity=2, line_size_bytes=64)

        # Core 1 loads line 0x1000 into Exclusive state
        tag, s_idx, _ = c1.decompose_address(0x1000)
        line1 = c1.sets[s_idx].lines[0]
        line1.tag = tag
        line1.state = MESIState.EXCLUSIVE

        # Core 2 performs BusRd on 0x1000 -> Core 1 snoops and demotes Exclusive to Shared
        hit, _ = c1.snoop(BusTransaction.BUS_RD, 0x1000)
        self.assertTrue(hit)
        self.assertEqual(line1.state, MESIState.SHARED)

        # Core 2 performs BusRdX (write miss) -> Core 1 snoops and invalidates its line
        hit, _ = c1.snoop(BusTransaction.BUS_RDX, 0x1000)
        self.assertTrue(hit)
        self.assertEqual(line1.state, MESIState.INVALID)


if __name__ == "__main__":
    unittest.main()
