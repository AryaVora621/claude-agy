"""
Unit tests for NexusOS Virtual Memory, MMU, TLB, Frame Allocator, and COW.
"""

import unittest
from nexus.types import (
    PAGE_SIZE, PageFlags, PrivilegeRing
)
from nexus.mmu import (
    PageTable, PhysicalFrameAllocator, SwapDevice, TLB, MMU, PageFaultError
)


class TestMMU(unittest.TestCase):
    def setUp(self):
        self.allocator = PhysicalFrameAllocator(num_frames=32)
        self.swap = SwapDevice(capacity_frames=64)
        self.mmu = MMU(self.allocator, self.swap)

    def test_frame_allocator(self):
        pfn1 = self.allocator.allocate()
        self.assertIsNotNone(pfn1)
        self.assertEqual(self.allocator.get_ref_count(pfn1), 1)

        self.allocator.retain(pfn1)
        self.assertEqual(self.allocator.get_ref_count(pfn1), 2)

        freed = self.allocator.release(pfn1)
        self.assertFalse(freed)
        self.assertEqual(self.allocator.get_ref_count(pfn1), 1)

        freed = self.allocator.release(pfn1)
        self.assertTrue(freed)
        self.assertEqual(self.allocator.get_ref_count(pfn1), 0)

    def test_page_table_mapping(self):
        pt = PageTable()
        vpn = 0x1234
        pfn = 5
        flags = PageFlags.READABLE | PageFlags.WRITABLE | PageFlags.USER_ACCESSIBLE

        entry = pt.map_page(vpn, pfn, flags)
        self.assertEqual(entry.pfn, pfn)
        self.assertTrue(entry.flags & PageFlags.PRESENT)

        looked_up = pt.lookup(vpn)
        self.assertIsNotNone(looked_up)
        self.assertEqual(looked_up.pfn, pfn)

        unmapped = pt.unmap_page(vpn)
        self.assertEqual(unmapped.pfn, pfn)
        self.assertIsNone(pt.lookup(vpn))

    def test_tlb_caching(self):
        tlb = TLB(capacity=4)
        tlb.insert(1, 10, 0x01)
        tlb.insert(2, 20, 0x01)

        res = tlb.lookup(1)
        self.assertEqual(res, (10, 0x01))
        self.assertEqual(tlb.hits, 1)

        res = tlb.lookup(99)
        self.assertIsNone(res)
        self.assertEqual(tlb.misses, 1)

        tlb.invalidate(1)
        self.assertIsNone(tlb.lookup(1))

    def test_mmu_translation_and_permissions(self):
        pt = PageTable()
        vpn = 0x05
        pfn = self.allocator.allocate()
        # Map as Kernel-only page (no USER_ACCESSIBLE)
        pt.map_page(vpn, pfn, PageFlags.READABLE | PageFlags.WRITABLE)
        self.mmu.switch_address_space(pt)

        vaddr = (vpn << 12) | 0x42

        # Ring 0 should succeed
        paddr = self.mmu.translate(vaddr, mode="r", ring=PrivilegeRing.RING_0)
        self.assertEqual(paddr, (pfn << 12) | 0x42)

        # Ring 3 should raise PageFaultError (Permission Denied)
        with self.assertRaises(PageFaultError) as ctx:
            self.mmu.translate(vaddr, mode="r", ring=PrivilegeRing.RING_3)
        self.assertTrue(ctx.exception.permission_denied)

    def test_copy_on_write_divergence(self):
        pt = PageTable()
        vpn = 0x10
        pfn = self.allocator.allocate()
        pt.map_page(vpn, pfn, PageFlags.READABLE | PageFlags.COW | PageFlags.USER_ACCESSIBLE)
        self.mmu.switch_address_space(pt)

        vaddr = (vpn << 12)

        # Write to physical frame first
        self.allocator.write_bytes(pfn, 0, b"ORIGINAL_DATA")
        self.allocator.retain(pfn)  # Ref count = 2 (simulating shared frame across fork)

        # Attempt write: raises PageFault COW
        with self.assertRaises(PageFaultError) as ctx:
            self.mmu.translate(vaddr, mode="w", ring=PrivilegeRing.RING_3)
        self.assertTrue(ctx.exception.is_cow)

        # Handle COW fault
        success = self.mmu.handle_cow_fault(vpn)
        self.assertTrue(success)

        # Frame should now be unique and writable
        entry = pt.lookup(vpn)
        self.assertNotEqual(entry.pfn, pfn)
        self.assertTrue(entry.flags & PageFlags.WRITABLE)
        self.assertFalse(entry.flags & PageFlags.COW)

        # Writing memory now succeeds
        self.mmu.write_memory(vaddr, b"NEW_DATA")
        data = self.mmu.read_memory(vaddr, 8)
        self.assertEqual(data, b"NEW_DATA")

        # Original frame still has original bytes
        orig_bytes = self.allocator.read_bytes(pfn, 0, 13)
        self.assertEqual(orig_bytes, b"ORIGINAL_DATA")

    def test_swap_device(self):
        data = b"\xAB" * PAGE_SIZE
        slot = self.swap.swap_out(data)
        self.assertIsNotNone(slot)

        restored = self.swap.swap_in(slot)
        self.assertEqual(restored, data)
        # Slot is now free
        self.assertIsNone(self.swap.swap_in(slot))


if __name__ == "__main__":
    unittest.main()
