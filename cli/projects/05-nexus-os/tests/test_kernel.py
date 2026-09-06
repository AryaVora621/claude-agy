"""
Integration tests for NexusOS Microkernel.
Tests system call execution, fork-exec lifecycle, COW memory isolation,
IPC rendezvous blocking, and pipe communication across process boundaries.
"""

import unittest
from nexus.types import PageFlags, Errno, ProcessState
from nexus.kernel import NexusKernel


class TestNexusKernel(unittest.TestCase):
    def setUp(self):
        self.kernel = NexusKernel(num_frames=64)

    def test_process_creation(self):
        proc = self.kernel.create_process("test_daemon")
        self.assertEqual(proc.pid, 1)
        self.assertEqual(proc.name, "test_daemon")
        self.assertEqual(proc.state, ProcessState.RUNNING)
        self.assertIn(1, self.kernel.processes)

    def test_mmap_syscall(self):
        proc = self.kernel.create_process("mmap_tester")
        vaddr = 0x00050000
        status, mapped_addr = self.kernel.sys_mmap(vaddr, num_pages=2, flags=PageFlags.READABLE | PageFlags.WRITABLE)

        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(mapped_addr, vaddr)

        # Write and read back across mapped pages
        self.kernel.mmu.write_memory(vaddr, b"PAGE_DATA_SPAN")
        read_back = self.kernel.mmu.read_memory(vaddr, 14)
        self.assertEqual(read_back, b"PAGE_DATA_SPAN")

    def test_cow_fork_isolation(self):
        parent = self.kernel.create_process("parent")
        vaddr = 0x00040000
        self.kernel.sys_mmap(vaddr, num_pages=1, flags=PageFlags.READABLE | PageFlags.WRITABLE)
        self.kernel.mmu.write_memory(vaddr, b"SHARED_SECRET")

        # Fork child
        status, child_pid = self.kernel.sys_fork()
        self.assertEqual(status, Errno.SUCCESS)
        child = self.kernel.processes[child_pid]

        # Before write: both parent and child page table entries point to same physical frame
        p_entry = parent.address_space.page_table.lookup(vaddr >> 12)
        c_entry = child.address_space.page_table.lookup(vaddr >> 12)
        self.assertEqual(p_entry.pfn, c_entry.pfn)
        self.assertEqual(self.kernel.frame_allocator.get_ref_count(p_entry.pfn), 2)

        # Child writes new data -> triggers COW page fault
        self.kernel.set_active_process(child_pid)
        self.kernel.mmu.write_memory(vaddr, b"CHILD_SECRET!")

        # After write: child should have distinct physical frame
        new_c_entry = child.address_space.page_table.lookup(vaddr >> 12)
        self.assertNotEqual(p_entry.pfn, new_c_entry.pfn)

        # Parent data untouched
        self.kernel.set_active_process(parent.pid)
        p_data = self.kernel.mmu.read_memory(vaddr, 13)
        self.assertEqual(p_data, b"SHARED_SECRET")

        # Child data verified
        self.kernel.set_active_process(child_pid)
        c_data = self.kernel.mmu.read_memory(vaddr, 13)
        self.assertEqual(c_data, b"CHILD_SECRET!")

    def test_ipc_rendezvous_syscalls(self):
        server = self.kernel.create_process("server")
        client = self.kernel.create_process("client")

        # Server attempts to receive: blocks because no message is waiting
        self.kernel.set_active_process(server.pid)
        status, msg = self.kernel.sys_ipc_recv("calc_service")
        self.assertEqual(status, Errno.EAGAIN)
        self.assertEqual(server.state, ProcessState.BLOCKED_IPC)

        # Client sends request: unblocks server!
        self.kernel.set_active_process(client.pid)
        status = self.kernel.sys_ipc_send("calc_service", {"op": "ADD", "a": 10, "b": 20})
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(server.state, ProcessState.READY)
        self.assertEqual(server.ipc_message_buffer.payload, {"op": "ADD", "a": 10, "b": 20})

    def test_pipe_and_waitpid(self):
        parent = self.kernel.create_process("parent")
        status, r_fd, w_fd = self.kernel.sys_pipe()

        status, child_pid = self.kernel.sys_fork()
        child = self.kernel.processes[child_pid]

        # Child writes to pipe and exits
        self.kernel.set_active_process(child_pid)
        status, written = self.kernel.sys_write(w_fd, b"DATA_FROM_CHILD")
        self.assertEqual(status, Errno.SUCCESS)
        self.kernel.sys_close(w_fd)
        self.kernel.sys_exit(99)
        self.assertEqual(child.state, ProcessState.ZOMBIE)

        # Parent reads from pipe
        self.kernel.set_active_process(parent.pid)
        status, read_bytes = self.kernel.sys_read(r_fd, 32)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(read_bytes, b"DATA_FROM_CHILD")

        # Parent reaps child
        status, reaped_pid, code = self.kernel.sys_waitpid(child_pid)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(reaped_pid, child_pid)
        self.assertEqual(code, 99)
        self.assertNotIn(child_pid, self.kernel.processes)


if __name__ == "__main__":
    unittest.main()
