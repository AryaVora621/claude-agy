"""
NexusOS: Microkernel & Virtual Memory Operating System Simulator.
Built from first principles in zero-dependency Python.
"""

from nexus.types import (
    PAGE_SIZE, PrivilegeRing, ProcessState, SyscallNum, PageFlags,
    CapabilityRights, Errno, SyscallResult
)
from nexus.mmu import PageTable, PhysicalFrameAllocator, SwapDevice, TLB, MMU, PageFaultError
from nexus.process import ProcessControlBlock, AddressSpace, MemoryRegion, Capability, CSpace
from nexus.scheduler import MLFQScheduler
from nexus.ipc import IPCManager, IPCMessage, SynchronousEndpoint, AsynchronousMailbox
from nexus.vfs import VirtualFileSystem, Inode, FileDescriptor, UnixPipe
from nexus.kernel import NexusKernel

__all__ = [
    "PAGE_SIZE", "PrivilegeRing", "ProcessState", "SyscallNum", "PageFlags",
    "CapabilityRights", "Errno", "SyscallResult", "PageTable", "PhysicalFrameAllocator",
    "SwapDevice", "TLB", "MMU", "PageFaultError", "ProcessControlBlock",
    "AddressSpace", "MemoryRegion", "Capability", "CSpace", "MLFQScheduler",
    "IPCManager", "IPCMessage", "SynchronousEndpoint", "AsynchronousMailbox",
    "VirtualFileSystem", "Inode", "FileDescriptor", "UnixPipe", "NexusKernel"
]
