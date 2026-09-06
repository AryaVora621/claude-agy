"""
NexusOS: Core Types, Enums, and Architectural Constants.
Defines processor rings, traps, system call numbers, page flags, and capabilities.
"""

from enum import Enum, IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


# Architectural Constants
PAGE_SIZE = 4096            # 4KB Virtual & Physical Pages
PAGE_SHIFT = 12             # Offset bits (2^12 = 4096)
PAGE_MASK = 0xFFF           # Offset mask
NUM_PHYSICAL_FRAMES = 256   # 1MB total simulated RAM (256 * 4KB)
SWAP_CAPACITY_FRAMES = 512  # 2MB simulated swap partition

MAX_MLFQ_QUEUES = 4         # Q0 (highest) to Q3 (lowest)
BOOST_INTERVAL_TICKS = 50   # Starvation prevention boost period


class PrivilegeRing(IntEnum):
    """Hardware privilege rings (x86/ARM style)."""
    RING_0 = 0  # Kernel space (supervisor mode, full hardware access)
    RING_3 = 3  # User space (unprivileged, restricted memory access)


class ProcessState(Enum):
    """Lifecycle states of a thread / process."""
    EMBRYO = auto()
    READY = auto()
    RUNNING = auto()
    BLOCKED_IPC = auto()
    BLOCKED_IO = auto()
    SLEEPING = auto()
    ZOMBIE = auto()


class TrapCode(IntEnum):
    """Hardware trap & interrupt vectors."""
    TRAP_TIMER = 0x20
    TRAP_SYSCALL = 0x80
    TRAP_PAGE_FAULT = 0x0E
    TRAP_ILLEGAL_INSTRUCTION = 0x06
    TRAP_PERMISSION_DENIED = 0x0D


class SyscallNum(IntEnum):
    """Standard system call interface numbers."""
    # Process management
    SYS_YIELD = 1
    SYS_FORK = 2
    SYS_EXEC = 3
    SYS_EXIT = 4
    SYS_WAITPID = 5
    SYS_GETPID = 6
    SYS_SLEEP = 7

    # Memory management
    SYS_MMAP = 10
    SYS_MUNMAP = 11
    SYS_BRK = 12

    # IPC & Capabilities
    SYS_IPC_SEND = 20
    SYS_IPC_RECV = 21
    SYS_IPC_CALL = 22  # Synchronous rendezvous: send and wait for reply

    # Virtual File System
    SYS_OPEN = 30
    SYS_READ = 31
    SYS_WRITE = 32
    SYS_CLOSE = 33
    SYS_PIPE = 34
    SYS_DUP2 = 35


class PageFlags(IntEnum):
    """Page Table Entry bit flags."""
    PRESENT = 1 << 0
    READABLE = 1 << 1
    WRITABLE = 1 << 2
    USER_ACCESSIBLE = 1 << 3
    ACCESSED = 1 << 4
    DIRTY = 1 << 5
    COW = 1 << 6         # Copy-On-Write marker for lazy memory duplication


class CapabilityRights(IntEnum):
    """Access rights for capability tokens in process CSpace."""
    READ = 1 << 0
    WRITE = 1 << 1
    EXECUTE = 1 << 2
    GRANT = 1 << 3       # Can delegate this capability to another process
    INVOKE = 1 << 4      # Can trigger IPC calls through this endpoint


class Errno(IntEnum):
    """Unix-style error return codes."""
    SUCCESS = 0
    EPERM = -1           # Operation not permitted
    ENOENT = -2          # No such file or directory
    ESRCH = -3           # No such process
    EBADF = -9           # Bad file descriptor
    EAGAIN = -11         # Resource temporarily unavailable / would block
    ENOMEM = -12         # Out of physical / virtual memory
    EACCES = -13         # Permission denied
    EFAULT = -14         # Bad address (unmapped memory)
    EBUSY = -16          # Device or resource busy
    EINVAL = -22         # Invalid argument
    EPIPE = -32          # Broken pipe (no readers)


@dataclass
class CPUContext:
    """Simulated CPU register state saved on context switches."""
    pc: int = 0
    sp: int = 0
    r0: int = 0
    r1: int = 0
    r2: int = 0
    r3: int = 0
    flags: int = 0
    ring: PrivilegeRing = PrivilegeRing.RING_3


@dataclass
class SyscallResult:
    """Result of a system call invocation."""
    status: Errno
    value: Any = None
