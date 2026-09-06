"""
NexusOS: Microkernel Engine, System Call Dispatcher, and Hardware Event Loop.
Integrates Virtual Memory Management (MMU, COW), MLFQ Scheduler,
Capability-Based IPC Rendezvous, and Virtual File System with Unix Pipes.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from nexus.types import (
    PAGE_SIZE, PAGE_SHIFT, PageFlags, PrivilegeRing, ProcessState,
    SyscallNum, Errno, SyscallResult, CapabilityRights
)
from nexus.mmu import PhysicalFrameAllocator, SwapDevice, MMU, PageFaultError
from nexus.process import ProcessControlBlock, AddressSpace, Capability
from nexus.scheduler import MLFQScheduler
from nexus.ipc import IPCManager, IPCMessage
from nexus.vfs import VirtualFileSystem, FileDescriptor, InodeType


class NexusKernel:
    """
    NexusOS Microkernel Controller.
    Oversees hardware abstraction, memory isolation, process scheduling, and system calls.
    """
    def __init__(self, num_frames: int = 256):
        # Hardware & Core Subsystems
        self.frame_allocator = PhysicalFrameAllocator(num_frames=num_frames)
        self.swap_device = SwapDevice()
        self.mmu = MMU(self.frame_allocator, self.swap_device)
        self.scheduler = MLFQScheduler()
        self.ipc = IPCManager()
        self.vfs = VirtualFileSystem()

        # Process Registry
        self.processes: Dict[int, ProcessControlBlock] = {}
        self._next_pid = 1
        self.ticks = 0

        # Registered user-space task functions (simulated thread routines)
        self._task_routines: Dict[int, Callable[['NexusKernel', ProcessControlBlock], Any]] = {}

        # Waitpid waiters: pid -> list of waiting parent pids
        self._waiters: Dict[int, List[int]] = {}

    @property
    def current_process(self) -> Optional[ProcessControlBlock]:
        """Currently executing process on the CPU."""
        if self.scheduler.current_pid is None:
            return None
        return self.processes.get(self.scheduler.current_pid)

    def create_process(
        self,
        name: str,
        routine: Optional[Callable[['NexusKernel', ProcessControlBlock], Any]] = None,
        ring: PrivilegeRing = PrivilegeRing.RING_3,
        ppid: int = 0
    ) -> ProcessControlBlock:
        """Spawn a new process with private address space and standard file descriptors."""
        pid = self._next_pid
        self._next_pid += 1

        addr_space = AddressSpace()
        # Allocate initial text segment (1 page) and stack segment (1 page)
        addr_space.allocate_pages_for_region(
            vaddr=addr_space.text_start,
            num_pages=1,
            flags=PageFlags.READABLE | PageFlags.USER_ACCESSIBLE,
            frame_allocator=self.frame_allocator
        )
        addr_space.add_region(addr_space.text_start, PAGE_SIZE, PageFlags.READABLE | PageFlags.USER_ACCESSIBLE, "[text]")

        addr_space.allocate_pages_for_region(
            vaddr=addr_space.stack_limit,
            num_pages=1,
            flags=PageFlags.READABLE | PageFlags.WRITABLE | PageFlags.USER_ACCESSIBLE,
            frame_allocator=self.frame_allocator
        )
        addr_space.add_region(addr_space.stack_limit, PAGE_SIZE, PageFlags.READABLE | PageFlags.WRITABLE | PageFlags.USER_ACCESSIBLE, "[stack]")

        pcb = ProcessControlBlock(
            pid=pid,
            name=name,
            ppid=ppid,
            ring=ring,
            address_space=addr_space
        )

        # Standard file descriptors (0: stdin, 1: stdout, 2: stderr)
        _, dev_null = self.vfs.open("/dev/null", "rw")
        pcb.fd_table[0] = dev_null
        pcb.fd_table[1] = dev_null
        pcb.fd_table[2] = dev_null

        self.processes[pid] = pcb
        self.scheduler.add_process(pcb)
        self.scheduler.set_ready(pid)

        if routine is not None:
            self._task_routines[pid] = routine

        # Automatically switch to first process if CPU is idle
        if self.scheduler.current_pid is None:
            self.set_active_process(pid)

        return pcb

    def set_active_process(self, pid: int) -> Optional[ProcessControlBlock]:
        """Manually switch running CPU context to specified process."""
        pcb = self.processes.get(pid)
        if pcb:
            self.scheduler.current_pid = pid
            pcb.state = ProcessState.RUNNING
            self.mmu.switch_address_space(pcb.address_space.page_table)
        return pcb

    # -------------------------------------------------------------------------
    # System Call Interface
    # -------------------------------------------------------------------------

    def sys_getpid(self) -> int:
        proc = self.current_process
        return proc.pid if proc else -1

    def sys_yield(self) -> None:
        """Voluntarily yield CPU to next ready process."""
        self.scheduler.yield_current()

    def sys_sleep(self, ticks: int) -> None:
        """Suspend calling process for specified duration."""
        proc = self.current_process
        if proc:
            proc.sleep_ticks = ticks
            self.scheduler.set_blocked(proc.pid, ProcessState.SLEEPING, reason=f"Sleep({ticks})")

    def sys_fork(self) -> Tuple[Errno, int]:
        """
        Create a child process via Copy-On-Write memory cloning.
        Parent receives child PID; child receives 0.
        """
        parent = self.current_process
        if parent is None:
            return Errno.EPERM, -1

        child_pid = self._next_pid
        self._next_pid += 1

        # Clone AddressSpace with COW physical frame sharing
        child_addr_space = parent.address_space.clone_cow(self.frame_allocator)

        child = ProcessControlBlock(
            pid=child_pid,
            name=f"{parent.name}_child",
            ppid=parent.pid,
            ring=parent.ring,
            address_space=child_addr_space
        )

        # Clone Capabilities
        child.cspace = parent.cspace.clone()

        # Clone File Descriptors and adjust pipe reference counters
        for fd_num, fd_obj in parent.fd_table.items():
            if fd_obj is not None:
                child.fd_table[fd_num] = fd_obj
                if fd_obj.pipe is not None:
                    if fd_obj.is_pipe_reader:
                        fd_obj.pipe.reader_count += 1
                    if fd_obj.is_pipe_writer:
                        fd_obj.pipe.writer_count += 1

        # Register parent/child hierarchy
        parent.children.append(child_pid)
        self.processes[child_pid] = child

        # Inherit executable routine if present
        if parent.pid in self._task_routines:
            self._task_routines[child_pid] = self._task_routines[parent.pid]

        self.scheduler.add_process(child)
        self.scheduler.set_ready(child_pid)

        return Errno.SUCCESS, child_pid

    def sys_exit(self, code: int) -> None:
        """Terminate calling process, wake waiting parents, and cleanup resources."""
        proc = self.current_process
        if proc is None:
            return

        proc.exit_code = code
        proc.state = ProcessState.ZOMBIE
        self.scheduler.remove_process(proc.pid)
        self.ipc.cleanup_process(proc.pid)

        # Close all open file descriptors
        for fd_num in list(proc.fd_table.keys()):
            self.sys_close(fd_num)

        # Wake up parent waiting in waitpid
        parent_waiters = self._waiters.get(proc.pid, [])
        for p_pid in parent_waiters:
            self.scheduler.set_ready(p_pid)

    def sys_waitpid(self, target_pid: int) -> Tuple[Errno, int, int]:
        """
        Wait for child process to terminate.
        Returns (status, child_pid, exit_code).
        """
        curr = self.current_process
        if curr is None:
            return Errno.EPERM, -1, -1

        child = self.processes.get(target_pid)
        if child is None or child.ppid != curr.pid:
            return Errno.ESRCH, -1, -1

        if child.state == ProcessState.ZOMBIE:
            # Child already terminated: reap and clean up
            code = child.exit_code if child.exit_code is not None else 0
            child.address_space.destroy(self.frame_allocator)
            del self.processes[target_pid]
            if target_pid in self._task_routines:
                del self._task_routines[target_pid]
            return Errno.SUCCESS, target_pid, code

        # Child is still running: block calling process
        if target_pid not in self._waiters:
            self._waiters[target_pid] = []
        self._waiters[target_pid].append(curr.pid)
        self.scheduler.set_blocked(curr.pid, ProcessState.BLOCKED_IO, reason=f"Waitpid({target_pid})")
        return Errno.EAGAIN, target_pid, 0

    def sys_mmap(self, vaddr: int, num_pages: int, flags: int) -> Tuple[Errno, int]:
        """Allocate virtual pages in process address space."""
        curr = self.current_process
        if curr is None:
            return Errno.EPERM, 0

        success = curr.address_space.allocate_pages_for_region(
            vaddr=vaddr,
            num_pages=num_pages,
            flags=flags | PageFlags.USER_ACCESSIBLE,
            frame_allocator=self.frame_allocator
        )
        if not success:
            return Errno.ENOMEM, 0

        curr.address_space.add_region(vaddr, num_pages * PAGE_SIZE, flags, "[mmap]")
        return Errno.SUCCESS, vaddr

    # -------------------------------------------------------------------------
    # IPC System Calls
    # -------------------------------------------------------------------------

    def sys_ipc_send(self, endpoint_id: str, payload: Any) -> Errno:
        """Send message to synchronous rendezvous endpoint."""
        curr = self.current_process
        if curr is None:
            return Errno.EPERM

        msg = IPCMessage(sender_pid=curr.pid, msg_type=1, payload=payload)
        endpoint = self.ipc.get_or_create_endpoint(endpoint_id)
        status, woken_receiver = endpoint.send(curr.pid, msg)

        if status == Errno.SUCCESS and woken_receiver is not None:
            # Deliver directly to woken receiver
            receiver = self.processes.get(woken_receiver)
            if receiver:
                receiver.ipc_message_buffer = msg
                self.scheduler.set_ready(woken_receiver)
            return Errno.SUCCESS

        # Block sender until receiver arrives
        self.scheduler.set_blocked(curr.pid, ProcessState.BLOCKED_IPC, reason=f"Send({endpoint_id})")
        return Errno.EAGAIN

    def sys_ipc_recv(self, endpoint_id: str) -> Tuple[Errno, Optional[Any]]:
        """Receive message from synchronous rendezvous endpoint."""
        curr = self.current_process
        if curr is None:
            return Errno.EPERM, None

        endpoint = self.ipc.get_or_create_endpoint(endpoint_id)
        status, msg, woken_sender = endpoint.receive(curr.pid)

        if status == Errno.SUCCESS and msg is not None:
            if woken_sender is not None:
                self.scheduler.set_ready(woken_sender)
            return Errno.SUCCESS, msg.payload

        # Block receiver until sender arrives
        self.scheduler.set_blocked(curr.pid, ProcessState.BLOCKED_IPC, reason=f"Recv({endpoint_id})")
        return Errno.EAGAIN, None

    # -------------------------------------------------------------------------
    # Virtual File System & Pipe System Calls
    # -------------------------------------------------------------------------

    def sys_open(self, path: str, mode: str = "r") -> Tuple[Errno, int]:
        curr = self.current_process
        if curr is None:
            return Errno.EPERM, -1

        status, fd_obj = self.vfs.open(path, mode)
        if status != Errno.SUCCESS or fd_obj is None:
            return status, -1

        fd_num = max(curr.fd_table.keys(), default=-1) + 1
        fd_obj.fd_num = fd_num
        curr.fd_table[fd_num] = fd_obj
        return Errno.SUCCESS, fd_num

    def sys_pipe(self) -> Tuple[Errno, int, int]:
        """Create a Unix pipe returning (read_fd, write_fd)."""
        curr = self.current_process
        if curr is None:
            return Errno.EPERM, -1, -1

        read_fd_obj, write_fd_obj = self.vfs.create_pipe()
        base_fd = max(curr.fd_table.keys(), default=-1) + 1

        r_num = base_fd
        w_num = base_fd + 1

        read_fd_obj.fd_num = r_num
        write_fd_obj.fd_num = w_num

        curr.fd_table[r_num] = read_fd_obj
        curr.fd_table[w_num] = write_fd_obj
        return Errno.SUCCESS, r_num, w_num

    def sys_read(self, fd_num: int, length: int) -> Tuple[Errno, bytes]:
        curr = self.current_process
        if curr is None or fd_num not in curr.fd_table:
            return Errno.EBADF, b""

        fd = curr.fd_table[fd_num]
        if fd.pipe is not None:
            status, data = fd.pipe.read(length, reader_pid=curr.pid)
            if status == Errno.EAGAIN:
                self.scheduler.set_blocked(curr.pid, ProcessState.BLOCKED_IO, reason=f"PipeRead({fd_num})")
            return status, data

        if fd.inode is not None:
            data = bytes(fd.inode.data[fd.offset:fd.offset + length])
            fd.offset += len(data)
            return Errno.SUCCESS, data

        return Errno.EBADF, b""

    def sys_write(self, fd_num: int, data: bytes) -> Tuple[Errno, int]:
        curr = self.current_process
        if curr is None or fd_num not in curr.fd_table:
            return Errno.EBADF, 0

        fd = curr.fd_table[fd_num]
        if fd.pipe is not None:
            status, written = fd.pipe.write(data, writer_pid=curr.pid)
            if status == Errno.SUCCESS:
                # Wake any blocked readers
                while fd.pipe.waiting_readers:
                    reader_pid = fd.pipe.waiting_readers.popleft()
                    self.scheduler.set_ready(reader_pid)
            elif status == Errno.EAGAIN:
                self.scheduler.set_blocked(curr.pid, ProcessState.BLOCKED_IO, reason=f"PipeWrite({fd_num})")
            return status, written

        if fd.inode is not None:
            fd.inode.data.extend(data)
            fd.offset += len(data)
            return Errno.SUCCESS, len(data)

        return Errno.EBADF, 0

    def sys_close(self, fd_num: int) -> Errno:
        curr = self.current_process
        if curr is None or fd_num not in curr.fd_table:
            return Errno.EBADF

        fd = curr.fd_table.pop(fd_num)
        if fd.pipe is not None:
            if fd.is_pipe_reader:
                fd.pipe.close_read_end()
            if fd.is_pipe_writer:
                fd.pipe.close_write_end()
                # Wake blocked readers so they receive EOF
                while fd.pipe.waiting_readers:
                    r_pid = fd.pipe.waiting_readers.popleft()
                    self.scheduler.set_ready(r_pid)
        return Errno.SUCCESS

    # -------------------------------------------------------------------------
    # Execution & Hardware Event Loop
    # -------------------------------------------------------------------------

    def step(self) -> Tuple[Optional[ProcessControlBlock], bool]:
        """
        Advance simulated kernel by one tick:
        1. Schedule next process via MLFQ.
        2. Update MMU translation context.
        3. Execute process step routine if present.
        """
        pcb, preempted = self.scheduler.tick()
        self.ticks += 1

        if pcb and pcb.state == ProcessState.RUNNING:
            self.mmu.switch_address_space(pcb.address_space.page_table)
            routine = self._task_routines.get(pcb.pid)
            if routine is not None:
                try:
                    routine(self, pcb)
                except Exception as e:
                    # Kernel caught unhandled process exception: terminate process
                    self.sys_exit(-1)

        return pcb, preempted

    def run(self, max_ticks: int = 1000) -> int:
        """Run the kernel simulation until no active processes remain or max_ticks reached."""
        for _ in range(max_ticks):
            active_procs = [p for p in self.processes.values() if p.state != ProcessState.ZOMBIE]
            if not active_procs:
                break
            self.step()
        return self.ticks
