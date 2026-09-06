"""
NexusOS: Multi-Level Feedback Queue (MLFQ) CPU Scheduler.
Implements dynamic priority adjustment, time-slice accounting, anti-gaming allotment tracking,
round-robin dispatching within priority levels, and periodic priority boosting for starvation prevention.
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
from nexus.types import ProcessState, BOOST_INTERVAL_TICKS, MAX_MLFQ_QUEUES
from nexus.process import ProcessControlBlock


class MLFQScheduler:
    """
    Multi-Level Feedback Queue (MLFQ) Scheduler with 4 Priority Tiers:
    - Q0: Highest priority, short time slice (2 ticks), allotment 4 ticks (Interactive/I/O burst)
    - Q1: Medium priority, moderate slice (4 ticks), allotment 8 ticks
    - Q2: Low priority, longer slice (8 ticks), allotment 16 ticks
    - Q3: Background priority, longest slice (16 ticks), round-robin
    """
    TIME_SLICES = [2, 4, 8, 16]
    TIME_ALLOTMENTS = [4, 8, 16, 999999]

    def __init__(self, boost_interval: int = BOOST_INTERVAL_TICKS):
        self.num_queues = MAX_MLFQ_QUEUES
        self.boost_interval = boost_interval
        self.queues: List[deque[int]] = [deque() for _ in range(self.num_queues)]
        self.processes: Dict[int, ProcessControlBlock] = {}
        self.current_pid: Optional[int] = None
        self.ticks = 0
        self.boost_count = 0

    def add_process(self, pcb: ProcessControlBlock) -> None:
        """Register a new process into the highest priority queue Q0."""
        self.processes[pcb.pid] = pcb
        pcb.priority = 0
        pcb.time_slice_remaining = self.TIME_SLICES[0]
        pcb.allotment_remaining = self.TIME_ALLOTMENTS[0]
        if pcb.state == ProcessState.READY:
            self.queues[0].append(pcb.pid)

    def remove_process(self, pid: int) -> Optional[ProcessControlBlock]:
        """Deregister process from scheduler queues."""
        pcb = self.processes.pop(pid, None)
        if pcb:
            for q in self.queues:
                if pid in q:
                    q.remove(pid)
            if self.current_pid == pid:
                self.current_pid = None
        return pcb

    def set_ready(self, pid: int) -> None:
        """Mark a blocked/sleeping process as ready and enqueue at its current priority."""
        pcb = self.processes.get(pid)
        if pcb and pcb.state != ProcessState.READY:
            pcb.state = ProcessState.READY
            if pid not in self.queues[pcb.priority]:
                self.queues[pcb.priority].append(pid)

    def set_blocked(self, pid: int, state: ProcessState, reason: Optional[str] = None) -> None:
        """Transition process to a blocked state (e.g. waiting for IPC or I/O)."""
        pcb = self.processes.get(pid)
        if pcb:
            pcb.state = state
            pcb.blocked_on = reason
            for q in self.queues:
                if pid in q:
                    q.remove(pid)
            if self.current_pid == pid:
                self.current_pid = None

    def priority_boost(self) -> None:
        """
        Global Starvation Prevention:
        Rule 5: Periodically reset all live processes to top queue Q0 with fresh allotments.
        """
        self.boost_count += 1
        for q in self.queues:
            q.clear()

        for pid, pcb in self.processes.items():
            if pcb.is_alive():
                pcb.priority = 0
                pcb.time_slice_remaining = self.TIME_SLICES[0]
                pcb.allotment_remaining = self.TIME_ALLOTMENTS[0]
                if pcb.state == ProcessState.READY:
                    self.queues[0].append(pid)

    def pick_next(self) -> Optional[ProcessControlBlock]:
        """
        Select highest-priority ready process (Rule 1 & Rule 2).
        Traverses Q0 down to Q3.
        """
        for prio in range(self.num_queues):
            queue = self.queues[prio]
            while queue:
                pid = queue[0]
                pcb = self.processes.get(pid)
                if pcb is None or pcb.state != ProcessState.READY:
                    queue.popleft()
                    continue
                return pcb
        return None

    def yield_current(self) -> Optional[ProcessControlBlock]:
        """
        Voluntary CPU yield before time slice exhaustion.
        Preserves priority level, places process at back of its current queue.
        """
        if self.current_pid is None:
            return self.pick_next()

        curr_pcb = self.processes.get(self.current_pid)
        if curr_pcb and curr_pcb.state == ProcessState.RUNNING:
            curr_pcb.state = ProcessState.READY
            q = self.queues[curr_pcb.priority]
            if curr_pcb.pid in q:
                q.remove(curr_pcb.pid)
            q.append(curr_pcb.pid)

        self.current_pid = None
        return self.pick_next()

    def tick(self) -> Tuple[Optional[ProcessControlBlock], bool]:
        """
        Execute one scheduling clock tick:
        1. Wake up sleeping processes whose timer expired.
        2. Check for periodic priority boost.
        3. Account CPU time, slice, and allotment for running process.
        4. Demote if allotment spent; preempt if time slice spent.
        5. Return (current_pcb, was_preempted).
        """
        self.ticks += 1
        preempted = False

        # 1. Update sleeping processes
        for pcb in list(self.processes.values()):
            if pcb.state == ProcessState.SLEEPING:
                pcb.sleep_ticks -= 1
                if pcb.sleep_ticks <= 0:
                    self.set_ready(pcb.pid)

        # 2. Check periodic priority boost
        if self.ticks % self.boost_interval == 0:
            self.priority_boost()

        # 3. If CPU is idle, dispatch highest-priority ready process
        if self.current_pid is None:
            next_pcb = self.pick_next()
            if next_pcb:
                self.current_pid = next_pcb.pid
                next_pcb.state = ProcessState.RUNNING

        # 4. Account CPU usage for currently executing process
        curr_pcb = self.processes.get(self.current_pid) if self.current_pid is not None else None
        if curr_pcb and curr_pcb.state == ProcessState.RUNNING:
            curr_pcb.total_cpu_time += 1
            curr_pcb.time_slice_remaining -= 1
            curr_pcb.allotment_remaining -= 1

            # Rule 4: Allotment exhaustion -> Demotion to lower queue
            if curr_pcb.allotment_remaining <= 0 and curr_pcb.priority < self.num_queues - 1:
                old_q = self.queues[curr_pcb.priority]
                if curr_pcb.pid in old_q:
                    old_q.remove(curr_pcb.pid)

                curr_pcb.priority += 1
                curr_pcb.time_slice_remaining = self.TIME_SLICES[curr_pcb.priority]
                curr_pcb.allotment_remaining = self.TIME_ALLOTMENTS[curr_pcb.priority]
                curr_pcb.state = ProcessState.READY
                self.queues[curr_pcb.priority].append(curr_pcb.pid)
                self.current_pid = None
                preempted = True

            # Rule 2: Time-slice exhaustion -> Round-Robin rotation within same queue
            elif curr_pcb.time_slice_remaining <= 0:
                old_q = self.queues[curr_pcb.priority]
                if curr_pcb.pid in old_q:
                    old_q.remove(curr_pcb.pid)

                curr_pcb.time_slice_remaining = self.TIME_SLICES[curr_pcb.priority]
                curr_pcb.state = ProcessState.READY
                self.queues[curr_pcb.priority].append(curr_pcb.pid)
                self.current_pid = None
                preempted = True

        return curr_pcb, preempted
