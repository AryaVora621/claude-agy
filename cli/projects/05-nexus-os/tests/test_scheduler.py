"""
Unit tests for NexusOS MLFQ Scheduler.
Verifies priority dispatching, time-slice preemption, allotment demotions,
voluntary yield preservation, and periodic priority boosting.
"""

import unittest
from nexus.types import ProcessState, BOOST_INTERVAL_TICKS
from nexus.process import ProcessControlBlock
from nexus.scheduler import MLFQScheduler


class TestMLFQScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = MLFQScheduler(boost_interval=20)

    def test_initial_queue_and_dispatch(self):
        p1 = ProcessControlBlock(pid=1, name="p1")
        p2 = ProcessControlBlock(pid=2, name="p2")

        self.scheduler.add_process(p1)
        self.scheduler.add_process(p2)
        self.scheduler.set_ready(1)
        self.scheduler.set_ready(2)

        # Both should start in Q0
        self.assertEqual(p1.priority, 0)
        self.assertEqual(p2.priority, 0)

        # Dispatch should pick p1 first
        pcb, _ = self.scheduler.tick()
        self.assertEqual(pcb.pid, 1)

    def test_allotment_demotion(self):
        p1 = ProcessControlBlock(pid=1, name="cpu_intensive")
        self.scheduler.add_process(p1)
        self.scheduler.set_ready(1)

        # Initial Q0 has allotment 4
        self.assertEqual(p1.priority, 0)

        # Run for 4 ticks (exhausting allotment)
        for _ in range(4):
            pcb, _ = self.scheduler.tick()

        # p1 should now be demoted to Q1
        self.assertEqual(p1.priority, 1)
        self.assertEqual(p1.time_slice_remaining, MLFQScheduler.TIME_SLICES[1])

        # Run for 8 ticks in Q1
        for _ in range(8):
            pcb, _ = self.scheduler.tick()

        # p1 should now be demoted to Q2
        self.assertEqual(p1.priority, 2)

    def test_yield_preserves_priority(self):
        p1 = ProcessControlBlock(pid=1, name="interactive")
        self.scheduler.add_process(p1)
        self.scheduler.set_ready(1)

        # Run 1 tick in Q0
        self.scheduler.tick()
        self.assertEqual(p1.priority, 0)

        # Yield voluntarily before slice/allotment runs out
        self.scheduler.yield_current()

        # Should remain in Q0
        self.assertEqual(p1.priority, 0)

    def test_priority_boost(self):
        p1 = ProcessControlBlock(pid=1, name="starved_proc")
        self.scheduler.add_process(p1)
        self.scheduler.set_ready(1)

        # Demote to Q2
        p1.priority = 2
        self.scheduler.queues[0].clear()
        self.scheduler.queues[2].append(1)

        # Trigger priority boost
        self.scheduler.priority_boost()

        # p1 should be boosted back to Q0
        self.assertEqual(p1.priority, 0)
        self.assertIn(1, self.scheduler.queues[0])
        self.assertEqual(p1.allotment_remaining, MLFQScheduler.TIME_ALLOTMENTS[0])

    def test_sleep_wakeup(self):
        p1 = ProcessControlBlock(pid=1, name="sleeper")
        self.scheduler.add_process(p1)
        self.scheduler.set_ready(1)

        # Put to sleep for 3 ticks
        p1.sleep_ticks = 3
        self.scheduler.set_blocked(1, ProcessState.SLEEPING)
        self.assertNotIn(1, self.scheduler.queues[0])

        # Advance 2 ticks
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(p1.state, ProcessState.SLEEPING)

        # Advance 3rd tick: should wake up and enter READY or be dispatched to RUNNING
        self.scheduler.tick()
        self.assertIn(p1.state, (ProcessState.READY, ProcessState.RUNNING))
        self.assertTrue(p1.is_alive())


if __name__ == "__main__":
    unittest.main()
