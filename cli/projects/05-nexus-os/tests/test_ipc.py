"""
Unit tests for NexusOS Inter-Process Communication (IPC) and Capabilities.
Tests synchronous rendezvous endpoints, asynchronous mailboxes, and capability rights.
"""

import unittest
from nexus.types import Errno, CapabilityRights
from nexus.process import Capability, CSpace
from nexus.ipc import IPCManager, IPCMessage, SynchronousEndpoint, AsynchronousMailbox


class TestIPC(unittest.TestCase):
    def setUp(self):
        self.ipc = IPCManager()

    def test_sync_rendezvous_sender_first(self):
        ep = self.ipc.get_or_create_endpoint("service_auth")
        msg = IPCMessage(sender_pid=10, msg_type=1, payload={"token": "XYZ"})

        # Sender arrives first: no receiver waiting, sender should block
        status, woken = ep.send(10, msg)
        self.assertEqual(status, Errno.EAGAIN)
        self.assertIsNone(woken)

        # Receiver arrives: should get sender's message and wake sender (PID 10)
        status, received_msg, woken_sender = ep.receive(20)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertIsNotNone(received_msg)
        self.assertEqual(received_msg.payload, {"token": "XYZ"})
        self.assertEqual(woken_sender, 10)

    def test_sync_rendezvous_receiver_first(self):
        ep = self.ipc.get_or_create_endpoint("service_db")

        # Receiver arrives first: no sender waiting, receiver should block
        status, msg, woken = ep.receive(20)
        self.assertEqual(status, Errno.EAGAIN)
        self.assertIsNone(msg)

        # Sender arrives: delivers message and wakes receiver (PID 20)
        send_msg = IPCMessage(sender_pid=10, msg_type=2, payload="SELECT 1")
        status, woken_receiver = ep.send(10, send_msg)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(woken_receiver, 20)

    def test_async_mailbox(self):
        mb = self.ipc.get_or_create_mailbox("event_bus", capacity=2)

        msg1 = IPCMessage(sender_pid=1, msg_type=1, payload="EVENT_1")
        msg2 = IPCMessage(sender_pid=1, msg_type=1, payload="EVENT_2")
        msg3 = IPCMessage(sender_pid=1, msg_type=1, payload="EVENT_3")

        # First 2 should buffer without blocking
        status, _ = mb.send(1, msg1)
        self.assertEqual(status, Errno.SUCCESS)
        status, _ = mb.send(1, msg2)
        self.assertEqual(status, Errno.SUCCESS)

        # 3rd should block because capacity is 2
        status, _ = mb.send(1, msg3)
        self.assertEqual(status, Errno.EAGAIN)

        # Read first message: should succeed and pop EVENT_1
        status, r_msg, woken_writer = mb.receive(2)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(r_msg.payload, "EVENT_1")
        # Writer 1 should be woken to place EVENT_3 into buffer
        self.assertEqual(woken_writer, 1)

    def test_capabilities(self):
        cspace = CSpace()
        cap = Capability(
            token_id="tok_1",
            resource_type="ENDPOINT",
            target_id="kernel_logger",
            rights=CapabilityRights.READ | CapabilityRights.WRITE
        )
        slot = cspace.insert(cap)

        self.assertTrue(cspace.check_access(slot, CapabilityRights.READ))
        self.assertTrue(cspace.check_access(slot, CapabilityRights.WRITE))
        self.assertFalse(cspace.check_access(slot, CapabilityRights.GRANT))
        self.assertFalse(cspace.check_access(slot, CapabilityRights.INVOKE))

        # Clone on fork
        cloned_cspace = cspace.clone()
        self.assertTrue(cloned_cspace.check_access(slot, CapabilityRights.WRITE))


if __name__ == "__main__":
    unittest.main()
