"""
HydraNet: Unit Tests for ARP Resolution and IPv4 Routing / Fragmentation.
"""

import unittest
import time
from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetType,
    EthernetFrame,
    ARPOpcode,
    ARPHeader,
    IPProtocol,
    IPv4Header
)
from hydra.device import VirtualNetDevice, VirtualEthernetBus
from hydra.arp import ARPEngine
from hydra.ip import IPEngine, RouteEntry


class TestARPandIP(unittest.TestCase):
    def test_arp_cache_and_expiration(self):
        mac = MACAddress("00:11:22:33:44:55")
        ip = IPAddress("192.168.1.1")
        dev = VirtualNetDevice("eth0", mac, ip)
        arp = ARPEngine(dev, cache_timeout=0.1)

        # Static entry does not expire
        arp.insert_static(IPAddress("192.168.1.254"), MACAddress("fe:ed:fa:ce:ca:fe"))
        self.assertEqual(str(arp.resolve(IPAddress("192.168.1.254"))), "fe:ed:fa:ce:ca:fe")

        # Dynamic learning
        reply_pkt = ARPHeader(
            opcode=ARPOpcode.REPLY,
            sender_mac=MACAddress("12:34:56:78:90:ab"),
            sender_ip=IPAddress("192.168.1.50"),
            target_mac=mac,
            target_ip=ip
        ).pack()

        arp.handle_arp_frame(reply_pkt)
        self.assertEqual(str(arp.resolve(IPAddress("192.168.1.50"))), "12:34:56:78:90:ab")

        # Sleep past timeout and confirm expiration
        time.sleep(0.12)
        self.assertIsNone(arp.resolve(IPAddress("192.168.1.50")))
        # Static entry still present
        self.assertIsNotNone(arp.resolve(IPAddress("192.168.1.254")))

    def test_arp_request_and_reply_exchange(self):
        bus = VirtualEthernetBus("testbus")
        dev1 = VirtualNetDevice("eth0", MACAddress("00:00:00:00:00:01"), IPAddress("10.0.0.1"))
        dev2 = VirtualNetDevice("eth1", MACAddress("00:00:00:00:00:02"), IPAddress("10.0.0.2"))
        bus.attach(dev1)
        bus.attach(dev2)

        arp1 = ARPEngine(dev1)
        arp2 = ARPEngine(dev2)

        # dev1 sends ARP request for 10.0.0.2
        arp1.send_request(IPAddress("10.0.0.2"))

        # dev2 polls frame and handles it
        f_req = dev2.poll_frame()
        self.assertIsNotNone(f_req)
        arp2.handle_arp_frame(f_req.payload)

        # dev2 automatically transmitted ARP reply to dev1
        f_reply = dev1.poll_frame()
        self.assertIsNotNone(f_reply)
        arp1.handle_arp_frame(f_reply.payload)

        # Both devices now have each other in cache
        self.assertEqual(str(arp1.resolve(IPAddress("10.0.0.2"))), "00:00:00:00:00:02")
        self.assertEqual(str(arp2.resolve(IPAddress("10.0.0.1"))), "00:00:00:00:00:01")

    def test_ipv4_longest_prefix_match(self):
        ip_engine = IPEngine()
        dev = VirtualNetDevice("eth0", MACAddress("00:00:00:00:00:01"), IPAddress("10.0.0.1"))

        # Route A: Default route 0.0.0.0/0
        ip_engine.add_route(IPAddress("0.0.0.0"), IPAddress("0.0.0.0"), IPAddress("10.0.0.254"), dev)
        # Route B: 10.0.0.0/8
        ip_engine.add_route(IPAddress("10.0.0.0"), IPAddress("255.0.0.0"), None, dev)
        # Route C: 10.1.0.0/16
        ip_engine.add_route(IPAddress("10.1.0.0"), IPAddress("255.255.0.0"), None, dev)
        # Route D: 10.1.2.0/24 (Most specific for 10.1.2.5)
        ip_engine.add_route(IPAddress("10.1.2.0"), IPAddress("255.255.255.0"), None, dev)

        match_specific = ip_engine.find_route(IPAddress("10.1.2.5"))
        self.assertEqual(match_specific.prefix_length(), 24)

        match_16 = ip_engine.find_route(IPAddress("10.1.99.5"))
        self.assertEqual(match_16.prefix_length(), 16)

        match_default = ip_engine.find_route(IPAddress("172.16.0.1"))
        self.assertEqual(match_default.prefix_length(), 0)

    def test_ipv4_fragmentation_and_reassembly(self):
        bus = VirtualEthernetBus("frag_bus")
        dev1 = VirtualNetDevice("eth0", MACAddress("00:00:00:00:00:01"), IPAddress("192.168.1.1"), mtu=500)
        dev2 = VirtualNetDevice("eth1", MACAddress("00:00:00:00:00:02"), IPAddress("192.168.1.2"), mtu=500)
        bus.attach(dev1)
        bus.attach(dev2)

        arp1 = ARPEngine(dev1)
        arp2 = ARPEngine(dev2)
        arp1.insert_static(IPAddress("192.168.1.2"), MACAddress("00:00:00:00:00:02"))
        arp2.insert_static(IPAddress("192.168.1.1"), MACAddress("00:00:00:00:00:01"))

        ip1 = IPEngine()
        ip1.register_device(dev1, arp1)
        ip1.add_route(IPAddress("192.168.1.0"), IPAddress("255.255.255.0"), None, dev1)

        ip2 = IPEngine()
        ip2.register_device(dev2, arp2)
        ip2.add_route(IPAddress("192.168.1.0"), IPAddress("255.255.255.0"), None, dev2)

        received_packets = []
        def on_tcp(hdr: IPv4Header, payload: bytes):
            received_packets.append(payload)

        ip2.register_protocol(IPProtocol.TCP, on_tcp)

        # Send 1200 bytes payload over MTU=500 interface (max payload per frag = 480 bytes)
        # Expected fragments: 480 + 480 + 240 = 1200 bytes across 3 packets
        large_payload = b"ABCDEFGHIJ" * 120  # 1200 bytes
        sent = ip1.send_packet(
            src_ip=IPAddress("192.168.1.1"),
            dst_ip=IPAddress("192.168.1.2"),
            protocol=IPProtocol.TCP,
            payload=large_payload
        )
        self.assertTrue(sent)

        # Receive and pass fragments to receiver
        while True:
            frame = dev2.poll_frame()
            if frame is None:
                break
            ip2.handle_ip_packet(frame.payload)

        self.assertEqual(len(received_packets), 1)
        self.assertEqual(received_packets[0], large_payload)


if __name__ == "__main__":
    unittest.main()
