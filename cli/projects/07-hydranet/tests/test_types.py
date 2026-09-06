"""
HydraNet: Unit Tests for Protocol Types, Headers, and RFC 1071 Checksums.
"""

import unittest
from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetType,
    EthernetFrame,
    ARPOpcode,
    ARPHeader,
    IPProtocol,
    IPv4Header,
    TCPFlags,
    TCPHeader,
    compute_checksum,
    compute_tcp_checksum
)


class TestTypesAndChecksum(unittest.TestCase):
    def test_mac_address(self):
        mac = MACAddress("00:1a:2b:3c:4d:5e")
        self.assertEqual(str(mac), "00:1a:2b:3c:4d:5e")
        self.assertEqual(mac.raw, b"\x00\x1a\x2b\x3c\x4d\x5e")
        self.assertFalse(mac.is_broadcast())

        bcast = MACAddress("ff:ff:ff:ff:ff:ff")
        self.assertTrue(bcast.is_broadcast())

        # Equality & hash
        mac2 = MACAddress.from_bytes(b"\x00\x1a\x2b\x3c\x4d\x5e")
        self.assertEqual(mac, mac2)
        self.assertEqual(hash(mac), hash(mac2))

    def test_ip_address(self):
        ip = IPAddress("192.168.1.100")
        self.assertEqual(str(ip), "192.168.1.100")
        self.assertEqual(ip.int_val, (192 << 24) + (168 << 16) + (1 << 8) + 100)
        self.assertFalse(ip.is_loopback())

        loop = IPAddress("127.0.0.1")
        self.assertTrue(loop.is_loopback())

        ip2 = IPAddress.from_bytes(bytes([192, 168, 1, 100]))
        self.assertEqual(ip, ip2)
        self.assertEqual(hash(ip), hash(ip2))

    def test_rfc1071_checksum(self):
        # Known test vector from RFC 1071
        # Two 16-bit words: 0x0001, 0x0002 -> sum = 0x0003, complement = 0xFFFC
        data = b"\x00\x01\x00\x02"
        chk = compute_checksum(data)
        self.assertEqual(chk, 0xFFFC)

        # Checksum over data + checksum word must be zero (one's complement identity)
        import struct
        data_with_chk = data + struct.pack("!H", chk)
        self.assertEqual(compute_checksum(data_with_chk), 0)

        # Odd-length payload handling
        odd_data = b"\x01\x02\x03"
        chk_odd = compute_checksum(odd_data)
        self.assertIsInstance(chk_odd, int)

    def test_ethernet_frame_pack_unpack(self):
        dst = MACAddress("aa:bb:cc:dd:ee:ff")
        src = MACAddress("11:22:33:44:55:66")
        payload = b"Hello Ethernet World!"
        frame = EthernetFrame(dst_mac=dst, src_mac=src, ethertype=int(EthernetType.IPV4), payload=payload)

        raw = frame.pack()
        self.assertEqual(len(raw), 14 + len(payload))

        unpacked, unp_payload = EthernetFrame.unpack(raw)
        self.assertEqual(unpacked.dst_mac, dst)
        self.assertEqual(unpacked.src_mac, src)
        self.assertEqual(unpacked.ethertype, int(EthernetType.IPV4))
        self.assertEqual(unp_payload, payload)

    def test_arp_header_pack_unpack(self):
        s_mac = MACAddress("00:11:22:33:44:55")
        s_ip = IPAddress("10.0.0.1")
        t_mac = MACAddress("ff:ff:ff:ff:ff:ff")
        t_ip = IPAddress("10.0.0.2")

        arp = ARPHeader(
            opcode=ARPOpcode.REQUEST,
            sender_mac=s_mac,
            sender_ip=s_ip,
            target_mac=t_mac,
            target_ip=t_ip
        )
        raw = arp.pack()
        self.assertEqual(len(raw), 28)

        unpacked = ARPHeader.unpack(raw)
        self.assertEqual(unpacked.opcode, ARPOpcode.REQUEST)
        self.assertEqual(unpacked.sender_mac, s_mac)
        self.assertEqual(unpacked.sender_ip, s_ip)
        self.assertEqual(unpacked.target_mac, t_mac)
        self.assertEqual(unpacked.target_ip, t_ip)

    def test_ipv4_header_pack_unpack_and_checksum(self):
        src = IPAddress("192.168.1.1")
        dst = IPAddress("192.168.1.2")
        payload = b"Sample IPv4 Payload"

        ip_hdr = IPv4Header(
            src_ip=src,
            dst_ip=dst,
            protocol=IPProtocol.TCP,
            total_length=20 + len(payload),
            identification=1234
        )
        raw_hdr = ip_hdr.pack(compute_sum=True)
        self.assertEqual(len(raw_hdr), 20)

        # Checksum over valid header must evaluate to zero
        self.assertEqual(compute_checksum(raw_hdr), 0)

        unpacked_hdr, unpacked_payload = IPv4Header.unpack(raw_hdr + payload)
        self.assertEqual(unpacked_hdr.src_ip, src)
        self.assertEqual(unpacked_hdr.dst_ip, dst)
        self.assertEqual(unpacked_hdr.protocol, IPProtocol.TCP)
        self.assertEqual(unpacked_hdr.identification, 1234)
        self.assertEqual(unpacked_payload, payload)

    def test_tcp_header_pack_unpack_and_pseudo_checksum(self):
        src_ip = IPAddress("10.0.0.1")
        dst_ip = IPAddress("10.0.0.2")
        payload = b"HydraNet TCP Payload Data"

        tcp_hdr = TCPHeader(
            src_port=8080,
            dst_port=443,
            seq_num=1000,
            ack_num=500,
            flags=int(TCPFlags.SYN) | int(TCPFlags.ACK),
            window_size=32768
        )
        raw_tcp_hdr = tcp_hdr.pack(src_ip=src_ip, dst_ip=dst_ip, payload=payload)
        self.assertEqual(len(raw_tcp_hdr), 20)

        # Verify TCP pseudo-header checksum verification
        pseudo_check = compute_tcp_checksum(src_ip, dst_ip, raw_tcp_hdr + payload)
        self.assertEqual(pseudo_check, 0)

        unp_tcp, unp_payload = TCPHeader.unpack(raw_tcp_hdr + payload)
        self.assertEqual(unp_tcp.src_port, 8080)
        self.assertEqual(unp_tcp.dst_port, 443)
        self.assertEqual(unp_tcp.seq_num, 1000)
        self.assertEqual(unp_tcp.ack_num, 500)
        self.assertTrue(unp_tcp.has_flag(TCPFlags.SYN))
        self.assertTrue(unp_tcp.has_flag(TCPFlags.ACK))
        self.assertFalse(unp_tcp.has_flag(TCPFlags.FIN))
        self.assertEqual(unp_tcp.window_size, 32768)
        self.assertEqual(unp_payload, payload)


if __name__ == "__main__":
    unittest.main()
