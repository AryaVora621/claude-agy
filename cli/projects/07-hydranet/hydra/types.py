"""
HydraNet: Protocol Types, Binary Framing, and RFC 1071 Checksum Engine.
Implements Ethernet II frames, ARP headers, IPv4 packet framing with fragmentation flags,
and RFC 793 TCP segment headers with pseudo-header checksum verification.
"""

import struct
from enum import IntEnum
from typing import Tuple, Optional


class EthernetType(IntEnum):
    IPV4 = 0x0800
    ARP = 0x0806


class ARPOpcode(IntEnum):
    REQUEST = 1
    REPLY = 2


class IPProtocol(IntEnum):
    ICMP = 1
    TCP = 6
    UDP = 17


class TCPFlags(IntEnum):
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80


def compute_checksum(data: bytes) -> int:
    """
    RFC 1071: 16-bit One's Complement Sum of 16-bit words.
    Pads odd-length payloads with a trailing zero byte.
    """
    if len(data) % 2 != 0:
        data = data + b"\x00"

    total = 0
    for i in range(0, len(data), 2):
        word = (data[i] << 8) + data[i + 1]
        total += word
        # Fold carry bits into lower 16 bits
        while total > 0xFFFF:
            total = (total & 0xFFFF) + (total >> 16)

    # Invert bits to produce one's complement
    return ~total & 0xFFFF


class MACAddress:
    """6-byte IEEE 802.3 MAC hardware address."""
    __slots__ = ("raw",)

    def __init__(self, addr: str):
        parts = addr.split(":")
        if len(parts) != 6:
            raise ValueError(f"Invalid MAC address string: {addr}")
        self.raw = bytes(int(p, 16) for p in parts)

    @classmethod
    def from_bytes(cls, b: bytes) -> "MACAddress":
        if len(b) != 6:
            raise ValueError(f"MAC bytes must be length 6, got {len(b)}")
        obj = cls.__new__(cls)
        obj.raw = b
        return obj

    def __str__(self) -> str:
        return ":".join(f"{b:02x}" for b in self.raw)

    def __repr__(self) -> str:
        return f"MACAddress('{str(self)}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MACAddress):
            return self.raw == other.raw
        return False

    def __hash__(self) -> int:
        return hash(self.raw)

    def is_broadcast(self) -> bool:
        return self.raw == b"\xff\xff\xff\xff\xff\xff"


class IPAddress:
    """4-byte IPv4 dotted-quad address."""
    __slots__ = ("raw", "int_val")

    def __init__(self, addr: str):
        parts = addr.split(".")
        if len(parts) != 4:
            raise ValueError(f"Invalid IPv4 address string: {addr}")
        self.raw = bytes(int(p) for p in parts)
        self.int_val = struct.unpack("!I", self.raw)[0]

    @classmethod
    def from_bytes(cls, b: bytes) -> "IPAddress":
        if len(b) != 4:
            raise ValueError(f"IPv4 bytes must be length 4, got {len(b)}")
        obj = cls.__new__(cls)
        obj.raw = b
        obj.int_val = struct.unpack("!I", b)[0]
        return obj

    @classmethod
    def from_int(cls, val: int) -> "IPAddress":
        obj = cls.__new__(cls)
        obj.int_val = val & 0xFFFFFFFF
        obj.raw = struct.pack("!I", obj.int_val)
        return obj

    def __str__(self) -> str:
        return ".".join(str(b) for b in self.raw)

    def __repr__(self) -> str:
        return f"IPAddress('{str(self)}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IPAddress):
            return self.int_val == other.int_val
        return False

    def __hash__(self) -> int:
        return hash(self.int_val)

    def is_loopback(self) -> bool:
        return (self.int_val >> 24) == 127


class EthernetFrame:
    """Ethernet II frame (14-byte header + payload)."""
    __slots__ = ("dst_mac", "src_mac", "ethertype", "payload")

    def __init__(self, dst_mac: MACAddress, src_mac: MACAddress, ethertype: int, payload: bytes = b""):
        self.dst_mac = dst_mac
        self.src_mac = src_mac
        self.ethertype = ethertype
        self.payload = payload

    def pack(self) -> bytes:
        header = self.dst_mac.raw + self.src_mac.raw + struct.pack("!H", self.ethertype)
        return header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> Tuple["EthernetFrame", bytes]:
        if len(data) < 14:
            raise ValueError("Data too short for Ethernet frame")
        dst = MACAddress.from_bytes(data[0:6])
        src = MACAddress.from_bytes(data[6:12])
        ethertype = struct.unpack("!H", data[12:14])[0]
        payload = data[14:]
        return cls(dst, src, ethertype, payload), payload


class ARPHeader:
    """RFC 826 Address Resolution Protocol (28 bytes)."""
    __slots__ = ("hw_type", "proto_type", "hw_size", "proto_size", "opcode",
                 "sender_mac", "sender_ip", "target_mac", "target_ip")

    def __init__(
        self,
        opcode: ARPOpcode,
        sender_mac: MACAddress,
        sender_ip: IPAddress,
        target_mac: MACAddress,
        target_ip: IPAddress,
        hw_type: int = 1,
        proto_type: int = int(EthernetType.IPV4),
        hw_size: int = 6,
        proto_size: int = 4
    ):
        self.hw_type = hw_type
        self.proto_type = proto_type
        self.hw_size = hw_size
        self.proto_size = proto_size
        self.opcode = opcode
        self.sender_mac = sender_mac
        self.sender_ip = sender_ip
        self.target_mac = target_mac
        self.target_ip = target_ip

    def pack(self) -> bytes:
        prefix = struct.pack("!HHBBH", self.hw_type, self.proto_type, self.hw_size, self.proto_size, self.opcode)
        return prefix + self.sender_mac.raw + self.sender_ip.raw + self.target_mac.raw + self.target_ip.raw

    @classmethod
    def unpack(cls, data: bytes) -> "ARPHeader":
        if len(data) < 28:
            raise ValueError("Data too short for ARP packet")
        hw_type, proto_type, hw_size, proto_size, opcode = struct.unpack("!HHBBH", data[:8])
        sender_mac = MACAddress.from_bytes(data[8:14])
        sender_ip = IPAddress.from_bytes(data[14:18])
        target_mac = MACAddress.from_bytes(data[18:24])
        target_ip = IPAddress.from_bytes(data[24:28])
        return cls(
            opcode=ARPOpcode(opcode),
            sender_mac=sender_mac,
            sender_ip=sender_ip,
            target_mac=target_mac,
            target_ip=target_ip,
            hw_type=hw_type,
            proto_type=proto_type,
            hw_size=hw_size,
            proto_size=proto_size
        )


class IPv4Header:
    """RFC 791 IPv4 Header (minimum 20 bytes)."""
    __slots__ = ("version", "ihl", "tos", "total_length", "identification",
                 "flags", "fragment_offset", "ttl", "protocol", "checksum",
                 "src_ip", "dst_ip", "options")

    def __init__(
        self,
        src_ip: IPAddress,
        dst_ip: IPAddress,
        protocol: IPProtocol,
        total_length: int = 20,
        identification: int = 0,
        flags: int = 2,  # DF (Don't Fragment) default
        fragment_offset: int = 0,
        ttl: int = 64,
        tos: int = 0,
        version: int = 4,
        ihl: int = 5,
        checksum: int = 0,
        options: bytes = b""
    ):
        self.version = version
        self.ihl = ihl
        self.tos = tos
        self.total_length = total_length
        self.identification = identification
        self.flags = flags
        self.fragment_offset = fragment_offset
        self.ttl = ttl
        self.protocol = protocol
        self.checksum = checksum
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.options = options

    def pack(self, compute_sum: bool = True) -> bytes:
        v_ihl = (self.version << 4) | self.ihl
        flags_frag = ((self.flags & 0x07) << 13) | (self.fragment_offset & 0x1FFF)

        chk = 0
        raw_pre = struct.pack(
            "!BBHHHBBH",
            v_ihl,
            self.tos,
            self.total_length,
            self.identification,
            flags_frag,
            self.ttl,
            int(self.protocol),
            chk
        ) + self.src_ip.raw + self.dst_ip.raw + self.options

        if compute_sum:
            chk = compute_checksum(raw_pre)
            self.checksum = chk
            # Repack with real checksum
            return struct.pack(
                "!BBHHHBBH",
                v_ihl,
                self.tos,
                self.total_length,
                self.identification,
                flags_frag,
                self.ttl,
                int(self.protocol),
                chk
            ) + self.src_ip.raw + self.dst_ip.raw + self.options
        return raw_pre

    @classmethod
    def unpack(cls, data: bytes) -> Tuple["IPv4Header", bytes]:
        if len(data) < 20:
            raise ValueError("Data too short for IPv4 header")
        v_ihl, tos, tot_len, ident, flags_frag, ttl, proto, chk = struct.unpack("!BBHHHBBH", data[:12])
        version = v_ihl >> 4
        ihl = v_ihl & 0x0F
        header_len = ihl * 4
        if len(data) < header_len:
            raise ValueError(f"Packet smaller than declared IPv4 IHL ({header_len} bytes)")

        flags = (flags_frag >> 13) & 0x07
        frag_offset = flags_frag & 0x1FFF
        src_ip = IPAddress.from_bytes(data[12:16])
        dst_ip = IPAddress.from_bytes(data[16:20])
        options = data[20:header_len]
        payload = data[header_len:tot_len] if tot_len <= len(data) else data[header_len:]

        header = cls(
            src_ip=src_ip,
            dst_ip=dst_ip,
            protocol=IPProtocol(proto),
            total_length=tot_len,
            identification=ident,
            flags=flags,
            fragment_offset=frag_offset,
            ttl=ttl,
            tos=tos,
            version=version,
            ihl=ihl,
            checksum=chk,
            options=options
        )
        return header, payload


def compute_tcp_checksum(src_ip: IPAddress, dst_ip: IPAddress, tcp_raw: bytes) -> int:
    """
    RFC 793 TCP Checksum over Pseudo-Header + TCP Header + Data.
    Pseudo-Header (12 bytes):
      Source IP (4B) | Dest IP (4B) | Zero (1B) | Protocol 6 (1B) | TCP Length (2B)
    """
    pseudo_hdr = src_ip.raw + dst_ip.raw + struct.pack("!BBH", 0, int(IPProtocol.TCP), len(tcp_raw))
    return compute_checksum(pseudo_hdr + tcp_raw)


class TCPHeader:
    """RFC 793 Transmission Control Protocol Header (minimum 20 bytes)."""
    __slots__ = ("src_port", "dst_port", "seq_num", "ack_num", "data_offset",
                 "flags", "window_size", "checksum", "urgent_ptr", "options")

    def __init__(
        self,
        src_port: int,
        dst_port: int,
        seq_num: int = 0,
        ack_num: int = 0,
        flags: int = 0,
        window_size: int = 65535,
        data_offset: int = 5,
        checksum: int = 0,
        urgent_ptr: int = 0,
        options: bytes = b""
    ):
        self.src_port = src_port
        self.dst_port = dst_port
        self.seq_num = seq_num & 0xFFFFFFFF
        self.ack_num = ack_num & 0xFFFFFFFF
        self.flags = flags & 0xFF
        self.window_size = window_size & 0xFFFF
        self.data_offset = data_offset
        self.checksum = checksum
        self.urgent_ptr = urgent_ptr
        self.options = options

    def has_flag(self, flag: TCPFlags) -> bool:
        return bool(self.flags & flag)

    def pack(self, src_ip: Optional[IPAddress] = None, dst_ip: Optional[IPAddress] = None, payload: bytes = b"") -> bytes:
        data_offset_res = (self.data_offset << 4) & 0xF0
        raw_header = struct.pack(
            "!HHIIBBHHH",
            self.src_port,
            self.dst_port,
            self.seq_num,
            self.ack_num,
            data_offset_res,
            self.flags,
            self.window_size,
            0,  # Zero checksum for calculation
            self.urgent_ptr
        ) + self.options

        if src_ip is not None and dst_ip is not None:
            chk = compute_tcp_checksum(src_ip, dst_ip, raw_header + payload)
            self.checksum = chk
            return struct.pack(
                "!HHIIBBHHH",
                self.src_port,
                self.dst_port,
                self.seq_num,
                self.ack_num,
                data_offset_res,
                self.flags,
                self.window_size,
                chk,
                self.urgent_ptr
            ) + self.options
        return raw_header

    @classmethod
    def unpack(cls, data: bytes) -> Tuple["TCPHeader", bytes]:
        if len(data) < 20:
            raise ValueError("Data too short for TCP header")
        src_port, dst_port, seq, ack, offset_res, flags, win, chk, urg = struct.unpack("!HHIIBBHHH", data[:20])
        data_offset = offset_res >> 4
        header_len = data_offset * 4
        if len(data) < header_len:
            raise ValueError(f"Packet smaller than declared TCP data offset ({header_len} bytes)")

        options = data[20:header_len]
        payload = data[header_len:]
        header = cls(
            src_port=src_port,
            dst_port=dst_port,
            seq_num=seq,
            ack_num=ack,
            flags=flags,
            window_size=win,
            data_offset=data_offset,
            checksum=chk,
            urgent_ptr=urg,
            options=options
        )
        return header, payload
