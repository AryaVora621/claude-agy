"""
HydraNet: IPv4 Layer 3 Routing, Fragmentation, and Reassembly Engine.
Implements RFC 791 IPv4 packet routing with Longest Prefix Match (LPM),
dynamic IP fragmentation across lower-layer MTU boundaries, and hole-filling reassembly.
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Callable
from hydra.types import (
    IPAddress,
    MACAddress,
    EthernetType,
    IPProtocol,
    IPv4Header,
    compute_checksum
)
from hydra.device import VirtualNetDevice
from hydra.arp import ARPEngine


class RouteEntry:
    """Routing table entry mapping destination subnets to next-hop gateways and interfaces."""
    __slots__ = ("network", "netmask", "gateway", "device", "metric")

    def __init__(
        self,
        network: IPAddress,
        netmask: IPAddress,
        gateway: Optional[IPAddress],
        device: VirtualNetDevice,
        metric: int = 1
    ):
        self.network = network
        self.netmask = netmask
        self.gateway = gateway
        self.device = device
        self.metric = metric

    def matches(self, dest: IPAddress) -> bool:
        """Verify if destination IP matches this network/netmask mask."""
        return (dest.int_val & self.netmask.int_val) == (self.network.int_val & self.netmask.int_val)

    def prefix_length(self) -> int:
        """Count leading 1 bits in netmask for Longest Prefix Match (LPM)."""
        mask = self.netmask.int_val
        length = 0
        while mask & 0x80000000:
            length += 1
            mask = (mask << 1) & 0xFFFFFFFF
        return length


class FragmentBuffer:
    """Tracks arriving IPv4 fragments for reassembly of a segmented datagram."""

    def __init__(self, total_len: Optional[int] = None):
        self.fragments: Dict[int, bytes] = {}  # offset_bytes -> payload
        self.total_len: Optional[int] = total_len
        self.created_at: float = time.time()

    def add_fragment(self, offset_bytes: int, payload: bytes, is_last: bool) -> None:
        self.fragments[offset_bytes] = payload
        if is_last:
            self.total_len = offset_bytes + len(payload)

    def is_complete(self) -> bool:
        if self.total_len is None:
            return False
        # Verify contiguous coverage from offset 0 to total_len
        curr = 0
        sorted_offsets = sorted(self.fragments.keys())
        for off in sorted_offsets:
            if off > curr:
                return False  # Missing fragment gap
            curr = max(curr, off + len(self.fragments[off]))
        return curr >= self.total_len

    def assemble(self) -> bytes:
        sorted_offsets = sorted(self.fragments.keys())
        buf = bytearray()
        for off in sorted_offsets:
            frag = self.fragments[off]
            if len(buf) < off:
                buf.extend(b"\x00" * (off - len(buf)))
            buf[off : off + len(frag)] = frag
        return bytes(buf)


class IPEngine:
    """
    Core IPv4 Routing and Packet Dispatching Engine.
    """

    def __init__(self, default_ttl: int = 64):
        self.default_ttl = default_ttl
        self.routes: List[RouteEntry] = []
        self.devices: List[VirtualNetDevice] = []
        self.arp_engines: Dict[VirtualNetDevice, ARPEngine] = {}
        self.next_ident = 1000
        # Reassembly table: (src_ip, dst_ip, ident, protocol) -> FragmentBuffer
        self.reassembly_table: Dict[Tuple[IPAddress, IPAddress, int, int], FragmentBuffer] = {}
        self.protocol_handlers: Dict[IPProtocol, Callable[[IPv4Header, bytes], None]] = {}

    def register_device(self, device: VirtualNetDevice, arp: ARPEngine) -> None:
        self.devices.append(device)
        self.arp_engines[device] = arp

    def add_route(
        self,
        network: IPAddress,
        netmask: IPAddress,
        gateway: Optional[IPAddress],
        device: VirtualNetDevice,
        metric: int = 1
    ) -> None:
        entry = RouteEntry(network, netmask, gateway, device, metric)
        self.routes.append(entry)
        # Sort routes by longest prefix length descending, then metric ascending
        self.routes.sort(key=lambda r: (-r.prefix_length(), r.metric))

    def register_protocol(self, proto: IPProtocol, handler: Callable[[IPv4Header, bytes], None]) -> None:
        self.protocol_handlers[proto] = handler

    def find_route(self, dest: IPAddress) -> Optional[RouteEntry]:
        """Perform Longest Prefix Match (LPM) lookup for destination IP."""
        for r in self.routes:
            if r.matches(dest):
                return r
        return None

    def send_packet(
        self,
        src_ip: IPAddress,
        dst_ip: IPAddress,
        protocol: IPProtocol,
        payload: bytes,
        dont_fragment: bool = False
    ) -> bool:
        """
        Route and transmit an IPv4 datagram.
        Performs fragmentation if payload exceeds path MTU.
        """
        route = self.find_route(dst_ip)
        if route is None:
            return False

        dev = route.device
        arp = self.arp_engines.get(dev)
        if arp is None:
            return False

        # Determine next-hop L2 destination
        next_hop = route.gateway if route.gateway is not None else dst_ip
        dst_mac = arp.resolve(next_hop)
        if dst_mac is None:
            # Trigger ARP request for next hop
            arp.send_request(next_hop)
            stack = getattr(dev, "stack", None)
            if stack is not None:
                stack.step()
            dst_mac = arp.resolve(next_hop)
            if dst_mac is None:
                return False

        max_ip_payload = dev.mtu - 20
        ident = self.next_ident
        self.next_ident = (self.next_ident + 1) & 0xFFFF

        # Case 1: Fits in single MTU frame without fragmentation
        if len(payload) <= max_ip_payload:
            hdr = IPv4Header(
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=protocol,
                total_length=20 + len(payload),
                identification=ident,
                flags=2 if dont_fragment else 0,
                fragment_offset=0,
                ttl=self.default_ttl
            )
            ip_packet = hdr.pack() + payload
            return dev.send_frame(dst_mac, int(EthernetType.IPV4), ip_packet)

        # Case 2: Must Fragment
        if dont_fragment:
            return False  # Fragmentation needed but DF bit set

        # Fragment size must be multiple of 8 bytes
        frag_size = (max_ip_payload // 8) * 8
        offset = 0
        total_len = len(payload)

        while offset < total_len:
            chunk = payload[offset : offset + frag_size]
            is_last = (offset + len(chunk)) >= total_len
            flags = 0 if is_last else 1  # MF (More Fragments) bit

            hdr = IPv4Header(
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=protocol,
                total_length=20 + len(chunk),
                identification=ident,
                flags=flags,
                fragment_offset=offset // 8,
                ttl=self.default_ttl
            )
            ip_packet = hdr.pack() + chunk
            success = dev.send_frame(dst_mac, int(EthernetType.IPV4), ip_packet)
            if not success:
                return False
            offset += len(chunk)

        return True

    def handle_ip_packet(self, raw_ip: bytes) -> None:
        """
        Process incoming IPv4 packet from Layer 2.
        Validates checksum, reassembles fragments, and dispatches to upper-layer protocol.
        """
        try:
            hdr, payload = IPv4Header.unpack(raw_ip)
        except ValueError:
            return

        # Verify IPv4 header checksum
        if compute_checksum(raw_ip[: hdr.ihl * 4]) != 0:
            return  # Corrupted header discarded

        # Handle Fragmentation & Reassembly
        is_fragment = (hdr.flags & 1) != 0 or (hdr.fragment_offset > 0)
        if is_fragment:
            key = (hdr.src_ip, hdr.dst_ip, hdr.identification, int(hdr.protocol))
            if key not in self.reassembly_table:
                self.reassembly_table[key] = FragmentBuffer()

            buf = self.reassembly_table[key]
            is_last = (hdr.flags & 1) == 0
            buf.add_fragment(hdr.fragment_offset * 8, payload, is_last)

            if buf.is_complete():
                full_payload = buf.assemble()
                del self.reassembly_table[key]
                self._dispatch(hdr, full_payload)
        else:
            self._dispatch(hdr, payload)

    def _dispatch(self, hdr: IPv4Header, payload: bytes) -> None:
        handler = self.protocol_handlers.get(hdr.protocol)
        if handler is not None:
            handler(hdr, payload)
