"""
HydraNet: Address Resolution Protocol (ARP, RFC 826) Engine.
Maintains Layer 2 hardware address resolution table, dynamic cache timeouts,
and automatic ARP Request broadcast / Reply generation.
"""

import time
from typing import Dict, Optional, Tuple
from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetType,
    ARPHeader,
    ARPOpcode
)
from hydra.device import VirtualNetDevice


class ARPCacheEntry:
    """Stores resolved hardware address with lease expiration timestamp."""
    __slots__ = ("mac", "timestamp", "is_static")

    def __init__(self, mac: MACAddress, timestamp: float, is_static: bool = False):
        self.mac = mac
        self.timestamp = timestamp
        self.is_static = is_static

    def is_expired(self, current_time: float, timeout_sec: float) -> bool:
        if self.is_static:
            return False
        return (current_time - self.timestamp) > timeout_sec


class ARPEngine:
    """
    Manages ARP resolution table and packet processing for a network interface.
    """

    def __init__(self, device: VirtualNetDevice, cache_timeout: float = 300.0):
        self.device = device
        self.cache_timeout = cache_timeout
        self.cache: Dict[IPAddress, ARPCacheEntry] = {}

    def insert_static(self, ip: IPAddress, mac: MACAddress) -> None:
        """Insert permanent static ARP entry."""
        self.cache[ip] = ARPCacheEntry(mac, time.time(), is_static=True)

    def resolve(self, ip: IPAddress) -> Optional[MACAddress]:
        """
        Lookup target IP address in the local ARP cache.
        Returns hardware MAC address if resolved and valid, None otherwise.
        """
        now = time.time()
        entry = self.cache.get(ip)
        if entry is not None:
            if entry.is_expired(now, self.cache_timeout):
                del self.cache[ip]
                return None
            return entry.mac
        return None

    def send_request(self, target_ip: IPAddress) -> None:
        """Broadcast an ARP Request: 'Who has target_ip? Tell device.ip'."""
        broadcast_mac = MACAddress("ff:ff:ff:ff:ff:ff")
        zero_mac = MACAddress("00:00:00:00:00:00")

        arp_req = ARPHeader(
            opcode=ARPOpcode.REQUEST,
            sender_mac=self.device.mac,
            sender_ip=self.device.ip,
            target_mac=zero_mac,
            target_ip=target_ip
        )
        self.device.send_frame(
            dst_mac=broadcast_mac,
            ethertype=int(EthernetType.ARP),
            payload=arp_req.pack()
        )

    def handle_arp_frame(self, payload: bytes) -> Optional[ARPHeader]:
        """
        Process incoming ARP frame payload.
        Updates ARP cache and transmits replies to requests directed at this device.
        """
        try:
            arp = ARPHeader.unpack(payload)
        except ValueError:
            return None

        # Opportunistic learning: cache the sender's IP -> MAC mapping
        self.cache[arp.sender_ip] = ARPCacheEntry(arp.sender_mac, time.time())

        # If incoming frame is an ARP Request for this device's IP:
        if arp.opcode == ARPOpcode.REQUEST and arp.target_ip == self.device.ip:
            # Transmit unicast ARP Reply directly back to sender
            arp_reply = ARPHeader(
                opcode=ARPOpcode.REPLY,
                sender_mac=self.device.mac,
                sender_ip=self.device.ip,
                target_mac=arp.sender_mac,
                target_ip=arp.sender_ip
            )
            self.device.send_frame(
                dst_mac=arp.sender_mac,
                ethertype=int(EthernetType.ARP),
                payload=arp_reply.pack()
            )

        return arp
