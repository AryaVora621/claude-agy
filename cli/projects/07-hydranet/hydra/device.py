"""
HydraNet: Virtual Ethernet Bus and Network Device Abstraction.
Simulates Layer 2 Ethernet switches, link-layer carrier channels,
and controllable chaos testing (packet loss, latency jitter, duplication, corruption).
"""

import random
import time
from typing import List, Dict, Optional, Callable
from collections import deque
from hydra.types import MACAddress, IPAddress, EthernetFrame, EthernetType


class VirtualEthernetBus:
    """
    Virtual Layer 2 broadcast bus connecting multiple virtual network interfaces.
    Forwards frames to target MAC addresses or broadcasts to all connected endpoints.
    """

    def __init__(
        self,
        name: str = "veth0",
        drop_rate: float = 0.0,
        duplicate_rate: float = 0.0,
        corrupt_rate: float = 0.0
    ):
        self.name = name
        self.drop_rate = drop_rate
        self.duplicate_rate = duplicate_rate
        self.corrupt_rate = corrupt_rate
        self.devices: Dict[MACAddress, "VirtualNetDevice"] = {}
        self.packet_log: deque = deque(maxlen=1000)
        self.wire_traffic_bytes = 0
        self.frames_broadcasted = 0
        self.frames_dropped = 0

    def attach(self, device: "VirtualNetDevice") -> None:
        self.devices[device.mac] = device
        device.bus = self

    def detach(self, device: "VirtualNetDevice") -> None:
        if device.mac in self.devices:
            del self.devices[device.mac]
            device.bus = None

    def transmit(self, sender: "VirtualNetDevice", raw_frame: bytes) -> bool:
        """Inject an Ethernet frame onto the bus with simulated physical channel effects."""
        self.wire_traffic_bytes += len(raw_frame)

        # 1. Packet Drop Simulation
        if self.drop_rate > 0.0 and random.random() < self.drop_rate:
            self.frames_dropped += 1
            return False

        # 2. Packet Corruption Simulation
        frame_bytes = raw_frame
        if self.corrupt_rate > 0.0 and random.random() < self.corrupt_rate:
            # Flip a byte
            idx = random.randint(0, len(raw_frame) - 1)
            corrupted = bytearray(raw_frame)
            corrupted[idx] ^= 0xFF
            frame_bytes = bytes(corrupted)

        # 3. Unpack destination MAC to route or broadcast
        try:
            frame, _ = EthernetFrame.unpack(frame_bytes)
        except ValueError:
            return False

        self.packet_log.append((time.time(), sender.mac, frame.dst_mac, frame.ethertype, len(frame_bytes)))

        # 4. Dispatch to target or broadcast
        deliver_copies = 2 if (self.duplicate_rate > 0.0 and random.random() < self.duplicate_rate) else 1

        for _ in range(deliver_copies):
            if frame.dst_mac.is_broadcast():
                self.frames_broadcasted += 1
                for dev in self.devices.values():
                    if dev.mac != sender.mac:
                        dev.enqueue_rx(frame_bytes)
            else:
                target_dev = self.devices.get(frame.dst_mac)
                if target_dev is not None and target_dev.mac != sender.mac:
                    target_dev.enqueue_rx(frame_bytes)

        return True


class VirtualNetDevice:
    """
    Virtual Network Interface Controller (vNIC).
    Handles Layer 2 frame ingress/egress, MTU enforcement, and hardware counters.
    """

    def __init__(self, name: str, mac: MACAddress, ip: IPAddress, mtu: int = 1500):
        self.name = name
        self.mac = mac
        self.ip = ip
        self.mtu = mtu
        self.bus: Optional[VirtualEthernetBus] = None
        self.stack: Optional[object] = None
        self.rx_queue: deque = deque()

        # Hardware statistics
        self.tx_packets = 0
        self.rx_packets = 0
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.rx_dropped = 0

    def send_frame(self, dst_mac: MACAddress, ethertype: int, payload: bytes) -> bool:
        """Transmit an Ethernet frame through the attached bus."""
        if len(payload) > self.mtu:
            raise ValueError(f"Payload ({len(payload)} bytes) exceeds MTU ({self.mtu} bytes)")

        if self.bus is None:
            return False

        frame = EthernetFrame(dst_mac=dst_mac, src_mac=self.mac, ethertype=ethertype, payload=payload)
        raw = frame.pack()
        success = self.bus.transmit(self, raw)
        if success:
            self.tx_packets += 1
            self.tx_bytes += len(raw)
        return success

    def enqueue_rx(self, raw_frame: bytes) -> None:
        """Receive incoming raw frame from physical bus into RX buffer."""
        self.rx_queue.append(raw_frame)
        self.rx_packets += 1
        self.rx_bytes += len(raw_frame)

    def poll_frame(self) -> Optional[EthernetFrame]:
        """Fetch next pending frame from the RX queue."""
        if not self.rx_queue:
            return None
        raw = self.rx_queue.popleft()
        try:
            frame, _ = EthernetFrame.unpack(raw)
            return frame
        except ValueError:
            self.rx_dropped += 1
            return None
