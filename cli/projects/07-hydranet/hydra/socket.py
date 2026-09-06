"""
HydraNet: POSIX-Compatible Berkeley Socket API & Unified Network Host Stack.
Binds Layer 2 (Device/Bus), Layer 3 (ARP/IP), and Layer 4 (TCP FSM) into an
integrated protocol stack exposing socket(), bind(), listen(), accept(), connect(), send(), and recv().
"""

import time
import random
from typing import Dict, Tuple, Optional, List
from collections import deque
from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetType,
    IPProtocol,
    IPv4Header,
    TCPHeader,
    TCPFlags,
    compute_tcp_checksum
)
from hydra.device import VirtualNetDevice, VirtualEthernetBus
from hydra.arp import ARPEngine
from hydra.ip import IPEngine
from hydra.tcp import TCPConnection, TCPState


class TCPSocket:
    """
    POSIX-like Berkeley Socket interface.
    Operates in listening mode (accepting child sockets) or connected streaming mode.
    """

    def __init__(self, stack: "NetworkStack"):
        self.stack = stack
        self.local_ip: Optional[IPAddress] = None
        self.local_port: Optional[int] = None
        self.remote_ip: Optional[IPAddress] = None
        self.remote_port: Optional[int] = None

        self.conn: Optional[TCPConnection] = None
        self.is_listening = False
        self.backlog_queue: deque = deque()

    def bind(self, address: Tuple[str, int]) -> None:
        """Bind socket to local IP and port."""
        ip_str, port = address
        self.local_ip = IPAddress(ip_str)
        self.local_port = port
        self.stack.register_socket(self)

    def listen(self, backlog: int = 128) -> None:
        """Begin listening for incoming TCP connection requests."""
        if self.local_port is None:
            raise ValueError("Socket must be bound before listening")
        self.is_listening = True

    def accept(self, timeout: float = 5.0) -> Tuple["TCPSocket", Tuple[str, int]]:
        """Wait for and accept an inbound connection."""
        if not self.is_listening:
            raise ValueError("Socket is not in listening state")

        start = time.time()
        while time.time() - start < timeout:
            self.stack.step()
            if self.backlog_queue:
                child_conn = self.backlog_queue.popleft()
                child_sock = TCPSocket(self.stack)
                child_sock.local_ip = child_conn.local_ip
                child_sock.local_port = child_conn.local_port
                child_sock.remote_ip = child_conn.remote_ip
                child_sock.remote_port = child_conn.remote_port
                child_sock.conn = child_conn
                self.stack.active_connections[(child_sock.local_port, child_sock.remote_ip, child_sock.remote_port)] = child_conn
                return child_sock, (str(child_sock.remote_ip), child_sock.remote_port)
            time.sleep(0.001)

        raise TimeoutError("Accept timed out waiting for connection")

    def connect(self, address: Tuple[str, int], timeout: float = 5.0) -> None:
        """Initiate active TCP 3-way handshake to target server."""
        remote_ip_str, remote_port = address
        self.remote_ip = IPAddress(remote_ip_str)
        self.remote_port = remote_port

        if self.local_ip is None:
            self.local_ip = self.stack.device.ip
        if self.local_port is None:
            self.local_port = random.randint(49152, 65535)

        self.stack.register_socket(self)

        def send_segment_cb(hdr: TCPHeader, payload: bytes) -> bool:
            return self.stack.send_tcp(self.local_ip, self.remote_ip, hdr, payload)

        self.conn = TCPConnection(
            local_ip=self.local_ip,
            local_port=self.local_port,
            remote_ip=self.remote_ip,
            remote_port=self.remote_port,
            send_callback=send_segment_cb
        )
        self.stack.active_connections[(self.local_port, self.remote_ip, self.remote_port)] = self.conn
        self.conn.connect()

        # Wait for 3-way handshake to reach ESTABLISHED
        start = time.time()
        while time.time() - start < timeout:
            self.stack.step()
            if self.conn.state == TCPState.ESTABLISHED:
                return
            time.sleep(0.001)

        raise TimeoutError(f"Connection to {address} timed out (State: {self.conn.state.name})")

    def send(self, data: bytes) -> int:
        """Send data over established TCP stream."""
        if self.conn is None or self.conn.state != TCPState.ESTABLISHED:
            raise ConnectionError("Socket is not connected")
        sent = self.conn.send(data)
        self.stack.step()
        return sent

    def recv(self, bufsize: int = 4096, timeout: float = 5.0) -> bytes:
        """Read data from the TCP stream buffer."""
        if self.conn is None:
            raise ConnectionError("Socket is not connected")

        start = time.time()
        while time.time() - start < timeout:
            self.stack.step()
            data = self.conn.recv(bufsize)
            if data:
                return data
            if self.conn.state in (TCPState.CLOSE_WAIT, TCPState.CLOSED, TCPState.TIME_WAIT):
                return b""  # EOF (peer closed write end)
            time.sleep(0.001)

        return b""

    def close(self) -> None:
        """Close socket connection."""
        if self.conn is not None:
            self.conn.close()
            self.stack.step()


class NetworkStack:
    """
    Complete User-Space TCP/IP Host Network Node.
    Glues Ethernet device, ARP engine, IPv4 router, and TCP transport layer.
    """

    def __init__(self, name: str, mac: MACAddress, ip: IPAddress, netmask: IPAddress):
        self.name = name
        self.device = VirtualNetDevice(name, mac, ip)
        self.device.stack = self
        self.arp = ARPEngine(self.device)
        self.ip = IPEngine()
        self.ip.register_device(self.device, self.arp)
        # Direct subnet local route
        self.ip.add_route(ip, netmask, None, self.device)

        # Register higher-layer protocol handlers
        self.ip.register_protocol(IPProtocol.TCP, self._handle_tcp_packet)

        # TCP Port Demultiplexer
        self.listening_sockets: Dict[int, TCPSocket] = {}
        # active_connections: (local_port, remote_ip, remote_port) -> TCPConnection
        self.active_connections: Dict[Tuple[int, IPAddress, int], TCPConnection] = {}

    def register_socket(self, sock: TCPSocket) -> None:
        if sock.local_port is not None:
            self.listening_sockets[sock.local_port] = sock

    def send_tcp(self, src_ip: IPAddress, dst_ip: IPAddress, hdr: TCPHeader, payload: bytes) -> bool:
        """Package TCP segment and send via IPv4 engine."""
        tcp_raw = hdr.pack(src_ip, dst_ip, payload) + payload
        return self.ip.send_packet(src_ip, dst_ip, IPProtocol.TCP, tcp_raw)

    def _handle_tcp_packet(self, ip_hdr: IPv4Header, payload: bytes) -> None:
        """Route incoming TCP packet to active connection or listening socket."""
        try:
            tcp_hdr, tcp_payload = TCPHeader.unpack(payload)
        except ValueError:
            return

        # Check for active connection match (local_port, remote_ip, remote_port)
        conn_key = (tcp_hdr.dst_port, ip_hdr.src_ip, tcp_hdr.src_port)
        conn = self.active_connections.get(conn_key)
        if conn is not None:
            conn.handle_segment(tcp_hdr, tcp_payload)
            return

        # Check for listening server socket on target port
        listen_sock = self.listening_sockets.get(tcp_hdr.dst_port)
        if listen_sock is not None and listen_sock.is_listening:
            if tcp_hdr.has_flag(TCPFlags.SYN):
                # Spawn incoming child connection
                def send_child_segment(h: TCPHeader, p: bytes) -> bool:
                    return self.send_tcp(listen_sock.local_ip, ip_hdr.src_ip, h, p)

                new_conn = TCPConnection(
                    local_ip=listen_sock.local_ip,
                    local_port=listen_sock.local_port,
                    remote_ip=ip_hdr.src_ip,
                    remote_port=tcp_hdr.src_port,
                    send_callback=send_child_segment
                )
                new_conn.state = TCPState.LISTEN
                new_conn.handle_segment(tcp_hdr, tcp_payload)
                self.active_connections[conn_key] = new_conn
                listen_sock.backlog_queue.append(new_conn)

    def step(self) -> None:
        """Process pending ingress packets and drive TCP timers across all attached nodes."""
        if self.device.bus is not None:
            for dev in list(self.device.bus.devices.values()):
                stack = getattr(dev, "stack", None)
                if stack is not None:
                    stack._step_local()
        else:
            self._step_local()

    def _step_local(self) -> None:
        # 1. Drain Layer 2 receive frames
        while True:
            frame = self.device.poll_frame()
            if frame is None:
                break

            if frame.ethertype == int(EthernetType.ARP):
                self.arp.handle_arp_frame(frame.payload)
            elif frame.ethertype == int(EthernetType.IPV4):
                self.ip.handle_ip_packet(frame.payload)

        # 2. Advance TCP connection timers (RTO, TIME_WAIT)
        closed_keys = []
        for key, conn in list(self.active_connections.items()):
            conn.tick()
            if conn.state == TCPState.CLOSED:
                closed_keys.append(key)

        for k in closed_keys:
            if k in self.active_connections:
                del self.active_connections[k]
            if conn.state == TCPState.CLOSED:
                closed_keys.append(key)

        for k in closed_keys:
            del self.active_connections[k]
