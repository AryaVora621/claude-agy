#!/usr/bin/env python3
"""
HydraNet: Terminal Packet Sniffer and Protocol Dissector.
Features:
1. Wireshark-style summary packet capture log with protocol coloring.
2. Layer 2-4 Deep Packet Inspection tree (Ethernet II, ARP, IPv4, TCP).
3. ASCII Ladder / Sequence Diagram visualizer showing TCP handshakes and data flows.
"""

import sys
import os
import time
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetFrame,
    EthernetType,
    ARPHeader,
    ARPOpcode,
    IPv4Header,
    IPProtocol,
    TCPHeader,
    TCPFlags,
    compute_checksum
)
from hydra.device import VirtualEthernetBus, VirtualNetDevice
from hydra.socket import NetworkStack, TCPSocket


# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[37m"
BG_BLUE = "\033[44m"


class PacketDissector:
    """Dissects raw Ethernet frame bytes into structured protocol representations."""

    @staticmethod
    def dissect(raw_frame: bytes) -> dict:
        info = {
            "len": len(raw_frame),
            "eth": None,
            "arp": None,
            "ip": None,
            "tcp": None,
            "payload": b"",
            "proto_str": "ETH",
            "summary": ""
        }

        try:
            eth_frame, eth_payload = EthernetFrame.unpack(raw_frame)
            info["eth"] = eth_frame
        except ValueError:
            info["summary"] = "Corrupt Ethernet Frame"
            return info

        if eth_frame.ethertype == int(EthernetType.ARP):
            info["proto_str"] = "ARP"
            try:
                arp = ARPHeader.unpack(eth_payload)
                info["arp"] = arp
                if arp.opcode == ARPOpcode.REQUEST:
                    info["summary"] = f"Who has {arp.target_ip}? Tell {arp.sender_ip}"
                else:
                    info["summary"] = f"{arp.sender_ip} is at {arp.sender_mac}"
            except ValueError:
                info["summary"] = "Malformed ARP Packet"
            return info

        elif eth_frame.ethertype == int(EthernetType.IPV4):
            try:
                ip_hdr, ip_payload = IPv4Header.unpack(eth_payload)
                info["ip"] = ip_hdr
            except ValueError:
                info["summary"] = "Malformed IPv4 Packet"
                return info

            if ip_hdr.protocol == IPProtocol.TCP:
                info["proto_str"] = "TCP"
                try:
                    tcp_hdr, tcp_payload = TCPHeader.unpack(ip_payload)
                    info["tcp"] = tcp_hdr
                    info["payload"] = tcp_payload

                    flags_list = []
                    if tcp_hdr.has_flag(TCPFlags.SYN):
                        flags_list.append("SYN")
                    if tcp_hdr.has_flag(TCPFlags.ACK):
                        flags_list.append("ACK")
                    if tcp_hdr.has_flag(TCPFlags.FIN):
                        flags_list.append("FIN")
                    if tcp_hdr.has_flag(TCPFlags.RST):
                        flags_list.append("RST")
                    if tcp_hdr.has_flag(TCPFlags.PSH):
                        flags_list.append("PSH")
                    flag_str = "+".join(flags_list) if flags_list else "NONE"

                    payload_len = len(tcp_payload)
                    info["summary"] = (
                        f"{tcp_hdr.src_port} -> {tcp_hdr.dst_port} [{flag_str}] "
                        f"Seq={tcp_hdr.seq_num} Ack={tcp_hdr.ack_num} Win={tcp_hdr.window_size} Len={payload_len}"
                    )
                except ValueError:
                    info["summary"] = "Malformed TCP Segment"
            else:
                info["proto_str"] = f"IP({ip_hdr.protocol.name})"
                info["summary"] = f"{ip_hdr.src_ip} -> {ip_hdr.dst_ip} Protocol={ip_hdr.protocol.name} Len={len(ip_payload)}"

        return info


def print_packet_table(dissected_packets: List[Tuple[float, dict]]) -> None:
    print(f"\n{BOLD}{BG_BLUE}{WHITE}  NO. |  TIME (s) |      SOURCE IP / MAC      |   DESTINATION IP / MAC    | PROTO | LEN  | INFO {RESET}")
    print(f"{DIM}{'-' * 105}{RESET}")

    t_start = dissected_packets[0][0] if dissected_packets else 0.0

    for idx, (t, p) in enumerate(dissected_packets, 1):
        rel_time = t - t_start
        proto = p["proto_str"]

        if proto == "ARP":
            p_color = YELLOW
            src = str(p["arp"].sender_ip) if p["arp"] else str(p["eth"].src_mac)
            dst = str(p["arp"].target_ip) if p["arp"] else str(p["eth"].dst_mac)
        elif proto == "TCP":
            p_color = GREEN
            src = str(p["ip"].src_ip) if p["ip"] else str(p["eth"].src_mac)
            dst = str(p["ip"].dst_ip) if p["ip"] else str(p["eth"].dst_mac)
        else:
            p_color = CYAN
            src = str(p["eth"].src_mac)
            dst = str(p["eth"].dst_mac)

        print(
            f" {idx:4d} | {rel_time:9.4f} | {src:25s} | {dst:25s} | "
            f"{p_color}{proto:5s}{RESET} | {p['len']:4d} | {p['summary']}"
        )


def print_detailed_inspection(pkt: dict) -> None:
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}                 FRAME DETAILED PACKET DISSECTION                       {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")

    # Layer 2
    if pkt["eth"]:
        eth = pkt["eth"]
        print(f"\n{BOLD}[+] Layer 2: Ethernet II{RESET}")
        print(f"    |-- Destination MAC : {eth.dst_mac}")
        print(f"    |-- Source MAC      : {eth.src_mac}")
        print(f"    \\-- EtherType       : 0x{eth.ethertype:04x} ({EthernetType(eth.ethertype).name if eth.ethertype in EthernetType._value2member_map_ else 'Unknown'})")

    # ARP
    if pkt["arp"]:
        arp = pkt["arp"]
        print(f"\n{BOLD}[+] Layer 2.5: Address Resolution Protocol (ARP){RESET}")
        print(f"    |-- Opcode          : {arp.opcode.name} ({int(arp.opcode)})")
        print(f"    |-- Sender MAC      : {arp.sender_mac}")
        print(f"    |-- Sender IP       : {arp.sender_ip}")
        print(f"    |-- Target MAC      : {arp.target_mac}")
        print(f"    \\-- Target IP       : {arp.target_ip}")

    # Layer 3 IPv4
    if pkt["ip"]:
        ip = pkt["ip"]
        print(f"\n{BOLD}[+] Layer 3: Internet Protocol Version 4 (IPv4){RESET}")
        print(f"    |-- Header Length   : {ip.ihl * 4} bytes (IHL={ip.ihl})")
        print(f"    |-- Total Length    : {ip.total_length} bytes")
        print(f"    |-- Identification  : 0x{ip.identification:04x} ({ip.identification})")
        df = bool(ip.flags & 2)
        mf = bool(ip.flags & 1)
        print(f"    |-- Flags           : [DF={df}, MF={mf}]")
        print(f"    |-- Fragment Offset : {ip.fragment_offset * 8} bytes")
        print(f"    |-- Time To Live    : {ip.ttl}")
        print(f"    |-- Protocol        : {ip.protocol.name} ({int(ip.protocol)})")
        print(f"    |-- Header Checksum : 0x{ip.checksum:04x}")
        print(f"    |-- Source Address  : {ip.src_ip}")
        print(f"    \\-- Dest Address    : {ip.dst_ip}")

    # Layer 4 TCP
    if pkt["tcp"]:
        tcp = pkt["tcp"]
        print(f"\n{BOLD}[+] Layer 4: Transmission Control Protocol (TCP){RESET}")
        print(f"    |-- Source Port     : {tcp.src_port}")
        print(f"    |-- Destination Port: {tcp.dst_port}")
        print(f"    |-- Sequence Number : {tcp.seq_num}")
        print(f"    |-- Acknowledgment  : {tcp.ack_num}")
        print(f"    |-- Header Length   : {tcp.data_offset * 4} bytes")
        flags = []
        for f in (TCPFlags.SYN, TCPFlags.ACK, TCPFlags.FIN, TCPFlags.RST, TCPFlags.PSH, TCPFlags.URG):
            if tcp.has_flag(f):
                flags.append(f.name)
        print(f"    |-- Flags           : 0x{tcp.flags:02x} ({', '.join(flags)})")
        print(f"    |-- Window Size     : {tcp.window_size} bytes")
        print(f"    |-- Checksum        : 0x{tcp.checksum:04x}")
        print(f"    \\-- Urgent Pointer  : {tcp.urgent_ptr}")

    # Payload
    if pkt["payload"]:
        payload = pkt["payload"]
        preview = payload[:64].decode("utf-8", errors="replace").replace("\r", "\\r").replace("\n", "\\n")
        print(f"\n{BOLD}[+] Payload ({len(payload)} bytes):{RESET}")
        print(f"    {preview}{'...' if len(payload) > 64 else ''}")


def print_ladder_diagram(dissected_packets: List[Tuple[float, dict]]) -> None:
    print(f"\n{BOLD}{MAGENTA}========================================================================{RESET}")
    print(f"{BOLD}{MAGENTA}              TCP INTERACTIVE LADDER / SEQUENCE DIAGRAM                 {RESET}")
    print(f"{BOLD}{MAGENTA}========================================================================{RESET}\n")

    print(f"  {'Client (192.168.1.20)':^25s}                 {'Server (192.168.1.10)':^25s}")
    print(f"  {'|':^25s}                 {'|':^25s}")

    for t, pkt in dissected_packets:
        if pkt["proto_str"] == "ARP":
            arp = pkt["arp"]
            if arp and arp.opcode == ARPOpcode.REQUEST:
                label = f"ARP Who has {arp.target_ip}?"
                print(f"  {'|':^25s}  ---[{label}]--->   {'|':^25s}")
            elif arp and arp.opcode == ARPOpcode.REPLY:
                label = f"ARP {arp.sender_ip} at {arp.sender_mac}"
                print(f"  {'|':^25s}   <---[{label}]---  {'|':^25s}")
        elif pkt["proto_str"] == "TCP":
            tcp = pkt["tcp"]
            ip = pkt["ip"]
            if not tcp or not ip:
                continue

            flags = []
            if tcp.has_flag(TCPFlags.SYN):
                flags.append("SYN")
            if tcp.has_flag(TCPFlags.ACK):
                flags.append("ACK")
            if tcp.has_flag(TCPFlags.FIN):
                flags.append("FIN")
            if tcp.has_flag(TCPFlags.PSH):
                flags.append("PSH")
            f_str = "+".join(flags)

            p_len = len(pkt["payload"])
            data_str = f" [{p_len}B]" if p_len > 0 else ""
            desc = f"{f_str} seq={tcp.seq_num} ack={tcp.ack_num}{data_str}"

            if str(ip.src_ip) == "192.168.1.20":
                # Client to Server ->
                arrow = f"--------[{desc}]-------->"
                print(f"  {'|':^25s}  {arrow:<36s} {'|':^25s}")
            else:
                # Server to Client <-
                arrow = f"<--------[{desc}]--------"
                print(f"  {'|':^25s}  {arrow:>36s} {'|':^25s}")

    print(f"  {'|':^25s}                 {'|':^25s}\n")


def simulate_http_exchange() -> List[Tuple[float, dict]]:
    """Runs a simulated HTTP client/server exchange over HydraNet, capturing all wire frames."""
    captured_frames: List[Tuple[float, bytes]] = []

    class SniffingBus(VirtualEthernetBus):
        def transmit(self, sender: VirtualNetDevice, raw_frame: bytes) -> bool:
            captured_frames.append((time.time(), raw_frame))
            return super().transmit(sender, raw_frame)

    bus = SniffingBus("eth_tap")
    srv = NetworkStack("srv", MACAddress("00:50:56:00:00:01"), IPAddress("192.168.1.10"), IPAddress("255.255.255.0"))
    cli = NetworkStack("cli", MACAddress("00:50:56:00:00:02"), IPAddress("192.168.1.20"), IPAddress("255.255.255.0"))
    bus.attach(srv.device)
    bus.attach(cli.device)

    # Server listen
    srv_sock = TCPSocket(srv)
    srv_sock.bind(("192.168.1.10", 80))
    srv_sock.listen()

    # Client connect
    cli_sock = TCPSocket(cli)
    cli_sock.connect(("192.168.1.10", 80))

    conn_sock, _ = srv_sock.accept()

    # Client HTTP GET
    http_req = b"GET /index.html HTTP/1.1\r\nHost: 192.168.1.10\r\n\r\n"
    cli_sock.send(http_req)

    # Server receives & replies
    req_recv = conn_sock.recv(1024)
    http_resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: 18\r\n\r\n"
        b"Hello from HydraNet"
    )
    conn_sock.send(http_resp)

    # Client receives response
    resp_recv = cli_sock.recv(1024)

    # Teardown
    cli_sock.close()
    conn_sock.close()
    srv_sock.close()

    # Dissect captured frames
    dissected = [(t, PacketDissector.dissect(raw)) for t, raw in captured_frames]
    return dissected


def main() -> None:
    print(f"{BOLD}{BLUE}========================================================================{RESET}")
    print(f"{BOLD}{BLUE}       HYDRANET: TERMINAL PACKET SNIFFER & PROTOCOL DISSECTOR           {RESET}")
    print(f"{BOLD}{BLUE}========================================================================{RESET}")

    dissected = simulate_http_exchange()

    # 1. Wireshark-Style Packet Summary Table
    print_packet_table(dissected)

    # 2. Sequence / Ladder Flow
    print_ladder_diagram(dissected)

    # 3. Deep Packet Inspection on interesting packets (e.g. SYN, HTTP Request, HTTP Response)
    syn_pkt = next(p for t, p in dissected if p["tcp"] and p["tcp"].has_flag(TCPFlags.SYN) and not p["tcp"].has_flag(TCPFlags.ACK))
    http_req_pkt = next(p for t, p in dissected if p["payload"] and b"GET" in p["payload"])
    http_resp_pkt = next(p for t, p in dissected if p["payload"] and b"HTTP/1.1 200" in p["payload"])

    print(f"\n{BOLD}{YELLOW}>>> Inspecting Sample Packet: TCP SYN Handshake Initiator{RESET}")
    print_detailed_inspection(syn_pkt)

    print(f"\n{BOLD}{YELLOW}>>> Inspecting Sample Packet: HTTP Client Request Segment{RESET}")
    print_detailed_inspection(http_req_pkt)

    print(f"\n{BOLD}{YELLOW}>>> Inspecting Sample Packet: HTTP Server Response Segment{RESET}")
    print_detailed_inspection(http_resp_pkt)


if __name__ == "__main__":
    main()
