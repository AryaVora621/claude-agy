# HydraNet: Zero-Dependency User-Space TCP/IP Protocol Stack

HydraNet is an RFC-compliant, zero-dependency, user-space TCP/IP protocol stack implemented from first principles in the pure Python standard library.

It models the full 4-layer networking stack: Layer 2 Ethernet II framing with simulated virtual carrier channels and link impairment, Layer 2.5 Address Resolution Protocol (ARP), Layer 3 IPv4 packet routing with Longest Prefix Match (LPM) and dynamic fragmentation/reassembly, Layer 4 Transmission Control Protocol (TCP) with a complete 11-state RFC 793 Finite State Machine, sliding window flow control, RFC 5681 TCP Reno congestion control, RFC 6298 RTT estimation, and a standard POSIX-compatible Berkeley Socket API.

---

## Architectural Overview

```
+--------------------------------------------------------------------------------+
|                         APPLICATION LAYER                                      |
|    HTTP Client/Server, Terminal Echo, Diagnostic Packet Sniffers               |
+--------------------------------------------------------------------------------+
                                       |
                       POSIX Berkeley Socket Interface
                 bind(), listen(), accept(), connect(), send(), recv()
                                       |
+--------------------------------------------------------------------------------+
|                   LAYER 4: TRANSMISSION CONTROL PROTOCOL (TCP)                 |
|  - RFC 793 11-State FSM (CLOSED, LISTEN, SYN_SENT, ESTABLISHED, FIN_WAIT, ...) |
|  - Sliding Window Buffers (SND.UNA, SND.NXT, RCV.NXT, Out-of-Order Reassembly)  |
|  - RFC 5681 TCP Reno Congestion Control (Slow Start, Congestion Avoidance,     |
|    Fast Retransmit on 3 DupACKs, Fast Recovery window inflation)               |
|  - RFC 6298 Jacobson/Karels Smoothed RTT & Karn's Algorithm RTO Backoff       |
+--------------------------------------------------------------------------------+
                                       |
+--------------------------------------------------------------------------------+
|                      LAYER 3: INTERNET PROTOCOL (IPv4)                         |
|  - RFC 791 IPv4 Header Serialization & Validation                              |
|  - Longest Prefix Match (LPM) CIDR Subnet Routing Table                        |
|  - Dynamic MTU Fragmentation & Hole-Filling Datagram Reassembly Buffer         |
|  - RFC 1071 16-bit One's Complement Header & Pseudo-Header Checksumming        |
+--------------------------------------------------------------------------------+
                      |                                       |
          +-----------------------+               +-----------------------+
          |  LAYER 2.5: ARP       |               |  LAYER 2: LINK LAYER  |
          |  - RFC 826 ARP Table  |               |  - Ethernet II Frame  |
          |  - Hardware Cache     |               |    (14-byte headers)  |
          |  - Request/Reply FSM  |               |  - MAC Unicast/Bcast  |
          +-----------------------+               +-----------------------+
                      \                                       /
                       \                                     /
+--------------------------------------------------------------------------------+
|                   VIRTUAL PHYSICAL LAYER (VirtualEthernetBus)                  |
|    Multi-node broadcast bus with configurable packet drop, corruption, and     |
|    duplication for chaos testing and link degradation modeling                 |
+--------------------------------------------------------------------------------+
```

---

## Key Protocol Components

### 1. Layer 2: Virtual Ethernet Bus & Device
- **Ethernet II Framing**: 14-byte standard Ethernet headers (6-byte destination MAC, 6-byte source MAC, 2-byte EtherType).
- **Virtual Carrier Bus**: Connects multiple virtual network interface controllers (vNICs), supporting unicast MAC routing and broadcast delivery (`ff:ff:ff:ff:ff:ff`).
- **Chaos Injection**: Configurable physical channel impairments:
  - Random packet drop rate ($P_{drop}$).
  - Byte-level bit-flipping packet corruption ($P_{corrupt}$).
  - Link packet duplication ($P_{duplicate}$).

### 2. Layer 2.5: Address Resolution Protocol (RFC 826)
- **Dynamic ARP Cache**: Dynamic IP-to-MAC mapping with configurable entry expiration timeouts.
- **Hardware Resolution Exchange**: Generates standard ARP Request broadcasts and unicast ARP Replies.
- **Static ARP Table Entries**: Support for permanent pre-configured gateway bindings.

### 3. Layer 3: IPv4 Routing & Fragmentation (RFC 791)
- **Longest Prefix Match (LPM)**: Hierarchical routing table supporting arbitrary CIDR masks (/8, /16, /24, /30, /32, and default 0.0.0.0/0). Sorts routes by prefix length descending, then metric.
- **Dynamic MTU Fragmentation**: Splits IP payloads exceeding interface MTU (1500 bytes) into 8-byte aligned fragment slices with DF (Don't Fragment) and MF (More Fragments) flags.
- **Datagram Reassembly Buffer**: Tracks incoming out-of-order fragment slices, detects gaps, and reassembles complete datagrams upon arrival of final slice.

### 4. Layer 4: Transmission Control Protocol (RFC 793)
- **Complete 11-State Finite State Machine**:
  - `CLOSED`, `LISTEN`, `SYN_SENT`, `SYN_RCVD`, `ESTABLISHED`.
  - `FIN_WAIT_1`, `FIN_WAIT_2`, `CLOSE_WAIT`, `CLOSING`, `LAST_ACK`, `TIME_WAIT`.
- **Sliding Window Flow Control**:
  - `SendBuffer`: Slices outgoing byte streams into Maximum Segment Size (MSS) packets, tracks unacknowledged sequence range (`SND.UNA` to `SND.NXT`), and supports selective retransmission.
  - `ReceiveBuffer`: Tracks contiguous received bytes (`RCV.NXT`), buffers out-of-order segments, stitches sequence gaps automatically, and computes dynamic advertised receive window.
- **RFC 5681 TCP Reno Congestion Control**:
  - **Slow Start**: Exponential congestion window growth ($CWND \leftarrow CWND + MSS$) on every ACK while $CWND < ssthresh$.
  - **Congestion Avoidance**: Additive increase linear expansion ($CWND \leftarrow CWND + \frac{MSS \times MSS}{CWND}$) when $CWND \ge ssthresh$.
  - **Fast Retransmit**: Detects 3 duplicate ACKs, sets $ssthresh \leftarrow \max(2 \times MSS, \frac{CWND}{2})$, and retransmits missing segment immediately without waiting for RTO.
  - **Fast Recovery**: Inflates $CWND$ for each additional duplicate ACK, then resets to $ssthresh$ upon receiving recovery ACK.
  - **Timeout Window Collapse**: On RTO expiry, collapses $CWND \leftarrow 1 \times MSS$ and applies Karn's exponential backoff ($RTO \leftarrow \min(60.0, 2 \times RTO)$).
- **RFC 6298 Round-Trip Time Estimation**:
  - Computes smoothed round-trip time ($SRTT$) and variation ($RTTVAR$):
    $$RTTVAR \leftarrow (1 - \beta) \times RTTVAR + \beta \times |SRTT - R'|$$
    $$SRTT \leftarrow (1 - \alpha) \times SRTT + \alpha \times R'$$
  - Derives Retransmission Timeout: $RTO \leftarrow \max(RTO_{min}, \min(RTO_{max}, SRTT + 4 \times RTTVAR))$.

### 5. POSIX Berkeley Socket API
- Standard socket operations:
  ```python
  sock = TCPSocket(stack)
  sock.bind(("192.168.1.10", 8080))
  sock.listen()
  conn, client_addr = sock.accept()
  conn.send(b"HTTP/1.1 200 OK\r\n\r\nHello!")
  data = conn.recv(4096)
  conn.close()
  ```

---

## Directory Layout

```
projects/07-hydranet/
├── hydra/
│   ├── __init__.py           # Package exports
│   ├── types.py              # MACAddress, IPAddress, Headers, RFC 1071 Checksums
│   ├── device.py             # VirtualEthernetBus, VirtualNetDevice, Impairment
│   ├── arp.py                # RFC 826 ARP Table & Hardware Resolution
│   ├── ip.py                 # RFC 791 IPv4 Routing, LPM, Dynamic Fragmentation
│   ├── flow.py               # Send/Receive Sliding Window Buffers & Reassembly
│   ├── congestion.py         # RFC 5681 TCP Reno & RFC 6298 RTT Estimator
│   ├── tcp.py                # RFC 793 11-State TCP FSM & Connection TCB
│   └── socket.py             # POSIX Berkeley Socket API & NetworkStack Host Node
├── examples/
│   ├── packet_sniffer.py     # Terminal Wireshark dissector & ASCII ladder diagrams
│   └── echo_client_server.py # Client/Server duplex stream & lossy wire recovery demo
├── benchmarks/
│   └── bench_network.py      # Stack throughput, framing, routing, and TCP benchmarks
├── tests/
│   ├── test_types.py         # MAC, IP, Frame pack/unpack, and RFC 1071 checksums
│   ├── test_arp_ip.py        # ARP requests, LPM routing, and IPv4 fragmentation
│   ├── test_flow_congestion.py # Sliding window, Reno Reno recovery, RTT estimator
│   ├── test_tcp_fsm.py       # 3-way handshake and 4-way FIN teardown FSM
│   └── test_socket_e2e.py    # End-to-end socket echo and multi-segment stream
└── README.md
```

---

## Performance Benchmarks

Evaluated on an Apple Silicon host running Python 3.13 standard library:

| Benchmark Subsystem | Metric | Measured Performance |
|---|---|---|
| **RFC 1071 Checksum** | One's complement 16-bit sum | **21.7 MB/sec** |
| **Ethernet II Framing** | Pack and unpack operations | **1,252,893 frames/sec** (169.7 MB/s) |
| **IPv4 LPM Routing Table** | Longest Prefix Match (250 CIDR subnets) | **82,286 lookups/sec** |
| **IPv4 Fragmentation & Reassembly** | 12 KB datagrams segmented over MTU=1500 | **14,441 datagrams/sec** (165.3 MB/s) |
| **TCP Socket Stream Goodput** | End-to-end client/server stream over virtual bus | **15,518 KB/sec** (~15.5 MB/s) |

To run the complete benchmark suite:
```bash
python3 projects/07-hydranet/benchmarks/bench_network.py
```

---

## Interactive Wireshark Packet Sniffer

HydraNet includes a terminal packet sniffer and sequence ladder visualizer in `examples/packet_sniffer.py`. It inspects simulated wire traffic and outputs:

1. **Summary Table**: Wireshark-style colorized columns (Packet No., Relative Time, Source, Destination, Protocol, Length, Info).
2. **Interactive Ladder Sequence Flow**: Visual ASCII ladder diagram tracing ARP lookups, TCP 3-way handshakes, data transmissions, and teardowns.
3. **Deep Packet Inspection**: Hierarchical tree breakdown of Ethernet II, ARP, IPv4, and TCP headers with bitmask flags and checksum validations.

To run:
```bash
python3 projects/07-hydranet/examples/packet_sniffer.py
```

Sample output excerpt:
```text
  NO. |  TIME (s) |      SOURCE IP / MAC      |   DESTINATION IP / MAC    | PROTO | LEN  | INFO
---------------------------------------------------------------------------------------------------------
    1 |    0.0000 | 192.168.1.20              | 192.168.1.10              | ARP   |   42 | Who has 192.168.1.10? Tell 192.168.1.20
    2 |    0.0000 | 192.168.1.10              | 192.168.1.20              | ARP   |   42 | 192.168.1.10 is at 00:50:56:00:00:01
    3 |    0.0000 | 192.168.1.20              | 192.168.1.10              | TCP   |   54 | 52331 -> 80 [SYN] Seq=629152 Ack=0 Win=65535 Len=0
    4 |    0.0001 | 192.168.1.10              | 192.168.1.20              | TCP   |   54 | 80 -> 52331 [SYN+ACK] Seq=910679 Ack=629153 Win=65535 Len=0
    5 |    0.0001 | 192.168.1.20              | 192.168.1.10              | TCP   |   54 | 52331 -> 80 [ACK] Seq=629153 Ack=910680 Win=65535 Len=0

              TCP INTERACTIVE LADDER / SEQUENCE DIAGRAM
    Client (192.168.1.20)                     Server (192.168.1.10)
              |                                         |
              |              ---[ARP Who has 192.168.1.10?]--->               |
              |               <---[ARP 192.168.1.10 at 00:50:56:00:00:01]---  |
              |              --------[SYN seq=629152 ack=0]-------->          |
              |              <--------[SYN+ACK seq=910679 ack=629153]-------- |
              |              --------[ACK seq=629153 ack=910680]-------->     |
              |              --------[ACK+PSH seq=629153 ack=910680 [48B]]--->|
              |              <--------[ACK seq=910680 ack=629201]------------ |
              |              <--------[ACK+PSH seq=910680 ack=629201 [84B]]-- |
              |              --------[ACK seq=629201 ack=910764]-------->     |
              |              --------[ACK+FIN seq=629201 ack=910764]--------> |
              |              <--------[ACK seq=910764 ack=629202]------------ |
              |              <--------[ACK+FIN seq=910764 ack=629202]-------- |
              |              --------[ACK seq=629202 ack=910765]-------->     |
```

---

## Lossy Wire Recovery & Reno Telemetry

The `examples/echo_client_server.py` demonstration runs over an impaired Ethernet bus with an 8% physical packet drop rate:
- Demonstrates TCP Reno Fast Retransmit and RTO backoff handling lost packets.
- Telemetry outputs live congestion window dynamics:
  ```text
  [Reno Telemetry] CWND: 3650.0 B ( 2.5 MSS) | ssthresh: 2920.0 B | DupACKs: 2 | FastRecovery: False | Wire Drops: 2
  [Reno Telemetry] CWND: 1460.0 B ( 1.0 MSS) | ssthresh: 2920.0 B | DupACKs: 0 | FastRecovery: False | Wire Drops: 2
  [✓] Data Transfer Completed Across Lossy Wire!
      -> Bytes Transferred : 16,000 / 16,000
      -> Frames Dropped    : 4 frames
      -> Integrity Match   : True
  ```

---

## Test Suite

The unit and integration test suite verifies:
1. Ethernet II and ARP byte-level serialization and RFC 1071 checksum calculations.
2. ARP cache timeouts, expiration leasing, and resolution cycles.
3. IPv4 Longest Prefix Match (LPM) routing and multi-fragment reassembly.
4. Out-of-order receive buffer gap stitching and sliding send window ACK advancing.
5. TCP Reno Slow Start, Congestion Avoidance, 3-dup-ACK Fast Retransmit, and RTO collapse.
6. RFC 793 11-state machine transitions: 3-way handshake and 4-way teardown.
7. End-to-end POSIX Berkeley Socket duplex exchange and multi-segment streaming.

To execute tests:
```bash
PYTHONPATH="projects/07-hydranet" python3 -m unittest discover -s projects/07-hydranet/tests -v
```

Output:
```text
Ran 21 tests in 0.134s
OK
```
