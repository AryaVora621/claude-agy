"""Static Timing Analysis (STA) Engine with NLDM Delay & Slack Propagation.

Features:
1. Directed timing graph construction from mapped gate-level netlists.
2. Forward Arrival Time (AT) and slew propagation using NLDM 2D lookup tables.
3. Backward Required Arrival Time (RAT) propagation from timing endpoints.
4. Pin-level slack calculation: Slack = RAT - AT.
5. Setup timing closure analysis: Worst Negative Slack (WNS) and Total Negative Slack (TNS).
6. Physical critical path extraction and maximum frequency (F_max) reporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .liberty import StandardCell, TimingArc
from .netlist import CellInstance, Net, Netlist


@dataclass
class TimingPin:
    """Timing node representing a cell pin or primary I/O port."""
    name: str                   # "PORT:A", "U1:A", "U1:Y", etc.
    is_primary_input: bool = False
    is_primary_output: bool = False
    cell_inst: Optional[str] = None
    pin_name: Optional[str] = None
    arrival_time: float = 0.0   # ps
    required_time: float = 0.0  # ps
    slack: float = 0.0          # ps (RAT - AT)
    slew: float = 20.0          # ps
    capacitance: float = 0.0    # fF
    fanin_pins: List[str] = field(default_factory=list)
    fanout_pins: List[str] = field(default_factory=list)


@dataclass
class TimingArcEdge:
    """Directed edge in the timing graph representing cell delay or net traversal."""
    src: str                    # Source pin name
    dst: str                    # Destination pin name
    delay: float = 0.0          # Propagation delay in ps
    output_slew: float = 20.0   # Transition time in ps
    is_cell_arc: bool = False
    timing_arc: Optional[TimingArc] = None


@dataclass
class PathSegment:
    """Single hop along a timing path."""
    pin: str
    cell_name: str
    cell_type: str
    edge_delay: float
    arrival_time: float
    required_time: float
    slack: float
    slew: float


@dataclass
class TimingReport:
    """Complete static timing analysis report."""
    clock_period: float         # ps
    worst_negative_slack: float # ps
    total_negative_slack: float # ps
    critical_path: List[PathSegment]
    max_delay: float            # ps
    max_frequency_ghz: float    # GHz
    is_timing_met: bool
    num_endpoints: int
    violating_endpoints: int


class StaticTimingAnalyzer:
    """Static Timing Analysis engine for gate-level netlists."""

    def __init__(self, netlist: Netlist) -> None:
        self.netlist = netlist
        self.pins: Dict[str, TimingPin] = {}
        self.arcs: List[TimingArcEdge] = []
        self._build_timing_graph()

    def _pin_id(self, inst_or_port: str, pin: Optional[str] = None) -> str:
        if pin is None:
            return f"PORT:{inst_or_port}"
        return f"{inst_or_port}:{pin}"

    def _build_timing_graph(self) -> None:
        """Construct timing graph vertices and edges from netlist topology."""
        self.pins.clear()
        self.arcs.clear()

        # 1. Primary Inputs and Primary Outputs
        for pi_name in self.netlist.inputs:
            pid = self._pin_id(pi_name)
            self.pins[pid] = TimingPin(
                name=pid,
                is_primary_input=True,
                pin_name=pi_name,
                slew=20.0,
            )

        for po_name in self.netlist.outputs:
            pid = self._pin_id(po_name)
            self.pins[pid] = TimingPin(
                name=pid,
                is_primary_output=True,
                pin_name=po_name,
            )

        # 2. Cell Instance Pins
        for inst_name, inst in self.netlist.cells.items():
            cell = inst.cell_type
            for pin_name, pin_def in cell.pins.items():
                pid = self._pin_id(inst_name, pin_name)
                self.pins[pid] = TimingPin(
                    name=pid,
                    cell_inst=inst_name,
                    pin_name=pin_name,
                    capacitance=pin_def.capacitance,
                )

        # 3. Cell Internal Timing Arcs
        for inst_name, inst in self.netlist.cells.items():
            cell = inst.cell_type
            for arc in cell.timing_arcs:
                src_id = self._pin_id(inst_name, arc.from_pin)
                dst_id = self._pin_id(inst_name, arc.to_pin)
                if src_id in self.pins and dst_id in self.pins:
                    self.pins[src_id].fanout_pins.append(dst_id)
                    self.pins[dst_id].fanin_pins.append(src_id)
                    self.arcs.append(TimingArcEdge(
                        src=src_id,
                        dst=dst_id,
                        is_cell_arc=True,
                        timing_arc=arc,
                    ))

        # 4. Net Interconnect Edges
        for net_name, net in self.netlist.nets.items():
            driver_pin_id: Optional[str] = None
            if net.is_primary_input:
                driver_pin_id = self._pin_id(net_name)
            elif net.driver:
                driver_pin_id = self._pin_id(net.driver[0], net.driver[1])

            if driver_pin_id and driver_pin_id in self.pins:
                # Driver connects to all loads
                for load_inst, load_pin in net.loads:
                    sink_id = self._pin_id(load_inst, load_pin)
                    if sink_id in self.pins:
                        self.pins[driver_pin_id].fanout_pins.append(sink_id)
                        self.pins[sink_id].fanin_pins.append(driver_pin_id)
                        self.arcs.append(TimingArcEdge(
                            src=driver_pin_id,
                            dst=sink_id,
                            delay=net.wire_cap * 0.5, # 0.5 ps per fF wire delay
                            is_cell_arc=False,
                        ))

                if net.is_primary_output:
                    po_id = self._pin_id(net_name)
                    if po_id in self.pins and po_id != driver_pin_id:
                        self.pins[driver_pin_id].fanout_pins.append(po_id)
                        self.pins[po_id].fanin_pins.append(driver_pin_id)
                        self.arcs.append(TimingArcEdge(
                            src=driver_pin_id,
                            dst=po_id,
                            delay=0.5,
                            is_cell_arc=False,
                        ))

    def run_sta(
        self,
        clock_period: float = 1000.0, # ps (1.0 ns = 1.0 GHz default)
        input_slew: float = 20.0,     # ps
        output_load_cap: float = 5.0, # fF
    ) -> TimingReport:
        """Execute complete Static Timing Analysis pass."""
        self.netlist.compute_net_capacitances()

        # 1. Topological Sort of Pins
        order = self._topological_sort()

        # 2. Forward Propagation: Arrival Time (AT) and Slew
        # Initialize primary inputs
        for pi_name in self.netlist.inputs:
            pid = self._pin_id(pi_name)
            if pid in self.pins:
                self.pins[pid].arrival_time = 0.0
                self.pins[pid].slew = input_slew

        # Propagate forward
        for pin_id in order:
            curr_pin = self.pins[pin_id]

            # Find all outgoing edges from curr_pin
            for arc in self.arcs:
                if arc.src != pin_id:
                    continue

                dst_pin = self.pins[arc.dst]
                edge_delay = arc.delay
                out_slew = curr_pin.slew

                if arc.is_cell_arc and arc.timing_arc:
                    # Calculate driven net capacitance
                    inst = self.netlist.cells.get(curr_pin.cell_inst or "")
                    load_cap = output_load_cap
                    if inst and arc.timing_arc.to_pin in inst.pin_connections:
                        net_name = inst.pin_connections[arc.timing_arc.to_pin]
                        net = self.netlist.nets.get(net_name)
                        if net:
                            load_cap = max(0.5, net.total_cap)

                    # Lookup cell delay and output transition time
                    edge_delay = arc.timing_arc.lookup_delay(curr_pin.slew, load_cap)
                    out_slew = arc.timing_arc.lookup_slew(curr_pin.slew, load_cap)
                    arc.delay = edge_delay
                    arc.output_slew = out_slew

                new_at = curr_pin.arrival_time + edge_delay
                if new_at > dst_pin.arrival_time:
                    dst_pin.arrival_time = new_at
                    dst_pin.slew = out_slew

        # 3. Backward Propagation: Required Arrival Time (RAT)
        # Initialize endpoints (primary outputs)
        for po_name in self.netlist.outputs:
            poid = self._pin_id(po_name)
            if poid in self.pins:
                self.pins[poid].required_time = clock_period

        # Reverse topological traversal
        for pin_id in reversed(order):
            curr_pin = self.pins[pin_id]
            if curr_pin.is_primary_output:
                continue

            min_rat = float("inf")
            has_succ = False

            for arc in self.arcs:
                if arc.src != pin_id:
                    continue
                dst_pin = self.pins[arc.dst]
                has_succ = True
                cand_rat = dst_pin.required_time - arc.delay
                if cand_rat < min_rat:
                    min_rat = cand_rat

            if has_succ:
                curr_pin.required_time = min_rat
            else:
                curr_pin.required_time = clock_period

        # 4. Slack Calculation: Slack = RAT - AT
        endpoints: List[str] = [self._pin_id(name) for name in self.netlist.outputs if self._pin_id(name) in self.pins]

        for pin in self.pins.values():
            pin.slack = round(pin.required_time - pin.arrival_time, 3)

        worst_slack = min((self.pins[ep_id].slack for ep_id in endpoints), default=0.0)
        tns = sum(self.pins[ep_id].slack for ep_id in endpoints if self.pins[ep_id].slack < 0.0)
        violating_count = sum(1 for ep_id in endpoints if self.pins[ep_id].slack < 0.0)

        # 5. Critical Path Extraction
        critical_path = self._extract_critical_path(endpoints)
        max_delay = max((self.pins[ep_id].arrival_time for ep_id in endpoints), default=0.0)
        max_freq = 1000.0 / max_delay if max_delay > 0.0 else 0.0

        return TimingReport(
            clock_period=clock_period,
            worst_negative_slack=round(worst_slack, 3),
            total_negative_slack=round(tns, 3),
            critical_path=critical_path,
            max_delay=round(max_delay, 3),
            max_frequency_ghz=round(max_freq, 3),
            is_timing_met=(worst_slack >= 0.0),
            num_endpoints=len(endpoints),
            violating_endpoints=violating_count,
        )

    def _extract_critical_path(self, endpoints: List[str]) -> List[PathSegment]:
        """Backtrack from worst endpoint to extract the physical critical path."""
        if not endpoints:
            return []

        # Find endpoint with minimum slack
        worst_ep = min(endpoints, key=lambda ep: self.pins[ep].slack)
        path: List[str] = [worst_ep]

        curr = worst_ep
        visited_in_path: Set[str] = {curr}

        while not self.pins[curr].is_primary_input:
            fanins = self.pins[curr].fanin_pins
            if not fanins:
                break

            # Find fanin that minimizes slack (drives the critical delay)
            best_fanin: Optional[str] = None
            min_slack = float("inf")

            for fin in fanins:
                if fin in visited_in_path:
                    continue
                fin_slack = self.pins[fin].slack
                if fin_slack < min_slack:
                    min_slack = fin_slack
                    best_fanin = fin

            if best_fanin is None:
                break

            path.append(best_fanin)
            visited_in_path.add(best_fanin)
            curr = best_fanin

        path.reverse()

        segments: List[PathSegment] = []
        for i, pid in enumerate(path):
            pin = self.pins[pid]
            cell_inst = pin.cell_inst or "TOP"
            cell_type = "PORT"
            if pin.cell_inst:
                inst = self.netlist.cells.get(pin.cell_inst)
                if inst:
                    cell_type = inst.cell_type.name

            edge_delay = 0.0
            if i > 0:
                prev_pid = path[i - 1]
                for arc in self.arcs:
                    if arc.src == prev_pid and arc.dst == pid:
                        edge_delay = arc.delay
                        break

            segments.append(PathSegment(
                pin=pid,
                cell_name=cell_inst,
                cell_type=cell_type,
                edge_delay=round(edge_delay, 2),
                arrival_time=round(pin.arrival_time, 2),
                required_time=round(pin.required_time, 2),
                slack=round(pin.slack, 2),
                slew=round(pin.slew, 2),
            ))

        return segments

    def _topological_sort(self) -> List[str]:
        """Perform topological sort across timing graph vertices."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(pid: str) -> None:
            if pid in visited:
                return
            visited.add(pid)
            for fanout_id in self.pins[pid].fanout_pins:
                dfs(fanout_id)
            order.append(pid)

        for pi_name in self.netlist.inputs:
            pid = self._pin_id(pi_name)
            if pid in self.pins:
                dfs(pid)

        # Catch any disconnected components
        for pid in list(self.pins.keys()):
            if pid not in visited:
                dfs(pid)

        order.reverse()
        return order
