"""PyCircuit - First-Principles SPICE Analog Circuit Simulation Engine.

Implements Modified Nodal Analysis (MNA), companion models for reactive components
(Backward Euler integration for capacitors and inductors), piecewise non-linear
diodes, transistors, op-amps, and Gaussian elimination linear solvers in pure Python.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    if n == 0:
        return []
    if len(b) != n:
        raise ValueError(f"Dimension mismatch: A is {n}x{n}, b has {len(b)} elements")

    # Create deep copies of augmented matrix
    M = [list(row) for row in A]
    x_b = list(b)

    for i in range(n):
        # Partial pivoting: find maximum pivot in column i
        max_row = i
        max_val = abs(M[i][i])
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k

        # Swap rows if necessary
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            x_b[i], x_b[max_row] = x_b[max_row], x_b[i]

        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            # Singular or near-singular: apply tiny regularizer to prevent division by zero
            pivot = 1e-12 if pivot >= 0 else -1e-12
            M[i][i] = pivot

        # Eliminate rows below pivot
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            if abs(factor) > 1e-15:
                for col in range(i, n):
                    M[j][col] -= factor * M[i][col]
                x_b[j] -= factor * x_b[i]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        pivot = M[i][i]
        if abs(pivot) < 1e-14:
            x[i] = 0.0
        else:
            x[i] = (x_b[i] - s) / pivot

    return x


@dataclass
class CircuitComponent:
    """Base class for all physical schematic circuit components."""
    name: str
    n1: int  # Positive / Input node index
    n2: int  # Negative / Output node index
    value: float = 1.0  # Resistance (Ohms), Capacitance (F), Inductance (H), Voltage (V)
    current: float = 0.0  # Calculated branch current (Amperes)
    voltage_drop: float = 0.0  # V(n1) - V(n2)

    # Optional UI schematic positioning
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass
class Resistor(CircuitComponent):
    """Linear passive resistor."""
    pass


@dataclass
class Capacitor(CircuitComponent):
    """Capacitor companion model using Backward Euler numerical integration."""
    prev_v: float = 0.0
    g_eq: float = 0.0
    i_eq: float = 0.0

    def prepare_step(self, dt: float) -> None:
        """Calculate companion conductance and history current."""
        if dt <= 0:
            return
        self.g_eq = self.value / dt
        self.i_eq = self.g_eq * self.prev_v


@dataclass
class Inductor(CircuitComponent):
    """Inductor companion model using Backward Euler numerical integration."""
    prev_i: float = 0.0
    g_eq: float = 0.0
    i_eq: float = 0.0

    def prepare_step(self, dt: float) -> None:
        """Calculate companion conductance and history current."""
        if dt <= 0:
            return
        self.g_eq = dt / max(1e-9, self.value)
        self.i_eq = self.prev_i


@dataclass
class VoltageSourceDC(CircuitComponent):
    """Constant Direct Current (DC) independent voltage source."""
    pass


@dataclass
class VoltageSourceAC(CircuitComponent):
    """Sinusoidal Alternating Current (AC) independent voltage source."""
    frequency: float = 60.0  # Hz
    phase: float = 0.0       # Degrees
    offset: float = 0.0      # DC offset

    def get_voltage(self, t: float) -> float:
        """Instantaneous voltage at time t."""
        rad = 2.0 * math.pi * self.frequency * t + math.radians(self.phase)
        return self.offset + self.value * math.sin(rad)


@dataclass
class VoltageSourceClock(CircuitComponent):
    """Square wave clock generator."""
    frequency: float = 1000.0  # Hz
    duty_cycle: float = 0.5    # 0.0 to 1.0

    def get_voltage(self, t: float) -> float:
        period = 1.0 / max(1.0, self.frequency)
        t_mod = t % period
        return self.value if t_mod < (period * self.duty_cycle) else 0.0


@dataclass
class Diode(CircuitComponent):
    """Semiconductor PN junction diode with forward threshold and reverse blocking."""
    v_drop: float = 0.7        # Forward conduction drop (Volts)
    r_on: float = 1.0          # Forward dynamic resistance (Ohms)
    r_off: float = 1e6         # Reverse leakage resistance (Ohms)
    is_conducting: bool = False

    def get_dynamic_conductance(self, v_diff: float) -> Tuple[float, float]:
        """Return (conductance g, parallel current i_eq) for companion linearization."""
        if v_diff >= self.v_drop:
            self.is_conducting = True
            g = 1.0 / self.r_on
            i_eq = -self.v_drop / self.r_on
            return g, i_eq
        else:
            self.is_conducting = False
            g = 1.0 / self.r_off
            return g, 0.0


@dataclass
class Switch(CircuitComponent):
    """Interactive toggleable switch."""
    closed: bool = True
    r_closed: float = 1e-3
    r_open: float = 1e7

    @property
    def resistance(self) -> float:
        return self.r_closed if self.closed else self.r_open


@dataclass
class OpAmp:
    """Operational Amplifier with differential inputs and open-loop gain."""
    name: str
    n_in_pos: int  # Non-inverting input (+)
    n_in_neg: int  # Inverting input (-)
    n_out: int     # Output node
    v_supply_pos: float = 15.0
    v_supply_neg: float = -15.0
    gain: float = 1e5
    current_out: float = 0.0


class Circuit:
    """SPICE-class Modified Nodal Analysis (MNA) analog circuit simulator."""

    def __init__(self):
        self.components: List[CircuitComponent] = []
        self.opamps: List[OpAmp] = []
        self.num_nodes: int = 1  # Node 0 is permanently Ground (0.0 V)
        self.time: float = 0.0
        self.dt: float = 5e-5    # 50 microseconds default time step (20 kHz)
        self.node_voltages: List[float] = [0.0]
        self.history_nodes: Dict[int, List[float]] = {}
        self.history_time: List[float] = []

    def clear(self) -> None:
        """Reset circuit to pristine state."""
        self.components.clear()
        self.opamps.clear()
        self.num_nodes = 1
        self.time = 0.0
        self.node_voltages = [0.0]
        self.history_nodes.clear()
        self.history_time.clear()

    def add_node(self) -> int:
        """Allocate a new circuit node."""
        idx = self.num_nodes
        self.num_nodes += 1
        self.node_voltages.append(0.0)
        return idx

    def ensure_nodes(self, required_max: int) -> None:
        """Ensure node voltage array holds up to required_max node index."""
        while self.num_nodes <= required_max:
            self.add_node()

    def add_resistor(self, name: str, n1: int, n2: int, resistance: float) -> Resistor:
        self.ensure_nodes(max(n1, n2))
        r = Resistor(name=name, n1=n1, n2=n2, value=max(1e-4, resistance))
        self.components.append(r)
        return r

    def add_capacitor(self, name: str, n1: int, n2: int, capacitance: float) -> Capacitor:
        self.ensure_nodes(max(n1, n2))
        c = Capacitor(name=name, n1=n1, n2=n2, value=max(1e-12, capacitance))
        self.components.append(c)
        return c

    def add_inductor(self, name: str, n1: int, n2: int, inductance: float) -> Inductor:
        self.ensure_nodes(max(n1, n2))
        l = Inductor(name=name, n1=n1, n2=n2, value=max(1e-9, inductance))
        self.components.append(l)
        return l

    def add_dc_source(self, name: str, n1: int, n2: int, voltage: float) -> VoltageSourceDC:
        self.ensure_nodes(max(n1, n2))
        src = VoltageSourceDC(name=name, n1=n1, n2=n2, value=voltage)
        self.components.append(src)
        return src

    def add_ac_source(self, name: str, n1: int, n2: int, amplitude: float, frequency: float, phase: float = 0.0) -> VoltageSourceAC:
        self.ensure_nodes(max(n1, n2))
        src = VoltageSourceAC(name=name, n1=n1, n2=n2, value=amplitude, frequency=frequency, phase=phase)
        self.components.append(src)
        return src

    def add_clock_source(self, name: str, n1: int, n2: int, amplitude: float, frequency: float, duty_cycle: float = 0.5) -> VoltageSourceClock:
        self.ensure_nodes(max(n1, n2))
        src = VoltageSourceClock(name=name, n1=n1, n2=n2, value=amplitude, frequency=frequency, duty_cycle=duty_cycle)
        self.components.append(src)
        return src

    def add_diode(self, name: str, n1: int, n2: int, v_drop: float = 0.7) -> Diode:
        self.ensure_nodes(max(n1, n2))
        d = Diode(name=name, n1=n1, n2=n2, v_drop=v_drop)
        self.components.append(d)
        return d

    def add_switch(self, name: str, n1: int, n2: int, closed: bool = True) -> Switch:
        self.ensure_nodes(max(n1, n2))
        sw = Switch(name=name, n1=n1, n2=n2, closed=closed)
        self.components.append(sw)
        return sw

    def add_opamp(self, name: str, n_pos: int, n_neg: int, n_out: int, v_pos: float = 15.0, v_neg: float = -15.0) -> OpAmp:
        self.ensure_nodes(max(n_pos, n_neg, n_out))
        op = OpAmp(name=name, n_in_pos=n_pos, n_in_neg=n_neg, n_out=n_out, v_supply_pos=v_pos, v_supply_neg=v_neg)
        self.opamps.append(op)
        return op

    def _stamp_conductance(self, G: List[List[float]], n1: int, n2: int, g: float) -> None:
        """Stamp two-terminal conductance g into the G matrix."""
        # Node 0 is ground; equations are indexed for non-zero nodes (1 to num_nodes-1)
        if n1 > 0:
            G[n1 - 1][n1 - 1] += g
        if n2 > 0:
            G[n2 - 1][n2 - 1] += g
        if n1 > 0 and n2 > 0:
            G[n1 - 1][n2 - 1] -= g
            G[n2 - 1][n1 - 1] -= g

    def _stamp_current(self, I: List[float], n1: int, n2: int, i_val: float) -> None:
        """Stamp current flowing from n1 into n2 (injected into n2, extracted from n1)."""
        if n1 > 0:
            I[n1 - 1] -= i_val
        if n2 > 0:
            I[n2 - 1] += i_val

    def step(self) -> None:
        """Advance simulation by dt using Modified Nodal Analysis (MNA)."""
        dt = self.dt
        self.time += dt

        # Identify independent voltage sources
        v_sources: List[Tuple[CircuitComponent, float]] = []
        for c in self.components:
            if isinstance(c, VoltageSourceDC):
                v_sources.append((c, c.value))
            elif isinstance(c, VoltageSourceAC):
                v_sources.append((c, c.get_voltage(self.time)))
            elif isinstance(c, VoltageSourceClock):
                v_sources.append((c, c.get_voltage(self.time)))

        # Also op-amps introduce voltage source at their output
        num_v_sources = len(v_sources) + len(self.opamps)
        num_vars = (self.num_nodes - 1) + num_v_sources

        if num_vars <= 0:
            return

        # Prepare companion models for reactive components
        for c in self.components:
            if isinstance(c, (Capacitor, Inductor)):
                c.prepare_step(dt)

        # Allocate MNA matrices [G, B; C, D] * [v; j] = [i; e]
        M = [[0.0] * num_vars for _ in range(num_vars)]
        RHS = [0.0] * num_vars

        # 1. Stamp passive and companion conductances
        for c in self.components:
            n1, n2 = c.n1, c.n2
            if isinstance(c, Resistor):
                g = 1.0 / max(1e-6, c.value)
                self._stamp_conductance(M, n1, n2, g)
            elif isinstance(c, Switch):
                g = 1.0 / max(1e-6, c.resistance)
                self._stamp_conductance(M, n1, n2, g)
            elif isinstance(c, Capacitor):
                self._stamp_conductance(M, n1, n2, c.g_eq)
                self._stamp_current(RHS, n2, n1, c.i_eq)
            elif isinstance(c, Inductor):
                self._stamp_conductance(M, n1, n2, c.g_eq)
                self._stamp_current(RHS, n1, n2, c.i_eq)
            elif isinstance(c, Diode):
                v_diff = self.node_voltages[n1] - self.node_voltages[n2]
                g, i_eq = c.get_dynamic_conductance(v_diff)
                self._stamp_conductance(M, n1, n2, g)
                self._stamp_current(RHS, n1, n2, i_eq)

        # Add tiny ground shunt resistor to all nodes to ensure numerical solvability of floating subcircuits
        for i in range(1, self.num_nodes):
            M[i - 1][i - 1] += 1e-9

        # 2. Stamp Independent Voltage Sources
        node_offset = self.num_nodes - 1
        for idx, (v_src, v_val) in enumerate(v_sources):
            row = node_offset + idx
            n1, n2 = v_src.n1, v_src.n2

            # V(n1) - V(n2) = v_val
            if n1 > 0:
                M[n1 - 1][row] += 1.0
                M[row][n1 - 1] += 1.0
            if n2 > 0:
                M[n2 - 1][row] -= 1.0
                M[row][n2 - 1] -= 1.0

            RHS[row] = v_val

        # 3. Stamp Op-Amps (Textbook MNA formulation: V+ - V- - Vout/A = 0, current injected into n_out)
        op_offset = node_offset + len(v_sources)
        for idx, op in enumerate(self.opamps):
            row = op_offset + idx
            n_out = op.n_out
            n_pos = op.n_in_pos
            n_neg = op.n_in_neg

            # Output node: current from op-amp injected into n_out
            if n_out > 0:
                M[n_out - 1][row] += 1.0

            # Differential equation: V(n_pos) - V(n_neg) - V(n_out)/gain = 0
            if n_pos > 0:
                M[row][n_pos - 1] += 1.0
            if n_neg > 0:
                M[row][n_neg - 1] -= 1.0
            if op.gain > 0 and n_out > 0:
                M[row][n_out - 1] -= 1.0 / op.gain
            RHS[row] = 0.0

        # Solve linear system
        solution = solve_linear_system(M, RHS)

        # Extract node voltages
        self.node_voltages[0] = 0.0
        for i in range(1, self.num_nodes):
            self.node_voltages[i] = solution[i - 1] if i - 1 < len(solution) else 0.0

        # Op-amp rail saturation check
        for op in self.opamps:
            if op.n_out > 0:
                vout = self.node_voltages[op.n_out]
                if vout > op.v_supply_pos:
                    self.node_voltages[op.n_out] = op.v_supply_pos
                elif vout < op.v_supply_neg:
                    self.node_voltages[op.n_out] = op.v_supply_neg

        # Extract voltage source currents
        for idx, (v_src, _) in enumerate(v_sources):
            row = node_offset + idx
            v_src.current = solution[row] if row < len(solution) else 0.0

        for idx, op in enumerate(self.opamps):
            row = op_offset + idx
            op.current_out = solution[row] if row < len(solution) else 0.0

        # Update component voltages and currents for passive components
        for c in self.components:
            v1 = self.node_voltages[c.n1]
            v2 = self.node_voltages[c.n2]
            c.voltage_drop = v1 - v2

            if isinstance(c, Resistor):
                c.current = c.voltage_drop / max(1e-6, c.value)
            elif isinstance(c, Switch):
                c.current = c.voltage_drop / max(1e-6, c.resistance)
            elif isinstance(c, Capacitor):
                # i_C = g_eq * (v - prev_v)
                c.current = c.g_eq * (c.voltage_drop - c.prev_v)
                c.prev_v = c.voltage_drop
            elif isinstance(c, Inductor):
                # i_L = prev_i + g_eq * v
                c.current = c.prev_i + c.g_eq * c.voltage_drop
                c.prev_i = c.current
            elif isinstance(c, Diode):
                if c.voltage_drop >= c.v_drop:
                    c.current = (c.voltage_drop - c.v_drop) / c.r_on
                else:
                    c.current = c.voltage_drop / c.r_off

        # Record history for oscilloscope probes
        self.history_time.append(self.time)
        for i in range(self.num_nodes):
            if i not in self.history_nodes:
                self.history_nodes[i] = []
            self.history_nodes[i].append(self.node_voltages[i])

        # Limit history buffer to last 2000 points
        if len(self.history_time) > 2000:
            self.history_time.pop(0)
            for i in range(self.num_nodes):
                if self.history_nodes[i]:
                    self.history_nodes[i].pop(0)
