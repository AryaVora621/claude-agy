"""Liberty Standard Cell Library & Non-Linear Delay Model (NLDM) Engine.

Features:
1. Standard cell models: INV, BUF, NAND2, NOR2, AND2, OR2, XOR2, AOI21, OAI21, DFF.
2. Non-Linear Delay Model (NLDM) with 2D lookup tables for propagation delay and output slew.
3. High-precision 2D bilinear interpolation over (input_transition, capacitive_load).
4. AIG pattern definitions for dynamic programming technology mapping.
5. Calibrated 45nm/28nm-class standard cell library with physical area, power, and pin capacitances.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class NLDMTable:
    """2D Non-Linear Delay Model table over input_slew (ps) and load_cap (fF)."""
    slew_indices: List[float]  # Row indices (ps)
    cap_indices: List[float]   # Column indices (fF)
    values: List[List[float]]  # 2D table of delay/slew values (ps)

    def lookup(self, input_slew: float, load_cap: float) -> float:
        """Evaluate table via 2D bilinear interpolation."""
        # Clamp slew
        s_idx = self._find_index(input_slew, self.slew_indices)
        c_idx = self._find_index(load_cap, self.cap_indices)

        s0 = self.slew_indices[s_idx]
        s1 = self.slew_indices[s_idx + 1]
        c0 = self.cap_indices[c_idx]
        c1 = self.cap_indices[c_idx + 1]

        u = (input_slew - s0) / (s1 - s0) if s1 > s0 else 0.0
        v = (load_cap - c0) / (c1 - c0) if c1 > c0 else 0.0

        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        v00 = self.values[s_idx][c_idx]
        v10 = self.values[s_idx + 1][c_idx]
        v01 = self.values[s_idx][c_idx + 1]
        v11 = self.values[s_idx + 1][c_idx + 1]

        # Bilinear combination
        return (
            (1.0 - u) * (1.0 - v) * v00
            + u * (1.0 - v) * v10
            + (1.0 - u) * v * v01
            + u * v * v11
        )

    def _find_index(self, val: float, indices: List[float]) -> int:
        if val <= indices[0]:
            return 0
        if val >= indices[-1]:
            return len(indices) - 2
        for i in range(len(indices) - 1):
            if indices[i] <= val <= indices[i + 1]:
                return i
        return 0


# Alias for backwards compatibility
LookupTable2D = NLDMTable


@dataclass
class TimingArc:
    """Timing arc from an input pin to an output pin."""
    from_pin: str
    to_pin: str
    cell_delay_table: NLDMTable
    output_slew_table: NLDMTable

    def compute_delay_and_slew(self, input_slew: float, load_cap: float) -> Tuple[float, float]:
        """Compute propagation delay (ps) and output transition time (ps)."""
        delay = self.cell_delay_table.lookup(input_slew, load_cap)
        slew = self.output_slew_table.lookup(input_slew, load_cap)
        return delay, slew

    def lookup_delay(self, input_slew: float, load_cap: float) -> float:
        return self.cell_delay_table.lookup(input_slew, load_cap)

    def lookup_slew(self, input_slew: float, load_cap: float) -> float:
        return self.output_slew_table.lookup(input_slew, load_cap)


@dataclass
class CellPin:
    """Standard cell pin definition."""
    name: str
    is_input: bool
    capacitance: float = 1.0  # Input capacitance in femtofarads (fF)


@dataclass
class StandardCell:
    """Physical standard cell in the Liberty library."""
    name: str
    area: float             # Silicon area in square micrometers (um^2)
    leakage_power: float    # Static leakage power in nanowatts (nW)
    inputs: List[str]       # Input pin names
    outputs: List[str]      # Output pin names
    pins: Dict[str, CellPin] = field(default_factory=dict)
    timing_arcs: List[TimingArc] = field(default_factory=list)
    is_sequential: bool = False
    setup_time: float = 0.0   # Setup time constraint (ps) for sequential cells
    hold_time: float = 0.0    # Hold time constraint (ps) for sequential cells
    clk_to_q: float = 0.0     # Clock to Q propagation delay (ps)
    # AIG pattern for matching: tuple representation
    # Format: ('AND', left, right), ('NOT', child), or input string pin name
    aig_pattern: Optional[Tuple] = None

    def get_timing_arc(self, from_pin: str, to_pin: str) -> Optional[TimingArc]:
        """Retrieve timing arc between specified pins."""
        for arc in self.timing_arcs:
            if arc.from_pin == from_pin and arc.to_pin == to_pin:
                return arc
        return None


def create_nldm_table(base_delay: float, slew_coeff: float, cap_coeff: float) -> NLDMTable:
    """Generate calibrated NLDM lookup table from linear delay coefficients."""
    slews = [10.0, 30.0, 80.0, 150.0]
    caps = [1.0, 5.0, 15.0, 40.0]
    values = []
    for s in slews:
        row = []
        for c in caps:
            val = base_delay + (slew_coeff * s) + (cap_coeff * c)
            row.append(round(val, 2))
        values.append(row)
    return NLDMTable(slew_indices=slews, cap_indices=caps, values=values)


class LibertyLibrary:
    """Standard cell library container with cell index and pattern matching database."""

    def __init__(self, name: str = "Standard45nm") -> None:
        self.name = name
        self.cells: Dict[str, StandardCell] = {}

    def add_cell(self, cell: StandardCell) -> None:
        self.cells[cell.name] = cell

    def get_cell(self, name: str) -> StandardCell:
        if name not in self.cells:
            raise KeyError(f"Cell '{name}' not found in library '{self.name}'")
        return self.cells[name]


def get_default_library() -> LibertyLibrary:
    """Construct a comprehensive calibrated 45nm-style standard cell library."""
    lib = LibertyLibrary("TSMC45_Simulated")

    # Inverter: INV_X1
    # Y = ~A
    inv_x1 = StandardCell(
        name="INV_X1",
        area=1.0,
        leakage_power=0.4,
        inputs=["A"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.2),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", "A"),
    )
    inv_x1.timing_arcs.append(TimingArc(
        from_pin="A", to_pin="Y",
        cell_delay_table=create_nldm_table(base_delay=12.0, slew_coeff=0.08, cap_coeff=0.65),
        output_slew_table=create_nldm_table(base_delay=15.0, slew_coeff=0.10, cap_coeff=0.85),
    ))
    lib.add_cell(inv_x1)

    # Inverter: INV_X2 (Higher drive strength, larger area)
    inv_x2 = StandardCell(
        name="INV_X2",
        area=1.6,
        leakage_power=0.7,
        inputs=["A"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=2.0),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", "A"),
    )
    inv_x2.timing_arcs.append(TimingArc(
        from_pin="A", to_pin="Y",
        cell_delay_table=create_nldm_table(base_delay=10.0, slew_coeff=0.06, cap_coeff=0.35),
        output_slew_table=create_nldm_table(base_delay=12.0, slew_coeff=0.08, cap_coeff=0.45),
    ))
    lib.add_cell(inv_x2)

    # Buffer: BUF_X1
    # Y = A
    buf_x1 = StandardCell(
        name="BUF_X1",
        area=1.8,
        leakage_power=0.8,
        inputs=["A"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.4),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", ("NOT", "A")),
    )
    buf_x1.timing_arcs.append(TimingArc(
        from_pin="A", to_pin="Y",
        cell_delay_table=create_nldm_table(base_delay=22.0, slew_coeff=0.12, cap_coeff=0.55),
        output_slew_table=create_nldm_table(base_delay=16.0, slew_coeff=0.09, cap_coeff=0.50),
    ))
    lib.add_cell(buf_x1)

    # BUF: BUF_X2 (higher drive strength)
    buf_x2 = StandardCell(
        name="BUF_X2",
        area=2.0,
        leakage_power=0.8,
        inputs=["A"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=2.0),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", ("NOT", "A")),
    )
    buf_x2.timing_arcs.append(TimingArc(
        from_pin="A", to_pin="Y",
        cell_delay_table=create_nldm_table(base_delay=18.0, slew_coeff=0.10, cap_coeff=0.35),
        output_slew_table=create_nldm_table(base_delay=12.0, slew_coeff=0.07, cap_coeff=0.30),
    ))
    lib.add_cell(buf_x2)

    # NAND2: NAND2_X1
    # Y = ~(A & B)
    nand2_x1 = StandardCell(
        name="NAND2_X1",
        area=1.5,
        leakage_power=0.6,
        inputs=["A", "B"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.3),
            "B": CellPin(name="B", is_input=True, capacitance=1.3),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", ("AND", "A", "B")),
    )
    for inp in ["A", "B"]:
        nand2_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=15.0, slew_coeff=0.09, cap_coeff=0.70),
            output_slew_table=create_nldm_table(base_delay=18.0, slew_coeff=0.11, cap_coeff=0.90),
        ))
    lib.add_cell(nand2_x1)

    # AND2: AND2_X1
    # Y = A & B
    and2_x1 = StandardCell(
        name="AND2_X1",
        area=2.0,
        leakage_power=0.9,
        inputs=["A", "B"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.4),
            "B": CellPin(name="B", is_input=True, capacitance=1.4),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("AND", "A", "B"),
    )
    for inp in ["A", "B"]:
        and2_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=25.0, slew_coeff=0.14, cap_coeff=0.72),
            output_slew_table=create_nldm_table(base_delay=19.0, slew_coeff=0.10, cap_coeff=0.65),
        ))
    lib.add_cell(and2_x1)

    # NOR2: NOR2_X1
    # Y = ~(A | B) = (~A & ~B)
    nor2_x1 = StandardCell(
        name="NOR2_X1",
        area=1.6,
        leakage_power=0.65,
        inputs=["A", "B"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.5),
            "B": CellPin(name="B", is_input=True, capacitance=1.5),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("AND", ("NOT", "A"), ("NOT", "B")),
    )
    for inp in ["A", "B"]:
        nor2_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=18.0, slew_coeff=0.11, cap_coeff=0.85),
            output_slew_table=create_nldm_table(base_delay=20.0, slew_coeff=0.12, cap_coeff=0.95),
        ))
    lib.add_cell(nor2_x1)

    # OR2: OR2_X1
    # Y = A | B = ~(~A & ~B)
    or2_x1 = StandardCell(
        name="OR2_X1",
        area=2.1,
        leakage_power=0.95,
        inputs=["A", "B"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=1.4),
            "B": CellPin(name="B", is_input=True, capacitance=1.4),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", ("AND", ("NOT", "A"), ("NOT", "B"))),
    )
    for inp in ["A", "B"]:
        or2_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=26.0, slew_coeff=0.13, cap_coeff=0.75),
            output_slew_table=create_nldm_table(base_delay=20.0, slew_coeff=0.11, cap_coeff=0.70),
        ))
    lib.add_cell(or2_x1)

    # XOR2: XOR2_X1
    # Y = A ^ B
    xor2_x1 = StandardCell(
        name="XOR2_X1",
        area=3.2,
        leakage_power=1.4,
        inputs=["A", "B"],
        outputs=["Y"],
        pins={
            "A": CellPin(name="A", is_input=True, capacitance=2.1),
            "B": CellPin(name="B", is_input=True, capacitance=2.1),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        # (A & ~B) | (~A & B) = ~(~(A & ~B) & ~(~A & B))
        aig_pattern=("NOT", ("AND", ("NOT", ("AND", "A", ("NOT", "B"))), ("NOT", ("AND", ("NOT", "A"), "B")))),
    )
    for inp in ["A", "B"]:
        xor2_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=35.0, slew_coeff=0.18, cap_coeff=0.90),
            output_slew_table=create_nldm_table(base_delay=24.0, slew_coeff=0.14, cap_coeff=0.80),
        ))
    lib.add_cell(xor2_x1)

    # AOI21: AOI21_X1
    # Y = ~((A1 & A2) | B) = ~(A1 & A2) & ~B = NOT(AND(NOT(AND(NOT(AND(A1, A2)), NOT(B)))) ...
    # Y = ~((A1 & A2) | B) = ( ~(A1 & A2) ) & ~B
    aoi21_x1 = StandardCell(
        name="AOI21_X1",
        area=2.4,
        leakage_power=1.1,
        inputs=["A1", "A2", "B"],
        outputs=["Y"],
        pins={
            "A1": CellPin(name="A1", is_input=True, capacitance=1.5),
            "A2": CellPin(name="A2", is_input=True, capacitance=1.5),
            "B": CellPin(name="B", is_input=True, capacitance=1.5),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("AND", ("NOT", ("AND", "A1", "A2")), ("NOT", "B")),
    )
    for inp in ["A1", "A2", "B"]:
        aoi21_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=20.0, slew_coeff=0.12, cap_coeff=0.78),
            output_slew_table=create_nldm_table(base_delay=22.0, slew_coeff=0.13, cap_coeff=0.88),
        ))
    lib.add_cell(aoi21_x1)

    # OAI21: OAI21_X1
    # Y = ~((A1 | A2) & B)
    oai21_x1 = StandardCell(
        name="OAI21_X1",
        area=2.4,
        leakage_power=1.1,
        inputs=["A1", "A2", "B"],
        outputs=["Y"],
        pins={
            "A1": CellPin(name="A1", is_input=True, capacitance=1.5),
            "A2": CellPin(name="A2", is_input=True, capacitance=1.5),
            "B": CellPin(name="B", is_input=True, capacitance=1.5),
            "Y": CellPin(name="Y", is_input=False, capacitance=0.0),
        },
        aig_pattern=("NOT", ("AND", ("NOT", ("AND", ("NOT", "A1"), ("NOT", "A2"))), "B")),
    )
    for inp in ["A1", "A2", "B"]:
        oai21_x1.timing_arcs.append(TimingArc(
            from_pin=inp, to_pin="Y",
            cell_delay_table=create_nldm_table(base_delay=22.0, slew_coeff=0.13, cap_coeff=0.80),
            output_slew_table=create_nldm_table(base_delay=22.0, slew_coeff=0.13, cap_coeff=0.85),
        ))
    lib.add_cell(oai21_x1)

    # DFF: DFF_X1 (Sequential flip-flop)
    dff_x1 = StandardCell(
        name="DFF_X1",
        area=6.0,
        leakage_power=2.8,
        inputs=["D", "CLK"],
        outputs=["Q"],
        pins={
            "D": CellPin(name="D", is_input=True, capacitance=1.8),
            "CLK": CellPin(name="CLK", is_input=True, capacitance=2.5),
            "Q": CellPin(name="Q", is_input=False, capacitance=0.0),
        },
        is_sequential=True,
        setup_time=30.0,  # 30 ps setup time
        hold_time=10.0,   # 10 ps hold time
        clk_to_q=45.0,    # 45 ps clock-to-Q
    )
    dff_x1.timing_arcs.append(TimingArc(
        from_pin="CLK", to_pin="Q",
        cell_delay_table=create_nldm_table(base_delay=45.0, slew_coeff=0.15, cap_coeff=0.60),
        output_slew_table=create_nldm_table(base_delay=25.0, slew_coeff=0.12, cap_coeff=0.55),
    ))
    lib.add_cell(dff_x1)

    return lib
