"""LogicCraft EDA Synthesis & Timing Closure Interactive Workbench.

Demonstrates complete front-to-back industrial EDA digital design flow:
1. RTL / Boolean Logic specification & ROBDD canonical formal analysis.
2. Multi-level logic synthesis onto And-Inverter Graphs (AIG) with strashing.
3. Technology mapping to TSMC 45nm standard cell library via DAGON tree covering.
4. Physical netlist generation with load parasitics & structural Verilog export.
5. Static Timing Analysis (STA) with NLDM bilinear interpolation & slack computation.
6. Physical critical path extraction and sub-pixel Braille delay profiling.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List, Tuple

# Ensure logiccraft is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logiccraft.bdd import BDDManager
from logiccraft.aig import AIGGraph
from logiccraft.liberty import get_default_library
from logiccraft.techmap import TechMapper
from logiccraft.sta import StaticTimingAnalyzer, TimingReport
from logiccraft.visualizer import CircuitVisualizer


def build_4bit_adder_aig() -> AIGGraph:
    """Construct 4-Bit Carry-Ripple / Lookahead Adder AIG."""
    aig = AIGGraph()
    a_inputs = [aig.create_pi(f"A_{i}") for i in range(4)]
    b_inputs = [aig.create_pi(f"B_{i}") for i in range(4)]
    c_in = aig.create_pi("CIN")

    carry = c_in
    for i in range(4):
        a = a_inputs[i]
        b = b_inputs[i]
        # Propagate: P = A ^ B
        p = aig.xor_(a, b)
        # Sum: S = P ^ Cin
        s = aig.xor_(p, carry)
        # Generate: G = A & B
        g = aig.and_(a, b)
        # Next carry: Cout = G | (P & Cin)
        carry = aig.or_(g, aig.and_(p, carry))
        aig.set_output(f"SUM_{i}", s)

    aig.set_output("COUT", carry)
    return aig


def build_parity_tree_aig() -> AIGGraph:
    """Construct 8-Bit Parity Generator & Majority Voter AIG."""
    aig = AIGGraph()
    d_inputs = [aig.create_pi(f"D_{i}") for i in range(8)]

    # 8-input parity tree
    curr_parity = d_inputs[0]
    for i in range(1, 8):
        curr_parity = aig.xor_(curr_parity, d_inputs[i])
    aig.set_output("PARITY", curr_parity)

    # 4-input majority voter on low 4 bits: >= 3 ones
    # Maj(a,b,c,d) = (a&b&c) | (a&b&d) | (a&c&d) | (b&c&d)
    a, b, c, d = d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3]
    t1 = aig.and_(aig.and_(a, b), c)
    t2 = aig.and_(aig.and_(a, b), d)
    t3 = aig.and_(aig.and_(a, c), d)
    t4 = aig.and_(aig.and_(b, c), d)
    maj = aig.or_(aig.or_(t1, t2), aig.or_(t3, t4))
    aig.set_output("MAJ4", maj)

    return aig


def build_priority_encoder_aig() -> AIGGraph:
    """Construct 4-to-2 Priority Encoder AIG."""
    aig = AIGGraph()
    req = [aig.create_pi(f"REQ_{i}") for i in range(4)]

    # Grant 3: REQ_3
    g3 = req[3]
    # Grant 2: REQ_2 & ~REQ_3
    g2 = aig.and_(req[2], aig.not_(req[3]))
    # Grant 1: REQ_1 & ~REQ_2 & ~REQ_3
    g1 = aig.and_(req[1], aig.and_(aig.not_(req[2]), aig.not_(req[3])))
    # Grant 0: REQ_0 & ~REQ_1 & ~REQ_2 & ~REQ_3
    g0 = aig.and_(req[0], aig.and_(aig.not_(req[1]), aig.and_(aig.not_(req[2]), aig.not_(req[3]))))

    aig.set_output("GNT_3", g3)
    aig.set_output("GNT_2", g2)
    aig.set_output("GNT_1", g1)
    aig.set_output("GNT_0", g0)

    # Any grant
    any_gnt = aig.or_(aig.or_(g0, g1), aig.or_(g2, g3))
    aig.set_output("VALID", any_gnt)

    return aig


def run_eda_flow(
    circuit_type: str = "adder4",
    target_metric: str = "area",
    clock_period: float = 800.0,
    show_verilog: bool = False,
) -> None:
    """Execute complete EDA synthesis and static timing analysis pipeline."""
    c_cyan = "\033[96m"
    c_green = "\033[92m"
    c_yellow = "\033[93m"
    c_bold = "\033[1m"
    c_dim = "\033[2m"
    c_reset = "\033[0m"

    print(f"\n{c_bold}{c_cyan}{'=' * 76}{c_reset}")
    print(f"{c_bold}LOGICCRAFT: AUTOMATED DIGITAL SYNTHESIS & TIMING CLOSURE WORKBENCH{c_reset}")
    print(f"{c_bold}{c_cyan}{'=' * 76}{c_reset}\n")

    # 1. Circuit Construction
    if circuit_type == "adder4":
        module_name = "cla_adder_4bit"
        circuit_desc = "4-Bit Carry-Lookahead Adder (9 Inputs, 5 Outputs)"
        aig = build_4bit_adder_aig()
    elif circuit_type == "parity":
        module_name = "parity_voter_8bit"
        circuit_desc = "8-Bit Parity Generator & Majority Voter (8 Inputs, 2 Outputs)"
        aig = build_parity_tree_aig()
    elif circuit_type == "encoder":
        module_name = "priority_encoder_4to2"
        circuit_desc = "4-to-2 Priority Arbiter / Encoder (4 Inputs, 5 Outputs)"
        aig = build_priority_encoder_aig()
    else:
        raise ValueError(f"Unknown circuit: {circuit_type}")

    print(f"{c_bold}[STEP 1/5] Logic Synthesis & And-Inverter Graph Decomposition{c_reset}")
    print(f"  Target Circuit:        {c_cyan}{circuit_desc}{c_reset}")
    print(f"  Internal AND Nodes:    {aig.num_and_nodes}")
    print(f"  Graph Logic Depth:     {aig.max_depth} levels")
    print(f"  Structural Hash Hits:  Active (On-The-Fly Redundancy Elimination)\n")

    # 2. Technology Mapping
    print(f"{c_bold}[STEP 2/5] Standard Cell Technology Mapping (DAGON Algorithm){c_reset}")
    lib = get_default_library()
    print(f"  Target Library:        {lib.name} (45nm Standard CMOS, VDD=1.1V, Temp=25C)")
    print(f"  Optimization Goal:     Minimum Silicon {target_metric.upper()}")

    mapper = TechMapper(lib, target_metric=target_metric)
    netlist = mapper.map_aig(aig, module_name=module_name)

    print(f"  Mapped Cells:          {netlist.num_cells} standard cells")
    print(f"  Physical Area:         {netlist.total_area:.2f} um^2")
    print(f"  Static Leakage Power:  {netlist.total_leakage_power:.2f} nW\n")

    # 3. Static Timing Analysis
    print(f"{c_bold}[STEP 3/5] Static Timing Analysis (NLDM 2D Bilinear Interpolation){c_reset}")
    print(f"  Target Clock Period:   {clock_period:.1f} ps (Requested Freq: {1000.0/clock_period:.2f} GHz)")

    sta = StaticTimingAnalyzer(netlist)
    report = sta.run_sta(clock_period=clock_period)

    # 4. Telemetry Dashboard
    print(f"\n{c_bold}[STEP 4/5] Timing Closure & Circuit Telemetry HUD{c_reset}")
    hud = CircuitVisualizer.render_timing_hud(netlist, report, target_name=module_name)
    print(hud)

    # 5. Critical Path & Delay Profile
    print(f"\n{c_bold}[STEP 5/5] Critical Path Timing Analysis & Delay Profile{c_reset}")
    waterfall = CircuitVisualizer.render_critical_path_waterfall(report)
    print(waterfall)

    # Braille delay progression
    if report.critical_path:
        delays = [seg.arrival_time for seg in report.critical_path]
        print(f"{c_dim}Sub-Pixel Braille Delay Curve (Arrival Time vs Critical Path Stage):{c_reset}")
        braille_curve = CircuitVisualizer.render_delay_curve_braille(delays, width=50, height=6)
        print(braille_curve)
        print(f"{c_dim}Start ({report.critical_path[0].pin}) {'─' * 30} End ({report.critical_path[-1].pin}){c_reset}\n")

    if show_verilog:
        print(f"\n{c_bold}STRUCTURAL VERILOG NETLIST EXPORT:{c_reset}")
        print(f"{c_dim}{'─' * 76}{c_reset}")
        print(netlist.export_verilog())
        print(f"{c_dim}{'─' * 76}{c_reset}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LogicCraft EDA Synthesis & Timing Closure Workbench")
    parser.add_argument(
        "--circuit",
        choices=["adder4", "parity", "encoder"],
        default="adder4",
        help="Circuit architecture to synthesize and analyze",
    )
    parser.add_argument(
        "--target",
        choices=["area", "delay"],
        default="area",
        help="Optimization metric during technology mapping",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=800.0,
        help="Target clock period in picoseconds (default: 800.0 ps)",
    )
    parser.add_argument(
        "--verilog",
        action="store_true",
        help="Print structural Verilog netlist",
    )

    args = parser.parse_args()
    run_eda_flow(
        circuit_type=args.circuit,
        target_metric=args.target,
        clock_period=args.clock,
        show_verilog=args.verilog,
    )


if __name__ == "__main__":
    main()
