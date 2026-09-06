"""
QuantaLab: ASCII Terminal Visualizers for Quantum Circuits, Bloch Spheres, and States.
Features:
1. CircuitRenderer: Multi-wire ASCII circuit diagrams with gate boxes and control lines.
2. BlochSphereVisualizer: 2D ASCII Bloch sphere projection showing single-qubit vector orientation.
3. StateVisualizer: Probability distribution bar charts and Dirac ket decomposition.
"""

import math
from typing import Dict, List, Tuple
from .types import StateVector, BasisState
from .circuit import QuantumCircuit


class CircuitRenderer:
    """Renders QuantumCircuit objects into clean ASCII terminal wire diagrams."""

    @staticmethod
    def render(circ: QuantumCircuit) -> str:
        lines = [f"q{q}: ─" for q in range(circ.n_qubits)]

        for op in circ.ops:
            name = op.name
            qs = op.qubits

            if name in ("h", "x", "y", "z", "s", "t", "sdg", "tdg"):
                gate_str = f"[{name.upper()}]─"
                for q in range(circ.n_qubits):
                    if q == qs[0]:
                        lines[q] += gate_str
                    else:
                        lines[q] += "─" * len(gate_str)

            elif name in ("rx", "ry", "rz", "p"):
                gate_str = f"[{name.upper()}]─"
                for q in range(circ.n_qubits):
                    if q == qs[0]:
                        lines[q] += gate_str
                    else:
                        lines[q] += "─" * len(gate_str)

            elif name == "cx":
                ctrl, tgt = qs[0], qs[1]
                min_q, max_q = min(ctrl, tgt), max(ctrl, tgt)
                for q in range(circ.n_qubits):
                    if q == ctrl:
                        lines[q] += "─●─"
                    elif q == tgt:
                        lines[q] += "─[X]─"
                    elif min_q < q < max_q:
                        lines[q] += "─┼─"
                    else:
                        lines[q] += "───"

            elif name == "cz":
                ctrl, tgt = qs[0], qs[1]
                min_q, max_q = min(ctrl, tgt), max(ctrl, tgt)
                for q in range(circ.n_qubits):
                    if q in (ctrl, tgt):
                        lines[q] += "─●─"
                    elif min_q < q < max_q:
                        lines[q] += "─┼─"
                    else:
                        lines[q] += "───"

            elif name == "ccx":
                c1, c2, tgt = qs[0], qs[1], qs[2]
                all_qs = sorted([c1, c2, tgt])
                min_q, max_q = all_qs[0], all_qs[-1]
                for q in range(circ.n_qubits):
                    if q in (c1, c2):
                        lines[q] += "─●─"
                    elif q == tgt:
                        lines[q] += "─[X]─"
                    elif min_q < q < max_q:
                        lines[q] += "─┼─"
                    else:
                        lines[q] += "───"

            elif name == "swap":
                q1, q2 = qs[0], qs[1]
                min_q, max_q = min(q1, q2), max(q1, q2)
                for q in range(circ.n_qubits):
                    if q in (q1, q2):
                        lines[q] += "─x─"
                    elif min_q < q < max_q:
                        lines[q] += "─│─"
                    else:
                        lines[q] += "───"

            elif name == "measure":
                for q in range(circ.n_qubits):
                    if q == qs[0]:
                        lines[q] += "─[M]─"
                    else:
                        lines[q] += "─────"

            elif name == "barrier":
                for q in range(circ.n_qubits):
                    lines[q] += "─╫─"

        # Append end line
        result = [line + "─" for line in lines]
        return "\n".join(result)


class BlochSphereVisualizer:
    """Renders single-qubit Bloch vector (x, y, z) into an ASCII 2D projection."""

    @staticmethod
    def render(x: float, y: float, z: float, radius: int = 6) -> str:
        # Polar angles: theta in [0, pi], phi in [0, 2*pi)
        r = math.sqrt(x * x + y * y + z * z)
        theta_deg = math.degrees(math.acos(max(-1.0, min(1.0, z / r)))) if r > 1e-6 else 0.0
        phi_deg = math.degrees(math.atan2(y, x)) % 360.0

        # Project 3D vector onto 2D ASCII circle
        # Screen coordinates: row corresponds to -Z (top is +Z), col corresponds to Y (right is +Y)
        # We also slightly skew by X for perspective
        grid_h = radius * 2 + 1
        grid_w = radius * 4 + 1
        grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

        center_r = radius
        center_c = radius * 2

        # Draw sphere boundary
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            r_pos = int(round(center_r - radius * math.sin(rad)))
            c_pos = int(round(center_c + radius * 2 * math.cos(rad)))
            if 0 <= r_pos < grid_h and 0 <= c_pos < grid_w:
                grid[r_pos][c_pos] = "."

        # Draw equator
        for c_pos in range(center_c - radius * 2, center_c + radius * 2 + 1):
            if 0 <= c_pos < grid_w:
                grid[center_r][c_pos] = "-"

        # Draw vertical axis
        for r_pos in range(grid_h):
            grid[r_pos][center_c] = "|"
        grid[center_r][center_c] = "+"

        # Plot vector point
        target_r = int(round(center_r - z * radius))
        target_c = int(round(center_c + y * (radius * 2) + x * radius * 0.5))
        target_r = max(0, min(grid_h - 1, target_r))
        target_c = max(0, min(grid_w - 1, target_c))
        grid[target_r][target_c] = "*"

        out_lines = []
        out_lines.append("           |0> (+z)")
        for row in grid:
            out_lines.append("  " + "".join(row))
        out_lines.append("           |1> (-z)")
        out_lines.append(f"\n  Bloch Vector: (x={x:+.3f}, y={y:+.3f}, z={z:+.3f}) | Length: {r:.3f}")
        out_lines.append(f"  Spherical Coordinates: theta={theta_deg:.1f} deg, phi={phi_deg:.1f} deg")

        return "\n".join(out_lines)


class StateVisualizer:
    """Renders measurement probability histograms and basis state decompositions."""

    @staticmethod
    def bar_chart(counts: Dict[str, int], max_width: int = 30) -> str:
        if not counts:
            return "(No measurements)"

        total = sum(counts.values())
        lines = []
        for bitstring, count in sorted(counts.items()):
            pct = (count / total) * 100.0
            bar_len = int(round((count / total) * max_width))
            bar = "#" * bar_len + " " * (max_width - bar_len)
            lines.append(f"  |{bitstring}>: [{bar}] {pct:5.1f}% ({count:,} shots)")
        return "\n".join(lines)
