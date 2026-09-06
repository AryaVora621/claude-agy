"""Curated analog circuit presets for PyCircuit."""

from typing import Dict, Any, Callable
try:
    from .engine import Circuit
except ImportError:
    from engine import Circuit


def build_rlc_resonance(c: Circuit) -> None:
    """RLC Series Resonant Tank Circuit.
    Resonant frequency f0 = 1 / (2 * pi * sqrt(L * C)) ~ 159 Hz.
    """
    c.clear()
    # Node 0 = Ground
    # Node 1 = AC Source out
    # Node 2 = Resistor out
    # Node 3 = Inductor out / Capacitor top
    src = c.add_ac_source("V_AC", n1=1, n2=0, amplitude=10.0, frequency=159.0)
    src.x1, src.y1, src.x2, src.y2 = 120, 260, 120, 140

    r = c.add_resistor("R1", n1=1, n2=2, resistance=50.0)
    r.x1, r.y1, r.x2, r.y2 = 120, 140, 280, 140

    l = c.add_inductor("L1", n1=2, n2=3, inductance=0.1)  # 100 mH
    l.x1, l.y1, l.x2, l.y2 = 280, 140, 440, 140

    cap = c.add_capacitor("C1", n1=3, n2=0, capacitance=10e-6)  # 10 uF
    cap.x1, cap.y1, cap.x2, cap.y2 = 440, 140, 440, 260


def build_bridge_rectifier(c: Circuit) -> None:
    """Full-Wave Diode Bridge Rectifier with Capacitor Filter.
    Converts 60 Hz AC to smoothed DC voltage across load.
    """
    c.clear()
    # AC input between Node 1 and Node 2
    src = c.add_ac_source("V_AC", n1=1, n2=2, amplitude=12.0, frequency=60.0)
    src.x1, src.y1, src.x2, src.y2 = 120, 240, 120, 140

    # Bridge Diodes
    # D1: Node 1 -> Node 3 (DC+)
    d1 = c.add_diode("D1", n1=1, n2=3)
    d1.x1, d1.y1, d1.x2, d1.y2 = 240, 140, 340, 80

    # D2: Node 2 -> Node 3 (DC+)
    d2 = c.add_diode("D2", n1=2, n2=3)
    d2.x1, d2.y1, d2.x2, d2.y2 = 240, 240, 340, 80

    # D3: Node 0 (Ground/DC-) -> Node 1
    d3 = c.add_diode("D3", n1=0, n2=1)
    d3.x1, d3.y1, d3.x2, d3.y2 = 340, 300, 240, 140

    # D4: Node 0 (Ground/DC-) -> Node 2
    d4 = c.add_diode("D4", n1=0, n2=2)
    d4.x1, d4.y1, d4.x2, d4.y2 = 340, 300, 240, 240

    # Smoothing Filter Capacitor: Node 3 -> Node 0
    cap = c.add_capacitor("C_Filter", n1=3, n2=0, capacitance=220e-6)  # 220 uF
    cap.x1, cap.y1, cap.x2, cap.y2 = 460, 80, 460, 300

    # Load Resistor: Node 3 -> Node 0
    r_load = c.add_resistor("R_Load", n1=3, n2=0, resistance=500.0)
    r_load.x1, r_load.y1, r_load.x2, r_load.y2 = 560, 80, 560, 300


def build_rc_step_transient(c: Circuit) -> None:
    """RC Charging and Discharging Transient Circuit.
    Time constant tau = R * C = 1000 * 10uF = 0.010 s (10 ms).
    """
    c.clear()
    src = c.add_dc_source("V_DC", n1=1, n2=0, voltage=10.0)
    src.x1, src.y1, src.x2, src.y2 = 120, 260, 120, 140

    sw = c.add_switch("SW1", n1=1, n2=2, closed=True)
    sw.x1, sw.y1, sw.x2, sw.y2 = 120, 140, 260, 140

    r = c.add_resistor("R1", n1=2, n2=3, resistance=1000.0)
    r.x1, r.y1, r.x2, r.y2 = 260, 140, 400, 140

    cap = c.add_capacitor("C1", n1=3, n2=0, capacitance=10e-6)  # 10 uF
    cap.x1, cap.y1, cap.x2, cap.y2 = 400, 140, 400, 260


def build_diode_clipper(c: Circuit) -> None:
    """Dual Symmetrical Diode Waveform Clipper.
    Clips +/-10V AC input to +/-0.7V threshold bars.
    """
    c.clear()
    src = c.add_ac_source("V_AC", n1=1, n2=0, amplitude=8.0, frequency=100.0)
    src.x1, src.y1, src.x2, src.y2 = 120, 260, 120, 140

    r = c.add_resistor("R_Limit", n1=1, n2=2, resistance=1000.0)
    r.x1, r.y1, r.x2, r.y2 = 120, 140, 300, 140

    # Diode 1 forward: Node 2 -> Node 0
    d1 = c.add_diode("D1_Top", n1=2, n2=0, v_drop=0.7)
    d1.x1, d1.y1, d1.x2, d1.y2 = 300, 140, 300, 260

    # Diode 2 reverse: Node 0 -> Node 2
    d2 = c.add_diode("D2_Bottom", n1=0, n2=2, v_drop=0.7)
    d2.x2, d2.y2, d2.x1, d2.y1 = 400, 140, 400, 260


def build_opamp_inverting(c: Circuit) -> None:
    """Op-Amp Inverting Linear Amplifier.
    Voltage gain Av = -Rf / Rin = -20k / 10k = -2.0x.
    """
    c.clear()
    # Input AC signal
    src = c.add_ac_source("V_IN", n1=1, n2=0, amplitude=2.5, frequency=200.0)
    src.x1, src.y1, src.x2, src.y2 = 100, 260, 100, 140

    # Input Resistor Rin: Node 1 -> Node 2 (Inverting input)
    rin = c.add_resistor("R_IN", n1=1, n2=2, resistance=10000.0)
    rin.x1, rin.y1, rin.x2, rin.y2 = 100, 140, 260, 140

    # Op-Amp: n_pos=0 (GND), n_neg=2, n_out=3
    op = c.add_opamp("OP1", n_pos=0, n_neg=2, n_out=3, v_pos=15.0, v_neg=-15.0)

    # Feedback Resistor Rf: Node 2 -> Node 3
    rf = c.add_resistor("R_Feedback", n1=2, n2=3, resistance=20000.0)
    rf.x1, rf.y1, rf.x2, rf.y2 = 260, 80, 460, 80

    # Load Resistor: Node 3 -> Node 0
    r_load = c.add_resistor("R_Load", n1=3, n2=0, resistance=2000.0)
    r_load.x1, r_load.y1, r_load.x2, r_load.y2 = 460, 160, 460, 260


PRESETS: Dict[str, Callable[[Circuit], None]] = {
    "RLC Resonant Tank": build_rlc_resonance,
    "Full-Wave Bridge Rectifier": build_bridge_rectifier,
    "RC Step Response Transient": build_rc_step_transient,
    "Diode Symmetrical Clipper": build_diode_clipper,
    "Op-Amp Inverting Amplifier": build_opamp_inverting,
}
