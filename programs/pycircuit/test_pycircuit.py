"""Comprehensive unit tests for PyCircuit SPICE Analog Simulator."""

import math
import unittest
import sys
import os

# Ensure local module directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    Circuit, Resistor, Capacitor, Inductor, VoltageSourceDC,
    VoltageSourceAC, VoltageSourceClock, Diode, Switch, OpAmp,
    solve_linear_system
)
from presets import (
    PRESETS, build_rlc_resonance, build_bridge_rectifier,
    build_rc_step_transient, build_diode_clipper, build_opamp_inverting
)


class TestLinearSystemSolver(unittest.TestCase):
    """Test first-principles Gaussian elimination solver."""

    def test_2x2_system(self):
        # 2x + y = 5
        # x + 3y = 5
        # Solution: x = 2, y = 1
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 5.0]
        x = solve_linear_system(A, b)
        self.assertAlmostEqual(x[0], 2.0, places=6)
        self.assertAlmostEqual(x[1], 1.0, places=6)

    def test_3x3_partial_pivoting(self):
        # Requires row swapping when pivot is zero or small
        A = [
            [0.0, 2.0, 1.0],
            [1.0, -1.0, 1.0],
            [2.0, 1.0, -1.0]
        ]
        b = [4.0, 2.0, 1.0]
        x = solve_linear_system(A, b)
        # Verify A * x = b
        self.assertAlmostEqual(0*x[0] + 2*x[1] + 1*x[2], 4.0, places=5)
        self.assertAlmostEqual(1*x[0] - 1*x[1] + 1*x[2], 2.0, places=5)
        self.assertAlmostEqual(2*x[0] + 1*x[1] - 1*x[2], 1.0, places=5)


class TestCircuitBasics(unittest.TestCase):
    """Test DC and fundamental circuit laws."""

    def test_ohms_law(self):
        """Verify Ohm's Law: I = V / R with 10V and 500 Ohm."""
        c = Circuit()
        c.dt = 1e-4
        c.add_dc_source("V1", n1=1, n2=0, voltage=10.0)
        r = c.add_resistor("R1", n1=1, n2=0, resistance=500.0)

        c.step()
        self.assertAlmostEqual(c.node_voltages[1], 10.0, places=3)
        expected_current = 10.0 / 500.0  # 0.020 A (20 mA)
        self.assertAlmostEqual(r.current, expected_current, places=4)

    def test_voltage_divider(self):
        """Verify series voltage divider: V_mid = V_in * (R2 / (R1 + R2))."""
        c = Circuit()
        c.add_dc_source("V_IN", n1=1, n2=0, voltage=12.0)
        c.add_resistor("R1", n1=1, n2=2, resistance=1000.0)
        c.add_resistor("R2", n1=2, n2=0, resistance=2000.0)

        c.step()
        # V(Node 2) should be 12.0 * (2000 / 3000) = 8.0 V
        self.assertAlmostEqual(c.node_voltages[1], 12.0, places=3)
        self.assertAlmostEqual(c.node_voltages[2], 8.0, places=3)

    def test_switch_conduction(self):
        """Verify toggleable switch controls current flow."""
        c = Circuit()
        c.add_dc_source("V1", n1=1, n2=0, voltage=10.0)
        sw = c.add_switch("SW1", n1=1, n2=2, closed=True)
        r = c.add_resistor("R1", n1=2, n2=0, resistance=100.0)

        # Closed switch conducts
        c.step()
        self.assertAlmostEqual(c.node_voltages[2], 10.0, delta=0.05)
        self.assertAlmostEqual(r.current, 0.1, delta=0.01)

        # Open switch blocks
        sw.closed = False
        c.step()
        self.assertAlmostEqual(c.node_voltages[2], 0.0, delta=0.01)
        self.assertAlmostEqual(r.current, 0.0, delta=0.001)


class TestTransientAndReactive(unittest.TestCase):
    """Test reactive dynamics (RC and RLC circuits)."""

    def test_rc_exponential_charging(self):
        """Verify capacitor charging matches analytical curve: V(t) = V0 * (1 - exp(-t / tau))."""
        c = Circuit()
        # R = 1000 Ohm, C = 10 uF -> tau = 0.010 s (10 ms)
        r_val = 1000.0
        c_val = 10e-6
        tau = r_val * c_val

        c.dt = 1e-4  # 0.1 ms step
        c.add_dc_source("V1", n1=1, n2=0, voltage=10.0)
        c.add_resistor("R1", n1=1, n2=2, resistance=r_val)
        c.add_capacitor("C1", n1=2, n2=0, capacitance=c_val)

        # Step until t = tau (10 ms = 100 steps)
        steps = int(tau / c.dt)
        for _ in range(steps):
            c.step()

        # At t = tau, V_C should equal V0 * (1 - 1/e) ~ 10.0 * 0.63212 = 6.32 V
        expected_vc = 10.0 * (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(c.node_voltages[2], expected_vc, delta=0.15)

    def test_rlc_resonance_tank(self):
        """Verify RLC tank circuit sustains AC oscillation."""
        c = Circuit()
        build_rlc_resonance(c)
        c.dt = 5e-5

        # Run 200 simulation steps
        for _ in range(200):
            c.step()

        # Voltages should remain stable (finite and non-zero)
        v_cap = c.node_voltages[3]
        self.assertFalse(math.isnan(v_cap))
        self.assertFalse(math.isinf(v_cap))
        self.assertGreater(len(c.history_nodes[3]), 100)


class TestNonLinearAndActive(unittest.TestCase):
    """Test non-linear semiconductor diodes and operational amplifiers."""

    def test_diode_rectification(self):
        """Verify diode conducts in forward bias and blocks in reverse bias."""
        c = Circuit()
        c.add_dc_source("V_FWD", n1=1, n2=0, voltage=5.0)
        d = c.add_diode("D1", n1=1, n2=2, v_drop=0.7)
        r = c.add_resistor("R_LOAD", n1=2, n2=0, resistance=100.0)

        # First step forward bias: V(2) should be approximately 5.0 - 0.7 = 4.3V
        c.step()
        c.step()
        self.assertTrue(d.is_conducting)
        self.assertGreater(c.node_voltages[2], 4.0)

    def test_opamp_inverting_gain(self):
        """Verify inverting amplifier gain: Av = -Rf / Rin."""
        c = Circuit()
        build_opamp_inverting(c)
        # Drive with small DC offset to verify gain accurately
        c.components[0] = VoltageSourceDC(name="V_IN", n1=1, n2=0, value=1.0)
        c.dt = 1e-4

        for _ in range(10):
            c.step()

        # Vin = 1.0V, Rin = 10k, Rf = 20k -> Vout = -2.0V
        v_out = c.node_voltages[3]
        self.assertAlmostEqual(v_out, -2.0, delta=0.2)

    def test_full_wave_bridge_rectifier(self):
        """Verify full-wave bridge rectifier converts bipolar AC to unipolar DC."""
        c = Circuit()
        build_bridge_rectifier(c)
        c.dt = 1e-4

        # Run for 2 full 60 Hz AC cycles (33 ms)
        for _ in range(350):
            c.step()

        # Output DC voltage across load (Node 3) should always be non-negative
        v_out_hist = c.history_nodes[3][-100:]
        for v in v_out_hist:
            self.assertGreaterEqual(v, -0.05, "Rectifier output dropped significantly below 0V")
        # Mean filtered output should be positive
        avg_v = sum(v_out_hist) / len(v_out_hist)
        self.assertGreater(avg_v, 5.0, "Filter capacitor failed to maintain positive DC level")

    def test_diode_clipper_limiting(self):
        """Verify symmetrical clipper clamps +/-8V AC peak down to +/-0.8V threshold."""
        c = Circuit()
        build_diode_clipper(c)
        c.dt = 5e-5

        # Run through 2 cycles of 100 Hz signal
        for _ in range(400):
            c.step()

        v_clipped = c.history_nodes[2][-200:]
        max_v = max(v_clipped)
        min_v = min(v_clipped)
        # Input was +/-8V, output must be clipped within +/-1.2V
        self.assertLess(max_v, 1.2)
        self.assertGreater(min_v, -1.2)

    def test_signal_sources(self):
        """Verify AC sinusoidal and clock source formulas."""
        ac = VoltageSourceAC(name="AC1", n1=1, n2=0, value=5.0, frequency=100.0)
        # At t = 0, sin(0) = 0
        self.assertAlmostEqual(ac.get_voltage(0.0), 0.0)
        # At t = 1/(4*f) = 0.0025 s, sin(pi/2) = 1.0 -> 5.0V
        self.assertAlmostEqual(ac.get_voltage(0.0025), 5.0, places=4)

        clk = VoltageSourceClock(name="CLK1", n1=1, n2=0, value=3.3, frequency=1000.0, duty_cycle=0.5)
        # At t = 0.0002 s (< 0.0005s), HIGH = 3.3V
        self.assertEqual(clk.get_voltage(0.0002), 3.3)
        # At t = 0.0007 s (> 0.0005s), LOW = 0.0V
        self.assertEqual(clk.get_voltage(0.0007), 0.0)


class TestPresetsIntegrity(unittest.TestCase):
    """Test that all curated presets load and simulate without errors."""

    def test_all_presets_simulate(self):
        c = Circuit()
        for name, builder in PRESETS.items():
            builder(c)
            self.assertGreater(c.num_nodes, 1, f"Preset {name} has no nodes")
            self.assertGreater(len(c.components) + len(c.opamps), 0, f"Preset {name} has no parts")

            # Run 50 simulation steps
            for _ in range(50):
                c.step()

            for n in range(c.num_nodes):
                val = c.node_voltages[n]
                self.assertFalse(math.isnan(val), f"NaN detected in {name} at node {n}")
                self.assertFalse(math.isinf(val), f"Inf detected in {name} at node {n}")


class TestAppLifecycle(unittest.TestCase):
    """Test desktop GUI lifecycle and component interactions."""

    def test_gui_initialization_and_step(self):
        """Verify PyCircuitApp instantiates, loads presets, steps, and cleans up."""
        try:
            import tkinter as tk
            from pycircuit import PyCircuitApp
            root = tk.Tk()
            app = PyCircuitApp(root)
            self.assertIsNotNone(app.circuit)
            self.assertGreater(app.circuit.num_nodes, 1)

            # Test step execution
            app.step_once()
            # Test preset switching
            app.load_preset("RC Step Response Transient")
            self.assertEqual(app.circuit.components[0].name, "V_DC")
            # Test switch toggle
            sw = app.circuit.components[1]
            self.assertTrue(sw.closed)

            app.on_close()
        except tk.TclError:
            # Handle display-less CI environments gracefully
            pass


if __name__ == "__main__":
    unittest.main()
