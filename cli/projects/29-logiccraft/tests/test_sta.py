"""Unit tests for Static Timing Analysis (STA) Engine."""

import unittest
from logiccraft.aig import AIGGraph
from logiccraft.liberty import get_default_library
from logiccraft.techmap import TechMapper
from logiccraft.sta import StaticTimingAnalyzer


class TestStaticTimingAnalysis(unittest.TestCase):
    """Test suite for STA arrival times, required times, slack, and critical path."""

    def setUp(self) -> None:
        self.lib = get_default_library()
        self.mapper = TechMapper(self.lib, target_metric="area")

    def test_inverter_chain_timing(self) -> None:
        """Verify timing propagation across a cascade of inverters."""
        aig = AIGGraph()
        in_a = aig.create_pi("A")
        # 3 inverters in series
        inv1 = aig.not_(in_a)
        inv2 = aig.not_(inv1)
        inv3 = aig.not_(inv2)
        aig.set_output("Y", inv3)

        netlist = self.mapper.map_aig(aig, module_name="inv_chain")
        sta = StaticTimingAnalyzer(netlist)
        report = sta.run_sta(clock_period=500.0)

        self.assertGreater(report.max_delay, 0.0)
        self.assertTrue(report.is_timing_met)
        self.assertGreater(report.worst_negative_slack, 0.0)
        self.assertEqual(report.total_negative_slack, 0.0)
        self.assertGreater(len(report.critical_path), 0)

    def test_timing_violation_detection(self) -> None:
        """Verify WNS turns negative when clock period is unrealistically tight."""
        aig = AIGGraph()
        a = aig.create_pi("A")
        b = aig.create_pi("B")
        out = aig.and_(a, b)
        aig.set_output("Y", out)

        netlist = self.mapper.map_aig(aig, module_name="tight_clock")
        sta = StaticTimingAnalyzer(netlist)

        # Impossibly tight clock period: 1.0 ps
        report = sta.run_sta(clock_period=1.0)
        self.assertFalse(report.is_timing_met)
        self.assertLess(report.worst_negative_slack, 0.0)
        self.assertLess(report.total_negative_slack, 0.0)
        self.assertEqual(report.violating_endpoints, 1)

    def test_critical_path_start_and_end(self) -> None:
        """Verify critical path starts at a primary input and terminates at a primary output."""
        aig = AIGGraph()
        a = aig.create_pi("A")
        b = aig.create_pi("B")
        c = aig.create_pi("C")
        t1 = aig.and_(a, b)
        t2 = aig.or_(t1, c)
        aig.set_output("Y", t2)

        netlist = self.mapper.map_aig(aig, module_name="comb_block")
        sta = StaticTimingAnalyzer(netlist)
        report = sta.run_sta(clock_period=1000.0)

        self.assertGreater(len(report.critical_path), 1)
        first_node = report.critical_path[0]
        last_node = report.critical_path[-1]

        self.assertTrue(first_node.pin.startswith("PORT:"))
        self.assertTrue(last_node.pin.startswith("PORT:"))


if __name__ == "__main__":
    unittest.main()
