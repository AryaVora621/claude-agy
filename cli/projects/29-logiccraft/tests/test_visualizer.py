"""Unit tests for BrailleCanvas and CircuitVisualizer."""

import unittest
from logiccraft.aig import AIGGraph
from logiccraft.liberty import get_default_library
from logiccraft.techmap import TechMapper
from logiccraft.sta import StaticTimingAnalyzer
from logiccraft.visualizer import BrailleCanvas, CircuitVisualizer


class TestVisualizer(unittest.TestCase):
    """Test suite for Unicode Braille rendering and timing HUD formatting."""

    def test_braille_canvas_set_pixel_and_render(self) -> None:
        """Verify sub-pixel setting and conversion to Braille characters."""
        canvas = BrailleCanvas(char_width=5, char_height=2)
        # Empty canvas should only have blank Braille characters U+2800
        rendered = canvas.render()
        self.assertEqual(len(rendered.split("\n")), 2)
        for ch in rendered.replace("\n", ""):
            self.assertEqual(ord(ch), 0x2800)

        # Set top-left pixel (0,0) -> dot 1 (0x01) -> 0x2801
        canvas.set_pixel(0, 0)
        rendered_one = canvas.render()
        first_char = rendered_one.split("\n")[0][0]
        self.assertEqual(ord(first_char), 0x2801)

    def test_braille_delay_curve_render(self) -> None:
        """Verify generation of Braille delay curves."""
        delays = [10.0, 25.0, 40.0, 60.0, 95.0]
        braille_str = CircuitVisualizer.render_delay_curve_braille(delays, width=20, height=4)
        self.assertGreater(len(braille_str), 0)
        self.assertEqual(len(braille_str.split("\n")), 4)

    def test_timing_hud_and_waterfall(self) -> None:
        """Verify rendering of telemetry HUD and critical path waterfall."""
        lib = get_default_library()
        mapper = TechMapper(lib)

        aig = AIGGraph()
        a = aig.create_pi("A")
        b = aig.create_pi("B")
        out = aig.and_(a, b)
        aig.set_output("Y", out)

        netlist = mapper.map_aig(aig, module_name="hud_test")
        sta = StaticTimingAnalyzer(netlist)
        report = sta.run_sta(clock_period=1000.0)

        hud = CircuitVisualizer.render_timing_hud(netlist, report, target_name="Test Block")
        self.assertIn("LOGICCRAFT EDA SYNTHESIS & STA TELEMETRY", hud)
        self.assertIn("Module Name:", hud)
        self.assertIn("Physical Area:", hud)

        waterfall = CircuitVisualizer.render_critical_path_waterfall(report)
        self.assertIn("CRITICAL PATH TIMING WATERFALL", waterfall)


if __name__ == "__main__":
    unittest.main()
