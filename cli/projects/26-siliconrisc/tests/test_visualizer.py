"""Unit tests for BrailleCanvas, PipelineHUD, RegisterInspector, and Workbench Dashboard."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.core import CPU, ExecutionMode
from siliconrisc.visualizer import (
    BrailleCanvas,
    CacheTelemetryHUD,
    PipelineHUD,
    RegisterInspector,
    SystemWorkbenchDashboard,
)


class TestVisualizer(unittest.TestCase):
    """Test Braille plotting and terminal HUD rendering."""

    def test_braille_canvas_basic_dots(self) -> None:
        canvas = BrailleCanvas(char_width=4, char_height=2)
        # Pixel dimensions: 8 x 8
        self.assertEqual(canvas.pixel_width, 8)
        self.assertEqual(canvas.pixel_height, 8)

        # Set pixel at (0, 0) -> dot 0 (0x01) -> U+2801 (⠁)
        canvas.set_pixel(0, 0)
        output = canvas.render()
        first_char = output.splitlines()[0][0]
        self.assertEqual(ord(first_char), 0x2801)

    def test_braille_canvas_sparkline(self) -> None:
        canvas = BrailleCanvas(char_width=10, char_height=3)
        data = [1.0, 1.2, 0.8, 1.5, 2.0, 1.1, 0.5, 1.8]
        canvas.plot_sparkline(data, min_val=0.0, max_val=2.0)
        rendered = canvas.render()
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(len(line), 10)

    def test_pipeline_hud_and_dashboard_render(self) -> None:
        cpu = CPU(mode=ExecutionMode.PIPELINED, initial_pc=0x80000000)
        # Load simple instructions
        insts = [
            0x00A00093,  # addi x1, x0, 10
            0x01400113,  # addi x2, x0, 20
            0x00100073,  # ebreak
        ]
        cpu.load_program(0x80000000, insts)
        status = cpu.step()

        # Render PipelineHUD
        hud = PipelineHUD.render(cpu.pipeline, status)
        self.assertIn("[IF] FETCH", hud)
        self.assertIn("[ID] DECODE", hud)
        self.assertIn("[EX] EXEC", hud)
        self.assertIn("[MEM] MEMORY", hud)
        self.assertIn("[WB] WRITE", hud)

        # Render RegisterInspector
        reg_str = RegisterInspector.render(cpu.pipeline.reg_file)
        self.assertIn("ra (x01)", reg_str)
        self.assertIn("sp (x02)", reg_str)

        # Render CacheTelemetryHUD
        cache_str = CacheTelemetryHUD.render(cpu)
        self.assertIn("L1-I Cache", cache_str)
        self.assertIn("L1-D Cache", cache_str)

        # Render SystemWorkbenchDashboard
        dashboard = SystemWorkbenchDashboard(cpu)
        rendered_dash = dashboard.render(status)
        self.assertIn("SILICONRISC RV64GC HARDWARE WORKBENCH", rendered_dash)


if __name__ == "__main__":
    unittest.main()
