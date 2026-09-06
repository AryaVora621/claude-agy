"""Unit tests for Standard Cell Library and Liberty NLDM Timing Models."""

import unittest
from logiccraft.liberty import (
    LookupTable2D,
    LibertyLibrary,
    get_default_library,
)


class TestLibertyModels(unittest.TestCase):
    """Test suite verifying Liberty standard cell definitions and 2D bilinear interpolation."""

    def setUp(self) -> None:
        self.lib = get_default_library()

    def test_default_cells_available(self) -> None:
        """Verify presence of core logic primitives in standard cell library."""
        expected_cells = [
            "INV_X1", "INV_X2", "BUF_X1", "BUF_X2",
            "NAND2_X1", "NOR2_X1", "AND2_X1", "OR2_X1",
            "XOR2_X1", "AOI21_X1", "OAI21_X1", "DFF_X1"
        ]
        for name in expected_cells:
            cell = self.lib.get_cell(name)
            self.assertIsNotNone(cell)
            self.assertGreater(cell.area, 0.0)
            self.assertGreater(cell.leakage_power, 0.0)
            self.assertTrue("Y" in cell.outputs or "Q" in cell.outputs)

    def test_bilinear_interpolation_grid_exact(self) -> None:
        """Verify exact lookup when test points coincide with table indices."""
        x_indices = [10.0, 50.0]
        y_indices = [2.0, 10.0]
        # values[row_x][col_y]
        values = [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
        lut = LookupTable2D(x_indices, y_indices, values)

        # Exact grid matches
        self.assertAlmostEqual(lut.lookup(10.0, 2.0), 10.0)
        self.assertAlmostEqual(lut.lookup(10.0, 10.0), 20.0)
        self.assertAlmostEqual(lut.lookup(50.0, 2.0), 30.0)
        self.assertAlmostEqual(lut.lookup(50.0, 10.0), 40.0)

    def test_bilinear_interpolation_midpoint(self) -> None:
        """Verify intermediate interpolation at center point."""
        x_indices = [10.0, 50.0]
        y_indices = [2.0, 10.0]
        values = [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
        lut = LookupTable2D(x_indices, y_indices, values)

        # Midpoint x=30.0 (halfway), y=6.0 (halfway)
        # Should evaluate to average of all 4 corners = (10+20+30+40)/4 = 25.0
        val = lut.lookup(30.0, 6.0)
        self.assertAlmostEqual(val, 25.0)

    def test_nldm_delay_scaling_with_load(self) -> None:
        """Verify propagation delay monotonically increases with capacitive load."""
        inv_cell = self.lib.get_cell("INV_X1")
        arc = inv_cell.timing_arcs[0]

        # Fix input slew at 20 ps, increase capacitive load from 1 fF to 30 fF
        d_light = arc.lookup_delay(input_slew=20.0, load_cap=1.0)
        d_medium = arc.lookup_delay(input_slew=20.0, load_cap=10.0)
        d_heavy = arc.lookup_delay(input_slew=20.0, load_cap=30.0)

        self.assertLess(d_light, d_medium)
        self.assertLess(d_medium, d_heavy)


if __name__ == "__main__":
    unittest.main()
