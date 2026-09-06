"""Unit tests for Technology Mapping Engine."""

import unittest
from logiccraft.aig import AIGGraph
from logiccraft.liberty import get_default_library
from logiccraft.techmap import TechMapper


class TestTechMapper(unittest.TestCase):
    """Test suite for DAGON tree covering technology mapper."""

    def setUp(self) -> None:
        self.lib = get_default_library()
        self.mapper = TechMapper(self.lib, target_metric="area")

    def test_map_inverter_and_buffer(self) -> None:
        """Verify mapping simple NOT logic to INV_X1."""
        aig = AIGGraph()
        in_a = aig.create_pi("A")
        out_y = aig.not_(in_a)
        aig.set_output("Y", out_y)

        netlist = self.mapper.map_aig(aig, module_name="inv_test")
        self.assertGreaterEqual(netlist.num_cells, 1)
        self.assertIn("A", netlist.inputs)
        self.assertIn("Y", netlist.outputs)
        self.assertGreater(netlist.total_area, 0.0)

    def test_map_nand_and_nor_gates(self) -> None:
        """Verify pattern matching for standard CMOS complementary gates."""
        # NAND2 circuit
        aig_nand = AIGGraph()
        a = aig_nand.create_pi("A")
        b = aig_nand.create_pi("B")
        out_nand = aig_nand.nand_(a, b)
        aig_nand.set_output("Y", out_nand)

        netlist_nand = self.mapper.map_aig(aig_nand, module_name="nand_test")
        cell_types = [inst.cell_type.name for inst in netlist_nand.cells.values()]
        self.assertIn("NAND2_X1", cell_types)

        # NOR2 circuit
        aig_nor = AIGGraph()
        c = aig_nor.create_pi("A")
        d = aig_nor.create_pi("B")
        out_nor = aig_nor.nor_(c, d)
        aig_nor.set_output("Y", out_nor)

        netlist_nor = self.mapper.map_aig(aig_nor, module_name="nor_test")
        cell_types_nor = [inst.cell_type.name for inst in netlist_nor.cells.values()]
        self.assertIn("NOR2_X1", cell_types_nor)

    def test_map_verilog_export(self) -> None:
        """Verify structural Verilog netlist generation."""
        aig = AIGGraph()
        a = aig.create_pi("A")
        b = aig.create_pi("B")
        out_y = aig.and_(a, b)
        aig.set_output("Y", out_y)

        netlist = self.mapper.map_aig(aig, module_name="and_test")
        vlog = netlist.export_verilog()
        self.assertIn("module and_test", vlog)
        self.assertIn("input A, B;", vlog)
        self.assertIn("output Y;", vlog)
        self.assertIn("endmodule", vlog)

    def test_map_complex_logic(self) -> None:
        """Verify technology mapping for compound logic networks."""
        aig = AIGGraph()
        a = aig.create_pi("A")
        b = aig.create_pi("B")
        c = aig.create_pi("C")
        d = aig.create_pi("D")

        # F1 = (A & B) | (C & D)
        term1 = aig.and_(a, b)
        term2 = aig.and_(c, d)
        f1 = aig.or_(term1, term2)
        aig.set_output("F1", f1)

        netlist = self.mapper.map_aig(aig, module_name="complex_test")
        self.assertGreater(netlist.num_cells, 0)
        self.assertEqual(len(netlist.inputs), 4)
        self.assertEqual(len(netlist.outputs), 1)


if __name__ == "__main__":
    unittest.main()
