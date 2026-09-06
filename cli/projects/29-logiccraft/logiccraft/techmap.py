"""Technology Mapping Engine via Dynamic Programming Tree Covering.

Maps an unmapped And-Inverter Graph (AIG) onto physical standard cells
from a Liberty library, optimizing for either:
- Minimum silicon area (um^2)
- Minimum critical path delay (ps)

Implements the DAGON algorithm with pattern matching and dynamic programming.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .aig import (
    AIGGraph,
    AIGNode,
    CONST_FALSE_LIT,
    CONST_TRUE_LIT,
    lit_is_inv,
    lit_node,
    lit_not,
    make_lit,
)
from .liberty import LibertyLibrary, StandardCell, get_default_library
from .netlist import Netlist


@dataclass
class MatchResult:
    """Record of a successful cell pattern match at an AIG literal."""
    cell: StandardCell
    cost: float
    # Mapping of cell input pin names to driven AIG literals
    pin_to_lit: Dict[str, int]


class TechMapper:
    """Dynamic programming technology mapper."""

    def __init__(self, library: LibertyLibrary, target_metric: str = "area") -> None:
        self.library = library
        self.target_metric = target_metric  # "area" or "delay"
        self.best_matches: Dict[int, MatchResult] = {}  # lit -> MatchResult
        self.best_costs: Dict[int, float] = {}          # lit -> min cost

    def map_aig(self, aig: AIGGraph, module_name: str = "mapped_top") -> Netlist:
        """Map complete AIG onto library cells and return gate-level Netlist."""
        self.best_matches.clear()
        self.best_costs.clear()

        # Primary inputs have zero cost
        for name, node_id in aig.pi_name_to_node.items():
            pi_lit = make_lit(node_id, False)
            self.best_costs[pi_lit] = 0.0

            # Inverted PI cost is cost of an INV
            inv_cell = self.library.get_cell("INV_X1")
            inv_cost = inv_cell.area if self.target_metric == "area" else 15.0
            inv_pi_lit = make_lit(node_id, True)
            self.best_costs[inv_pi_lit] = inv_cost
            self.best_matches[inv_pi_lit] = MatchResult(
                cell=inv_cell,
                cost=inv_cost,
                pin_to_lit={"A": pi_lit},
            )

        # Topological traversal through AIG nodes
        topo_order = aig.topological_sort()
        for node_id in topo_order:
            if node_id == 0 or aig.nodes[node_id].is_pi:
                continue
            # Evaluate both non-inverted and inverted polarity of node
            lit_pos = make_lit(node_id, False)
            lit_neg = make_lit(node_id, True)

            self._match_literal(aig, lit_pos)
            self._match_literal(aig, lit_neg)

        # Emit Netlist from outputs
        netlist = Netlist(module_name)
        for name in aig.pi_name_to_node:
            netlist.add_input(name)
        for name in aig.outputs:
            netlist.add_output(name)

        inst_counter = 1
        lit_to_net: Dict[int, str] = {}

        # Primary inputs connect to their net names
        for name, node_id in aig.pi_name_to_node.items():
            lit_to_net[make_lit(node_id, False)] = name

        def emit_lit(lit: int) -> str:
            nonlocal inst_counter
            if lit in lit_to_net:
                return lit_to_net[lit]

            match = self.best_matches.get(lit)
            if match is None:
                # If no direct match, synthesize via inverter from opposite polarity
                opp_lit = lit_not(lit)
                opp_net = emit_lit(opp_lit)
                inv_cell = self.library.get_cell("INV_X1")
                inst_name = f"U{inst_counter}"
                inst_counter += 1
                out_net = f"n_{lit}"
                netlist.add_cell(inst_name, inv_cell, {"A": opp_net, "Y": out_net})
                lit_to_net[lit] = out_net
                return out_net

            # Recursively emit driver nets for cell inputs
            pin_conns: Dict[str, str] = {}
            for pin_name, leaf_lit in match.pin_to_lit.items():
                pin_conns[pin_name] = emit_lit(leaf_lit)

            inst_name = f"U{inst_counter}"
            inst_counter += 1
            out_net = f"n_{lit}"
            pin_conns["Y"] = out_net
            netlist.add_cell(inst_name, match.cell, pin_conns)
            lit_to_net[lit] = out_net
            return out_net

        # Connect primary outputs
        for po_name, po_lit in aig.outputs.items():
            driven_net = emit_lit(po_lit)
            # If output net name differs from driven net, add a buffer or alias
            if driven_net != po_name:
                buf_cell = self.library.get_cell("BUF_X1")
                inst_name = f"U{inst_counter}"
                inst_counter += 1
                netlist.add_cell(inst_name, buf_cell, {"A": driven_net, "Y": po_name})

        netlist.compute_net_capacitances()
        return netlist

    def _match_literal(self, aig: AIGGraph, target_lit: int) -> None:
        """Find best standard cell match covering target_lit."""
        best_match: Optional[MatchResult] = None
        best_cost = float("inf")

        # Try all pattern matchers
        matchers = [
            self._try_match_inv,
            self._try_match_nand2,
            self._try_match_and2,
            self._try_match_nor2,
            self._try_match_or2,
            self._try_match_aoi21,
            self._try_match_xor2,
        ]

        for matcher in matchers:
            match = matcher(aig, target_lit)
            if match and match.cost < best_cost:
                best_cost = match.cost
                best_match = match

        # Also consider inverting opposite polarity if cheaper
        opp_lit = lit_not(target_lit)
        if opp_lit in self.best_costs:
            inv_cell = self.library.get_cell("INV_X1")
            cost_via_inv = self.best_costs[opp_lit] + (
                inv_cell.area if self.target_metric == "area" else 15.0
            )
            if cost_via_inv < best_cost:
                best_cost = cost_via_inv
                best_match = MatchResult(
                    cell=inv_cell,
                    cost=best_cost,
                    pin_to_lit={"A": opp_lit},
                )

        if best_match:
            self.best_matches[target_lit] = best_match
            self.best_costs[target_lit] = best_cost

    def _get_cell_cost(self, cell: StandardCell) -> float:
        if self.target_metric == "area":
            return cell.area
        # Base intrinsic delay metric
        return cell.timing_arcs[0].cell_delay_table.lookup(30.0, 5.0) if cell.timing_arcs else 20.0

    # Pattern Matchers for standard gate configurations

    def _try_match_inv(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match INV_X1: Y = NOT(A)."""
        if not lit_is_inv(target_lit):
            return None
        child_lit = lit_not(target_lit)
        if child_lit not in self.best_costs:
            return None
        cell = self.library.get_cell("INV_X1")
        total_cost = self._get_cell_cost(cell) + self.best_costs[child_lit]
        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": child_lit})

    def _try_match_and2(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match AND2_X1: Y = AND(A, B)."""
        if lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        in0, in1 = node.fanin0, node.fanin1
        if in0 not in self.best_costs or in1 not in self.best_costs:
            return None
        cell = self.library.get_cell("AND2_X1")
        total_cost = self._get_cell_cost(cell) + self.best_costs[in0] + self.best_costs[in1]
        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": in0, "B": in1})

    def _try_match_nand2(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match NAND2_X1: Y = NOT(AND(A, B))."""
        if not lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        in0, in1 = node.fanin0, node.fanin1
        if in0 not in self.best_costs or in1 not in self.best_costs:
            return None
        cell = self.library.get_cell("NAND2_X1")
        total_cost = self._get_cell_cost(cell) + self.best_costs[in0] + self.best_costs[in1]
        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": in0, "B": in1})

    def _try_match_nor2(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match NOR2_X1: Y = ~(A | B) = (~A & ~B)."""
        if lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        # Must be AND of two inverted literals: in0 = ~A, in1 = ~B
        in0, in1 = node.fanin0, node.fanin1
        if not (lit_is_inv(in0) and lit_is_inv(in1)):
            return None
        leaf_a, leaf_b = lit_not(in0), lit_not(in1)
        if leaf_a not in self.best_costs or leaf_b not in self.best_costs:
            return None
        cell = self.library.get_cell("NOR2_X1")
        total_cost = self._get_cell_cost(cell) + self.best_costs[leaf_a] + self.best_costs[leaf_b]
        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": leaf_a, "B": leaf_b})

    def _try_match_or2(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match OR2_X1: Y = A | B = ~(~A & ~B)."""
        if not lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        in0, in1 = node.fanin0, node.fanin1
        if not (lit_is_inv(in0) and lit_is_inv(in1)):
            return None
        leaf_a, leaf_b = lit_not(in0), lit_not(in1)
        if leaf_a not in self.best_costs or leaf_b not in self.best_costs:
            return None
        cell = self.library.get_cell("OR2_X1")
        total_cost = self._get_cell_cost(cell) + self.best_costs[leaf_a] + self.best_costs[leaf_b]
        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": leaf_a, "B": leaf_b})

    def _try_match_aoi21(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match AOI21_X1: Y = ~((A1 & A2) | B) = ~(A1 & A2) & ~B."""
        if lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        # One input is ~(A1 & A2) (i.e. an inverted AND), other is ~B
        in0, in1 = node.fanin0, node.fanin1
        # Check both permutations
        for cand_nand, cand_b in [(in0, in1), (in1, in0)]:
            if lit_is_inv(cand_nand) and lit_is_inv(cand_b):
                nand_node = aig.nodes.get(lit_node(cand_nand))
                if nand_node and not nand_node.is_pi:
                    a1, a2 = nand_node.fanin0, nand_node.fanin1
                    leaf_b = lit_not(cand_b)
                    if a1 in self.best_costs and a2 in self.best_costs and leaf_b in self.best_costs:
                        cell = self.library.get_cell("AOI21_X1")
                        total_cost = (
                            self._get_cell_cost(cell)
                            + self.best_costs[a1]
                            + self.best_costs[a2]
                            + self.best_costs[leaf_b]
                        )
                        return MatchResult(
                            cell=cell,
                            cost=total_cost,
                            pin_to_lit={"A1": a1, "A2": a2, "B": leaf_b},
                        )
        return None

    def _try_match_xor2(self, aig: AIGGraph, target_lit: int) -> Optional[MatchResult]:
        """Match XOR2_X1: Y = (A & ~B) | (~A & B)."""
        # XOR2 matches AIG XOR structure
        if not lit_is_inv(target_lit):
            return None
        node = aig.nodes.get(lit_node(target_lit))
        if not node or node.is_pi:
            return None
        in0, in1 = node.fanin0, node.fanin1
        if not (lit_is_inv(in0) and lit_is_inv(in1)):
            return None
        n0 = aig.nodes.get(lit_node(in0))
        n1 = aig.nodes.get(lit_node(in1))
        if not n0 or n0.is_pi or not n1 or n1.is_pi:
            return None
        # n0 is (A & ~B), n1 is (~A & B)
        # Verify cross-inversion
        p0, p1 = n0.fanin0, n0.fanin1
        q0, q1 = n1.fanin0, n1.fanin1
        candidates = [(p0, p1), (p1, p0)]
        for cand_a, cand_nb in candidates:
            if lit_is_inv(cand_nb):
                cand_b = lit_not(cand_nb)
                # Check if q matches (~cand_a & cand_b)
                if (q0 == lit_not(cand_a) and q1 == cand_b) or (q1 == lit_not(cand_a) and q0 == cand_b):
                    if cand_a in self.best_costs and cand_b in self.best_costs:
                        cell = self.library.get_cell("XOR2_X1")
                        total_cost = self._get_cell_cost(cell) + self.best_costs[cand_a] + self.best_costs[cand_b]
                        return MatchResult(cell=cell, cost=total_cost, pin_to_lit={"A": cand_a, "B": cand_b})
        return None
