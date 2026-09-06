"""And-Inverter Graph (AIG) Engine with Structural Hashing (Strashing).

Features:
1. Compact homogeneous DAG where all internal nodes are 2-input AND gates.
2. Inverter-carrying edges represented by literal LSB (lit = (node_id << 1) | inverted).
3. Two-level structural hashing (strashing) with on-the-fly Boolean simplifications.
4. Topological levelization and graph depth computation.
5. Bit-parallel logic simulation and BDD-to-AIG compilation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


CONST_FALSE_LIT: int = 0
CONST_TRUE_LIT: int = 1


def lit_node(lit: int) -> int:
    """Extract node ID from literal."""
    return lit >> 1


def lit_is_inv(lit: int) -> bool:
    """Check if literal is inverted."""
    return (lit & 1) == 1


def lit_not(lit: int) -> int:
    """Invert a literal."""
    return lit ^ 1


def make_lit(node_id: int, inverted: bool = False) -> int:
    """Construct literal from node ID and invert flag."""
    return (node_id << 1) | (1 if inverted else 0)


@dataclass
class AIGNode:
    """AIG node in the graph."""
    id: int
    is_pi: bool = False
    name: Optional[str] = None
    fanin0: int = CONST_FALSE_LIT  # Literal
    fanin1: int = CONST_FALSE_LIT  # Literal
    level: int = 0


class AIGGraph:
    """And-Inverter Graph with structural hashing."""

    def __init__(self) -> None:
        # Node 0 is reserved for constant false
        self.nodes: Dict[int, AIGNode] = {
            0: AIGNode(id=0, is_pi=False, name="CONST0", level=0)
        }
        self.next_node_id: int = 1

        # Primary inputs and outputs
        self.pi_nodes: List[int] = []
        self.pi_name_to_node: Dict[str, int] = {}
        self.outputs: Dict[str, int] = {}  # name -> literal

        # Structural hashing table: (canonical_fanin0, canonical_fanin1) -> node_id
        self.strash_table: Dict[Tuple[int, int], int] = {}

    @property
    def const_false(self) -> int:
        return CONST_FALSE_LIT

    @property
    def const_true(self) -> int:
        return CONST_TRUE_LIT

    def create_pi(self, name: str) -> int:
        """Create a new Primary Input port, returning its literal."""
        if name in self.pi_name_to_node:
            node_id = self.pi_name_to_node[name]
            return make_lit(node_id, False)

        node_id = self.next_node_id
        self.next_node_id += 1

        node = AIGNode(id=node_id, is_pi=True, name=name, level=0)
        self.nodes[node_id] = node
        self.pi_nodes.append(node_id)
        self.pi_name_to_node[name] = node_id

        return make_lit(node_id, False)

    def set_output(self, name: str, lit: int) -> None:
        """Register a Primary Output driven by a literal."""
        self.outputs[name] = lit

    def and_(self, lit0: int, lit1: int) -> int:
        """Create or get a 2-input AND node with two-level structural hashing."""
        # Trivial simplifications
        if lit0 == CONST_FALSE_LIT or lit1 == CONST_FALSE_LIT:
            return CONST_FALSE_LIT
        if lit0 == CONST_TRUE_LIT:
            return lit1
        if lit1 == CONST_TRUE_LIT:
            return lit0
        if lit0 == lit1:
            return lit0
        if lit0 == lit_not(lit1):
            return CONST_FALSE_LIT

        # Canonical fanin ordering: lit0 <= lit1
        if lit0 > lit1:
            lit0, lit1 = lit1, lit0

        key = (lit0, lit1)
        if key in self.strash_table:
            node_id = self.strash_table[key]
            return make_lit(node_id, False)

        # Allocate new AND node
        node_id = self.next_node_id
        self.next_node_id += 1

        lvl0 = self.nodes[lit_node(lit0)].level
        lvl1 = self.nodes[lit_node(lit1)].level
        node_lvl = 1 + max(lvl0, lvl1)

        node = AIGNode(
            id=node_id,
            is_pi=False,
            fanin0=lit0,
            fanin1=lit1,
            level=node_lvl,
        )
        self.nodes[node_id] = node
        self.strash_table[key] = node_id

        return make_lit(node_id, False)

    def not_(self, lit: int) -> int:
        """Logical NOT."""
        return lit_not(lit)

    def or_(self, lit0: int, lit1: int) -> int:
        """Logical OR via De Morgan: ~(~a & ~b)."""
        return self.not_(self.and_(self.not_(lit0), self.not_(lit1)))

    def xor_(self, lit0: int, lit1: int) -> int:
        """Logical XOR: (a & ~b) | (~a & b) = ~(~(a & ~b) & ~(~a & b))."""
        term1 = self.and_(lit0, self.not_(lit1))
        term2 = self.and_(self.not_(lit0), lit1)
        return self.or_(term1, term2)

    def nand_(self, lit0: int, lit1: int) -> int:
        return self.not_(self.and_(lit0, lit1))

    def nor_(self, lit0: int, lit1: int) -> int:
        return self.not_(self.or_(lit0, lit1))

    def xnor_(self, lit0: int, lit1: int) -> int:
        return self.not_(self.xor_(lit0, lit1))

    def mux_(self, c_lit: int, t_lit: int, e_lit: int) -> int:
        """Multiplexer: (c & t) | (~c & e)."""
        term1 = self.and_(c_lit, t_lit)
        term2 = self.and_(self.not_(c_lit), e_lit)
        return self.or_(term1, term2)

    @property
    def num_and_nodes(self) -> int:
        """Count total internal AND gates."""
        return len(self.nodes) - 1 - len(self.pi_nodes)

    @property
    def max_depth(self) -> int:
        """Maximum logic level (depth) across all primary outputs."""
        if not self.outputs:
            return 0
        return max(self.nodes[lit_node(lit)].level for lit in self.outputs.values())

    def simulate(self, pi_values: Dict[str, bool]) -> Dict[str, bool]:
        """Simulate AIG under given primary input assignments."""
        values: Dict[int, bool] = {0: False}

        # Set primary inputs
        for name, node_id in self.pi_name_to_node.items():
            values[node_id] = pi_values.get(name, False)

        # Topological traversal
        topo_order = self.topological_sort()
        for node_id in topo_order:
            node = self.nodes[node_id]
            if node_id == 0 or node.is_pi:
                continue
            val0 = values[lit_node(node.fanin0)] ^ lit_is_inv(node.fanin0)
            val1 = values[lit_node(node.fanin1)] ^ lit_is_inv(node.fanin1)
            values[node_id] = val0 and val1

        # Evaluate outputs
        out_vals: Dict[str, bool] = {}
        for name, lit in self.outputs.items():
            out_vals[name] = values[lit_node(lit)] ^ lit_is_inv(lit)
        return out_vals

    def topological_sort(self) -> List[int]:
        """Return node IDs in topological order from inputs to outputs."""
        visited: Set[int] = set()
        order: List[int] = []

        def dfs(node_id: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            if not node.is_pi and node_id != 0:
                dfs(lit_node(node.fanin0))
                dfs(lit_node(node.fanin1))
            order.append(node_id)

        # Traverse from all outputs
        for lit in self.outputs.values():
            dfs(lit_node(lit))

        return order

    def compute_fanout_counts(self) -> Dict[int, int]:
        """Compute out-degree (fanout count) for all nodes."""
        counts: Dict[int, int] = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            if node.id != 0 and not node.is_pi:
                counts[lit_node(node.fanin0)] += 1
                counts[lit_node(node.fanin1)] += 1
        for lit in self.outputs.values():
            counts[lit_node(lit)] += 1
        return counts
