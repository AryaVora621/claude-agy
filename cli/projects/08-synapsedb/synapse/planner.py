"""
SynapseDB: Logical Query Plan and Rule-Based Optimizer.
Features:
1. Translates SQL AST into a hierarchical tree of Logical Operators.
2. Predicate Pushdown: pushes simple column filters directly into table scan.
3. Projection Pruning: analyzes query AST and restricts scans to strictly needed columns.
4. Join Graph Construction: arranges hash join operators.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from synapse.parser import (
    SelectStatement,
    ASTNode,
    BinaryOp,
    ColumnRef,
    Literal,
    AggregateExpr,
    ProjectedItem
)
from synapse.storage import ColumnarTable


class LogicalPlanNode:
    def explain(self, indent: int = 0) -> str:
        raise NotImplementedError


class LogicalScan(LogicalPlanNode):
    def __init__(
        self,
        table_name: str,
        table: ColumnarTable,
        projected_columns: List[str],
        pushed_predicates: List[Tuple[str, str, Any]]
    ):
        self.table_name = table_name
        self.table = table
        self.projected_columns = projected_columns
        self.pushed_predicates = pushed_predicates

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        pred_str = f" filters={self.pushed_predicates}" if self.pushed_predicates else ""
        return f"{sp}-> LogicalScan: {self.table_name} [cols={self.projected_columns}{pred_str}]"


class LogicalFilter(LogicalPlanNode):
    def __init__(self, input_node: LogicalPlanNode, condition: ASTNode):
        self.input_node = input_node
        self.condition = condition

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}-> LogicalFilter: {self.condition}\n{self.input_node.explain(indent + 1)}"


class LogicalHashJoin(LogicalPlanNode):
    def __init__(
        self,
        left: LogicalPlanNode,
        right: LogicalPlanNode,
        join_type: str,
        left_key: str,
        right_key: str
    ):
        self.left = left
        self.right = right
        self.join_type = join_type
        self.left_key = left_key
        self.right_key = right_key

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return (
            f"{sp}-> LogicalHashJoin: {self.join_type} ON {self.left_key} = {self.right_key}\n"
            f"{self.left.explain(indent + 1)}\n"
            f"{self.right.explain(indent + 1)}"
        )


class LogicalAggregate(LogicalPlanNode):
    def __init__(
        self,
        input_node: LogicalPlanNode,
        group_by: List[str],
        aggregates: List[Tuple[str, Optional[str], str]]  # (func, arg_col, output_alias)
    ):
        self.input_node = input_node
        self.group_by = group_by
        self.aggregates = aggregates

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return (
            f"{sp}-> LogicalAggregate: group_by={self.group_by} agg={self.aggregates}\n"
            f"{self.input_node.explain(indent + 1)}"
        )


class LogicalSort(LogicalPlanNode):
    def __init__(self, input_node: LogicalPlanNode, order_by: List[Tuple[str, bool]]):
        self.input_node = input_node
        self.order_by = order_by

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}-> LogicalSort: {self.order_by}\n{self.input_node.explain(indent + 1)}"


class LogicalLimit(LogicalPlanNode):
    def __init__(self, input_node: LogicalPlanNode, limit: int):
        self.input_node = input_node
        self.limit = limit

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}-> LogicalLimit: {self.limit}\n{self.input_node.explain(indent + 1)}"


class LogicalProject(LogicalPlanNode):
    def __init__(self, input_node: LogicalPlanNode, projections: List[Tuple[str, str]]):
        self.input_node = input_node
        self.projections = projections  # (source_name, output_alias)

    def explain(self, indent: int = 0) -> str:
        sp = "  " * indent
        return f"{sp}-> LogicalProject: {self.projections}\n{self.input_node.explain(indent + 1)}"


class QueryPlanner:
    """Compiles SQL AST into an optimized Logical Plan."""

    def __init__(self, catalog: Dict[str, ColumnarTable]):
        self.catalog = catalog

    def plan(self, ast: SelectStatement) -> LogicalPlanNode:
        if ast.from_table not in self.catalog:
            raise KeyError(f"Table '{ast.from_table}' not found in catalog")

        base_table = self.catalog[ast.from_table]

        # 1. Collect all referenced column names across query to prune scan
        referenced_cols: Set[str] = set()
        for p in ast.projections:
            self._collect_columns(p.expr, referenced_cols)
        if ast.where:
            self._collect_columns(ast.where, referenced_cols)
        for join in ast.joins:
            self._collect_columns(join.condition, referenced_cols)
        for g in ast.group_by:
            referenced_cols.add(g.name)
        if ast.having:
            self._collect_columns(ast.having, referenced_cols)
        for ob_col, _ in ast.order_by:
            referenced_cols.add(ob_col)

        # Handle wildcard SELECT *
        for p in ast.projections:
            if isinstance(p.expr, ColumnRef) and p.expr.name == "*":
                referenced_cols = set(base_table.schema.columns)
                break

        # 2. Extract pushable predicates from WHERE
        pushed_predicates: List[Tuple[str, str, Any]] = []
        remaining_where = None

        if ast.where:
            pushed_predicates, remaining_where = self._extract_predicates(ast.where)

        scan_cols = [c for c in base_table.schema.columns if c in referenced_cols]
        if not scan_cols:
            scan_cols = [base_table.schema.columns[0]]  # At least one column for row counting

        root: LogicalPlanNode = LogicalScan(
            table_name=ast.from_table,
            table=base_table,
            projected_columns=scan_cols,
            pushed_predicates=pushed_predicates
        )

        # 3. Apply remaining unpushed filter
        if remaining_where is not None:
            root = LogicalFilter(root, remaining_where)

        # 4. Joins
        for join in ast.joins:
            if join.table not in self.catalog:
                raise KeyError(f"Joined table '{join.table}' not found in catalog")
            join_table = self.catalog[join.table]

            cond = join.condition
            if not isinstance(cond.left, ColumnRef) or not isinstance(cond.right, ColumnRef):
                raise NotImplementedError("Only column equality joins currently supported")

            left_key = cond.left.name
            right_key = cond.right.name

            join_scan = LogicalScan(
                table_name=join.table,
                table=join_table,
                projected_columns=list(join_table.schema.columns),
                pushed_predicates=[]
            )
            root = LogicalHashJoin(root, join_scan, join.join_type, left_key, right_key)

        # 5. Aggregation
        has_aggregates = any(isinstance(p.expr, AggregateExpr) for p in ast.projections)
        has_group_by = len(ast.group_by) > 0

        if has_aggregates or has_group_by:
            group_cols = [g.name for g in ast.group_by]
            aggs = []
            for p in ast.projections:
                if isinstance(p.expr, AggregateExpr):
                    arg_name = p.expr.arg.name if isinstance(p.expr.arg, ColumnRef) else None
                    aggs.append((p.expr.func, arg_name, p.output_name()))
            root = LogicalAggregate(root, group_cols, aggs)

        # 6. Sorting
        if ast.order_by:
            root = LogicalSort(root, ast.order_by)

        # 7. Final Projection
        final_projs = []
        for p in ast.projections:
            if isinstance(p.expr, ColumnRef) and p.expr.name == "*":
                # Expand wildcard
                for c in base_table.schema.columns:
                    final_projs.append((c, c))
            else:
                out_name = p.output_name()
                src_name = p.expr.name if isinstance(p.expr, ColumnRef) else out_name
                final_projs.append((src_name, out_name))

        root = LogicalProject(root, final_projs)

        # 8. Limit
        if ast.limit is not None:
            root = LogicalLimit(root, ast.limit)

        return root

    def _collect_columns(self, node: Optional[ASTNode], cols: Set[str]) -> None:
        if node is None:
            return
        if isinstance(node, ColumnRef):
            cols.add(node.name)
        elif isinstance(node, BinaryOp):
            self._collect_columns(node.left, cols)
            self._collect_columns(node.right, cols)
        elif isinstance(node, AggregateExpr):
            self._collect_columns(node.arg, cols)

    def _extract_predicates(self, node: ASTNode) -> Tuple[List[Tuple[str, str, Any]], Optional[ASTNode]]:
        predicates = []
        # Check if root is a simple binary comparison: col op literal
        if isinstance(node, BinaryOp):
            if node.op in ("=", "!=", "<", "<=", ">", ">="):
                if isinstance(node.left, ColumnRef) and isinstance(node.right, Literal):
                    predicates.append((node.left.name, node.op, node.right.value))
                    return predicates, None
                elif isinstance(node.right, ColumnRef) and isinstance(node.left, Literal):
                    # Flip operator
                    flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}
                    predicates.append((node.right.name, flip[node.op], node.left.value))
                    return predicates, None
            elif node.op == "AND":
                left_preds, left_rem = self._extract_predicates(node.left)
                right_preds, right_rem = self._extract_predicates(node.right)
                predicates.extend(left_preds)
                predicates.extend(right_preds)
                if left_rem and right_rem:
                    return predicates, BinaryOp(left_rem, "AND", right_rem)
                elif left_rem:
                    return predicates, left_rem
                elif right_rem:
                    return predicates, right_rem
                else:
                    return predicates, None

        return predicates, node
