"""
SynapseDB: Unit Tests for SQL Parser and Query Planner.
"""

import unittest
from synapse.types import DataType
from synapse.storage import ColumnarTable, TableSchema
from synapse.parser import parse_sql, ColumnRef, AggregateExpr
from synapse.planner import (
    QueryPlanner,
    LogicalScan,
    LogicalAggregate,
    LogicalFilter,
    LogicalHashJoin
)


class TestParserPlanner(unittest.TestCase):
    def setUp(self):
        schema = TableSchema({
            "id": DataType.INT64,
            "region": DataType.VARCHAR,
            "sales": DataType.FLOAT64
        })
        self.table = ColumnarTable("orders", schema)
        self.catalog = {"orders": self.table}

    def test_sql_parser_basic(self):
        sql = "SELECT id, region FROM orders WHERE sales > 100.0 ORDER BY id DESC LIMIT 10"
        ast = parse_sql(sql)

        self.assertEqual(ast.from_table, "orders")
        self.assertEqual(len(ast.projections), 2)
        self.assertEqual(ast.limit, 10)
        self.assertEqual(ast.order_by, [("id", True)])

    def test_sql_parser_aggregates_group_by(self):
        sql = "SELECT region, SUM(sales) AS total_rev, COUNT(*) AS count_orders FROM orders GROUP BY region"
        ast = parse_sql(sql)

        self.assertEqual(len(ast.group_by), 1)
        self.assertEqual(ast.group_by[0].name, "region")
        self.assertEqual(len(ast.projections), 3)

        # First projection: region
        self.assertEqual(ast.projections[0].output_name(), "region")
        # Second: SUM(sales) as total_rev
        self.assertEqual(ast.projections[1].alias, "total_rev")
        # Third: COUNT(*) as count_orders
        self.assertEqual(ast.projections[2].alias, "count_orders")

    def test_planner_predicate_pushdown_and_pruning(self):
        sql = "SELECT region, sales FROM orders WHERE sales > 50.0"
        ast = parse_sql(sql)
        planner = QueryPlanner(self.catalog)
        plan = planner.plan(ast)

        # Plan root should have pushed down predicate into LogicalScan
        curr = plan
        while not isinstance(curr, LogicalScan):
            curr = curr.input_node

        self.assertIsInstance(curr, LogicalScan)
        self.assertEqual(curr.pushed_predicates, [("sales", ">", 50.0)])
        # Projection pruning: id is not queried, so only region and sales should be scanned!
        self.assertNotIn("id", curr.projected_columns)
        self.assertIn("region", curr.projected_columns)
        self.assertIn("sales", curr.projected_columns)


if __name__ == "__main__":
    unittest.main()
