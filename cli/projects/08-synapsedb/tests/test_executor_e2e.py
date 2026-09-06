"""
SynapseDB: End-to-End Vectorized SQL Execution Tests.
"""

import unittest
from synapse.types import DataType
from synapse.storage import ColumnarTable, TableSchema
from synapse.executor import Engine


class TestExecutorEndToEnd(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()

        # Orders Table
        order_schema = TableSchema({
            "order_id": DataType.INT64,
            "cust_id": DataType.INT64,
            "region": DataType.VARCHAR,
            "amount": DataType.FLOAT64
        })
        self.orders = ColumnarTable("orders", order_schema, chunk_size=3)
        self.orders.insert_rows([
            {"order_id": 1, "cust_id": 101, "region": "East", "amount": 100.0},
            {"order_id": 2, "cust_id": 102, "region": "West", "amount": 250.0},
            {"order_id": 3, "cust_id": 101, "region": "East", "amount": 150.0},
            {"order_id": 4, "cust_id": 103, "region": "Central", "amount": 300.0},
            {"order_id": 5, "cust_id": 102, "region": "West", "amount": 50.0},
            {"order_id": 6, "cust_id": 104, "region": "East", "amount": 400.0}
        ])
        self.engine.register_table(self.orders)

        # Customers Table
        cust_schema = TableSchema({
            "cust_id": DataType.INT64,
            "cust_name": DataType.VARCHAR,
            "tier": DataType.VARCHAR
        })
        self.customers = ColumnarTable("customers", cust_schema, chunk_size=3)
        self.customers.insert_rows([
            {"cust_id": 101, "cust_name": "Alice", "tier": "Gold"},
            {"cust_id": 102, "cust_name": "Bob", "tier": "Silver"},
            {"cust_id": 103, "cust_name": "Charlie", "tier": "Bronze"},
            {"cust_id": 104, "cust_name": "Diana", "tier": "Platinum"}
        ])
        self.engine.register_table(self.customers)

    def test_simple_select_filter(self):
        res = self.engine.execute("SELECT order_id, amount FROM orders WHERE amount > 200.0")
        self.assertEqual(len(res), 3)
        amounts = sorted(res.column("amount").to_list())
        self.assertEqual(amounts, [250.0, 300.0, 400.0])

    def test_aggregation_and_group_by(self):
        sql = (
            "SELECT region, SUM(amount) AS total_sales, COUNT(*) AS num_orders, AVG(amount) AS avg_sales "
            "FROM orders GROUP BY region ORDER BY region ASC"
        )
        res = self.engine.execute(sql)
        self.assertEqual(len(res), 3)

        rows = res.to_rows()
        # Sort order: Central, East, West
        self.assertEqual(rows[0]["region"], "Central")
        self.assertEqual(rows[0]["total_sales"], 300.0)
        self.assertEqual(rows[0]["num_orders"], 1)

        self.assertEqual(rows[1]["region"], "East")
        self.assertEqual(rows[1]["total_sales"], 650.0)
        self.assertEqual(rows[1]["num_orders"], 3)
        self.assertAlmostEqual(rows[1]["avg_sales"], 650.0 / 3)

        self.assertEqual(rows[2]["region"], "West")
        self.assertEqual(rows[2]["total_sales"], 300.0)
        self.assertEqual(rows[2]["num_orders"], 2)

    def test_hash_join(self):
        sql = (
            "SELECT order_id, amount, cust_name, tier "
            "FROM orders JOIN customers ON cust_id = cust_id "
            "WHERE amount >= 250.0"
        )
        res = self.engine.execute(sql)
        self.assertEqual(len(res), 3)

        rows = res.to_rows()
        names = {r["cust_name"] for r in rows}
        self.assertEqual(names, {"Bob", "Charlie", "Diana"})

    def test_limit_and_order_by(self):
        sql = "SELECT order_id, amount FROM orders ORDER BY amount DESC LIMIT 2"
        res = self.engine.execute(sql)
        self.assertEqual(len(res), 2)
        self.assertEqual(res.column("amount").to_list(), [400.0, 300.0])


if __name__ == "__main__":
    unittest.main()
