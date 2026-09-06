#!/usr/bin/env python3
"""
SynapseDB: Interactive Columnar Analytics Dashboard & SQL Query Runner.
Features:
1. Automated TPC-H style e-commerce analytical queries over 25,000 columnar rows.
2. Formatted ASCII table outputs with column widths and execution metrics.
3. Logical Query Plan tree visualizer showing predicate pushdown and projection pruning.
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synapse.types import DataType, RecordBatch
from synapse.storage import ColumnarTable, TableSchema
from synapse.parser import parse_sql
from synapse.planner import QueryPlanner
from synapse.executor import Engine


# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def format_table(batch: RecordBatch, max_rows: int = 20) -> str:
    if len(batch) == 0:
        return f"{YELLOW}(0 rows returned){RESET}"

    rows = batch.to_rows()[:max_rows]
    cols = batch.names

    # Compute column widths
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            val_str = f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
            widths[c] = max(widths[c], len(val_str))

    # Header
    header_cells = [f" {c:<{widths[c]}} " for c in cols]
    header_line = f"|{'|'.join(header_cells)}|"
    sep_line = f"+{'+'.join('-' * (widths[c] + 2) for c in cols)}+"

    lines = [sep_line, header_line, sep_line]
    for r in rows:
        row_cells = []
        for c in cols:
            val = r[c]
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            row_cells.append(f" {val_str:<{widths[c]}} ")
        lines.append(f"|{'|'.join(row_cells)}|")
    lines.append(sep_line)

    if len(batch) > max_rows:
        lines.append(f"{YELLOW}(Showing first {max_rows} of {len(batch)} rows){RESET}")
    else:
        lines.append(f"{GREEN}({len(batch)} rows returned){RESET}")

    return "\n".join(lines)


def setup_ecommerce_database(num_orders: int = 25_000) -> Engine:
    print(f"[*] Generating {num_orders:,} row E-Commerce Columnar Dataset...")
    engine = Engine()

    # 1. Customers Table
    cust_schema = TableSchema({
        "cust_id": DataType.INT64,
        "cust_name": DataType.VARCHAR,
        "segment": DataType.VARCHAR,
        "country": DataType.VARCHAR
    })
    customers = ColumnarTable("customers", cust_schema, chunk_size=1024)

    segments = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
    countries = ["United States", "Germany", "United Kingdom", "Japan", "France"]
    cust_rows = []
    for c_id in range(1, 1001):
        cust_rows.append({
            "cust_id": c_id,
            "cust_name": f"Customer_{c_id:04d}",
            "segment": segments[c_id % len(segments)],
            "country": countries[c_id % len(countries)]
        })
    customers.insert_rows(cust_rows)
    engine.register_table(customers)

    # 2. Orders Table
    order_schema = TableSchema({
        "order_id": DataType.INT64,
        "cust_id": DataType.INT64,
        "channel": DataType.VARCHAR,
        "net_revenue": DataType.FLOAT64,
        "items_count": DataType.INT64
    })
    orders = ColumnarTable("orders", order_schema, chunk_size=2048)

    channels = ["Web", "Mobile_App", "API", "Direct"]
    order_rows = []
    random.seed(42)
    for o_id in range(1, num_orders + 1):
        order_rows.append({
            "order_id": o_id,
            "cust_id": random.randint(1, 1000),
            "channel": channels[random.randint(0, len(channels) - 1)],
            "net_revenue": round(random.uniform(15.0, 1500.0), 2),
            "items_count": random.randint(1, 12)
        })
    orders.insert_rows(order_rows)
    engine.register_table(orders)

    print(f"    -> Ingested {len(cust_rows):,} customers and {num_orders:,} orders into columnar storage.\n")
    return engine


def run_queries(engine: Engine) -> None:
    sample_queries = [
        (
            "Query 1: Channel Revenue Aggregation & Average Basket Size",
            (
                "SELECT channel, SUM(net_revenue) AS total_revenue, "
                "AVG(net_revenue) AS avg_order_val, COUNT(*) AS order_volume "
                "FROM orders GROUP BY channel ORDER BY total_revenue DESC"
            )
        ),
        (
            "Query 2: High-Value Order Filtering with Zone Map Pruning",
            (
                "SELECT order_id, channel, net_revenue, items_count "
                "FROM orders WHERE net_revenue > 1450.0 ORDER BY net_revenue DESC LIMIT 5"
            )
        ),
        (
            "Query 3: Relational Hash Join across Customers and Orders",
            (
                "SELECT order_id, cust_name, segment, country, net_revenue "
                "FROM orders JOIN customers ON cust_id = cust_id "
                "WHERE net_revenue > 1400.0 ORDER BY net_revenue DESC LIMIT 8"
            )
        )
    ]

    for title, sql in sample_queries:
        print(f"{BOLD}{CYAN}========================================================================{RESET}")
        print(f"{BOLD}{CYAN} {title}{RESET}")
        print(f"{BOLD}{CYAN}========================================================================{RESET}")
        print(f"{YELLOW}SQL:{RESET} {sql}\n")

        # Display optimized Logical Plan
        ast = parse_sql(sql)
        planner = QueryPlanner(engine.catalog)
        plan = planner.plan(ast)
        print(f"{BOLD}Execution Plan:{RESET}")
        print(plan.explain(0))
        print()

        t0 = time.perf_counter()
        result = engine.execute(sql)
        elapsed = time.perf_counter() - t0

        print(f"{BOLD}Query Result ({elapsed * 1000:.2f} ms):{RESET}")
        print(format_table(result))
        print("\n")


def main() -> None:
    print(f"{BOLD}{BLUE}========================================================================{RESET}")
    print(f"{BOLD}{BLUE}      SYNAPSEDB: COLUMNAR ANALYTICS ENGINE & VECTORIZED SQL RUNNER      {RESET}")
    print(f"{BOLD}{BLUE}========================================================================{RESET}\n")

    engine = setup_ecommerce_database(num_orders=25_000)
    run_queries(engine)


if __name__ == "__main__":
    main()
