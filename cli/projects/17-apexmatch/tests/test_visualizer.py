"""
Unit tests for ApexMatch Terminal Visualizer.
"""

import unittest
from apexmatch.types import Side, price_to_int
from apexmatch.order import OrderNode
from apexmatch.matching_engine import MatchingEngine
from apexmatch.visualizer import render_order_book_ladder


class TestVisualizer(unittest.TestCase):
    def test_render_empty_ladder(self):
        engine = MatchingEngine("AAPL")
        output = render_order_book_ladder(engine)
        self.assertIn("APEXMATCH", output)
        self.assertIn("NO SPREAD", output)

    def test_render_populated_ladder_with_tape(self):
        engine = MatchingEngine("NVDA")
        # Add Bids and Asks
        engine.process_order(OrderNode(1, "B1", "NVDA", Side.BUY, price_to_int(120.00), 500))
        engine.process_order(OrderNode(2, "B2", "NVDA", Side.BUY, price_to_int(119.50), 1200))
        engine.process_order(OrderNode(3, "S1", "NVDA", Side.SELL, price_to_int(120.50), 800))
        engine.process_order(OrderNode(4, "S2", "NVDA", Side.SELL, price_to_int(121.00), 1500))

        # Cross a trade
        engine.process_order(OrderNode(5, "Taker", "NVDA", Side.BUY, price_to_int(120.50), 300))

        output = render_order_book_ladder(engine, depth=5, tape_depth=3)
        self.assertIn("NVDA", output)
        self.assertIn("BID", output)
        self.assertIn("ASK", output)
        self.assertIn("Match #", output)
        self.assertIn("SPREAD", output)


if __name__ == "__main__":
    unittest.main()
