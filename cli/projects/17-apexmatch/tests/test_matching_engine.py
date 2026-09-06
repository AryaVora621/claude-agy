"""
Unit tests for ApexMatch Continuous Matching Engine, Order Types, and Risk Controls.
"""

import unittest
from apexmatch.types import Side, OrderType, TimeInForce, STPMode, price_to_int, int_to_price
from apexmatch.order import OrderNode
from apexmatch.matching_engine import MatchingEngine
from apexmatch.risk import RiskConfig


class TestMatchingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MatchingEngine("AAPL")

    def test_passive_resting_orders(self):
        b1 = OrderNode(1, "Trader1", "AAPL", Side.BUY, price_to_int(150.00), 100)
        accepted, trades, msg = self.engine.process_order(b1)
        self.assertTrue(accepted)
        self.assertEqual(len(trades), 0)
        self.assertEqual(self.engine.book.best_bid(), price_to_int(150.00))

    def test_aggressive_limit_order_full_fill(self):
        # Resting Ask: 100 shares @ 150.00
        a1 = OrderNode(1, "Seller", "AAPL", Side.SELL, price_to_int(150.00), 100)
        self.engine.process_order(a1)

        # Aggressive Bid: 100 shares @ 150.50
        b1 = OrderNode(2, "Buyer", "AAPL", Side.BUY, price_to_int(150.50), 100)
        accepted, trades, msg = self.engine.process_order(b1)

        self.assertTrue(accepted)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.maker_order_id, 1)
        self.assertEqual(t.taker_order_id, 2)
        # Execution price must be maker's price (150.00), NOT taker's price!
        self.assertEqual(t.price, price_to_int(150.00))
        self.assertEqual(t.qty, 100)

        # Both orders fully filled; book is empty
        self.assertIsNone(self.engine.book.best_bid())
        self.assertIsNone(self.engine.book.best_ask())

    def test_partial_fill_and_resting_remainder(self):
        # Resting Ask: 50 shares @ 150.00
        a1 = OrderNode(1, "Seller", "AAPL", Side.SELL, price_to_int(150.00), 50)
        self.engine.process_order(a1)

        # Aggressive Bid: 120 shares @ 150.00
        b1 = OrderNode(2, "Buyer", "AAPL", Side.BUY, price_to_int(150.00), 120)
        accepted, trades, msg = self.engine.process_order(b1)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].qty, 50)
        # Remainder (70 shares) should rest as best bid
        self.assertEqual(self.engine.book.best_bid(), price_to_int(150.00))
        self.assertEqual(self.engine.book.best_bid_volume(), 70)

    def test_ioc_order(self):
        # Resting Ask: 40 shares @ 150.00
        a1 = OrderNode(1, "Seller", "AAPL", Side.SELL, price_to_int(150.00), 40)
        self.engine.process_order(a1)

        # IOC Bid for 100 shares
        b1 = OrderNode(2, "Buyer", "AAPL", Side.BUY, price_to_int(150.00), 100, order_type=OrderType.IOC)
        accepted, trades, msg = self.engine.process_order(b1)

        self.assertTrue(accepted)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].qty, 40)
        # Remaining 60 shares should NOT rest on book
        self.assertIsNone(self.engine.book.best_bid())

    def test_fok_order(self):
        # Resting Ask: 60 shares @ 150.00
        a1 = OrderNode(1, "Seller", "AAPL", Side.SELL, price_to_int(150.00), 60)
        self.engine.process_order(a1)

        # FOK Bid for 100 shares (only 60 available -> REJECT!)
        fok_fail = OrderNode(2, "Buyer", "AAPL", Side.BUY, price_to_int(150.00), 100, order_type=OrderType.FOK)
        accepted, trades, msg = self.engine.process_order(fok_fail)
        self.assertFalse(accepted)
        self.assertEqual(len(trades), 0)
        self.assertIn("REJECT_FOK_INSUFFICIENT_LIQUIDITY", msg)
        # Resting ask is untouched
        self.assertEqual(self.engine.book.best_ask_volume(), 60)

        # FOK Bid for 50 shares (60 available -> SUCCEED!)
        fok_success = OrderNode(3, "Buyer", "AAPL", Side.BUY, price_to_int(150.00), 50, order_type=OrderType.FOK)
        accepted, trades, msg = self.engine.process_order(fok_success)
        self.assertTrue(accepted)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].qty, 50)
        self.assertEqual(self.engine.book.best_ask_volume(), 10)

    def test_iceberg_order_replenishment(self):
        # Iceberg Sell: 100 visible, 400 hidden reserve (500 total) @ 100.00
        iceberg = OrderNode(
            order_id=1,
            participant_id="Whale",
            symbol="AAPL",
            side=Side.SELL,
            price=price_to_int(100.00),
            qty=100,
            is_iceberg=True,
            visible_peak_qty=100,
            total_reserve_qty=400
        )
        self.engine.process_order(iceberg)
        # Visible volume in book should only be 100
        self.assertEqual(self.engine.book.best_ask_volume(), 100)

        # Buyer 1 buys 100 shares -> depletes visible peak -> triggers replenishment
        b1 = OrderNode(2, "Retail1", "AAPL", Side.BUY, price_to_int(100.00), 100)
        self.engine.process_order(b1)

        # Book should immediately replenish with next 100 shares from reserve
        self.assertEqual(self.engine.book.best_ask_volume(), 100)
        self.assertEqual(iceberg.total_reserve_qty, 300)

    def test_self_trade_prevention(self):
        # Resting Ask by Firm A
        a1 = OrderNode(1, "FirmA", "AAPL", Side.SELL, price_to_int(100.00), 100)
        self.engine.process_order(a1)

        # Aggressive Bid by SAME Firm A
        b1 = OrderNode(2, "FirmA", "AAPL", Side.BUY, price_to_int(100.00), 50)
        accepted, trades, msg = self.engine.process_order(b1)

        # Zero trades should occur under STP!
        self.assertEqual(len(trades), 0)

    def test_price_collar_rejection(self):
        # Set up a market at 100.00
        self.engine.process_order(OrderNode(1, "P1", "AAPL", Side.BUY, price_to_int(99.00), 100))
        self.engine.process_order(OrderNode(2, "P2", "AAPL", Side.SELL, price_to_int(101.00), 100))
        # Mid price = 100.00, collar = 10% (90.00 to 110.00)

        # Fat-finger order at 150.00 (> 10% collar!)
        bad_bid = OrderNode(3, "FatFinger", "AAPL", Side.BUY, price_to_int(150.00), 100)
        accepted, trades, msg = self.engine.process_order(bad_bid)
        self.assertFalse(accepted)
        self.assertIn("REJECT_PRICE_COLLAR_VIOLATION", msg)


if __name__ == "__main__":
    unittest.main()
