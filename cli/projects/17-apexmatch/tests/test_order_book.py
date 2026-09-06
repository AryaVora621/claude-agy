"""
Unit tests for ApexMatch Fixed-Point Arithmetic, OrderNode, and OrderBook.
"""

import unittest
from apexmatch.types import Side, OrderType, TimeInForce, price_to_int, int_to_price, format_price
from apexmatch.order import OrderNode, PriceLevel
from apexmatch.order_book import OrderBook


class TestOrderBook(unittest.TestCase):
    def test_fixed_point_precision(self):
        p_float = 150.2550
        p_int = price_to_int(p_float)
        self.assertEqual(p_int, 1502550)
        self.assertAlmostEqual(int_to_price(p_int), 150.2550, places=4)
        self.assertEqual(format_price(p_int), "$150.2550")

    def test_price_level_fifo_queue(self):
        level = PriceLevel(price=1000000)
        o1 = OrderNode(order_id=1, participant_id="P1", symbol="AAPL", side=Side.BUY, price=1000000, qty=100)
        o2 = OrderNode(order_id=2, participant_id="P2", symbol="AAPL", side=Side.BUY, price=1000000, qty=200)

        level.append(o1)
        level.append(o2)

        self.assertEqual(level.order_count, 2)
        self.assertEqual(level.total_volume, 300)
        self.assertEqual(level.head.order_id, 1)
        self.assertEqual(level.tail.order_id, 2)

        # Remove o1
        level.remove(o1)
        self.assertEqual(level.order_count, 1)
        self.assertEqual(level.total_volume, 200)
        self.assertEqual(level.head.order_id, 2)

    def test_order_book_insert_and_spread(self):
        book = OrderBook("AAPL")

        # Add bids: 100.00, 100.50, 99.50
        b1 = OrderNode(1, "P1", "AAPL", Side.BUY, price_to_int(100.00), 100)
        b2 = OrderNode(2, "P2", "AAPL", Side.BUY, price_to_int(100.50), 200)
        b3 = OrderNode(3, "P3", "AAPL", Side.BUY, price_to_int(99.50), 300)
        book.add_order(b1)
        book.add_order(b2)
        book.add_order(b3)

        # Best bid should be 100.50
        self.assertEqual(book.best_bid(), price_to_int(100.50))
        self.assertEqual(book.best_bid_volume(), 200)

        # Add asks: 101.00, 102.00
        a1 = OrderNode(4, "P4", "AAPL", Side.SELL, price_to_int(101.00), 150)
        a2 = OrderNode(5, "P5", "AAPL", Side.SELL, price_to_int(102.00), 250)
        book.add_order(a1)
        book.add_order(a2)

        # Best ask should be 101.00
        self.assertEqual(book.best_ask(), price_to_int(101.00))
        self.assertEqual(book.best_ask_volume(), 150)

        # Spread = 101.00 - 100.50 = 0.50
        self.assertEqual(book.spread(), price_to_int(0.50))
        # Mid price = (100.50 + 101.00) / 2 = 100.75
        self.assertEqual(book.mid_price(), price_to_int(100.75))

    def test_order_cancellation(self):
        book = OrderBook("AAPL")
        b1 = OrderNode(1, "P1", "AAPL", Side.BUY, price_to_int(100.00), 100)
        b2 = OrderNode(2, "P2", "AAPL", Side.BUY, price_to_int(100.00), 200)
        book.add_order(b1)
        book.add_order(b2)

        self.assertEqual(book.bids[price_to_int(100.00)].total_volume, 300)

        # Cancel b1
        cancelled = book.cancel_order(1)
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.order_id, 1)
        self.assertEqual(book.bids[price_to_int(100.00)].total_volume, 200)

        # Cancel b2 (level should be deleted)
        book.cancel_order(2)
        self.assertIsNone(book.best_bid())
        self.assertEqual(len(book.bid_prices), 0)

    def test_l2_snapshot(self):
        book = OrderBook("AAPL")
        book.add_order(OrderNode(1, "P1", "AAPL", Side.BUY, price_to_int(100.00), 50))
        book.add_order(OrderNode(2, "P2", "AAPL", Side.BUY, price_to_int(100.00), 75))
        book.add_order(OrderNode(3, "P3", "AAPL", Side.BUY, price_to_int(99.00), 200))
        book.add_order(OrderNode(4, "P4", "AAPL", Side.SELL, price_to_int(101.00), 120))

        bids_l2, asks_l2 = book.get_l2_snapshot(5)
        self.assertEqual(len(bids_l2), 2)
        self.assertEqual(bids_l2[0], (price_to_int(100.00), 125))
        self.assertEqual(bids_l2[1], (price_to_int(99.00), 200))

        self.assertEqual(len(asks_l2), 1)
        self.assertEqual(asks_l2[0], (price_to_int(101.00), 120))


if __name__ == "__main__":
    unittest.main()
