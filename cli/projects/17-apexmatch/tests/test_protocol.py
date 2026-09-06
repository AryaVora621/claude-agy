"""
Unit tests for ApexMatch Binary ITCH/OUCH Protocol Codec and Market Data Feed.
"""

import unittest
from apexmatch.protocol import (
    BinaryCodec,
    OUCHEnterOrder,
    OUCHCancelOrder,
    ITCHAddOrder,
    ITCHOrderExecuted,
    ITCHOrderCanceled,
    ITCHTrade,
)
from apexmatch.feed import MarketDataFeed, BookReconstructor
from apexmatch.types import Side, price_to_int
from apexmatch.order import OrderNode


class TestProtocolAndFeed(unittest.TestCase):
    def test_ouch_enter_order_roundtrip(self):
        msg = OUCHEnterOrder(
            order_id=123456789,
            side="B",
            qty=500,
            symbol="NVDA",
            price=1255000,
            order_type="L"
        )
        encoded = BinaryCodec.encode_ouch_enter(msg)
        self.assertEqual(len(encoded), 27)

        decoded = BinaryCodec.decode_ouch_enter(encoded)
        self.assertEqual(decoded.order_id, 123456789)
        self.assertEqual(decoded.side, "B")
        self.assertEqual(decoded.qty, 500)
        self.assertEqual(decoded.symbol, "NVDA")
        self.assertEqual(decoded.price, 1255000)
        self.assertEqual(decoded.order_type, "L")

    def test_itch_market_data_roundtrip(self):
        add = ITCHAddOrder(
            timestamp_ns=1700000000000,
            order_id=987654,
            side="S",
            shares=200,
            symbol="MSFT",
            price=4200000
        )
        enc_add = BinaryCodec.encode_itch_add(add)
        self.assertEqual(len(enc_add), 34)

        dec_add = BinaryCodec.decode_itch_add(enc_add)
        self.assertEqual(dec_add.order_id, 987654)
        self.assertEqual(dec_add.shares, 200)
        self.assertEqual(dec_add.symbol, "MSFT")

        trade = ITCHTrade(
            timestamp_ns=1700000000100,
            match_id=42,
            side="B",
            shares=150,
            symbol="MSFT",
            price=4200000
        )
        enc_trade = BinaryCodec.encode_itch_trade(trade)
        dec_trade = BinaryCodec.decode_itch_trade(enc_trade)
        self.assertEqual(dec_trade.match_id, 42)
        self.assertEqual(dec_trade.shares, 150)

    def test_book_reconstruction_via_itch_feed(self):
        feed = MarketDataFeed("TSLA")
        reconstructor = BookReconstructor("TSLA")
        feed.subscribe(reconstructor.process_message)

        # 1. Add Bid 100 shares @ 250.00
        o1 = OrderNode(1, "Trader1", "TSLA", Side.BUY, price_to_int(250.00), 100, timestamp_ns=100)
        feed.publish_order_add(o1)

        # 2. Add Ask 200 shares @ 252.00
        o2 = OrderNode(2, "Trader2", "TSLA", Side.SELL, price_to_int(252.00), 200, timestamp_ns=200)
        feed.publish_order_add(o2)

        bb, ba = reconstructor.get_top_of_book()
        self.assertEqual(bb, price_to_int(250.00))
        self.assertEqual(ba, price_to_int(252.00))

        # 3. Partial execution of ask (50 shares)
        feed.publish_order_exec(order_id=2, executed_qty=50, match_id=1, ts_ns=300)
        self.assertEqual(reconstructor.asks_l2[price_to_int(252.00)], 150)

        # 4. Cancel bid
        feed.publish_order_cancel(order_id=1, canceled_qty=100, ts_ns=400)
        bb, ba = reconstructor.get_top_of_book()
        self.assertIsNone(bb)
        self.assertEqual(ba, price_to_int(252.00))


if __name__ == "__main__":
    unittest.main()
