"""
ApexMatch: Level 3 (L3) Limit Order Book.
Maintains continuous price ladders, instant O(1) order cancellation,
and Level 2 aggregated depth queries.
"""

import bisect
from typing import Dict, List, Tuple, Optional
from apexmatch.types import Side
from apexmatch.order import OrderNode, PriceLevel


class OrderBook:
    """
    Full Level 3 Limit Order Book for a single trading symbol.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()

        # Price -> PriceLevel
        self.bids: Dict[int, PriceLevel] = {}
        self.asks: Dict[int, PriceLevel] = {}

        # Sorted price arrays for O(1) top-of-book and binary search
        # bid_prices: strictly descending (highest bid at index 0)
        # ask_prices: strictly ascending (lowest ask at index 0)
        self.bid_prices: List[int] = []
        self.ask_prices: List[int] = []

        # Order ID -> OrderNode for O(1) direct cancellation
        self.order_map: Dict[int, OrderNode] = {}

    def best_bid(self) -> Optional[int]:
        """Returns highest active bid price or None if book is empty."""
        return self.bid_prices[0] if self.bid_prices else None

    def best_ask(self) -> Optional[int]:
        """Returns lowest active ask price or None if book is empty."""
        return self.ask_prices[0] if self.ask_prices else None

    def best_bid_volume(self) -> int:
        best_p = self.best_bid()
        return self.bids[best_p].total_volume if best_p is not None else 0

    def best_ask_volume(self) -> int:
        best_p = self.best_ask()
        return self.asks[best_p].total_volume if best_p is not None else 0

    def spread(self) -> Optional[int]:
        """Returns bid-ask spread (best_ask - best_bid) in integer micro-cents."""
        bb = self.best_bid()
        ba = self.best_ask()
        return (ba - bb) if (bb is not None and ba is not None) else None

    def mid_price(self) -> Optional[int]:
        """Returns midpoint price (best_bid + best_ask) // 2."""
        bb = self.best_bid()
        ba = self.best_ask()
        return ((bb + ba) // 2) if (bb is not None and ba is not None) else None

    def micro_price(self) -> Optional[float]:
        """
        Volume-weighted mid-price:
        MicroPrice = (BestAsk * BidVol + BestBid * AskVol) / (BidVol + AskVol)
        """
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None

        bv = self.best_bid_volume()
        av = self.best_ask_volume()
        total_v = bv + av
        if total_v == 0:
            return float(self.mid_price() or 0)

        return (ba * bv + bb * av) / total_v

    def add_order(self, order: OrderNode) -> None:
        """
        Rests an order on the passive order book at its specified price level.
        """
        price = order.price
        side = order.side

        if side == Side.BUY:
            level = self.bids.get(price)
            if level is None:
                level = PriceLevel(price)
                self.bids[price] = level
                # Insert in descending order: negate for bisect
                # or insert using inverted search
                self._insert_bid_price(price)
            level.append(order)
        else:
            level = self.asks.get(price)
            if level is None:
                level = PriceLevel(price)
                self.asks[price] = level
                bisect.insort_left(self.ask_prices, price)
            level.append(order)

        self.order_map[order.order_id] = order

    def _insert_bid_price(self, price: int) -> None:
        """Inserts price into self.bid_prices maintaining descending order."""
        # Find position where price should be placed
        low = 0
        high = len(self.bid_prices)
        while low < high:
            mid = (low + high) // 2
            if self.bid_prices[mid] > price:
                low = mid + 1
            else:
                high = mid
        self.bid_prices.insert(low, price)

    def cancel_order(self, order_id: int) -> Optional[OrderNode]:
        """
        Cancels an order from the book in O(1) time.
        Returns the removed OrderNode or None if not found.
        """
        order = self.order_map.pop(order_id, None)
        if order is None:
            return None

        level = order.parent_level
        if level is not None:
            level.remove(order)
            if level.is_empty():
                self._delete_empty_level(order.side, order.price)

        return order

    def _delete_empty_level(self, side: Side, price: int) -> None:
        """Removes an empty price level and prunes sorted price arrays."""
        if side == Side.BUY:
            self.bids.pop(price, None)
            idx = self._binary_search(self.bid_prices, price, descending=True)
            if idx != -1:
                del self.bid_prices[idx]
        else:
            self.asks.pop(price, None)
            idx = self._binary_search(self.ask_prices, price, descending=False)
            if idx != -1:
                del self.ask_prices[idx]

    @staticmethod
    def _binary_search(arr: List[int], target: int, descending: bool = False) -> int:
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            val = arr[mid]
            if val == target:
                return mid
            if descending:
                if val > target:
                    low = mid + 1
                else:
                    high = mid - 1
            else:
                if val < target:
                    low = mid + 1
                else:
                    high = mid - 1
        return -1

    def get_l2_snapshot(self, depth: int = 10) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns aggregated Level 2 depth snapshot:
        (bids: [(price, volume), ...], asks: [(price, volume), ...])
        """
        bids_l2 = [
            (p, self.bids[p].total_volume)
            for p in self.bid_prices[:depth]
        ]
        asks_l2 = [
            (p, self.asks[p].total_volume)
            for p in self.ask_prices[:depth]
        ]
        return bids_l2, asks_l2

    def get_total_volume_at_or_better(self, side: Side, limit_price: int) -> int:
        """
        Computes total resting liquidity available to cross against an incoming order.
        For BUY: asks with price <= limit_price.
        For SELL: bids with price >= limit_price.
        """
        total_vol = 0
        if side == Side.BUY:
            for p in self.ask_prices:
                if p > limit_price:
                    break
                total_vol += self.asks[p].total_volume
        else:
            for p in self.bid_prices:
                if p < limit_price:
                    break
                total_vol += self.bids[p].total_volume
        return total_vol
