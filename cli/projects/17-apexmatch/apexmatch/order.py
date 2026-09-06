"""
ApexMatch: Doubly-Linked Order and Price Level Data Structures.
Guarantees O(1) order insertion at tail, O(1) order cancellation anywhere in queue,
and strict FIFO Price-Time priority execution.
"""

from typing import Optional
from apexmatch.types import Side, OrderType, TimeInForce


class PriceLevel:
    """
    Queue of resting orders at a single discrete price level.
    Maintains strict FIFO time priority via a doubly-linked list.
    """

    __slots__ = ("price", "head", "tail", "order_count", "total_volume")

    def __init__(self, price: int) -> None:
        self.price = price
        self.head: Optional["OrderNode"] = None
        self.tail: Optional["OrderNode"] = None
        self.order_count = 0
        self.total_volume = 0

    def append(self, node: "OrderNode") -> None:
        """Appends an order node to the tail of the FIFO queue in O(1) time."""
        node.parent_level = self
        node.prev = self.tail
        node.next = None

        if self.tail is not None:
            self.tail.next = node
        else:
            self.head = node

        self.tail = node
        self.order_count += 1
        self.total_volume += node.qty

    def remove(self, node: "OrderNode") -> None:
        """Removes an order node from anywhere in the queue in O(1) time."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        self.order_count -= 1
        self.total_volume -= node.qty

        node.prev = None
        node.next = None
        node.parent_level = None

    def is_empty(self) -> bool:
        return self.order_count == 0 or self.head is None

    def __len__(self) -> int:
        return self.order_count


class OrderNode:
    """
    Individual order resting in or crossing the limit order book.
    """

    __slots__ = (
        "order_id",
        "participant_id",
        "symbol",
        "side",
        "price",
        "qty",
        "initial_qty",
        "order_type",
        "time_in_force",
        "timestamp_ns",
        "prev",
        "next",
        "parent_level",
        "is_iceberg",
        "total_reserve_qty",
        "visible_peak_qty",
    )

    def __init__(
        self,
        order_id: int,
        participant_id: str,
        symbol: str,
        side: Side,
        price: int,
        qty: int,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.GTC,
        timestamp_ns: int = 0,
        is_iceberg: bool = False,
        visible_peak_qty: int = 0,
        total_reserve_qty: int = 0
    ) -> None:
        self.order_id = order_id
        self.participant_id = participant_id
        self.symbol = symbol
        self.side = side
        self.price = price
        self.qty = qty
        self.initial_qty = qty
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.timestamp_ns = timestamp_ns

        # Doubly-linked pointers
        self.prev: Optional["OrderNode"] = None
        self.next: Optional["OrderNode"] = None
        self.parent_level: Optional[PriceLevel] = None

        # Iceberg order configuration
        self.is_iceberg = is_iceberg
        self.visible_peak_qty = visible_peak_qty
        self.total_reserve_qty = total_reserve_qty

    def fill(self, fill_qty: int) -> int:
        """
        Deducts filled quantity from order node and updates its parent level.
        Returns actual filled quantity.
        """
        actual_fill = min(self.qty, fill_qty)
        self.qty -= actual_fill
        if self.parent_level is not None:
            self.parent_level.total_volume -= actual_fill
        return actual_fill

    def replenish_iceberg(self) -> int:
        """
        Replenishes visible peak from hidden reserve.
        Returns amount replenished.
        """
        if not self.is_iceberg or self.total_reserve_qty <= 0 or self.qty > 0:
            return 0

        replenish_amt = min(self.total_reserve_qty, self.visible_peak_qty)
        self.qty = replenish_amt
        self.total_reserve_qty -= replenish_amt
        if self.parent_level is not None:
            self.parent_level.total_volume += replenish_amt
        return replenish_amt
