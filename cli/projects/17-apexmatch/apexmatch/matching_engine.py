"""
ApexMatch: Continuous FIFO Order Matching Engine.
Executes incoming aggressive orders against resting book liquidity with
price-time priority, maker-pricing semantics, advanced order types (IOC, FOK, Iceberg),
and Self-Trade Prevention (STP).
"""

from typing import List, Tuple, Optional
from apexmatch.types import Side, OrderType, TimeInForce, STPMode, Trade
from apexmatch.order import OrderNode
from apexmatch.order_book import OrderBook
from apexmatch.risk import RiskGate, RiskConfig


class MatchingEngine:
    """
    High-frequency deterministic matching engine.
    """

    def __init__(
        self,
        symbol: str,
        risk_config: Optional[RiskConfig] = None,
        stp_mode: STPMode = STPMode.CANCEL_PASSIVE
    ) -> None:
        self.symbol = symbol.upper()
        self.book = OrderBook(self.symbol)
        self.risk_gate = RiskGate(risk_config)
        self.stp_mode = stp_mode

        self.match_id_counter = 0
        self.trades: List[Trade] = []
        self.total_traded_volume = 0
        self.total_traded_notional = 0

    def process_order(self, order: OrderNode) -> Tuple[bool, List[Trade], str]:
        """
        Processes an incoming order:
        1. Pre-trade risk evaluation.
        2. Pre-check liquidity for FOK orders.
        3. Match aggressive crossings against resting orders.
        4. Rest remaining quantity (for Limit/GTC) or cancel (for Market/IOC).
        Returns: (accepted: bool, trades: List[Trade], message: str)
        """
        # 1. Pre-trade risk check
        passed, reason = self.risk_gate.check_order(order, self.book)
        if not passed:
            return False, [], reason

        # 2. Fill-or-Kill (FOK) pre-check
        if order.order_type == OrderType.FOK or order.time_in_force == TimeInForce.FOK:
            available_liq = self.book.get_total_volume_at_or_better(order.side, order.price)
            if available_liq < order.qty:
                return False, [], f"REJECT_FOK_INSUFFICIENT_LIQUIDITY: required {order.qty}, available {available_liq}"

        trades: List[Trade] = []

        # 3. Aggressive crossing loop
        if order.side == Side.BUY:
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)

        # Record risk positions for executed trades
        for t in trades:
            self.risk_gate.record_fill(t.buyer_participant_id, Side.BUY, t.qty)
            self.risk_gate.record_fill(t.seller_participant_id, Side.SELL, t.qty)
            self.trades.append(t)
            self.total_traded_volume += t.qty
            self.total_traded_notional += t.price * t.qty

        # 4. Handle remaining unfulfilled quantity
        if order.qty > 0:
            if order.order_type in (OrderType.MARKET, OrderType.IOC) or order.time_in_force == TimeInForce.IOC:
                # Immediate or Cancel: remainder is dropped
                return True, trades, f"PARTIAL_FILL_IOC_REMAINDER_CANCELLED: {order.qty} cancelled"

            # Limit order rests on the passive book
            if order.is_iceberg:
                # Setup visible peak
                if order.visible_peak_qty > 0 and order.qty > order.visible_peak_qty:
                    order.total_reserve_qty += (order.qty - order.visible_peak_qty)
                    order.qty = order.visible_peak_qty

            self.book.add_order(order)
            status_msg = "FILLED" if not trades else "PARTIALLY_FILLED_AND_RESTING"
            return True, trades, status_msg

        return True, trades, "FILLED"

    def _match_buy(self, order: OrderNode) -> List[Trade]:
        """Matches an incoming BUY order against resting asks."""
        trades: List[Trade] = []

        while order.qty > 0 and self.book.ask_prices:
            best_ask_price = self.book.ask_prices[0]

            # Price limit check (Market orders match at any price)
            if order.order_type != OrderType.MARKET and best_ask_price > order.price:
                break

            level = self.book.asks[best_ask_price]
            maker_order = level.head

            if maker_order is None:
                break

            # Self-Trade Prevention (STP) check
            if maker_order.participant_id == order.participant_id:
                if self.stp_mode == STPMode.CANCEL_PASSIVE:
                    # Cancel resting order and continue matching
                    self.book.cancel_order(maker_order.order_id)
                    continue
                else:  # CANCEL_AGGRESSIVE
                    # Cancel the incoming aggressive order
                    order.qty = 0
                    break

            # Execute trade at maker's resting price!
            fill_qty = min(order.qty, maker_order.qty)
            self.match_id_counter += 1

            trade = Trade(
                match_id=self.match_id_counter,
                maker_order_id=maker_order.order_id,
                taker_order_id=order.order_id,
                buyer_participant_id=order.participant_id,
                seller_participant_id=maker_order.participant_id,
                symbol=self.symbol,
                price=best_ask_price,
                qty=fill_qty,
                timestamp_ns=order.timestamp_ns,
                taker_side=Side.BUY
            )
            trades.append(trade)

            # Deduct filled volume
            order.qty -= fill_qty
            maker_order.fill(fill_qty)

            # If maker is fully depleted
            if maker_order.qty == 0:
                if maker_order.is_iceberg and maker_order.replenish_iceberg() > 0:
                    # Move to tail of level (loses time priority for new peak)
                    if level.order_count > 1:
                        level.remove(maker_order)
                        level.append(maker_order)
                else:
                    self.book.cancel_order(maker_order.order_id)

        return trades

    def _match_sell(self, order: OrderNode) -> List[Trade]:
        """Matches an incoming SELL order against resting bids."""
        trades: List[Trade] = []

        while order.qty > 0 and self.book.bid_prices:
            best_bid_price = self.book.bid_prices[0]

            # Price limit check
            if order.order_type != OrderType.MARKET and best_bid_price < order.price:
                break

            level = self.book.bids[best_bid_price]
            maker_order = level.head

            if maker_order is None:
                break

            # Self-Trade Prevention (STP) check
            if maker_order.participant_id == order.participant_id:
                if self.stp_mode == STPMode.CANCEL_PASSIVE:
                    self.book.cancel_order(maker_order.order_id)
                    continue
                else:  # CANCEL_AGGRESSIVE
                    order.qty = 0
                    break

            # Execute trade at maker's resting price!
            fill_qty = min(order.qty, maker_order.qty)
            self.match_id_counter += 1

            trade = Trade(
                match_id=self.match_id_counter,
                maker_order_id=maker_order.order_id,
                taker_order_id=order.order_id,
                buyer_participant_id=maker_order.participant_id,
                seller_participant_id=order.participant_id,
                symbol=self.symbol,
                price=best_bid_price,
                qty=fill_qty,
                timestamp_ns=order.timestamp_ns,
                taker_side=Side.SELL
            )
            trades.append(trade)

            order.qty -= fill_qty
            maker_order.fill(fill_qty)

            if maker_order.qty == 0:
                if maker_order.is_iceberg and maker_order.replenish_iceberg() > 0:
                    if level.order_count > 1:
                        level.remove(maker_order)
                        level.append(maker_order)
                else:
                    self.book.cancel_order(maker_order.order_id)

        return trades

    def cancel_order(self, order_id: int) -> Optional[OrderNode]:
        """Cancels a resting order by ID."""
        return self.book.cancel_order(order_id)

    def vwap(self) -> Optional[float]:
        """Computes Volume-Weighted Average Price across all executed trades."""
        if self.total_traded_volume == 0:
            return None
        return (self.total_traded_notional / self.total_traded_volume) / 10_000.0
