# Engineering Specification: ApexMatch High-Frequency Exchange Matching Engine

## System Overview
ApexMatch is a high-performance, deterministic financial exchange matching engine and Level 3 (L3) limit order book implemented from first principles in pure Python (zero external dependencies). It is designed to model the exact operational standards of real-world electronic equities and derivatives exchanges (such as Nasdaq, BATS, and CME).

---

## 1. Fixed-Point Numeric Precision Engine
Financial matching engines cannot use IEEE-754 binary floating-point numbers due to non-associativity, representation drift, and rounding discrepancies. ApexMatch enforces strict 64-bit integer arithmetic:
* **Price Representation**: Stored as integer micro-cents ($10^{-4}$ precision). $1.0000 = 10,000$ base units.
* **Quantity Representation**: Stored as 64-bit unsigned integers.
* **Notional Value**: $\text{Notional} = (\text{Price} \times \text{Quantity}) // 10,000$.

---

## 2. L3 Limit Order Book Architecture
The Level 3 order book maintains full visibility of individual resting orders with strict Price-Time priority (FIFO):

### Data Structures:
1. **Doubly-Linked Order Node (`OrderNode`)**:
   - Fields: `order_id`, `participant_id`, `side`, `price`, `qty`, `initial_qty`, `timestamp`, `prev`, `next`.
   - Allows $O(1)$ removal from any position in the price level during cancellation.
2. **Price Level Queue (`PriceLevel`)**:
   - Doubly-linked queue maintaining order count, total volume, head, and tail pointers.
   - $O(1)$ append at tail for incoming passive orders.
   - $O(1)$ pop from head during matching.
3. **Price Level Index**:
   - Bids: Max-heap or sorted price array maintaining descending order ($P_{\text{best}} = \max(P)$).
   - Asks: Min-heap or sorted price array maintaining ascending order ($P_{\text{best}} = \min(P)$).
4. **Order Lookup Table (`order_map`)**:
   - Hash table mapping `order_id -> OrderNode`.
   - Enables instantaneous $O(1)$ lookup for cancellations and status modifications.

---

## 3. Order Types & Execution Semantics
* **Limit Order (GTC - Good-Till-Cancel)**:
  - Aggressive cross: If price crosses opposite best quote, matches immediately.
  - Passive rest: Any unfulfilled remaining quantity is enqueued at the tail of its price level.
* **Market Order**:
  - Sweeps the opposite book at prevailing prices until filled or liquidity is exhausted. Any unfulfilled remainder is cancelled immediately.
* **Immediate-or-Cancel (IOC)**:
  - Fills as much quantity as possible immediately against resting orders at or better than limit price; immediately cancels unfulfilled remainder without resting on book.
* **Fill-or-Kill (FOK)**:
  - Checks if cumulative available liquidity across all reachable price levels is $\ge$ order quantity. If yes, fills completely; if no, rejects order entirely with zero execution (all-or-nothing).
* **Iceberg Order**:
  - Contains `total_qty` and `visible_peak_qty`. Only `visible_peak_qty` is posted to the visible book.
  - When visible portion is exhausted, it is automatically replenished from the hidden reserve and placed at the tail of that price level (relinquishing time priority).
* **Self-Trade Prevention (STP)**:
  - Detects incoming orders that would cross against resting orders from the same participant ID.
  - Modes: Cancel Passive (resting order cancelled), Cancel Aggressive (incoming order rejected).

---

## 4. Pre-Trade Risk & Compliance Gate
Before entering the matching loop, every order must pass validation:
* **Price Collar Check**: Rejects orders whose limit price deviates more than $\pm 10\%$ from current midpoint price to prevent fat-finger errors.
* **Order Size Limit**: Verifies quantity $\le \text{MaxShares}$ (e.g., 100,000 shares).
* **Notional Exposure Limit**: Verifies total trade value $\le \text{MaxNotional}$ (e.g., $1,000,000).
* **Participant Balance/Credit Limit**: Rejects orders that exceed participant position thresholds.

---

## 5. Binary Wire Protocol (ITCH / OUCH)
Encodes binary packed records matching exchange wire specifications:
* **OUCH Order Entry**:
  - `EnterOrder`: Type 'O', OrderID (u64), Side ('B'/'S'), Quantity (u32), Symbol (8 bytes), Price (u32), OrderType ('L'/'M'/'I'/'F').
  - `CancelOrder`: Type 'X', OrderID (u64), Quantity (u32).
* **ITCH Market Data Dissemination**:
  - `SystemEvent`: Type 'S', Timestamp (u64), EventCode ('O'/'C').
  - `OrderAdded`: Type 'A', Timestamp (u64), OrderID (u64), Side ('B'/'S'), Quantity (u32), Price (u32).
  - `OrderExecuted`: Type 'E', Timestamp (u64), OrderID (u64), ExecutedQuantity (u32), MatchID (u64).
  - `OrderCanceled`: Type 'C', Timestamp (u64), OrderID (u64), CanceledQuantity (u32).
  - `TradeMessage`: Type 'P', Timestamp (u64), MatchID (u64), Side ('B'/'S'), Quantity (u32), Price (u32).

---

## 6. Real-Time Analytics & Terminal Visualizer
* **Metrics**:
  - Spread, Mid-price, Micro-price (volume-weighted top of book).
  - Volume-Weighted Average Price (VWAP): $\text{VWAP} = \frac{\sum P_i Q_i}{\sum Q_i}$.
  - Order Flow Imbalance (OFI).
* **Terminal Display**:
  - ANSI TrueColor Level 2 Depth Ladder with bid/ask quantity bars.
  - Live execution trade tape.
  - Latency and throughput diagnostics.
