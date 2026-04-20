"""
execution_engine.py - Institutional Execution Infrastructure
Features: Smart Order Routing, TWAP, VWAP, Iceberg,
          Latency optimization, Slippage modeling, Dark pools
"""

import asyncio
import aiohttp
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import hmac
import hashlib
import json
import threading

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET    = "market"
    LIMIT     = "limit"
    STOP      = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG   = "iceberg"
    TWAP      = "twap"
    VWAP      = "vwap"

class OrderStatus(Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    PARTIAL   = "partial_fill"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"
    EXPIRED   = "expired"

class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Canonical order representation"""
    order_id:     str
    symbol:       str
    side:         OrderSide
    order_type:   OrderType
    quantity:     float
    price:        float = 0.0
    stop_price:   float = 0.0
    time_in_force: str = "GTC"
    created_at:   float = field(default_factory=time.time)
    filled_qty:   float = 0.0
    avg_fill_px:  float = 0.0
    status:       OrderStatus = OrderStatus.PENDING
    exchange:     str = "primary"
    metadata:     Dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        ]

    @property
    def fill_rate(self) -> float:
        return self.filled_qty / (self.quantity + 1e-10)

@dataclass
class Fill:
    """Trade execution record"""
    fill_id:    str
    order_id:   str
    symbol:     str
    side:       str
    quantity:   float
    price:      float
    timestamp:  float
    exchange:   str
    fees:       float
    slippage:   float

# ─────────────────────────────────────────────
# SLIPPAGE MODEL
# ─────────────────────────────────────────────

class SlippageModel:
    """
    Estimates and minimizes market impact.
    Uses Almgren-Chriss optimal execution model.
    """

    def __init__(self):
        self.fill_history: deque = deque(maxlen=10000)

    def estimate_slippage(
        self,
        quantity:      float,
        market_price:  float,
        avg_volume:    float,
        volatility:    float,
        side:          str,
    ) -> float:
        """
        Almgren-Chriss slippage model:
        Temporary impact = η × (Q/V)^0.6 × σ
        """
        if avg_volume <= 0:
            return volatility * 0.01

        participation = quantity / (avg_volume + 1e-10)
        eta           = 0.1       # Temporary impact coefficient
        temp_impact   = eta * (participation ** 0.6) * volatility

        # Permanent impact (price pressure)
        gamma = 0.01
        perm_impact = gamma * participation * volatility

        total_impact = (temp_impact + perm_impact) * market_price
        return total_impact if side == "buy" else -total_impact

    def optimal_schedule(
        self,
        total_qty:     float,
        time_horizon:  float,
        risk_aversion: float = 0.01,
        volatility:    float = 0.01,
        avg_volume:    float = 1_000_000,
    ) -> np.ndarray:
        """
        Almgren-Chriss optimal execution trajectory.
        Balances market impact vs. execution risk.
        Returns array of quantities per time slice.
        """
        n_slices = max(10, int(time_horizon / 60))   # 1-min slices
        kappa    = np.sqrt(risk_aversion * volatility**2)
        t        = np.linspace(0, time_horizon, n_slices + 1)

        # Optimal execution trajectory
        sinh_kT  = np.sinh(kappa * time_horizon)
        schedule = (
            total_qty *
            np.sinh(kappa * (time_horizon - t[:-1])) / sinh_kT
        )
        return schedule

# ─────────────────────────────────────────────
# SMART ORDER ROUTER
# ─────────────────────────────────────────────

class SmartOrderRouter:
    """
    Routes orders to best execution venue:
    - Price improvement scanning
    - Liquidity aggregation
    - Dark pool access
    - Exchange fee optimization
    """

    def __init__(self, exchanges: List[Dict]):
        self.exchanges    = exchanges
        self.fill_quality: Dict[str, deque] = {
            e['name']: deque(maxlen=1000) for e in exchanges
        }
        self.latency: Dict[str, deque] = {
            e['name']: deque(maxlen=100) for e in exchanges
        }

    def select_venue(
        self,
        symbol:    str,
        side:      str,
        quantity:  float,
        quotes:    Dict[str, Dict],   # {exchange: {bid, ask, depth}}
    ) -> str:
        """
        Score each venue and select best execution destination.
        Scoring: price + depth + latency + fees + fill quality
        """
        scores = {}

        for exchange, quote in quotes.items():
            if not quote:
                continue

            # Price score (lower ask for buy, higher bid for sell)
            if side == "buy":
                price_score = -quote.get('ask', np.inf)
            else:
                price_score = quote.get('bid', 0)

            # Depth score: can the exchange absorb the order?
            depth = quote.get('depth', 0)
            depth_score = min(1.0, depth / (quantity + 1e-10))

            # Latency score (lower is better)
            avg_lat = (
                np.mean(self.latency[exchange])
                if exchange in self.latency and self.latency[exchange]
                else 0.1
            )
            lat_score = -avg_lat

            # Fill quality score
            fq = (
                np.mean(self.fill_quality[exchange])
                if exchange in self.fill_quality and self.fill_quality[exchange]
                else 0.5
            )

            # Fee score
            fee = quote.get('fee_rate', 0.0002)
            fee_score = -fee

            scores[exchange] = (
                0.40 * price_score / 1000 +  # Normalize
                0.25 * depth_score +
                0.15 * lat_score +
                0.10 * fq +
                0.10 * fee_score
            )

        if not scores:
            return self.exchanges[0]['name']
        return max(scores, key=scores.get)

# ─────────────────────────────────────────────
# EXECUTION ALGORITHMS
# ─────────────────────────────────────────────

class TWAPExecutor:
    """Time-Weighted Average Price execution"""

    def __init__(self, total_qty: float, duration_seconds: float, slices: int = 10):
        self.total_qty  = total_qty
        self.duration   = duration_seconds
        self.slices     = slices
        self.qty_per_slice = total_qty / slices
        self.interval   = duration_seconds / slices
        self.slice_idx  = 0
        self.filled     = 0.0

    def next_slice(self) -> Optional[float]:
        """Get quantity for next TWAP slice"""
        if self.slice_idx >= self.slices:
            return None
        qty = min(self.qty_per_slice, self.total_qty - self.filled)
        self.slice_idx += 1
        return qty if qty > 0 else None

class VWAPExecutor:
    """Volume-Weighted Average Price execution"""

    def __init__(
        self,
        total_qty:       float,
        volume_profile:  np.ndarray,   # Historical volume by minute
        duration_minutes: int,
    ):
        self.total_qty      = total_qty
        self.volume_profile = volume_profile / volume_profile.sum()
        self.duration       = duration_minutes
        self.schedule       = total_qty * self.volume_profile[:duration_minutes]
        self.slice_idx      = 0
        self.filled         = 0.0

    def next_slice(self) -> Optional[float]:
        if self.slice_idx >= len(self.schedule):
            return None
        return float(self.schedule[self.slice_idx])

# ─────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────

class PositionManager:
    """
    Tracks all open positions with:
    - Real-time P&L
    - Greeks (for options)
    - Correlation-adjusted sizing
    - Automatic stop management
    """

    @dataclass
    class Position:
        symbol:       str
        side:         str
        quantity:     float
        entry_price:  float
        entry_time:   float
        stop_loss:    float
        take_profit:  float
        current_price: float = 0.0
        unrealized_pnl: float = 0.0
        realized_pnl:   float = 0.0

        def update_price(self, price: float):
            self.current_price = price
            pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl = pnl if self.side == "long" else -pnl

        @property
        def should_stop(self) -> bool:
            if self.side == "long":
                return self.current_price <= self.stop_loss
            return self.current_price >= self.stop_loss

        @property
        def should_take_profit(self) -> bool:
            if self.side == "long":
                return self.current_price >= self.take_profit
            return self.current_price <= self.take_profit

    def __init__(self):
        self.positions: Dict[str, PositionManager.Position] = {}
        self._lock     = threading.Lock()

    def open_position(
        self,
        symbol:      str,
        side:        str,
        quantity:    float,
        price:       float,
        stop_loss:   float,
        take_profit: float,
    ):
        with self._lock:
            self.positions[symbol] = self.Position(
                symbol      = symbol,
                side        = side,
                quantity    = quantity,
                entry_price = price,
                entry_time  = time.time(),
                stop_loss   = stop_loss,
                take_profit = take_profit,
                current_price = price,
            )

    def update_prices(self, prices: Dict[str, float]) -> List[str]:
        """Update all position prices; return list of triggered stops"""
        triggered = []
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    pos.update_price(price)
                    if pos.should_stop or pos.should_take_profit:
                        triggered.append(symbol)
        return triggered

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_exposure(self) -> float:
        return sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
