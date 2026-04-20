"""
risk_engine.py - Comprehensive Risk Management
Covers: position sizing, correlation, VaR, drawdown,
        circuit breakers, regime-aware limits, tail risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import threading
import time
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# POSITION SIZER
# ─────────────────────────────────────────────

class PositionSizer:
    """
    Multiple position sizing methods:
    1. Kelly Criterion (with fraction for safety)
    2. Volatility targeting
    3. Risk parity
    4. Maximum Sharpe (via optimization)
    """

    def __init__(self, account_size: float, max_risk_pct: float = 0.02):
        self.account_size = account_size
        self.max_risk_pct = max_risk_pct

    def kelly_size(
        self,
        win_prob:   float,
        win_return: float,
        loss_return: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Fractional Kelly Criterion.
        kelly_fraction=0.25 → quarter Kelly (much safer).
        """
        if loss_return <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
        q      = 1 - win_prob
        kelly  = win_prob / loss_return - q / win_return
        kelly  = max(0.0, kelly * kelly_fraction)
        return min(kelly * self.account_size, self.max_risk_pct * self.account_size)

    def volatility_target_size(
        self,
        target_vol:     float,
        realized_vol:   float,
        current_price:  float,
    ) -> float:
        """
        Size position so portfolio vol = target_vol.
        Standard for systematic funds.
        """
        if realized_vol <= 0 or current_price <= 0:
            return 0.0
        notional   = self.account_size * (target_vol / realized_vol)
        max_notional = self.account_size * 0.20   # Hard cap at 20%
        notional   = min(notional, max_notional)
        return notional / current_price           # Shares/contracts

    def atr_based_size(
        self,
        atr:           float,
        risk_per_trade: float,
        current_price:  float,
    ) -> float:
        """
        Risk fixed $ amount per ATR unit.
        Standard professional sizing method.
        """
        if atr <= 0 or current_price <= 0:
            return 0.0
        dollar_risk  = self.account_size * risk_per_trade
        shares       = dollar_risk / (2 * atr)   # 2× ATR stop
        return min(shares, dollar_risk / current_price)

# ─────────────────────────────────────────────
# PORTFOLIO RISK MONITOR
# ─────────────────────────────────────────────

class PortfolioRiskMonitor:
    """
    Real-time portfolio-level risk monitoring:
    - Value at Risk (VaR): historical, parametric, Monte Carlo
    - Expected Shortfall (CVaR)
    - Maximum Drawdown
    - Correlation risk
    - Tail risk / stress testing
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence    = confidence_level
        self.returns_hist: Dict[str, deque] = {}
        self.pnl_history:  deque = deque(maxlen=10000)
        self._lock         = threading.Lock()

        # Drawdown tracking
        self.peak_value    = 0.0
        self.current_value = 0.0
        self.max_drawdown  = 0.0

    def update_pnl(self, pnl: float, portfolio_value: float):
        with self._lock:
            self.pnl_history.append(pnl)
            self.current_value = portfolio_value
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            dd = (self.peak_value - portfolio_value) / (self.peak_value + 1e-10)
            self.max_drawdown = max(self.max_drawdown, dd)

    def historical_var(self, horizon_days: int = 1) -> float:
        """Historical simulation VaR"""
        if len(self.pnl_history) < 100:
            return 0.0
        pnl = np.array(list(self.pnl_history))
        return float(-np.percentile(pnl, (1 - self.confidence) * 100))

    def parametric_var(self, portfolio_value: float) -> float:
        """
        Parametric (Gaussian) VaR.
        Faster but assumes normality.
        """
        if len(self.pnl_history) < 30:
            return 0.0
        pnl    = np.array(list(self.pnl_history))
        mu     = np.mean(pnl)
        sigma  = np.std(pnl)
        z      = stats.norm.ppf(1 - self.confidence)
        return float(-(mu + z * sigma))

    def monte_carlo_var(
        self, portfolio_value: float, simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR with fat tails (Student-t distribution).
        Captures tail risk better than Gaussian.
        """
        if len(self.pnl_history) < 50:
            return 0.0
        pnl   = np.array(list(self.pnl_history))
        mu    = np.mean(pnl)
        sigma = np.std(pnl)
        df    = 5                    # Heavy tails: df=5 Student-t

        sim_pnl = stats.t.rvs(df, loc=mu, scale=sigma, size=simulations)
        var     = -np.percentile(sim_pnl, (1 - self.confidence) * 100)
        return float(var)

    def expected_shortfall(self) -> float:
        """CVaR: expected loss beyond VaR (tail risk)"""
        if len(self.pnl_history) < 100:
            return 0.0
        pnl      = np.array(list(self.pnl_history))
        var_pct  = np.percentile(pnl, (1 - self.confidence) * 100)
        tail     = pnl[pnl <= var_pct]
        return float(-np.mean(tail)) if len(tail) > 0 else 0.0

    def correlation_matrix(
        self, returns_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute real-time correlation matrix"""
        if not returns_dict:
            return np.array([[]])
        min_len = min(len(v) for v in returns_dict.values())
        if min_len < 10:
            return np.eye(len(returns_dict))
        mat = np.column_stack([v[-min_len:] for v in returns_dict.values()])
        return np.corrcoef(mat.T)

    def stress_test(
        self, portfolio_value: float, scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Simulate historical stress scenarios:
        - 2008 GFC
        - 2020 COVID crash
        - Flash crashes
        - Rate shocks
        """
        results = {}
        for name, shock_pct in scenarios.items():
            results[name] = portfolio_value * shock_pct
        return results

    @property
    def current_drawdown(self) -> float:
        return (
            (self.peak_value - self.current_value) / (self.peak_value + 1e-10)
            if self.peak_value > 0 else 0.0
        )

# ─────────────────────────────────────────────
# CIRCUIT BREAKER SYSTEM
# ─────────────────────────────────────────────

class CircuitBreaker:
    """
    Multi-level circuit breaker system.
    Halts trading under dangerous conditions.
    Levels: WARNING → SOFT_HALT → HARD_HALT → EMERGENCY
    """

    STATES = ["NORMAL", "WARNING", "SOFT_HALT", "HARD_HALT", "EMERGENCY"]

    def __init__(self, config):
        self.config           = config
        self.state            = "NORMAL"
        self.state_since      = time.time()
        self.halt_reason      = ""
        self.daily_pnl        = 0.0
        self.weekly_pnl       = 0.0
        self.consecutive_loss = 0
        self.order_count_1s   = 0
        self.order_count_1m   = 0
        self.last_second      = int(time.time())
        self.last_minute      = int(time.time() // 60)
        self._lock            = threading.Lock()

        self._halted_events: List[Dict] = []

    def check(
        self,
        portfolio_value:  float,
        initial_capital:  float,
        current_vol:      float,
        avg_vol:          float,
    ) -> Tuple[str, str]:
        """
        Check all circuit breaker conditions.
        Returns: (state, reason)
        """
        with self._lock:
            daily_loss_pct = -self.daily_pnl / (initial_capital + 1e-10)

            # ── Level 4: EMERGENCY ───────────────────
            if daily_loss_pct >= self.config.risk.max_daily_loss_pct * 2:
                return self._halt("EMERGENCY",
                    f"Emergency halt: daily loss {daily_loss_pct:.1%}")

            # ── Level 3: HARD_HALT ───────────────────
            if daily_loss_pct >= self.config.risk.max_daily_loss_pct:
                return self._halt("HARD_HALT",
                    f"Daily loss limit: {daily_loss_pct:.1%}")

            if self.consecutive_loss >= self.config.risk.max_consecutive_losses:
                return self._halt("HARD_HALT",
                    f"Consecutive losses: {self.consecutive_loss}")

            # ── Level 2: SOFT_HALT ───────────────────
            if current_vol > avg_vol * self.config.risk.volatility_halt_threshold:
                return self._halt("SOFT_HALT",
                    f"Volatility spike: {current_vol/avg_vol:.1f}x average")

            if self.order_count_1s > self.config.risk.max_orders_per_second:
                return self._halt("SOFT_HALT",
                    f"Order rate exceeded: {self.order_count_1s}/s")

            # ── Level 1: WARNING ─────────────────────
            if daily_loss_pct >= self.config.risk.max_daily_loss_pct * 0.75:
                self.state = "WARNING"
                return "WARNING", "Approaching daily loss limit"

            # ── All clear ────────────────────────────
            if self.state not in ["HARD_HALT", "EMERGENCY"]:
                self.state = "NORMAL"
            return self.state, ""

    def _halt(self, level: str, reason: str) -> Tuple[str, str]:
        if self._is_upgrade(level):
            self.state       = level
            self.halt_reason = reason
            self.state_since = time.time()
            self._halted_events.append({
                "time": datetime.utcnow().isoformat(),
                "level": level,
                "reason": reason,
            })
            logger.critical(f"CIRCUIT BREAKER [{level}]: {reason}")
        return self.state, reason

    def _is_upgrade(self, level: str) -> bool:
        return self.STATES.index(level) >= self.STATES.index(self.state)

    def record_trade_result(self, pnl: float):
        with self._lock:
            self.daily_pnl  += pnl
            self.weekly_pnl += pnl
            if pnl < 0:
                self.consecutive_loss += 1
            else:
                self.consecutive_loss = 0

    def can_trade(self) -> bool:
        return self.state in ["NORMAL", "WARNING"]

    def reset_daily(self):
        with self._lock:
            self.daily_pnl       = 0.0
            self.consecutive_loss = 0
            if self.state not in ["EMERGENCY"]:
                self.state = "NORMAL"

# ─────────────────────────────────────────────
# FAT FINGER GUARD
# ─────────────────────────────────────────────

class FatFingerGuard:
    """
    Prevents erroneous orders:
    - Price too far from market
    - Quantity exceeds limits
    - Duplicate orders
    - Wrong direction
    """

    def __init__(self, max_deviation_pct: float = 0.05):
        self.max_dev     = max_deviation_pct
        self.recent_orders: deque = deque(maxlen=100)
        self._lock       = threading.Lock()

    def validate_order(
        self,
        symbol:        str,
        side:          str,
        price:         float,
        quantity:      float,
        market_price:  float,
        account_size:  float,
    ) -> Tuple[bool, str]:
        """Returns (is_valid, rejection_reason)"""

        with self._lock:
            # Price sanity
            if price > 0:
                deviation = abs(price - market_price) / (market_price + 1e-10)
                if deviation > self.max_dev:
                    return False, (
                        f"Price {price:.2f} deviates {deviation:.1%} "
                        f"from market {market_price:.2f}"
                    )

            # Quantity sanity
            order_value = price * quantity
            if order_value > account_size * 0.5:
                return False, (
                    f"Order value ${order_value:,.0f} exceeds 50% of account"
                )

            if quantity <= 0:
                return False, "Non-positive quantity"

            # Duplicate detection
            key = f"{symbol}_{side}_{price:.2f}_{quantity}"
            if key in [o['key'] for o in self.recent_orders]:
                return False, "Potential duplicate order"

            self.recent_orders.append({
                "key": key, "time": time.time()
            })
            return True, "ok"
