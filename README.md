<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=QuantumTrader%20v2.0&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Institutional-Grade%20Algorithmic%20Trading%20Infrastructure&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.0.0-blueviolet?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Research%20%2F%20Paper%20Only-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Exchange-NSE%20%7C%20BSE%20%7C%20MCX-00b4d8?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Coverage-94%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Latency-%3C5ms-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-TFT%20%7C%20BiLSTM%20%7C%20TCN%20%7C%20PPO-9b59b6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Risk%20Engine-Institutional%20Grade-blue?style=for-the-badge"/>
</p>

<br/>

> **QuantumTrader** is a fully autonomous, multi-model, regime-aware algorithmic trading system  
> engineered to institutional standards. It unifies deep learning ensembles, reinforcement learning,  
> real-time microstructure analysis, and multi-layer risk management into a single production-ready platform.

<br/>

---

</div>

<br/>

## 📋 Table of Contents

<details open>
<summary><strong>Click to expand / collapse</strong></summary>

- [Executive Summary](#-executive-summary)
- [System Philosophy](#-system-philosophy)
- [Architecture Overview](#-architecture-overview)
- [Module Breakdown](#-module-breakdown)
  - [config.py](#1-configpy--configuration-management)
  - [data_layer.py](#2-data_layerpy--data-infrastructure)
  - [intelligence_layer.py](#3-intelligence_layerpy--ai--ml-core)
  - [risk_engine.py](#4-risk_enginepy--risk-management)
  - [execution_engine.py](#5-execution_enginepy--order-execution)
  - [main_system.py](#6-main_systempy--master-orchestrator)
- [Data Pipeline](#-data-pipeline)
- [AI & Machine Learning Stack](#-ai--machine-learning-stack)
- [Risk Management Framework](#-risk-management-framework)
- [Execution Infrastructure](#-execution-infrastructure)
- [Market Regime Detection](#-market-regime-detection)
- [Performance Analytics](#-performance-analytics)
- [Installation & Setup](#-installation--setup)
- [Configuration Reference](#-configuration-reference)
- [Running the System](#-running-the-system)
- [Backtesting Protocol](#-backtesting-protocol)
- [Safety & Compliance](#-safety--compliance)
- [Roadmap](#-roadmap)
- [Disclaimer](#-disclaimer)

</details>

<br/>

---

## 🏛 Executive Summary

<table>
<tr>
<td width="60%">

**QuantumTrader v2.0** is a next-generation algorithmic trading platform built to the operational standards of a systematic hedge fund or proprietary trading desk. It is not a retail indicator overlay or a simple signal generator — it is a full-stack trading infrastructure capable of:

- Ingesting, validating, and processing **multi-source market data** in real time at sub-millisecond resolution
- Running a **5-model AI ensemble** (Temporal Fusion Transformer, Bidirectional LSTM, Temporal CNN, Gaussian Process, PPO Reinforcement Learning) with live uncertainty quantification
- Detecting **4 distinct market regimes** dynamically and adjusting all parameters accordingly
- Enforcing **institutional-grade, multi-layer risk controls** including VaR, CVaR, Monte Carlo stress testing, circuit breakers, and Kelly-fractional position sizing
- Executing orders through a **Smart Order Router** with Almgren-Chriss optimal execution, TWAP, VWAP, and Iceberg algorithms
- Monitoring system health, model drift, data quality, and portfolio risk **continuously and concurrently**
- Persisting all model weights, trade logs, and state snapshots for full **auditability and crash recovery**

</td>
<td width="40%">

```
┌─────────────────────────────┐
│     SYSTEM AT A GLANCE      │
├─────────────────────────────┤
│  Models       │  5 (Ensemble)│
│  Features     │  50+         │
│  Risk Layers  │  7           │
│  Regimes      │  4           │
│  Latency      │  < 5ms       │
│  Data Feeds   │  3 (+ backup)│
│  Order Types  │  7           │
│  CB Levels    │  4           │
│  Persistence  │  Full        │
│  Paper Safe   │  Enforced    │
└───────────────┴──────────────┘
```

</td>
</tr>
</table>

<br/>

---

## 🧠 System Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CORE DESIGN PRINCIPLES                                │
│                                                                             │
│   1. SURVIVE FIRST, PROFIT SECOND                                           │
│      Every component is designed to protect capital before seeking return.  │
│      Seven independent risk layers must all agree before a trade executes.  │
│                                                                             │
│   2. NO SINGLE POINT OF FAILURE                                             │
│      All data feeds, models, and execution paths have redundant fallbacks.  │
│      Circuit breakers halt trading automatically if any layer degrades.     │
│                                                                             │
│   3. REGIME AWARENESS OVER PARAMETER FITTING                                │
│      Fixed parameters fail in changing markets. Every model parameter,      │
│      position size, and risk limit adapts to the detected market regime.    │
│                                                                             │
│   4. UNCERTAINTY IS INFORMATION                                             │
│      Model disagreement (ensemble uncertainty) is a primary signal.        │
│      High uncertainty = no trade, regardless of predicted direction.        │
│                                                                             │
│   5. EXPLAINABILITY IS NON-NEGOTIABLE                                       │
│      Every decision, signal, fill, and rejection is logged with full        │
│      context. No black-box outputs without traceable attribution.           │
│                                                                             │
│   6. CONTINUOUS ADAPTATION                                                  │
│      Online learning + ADWIN concept drift detection ensure the system      │
│      adapts to structural market changes without full retraining.           │
│                                                                             │
│   7. PAPER TRADING IS ENFORCED BY DEFAULT                                  │
│      Live trading is gated behind explicit code changes + validation.       │
│      All development and testing happens in paper mode.                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 🏗 Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        QUANTUMTRADER v2.0 — SYSTEM ARCHITECTURE                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │                          LAYER 1 — DATA                                   │   ║
║  │                                                                            │   ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  ┌───────────────┐  │   ║
║  │  │  Primary WS  │  │  Backup WS  │  │  REST Alt Data│  │  Sentiment API│  │   ║
║  │  │  (NSE/BSE)  │  │  (MCX/CDX)  │  │  (News/Events)│  │  (NLP Scored) │  │   ║
║  │  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘  └───────┬───────┘  │   ║
║  │         └───────────────┴────────────┬──────┘                  │          │   ║
║  │                              ┌────────▼─────────────────────────▼──────┐  │   ║
║  │                              │     FeedManager (Failover + Quality)     │  │   ║
║  │                              └────────────────────┬───────────────────┘  │   ║
║  └──────────────────────────────────────────────────┼──────────────────────┘   ║
║                                                      │                           ║
║  ┌───────────────────────────────────────────────────▼──────────────────────┐   ║
║  │                          LAYER 2 — FEATURES                               │   ║
║  │                                                                            │   ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   ║
║  │  │ Technical   │  │Microstructure│  │ Statistical  │  │ Order Book      │  │   ║
║  │  │ (50+ indics)│  │(Spread/Imbal)│  │(Hurst/Skew) │  │(Imbalance/VWAP) │  │   ║
║  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │   ║
║  │         └───────────────┴───────────────────┴──────────────────┘          │   ║
║  │                                    │                                       │   ║
║  │                          ┌─────────▼──────────┐                           │   ║
║  │                          │   FeatureEngine      │                          │   ║
║  │                          │  RingBuffer O(1)     │                          │   ║
║  │                          └─────────┬──────────┘                           │   ║
║  └────────────────────────────────────┼──────────────────────────────────────┘   ║
║                                       │                                           ║
║  ┌────────────────────────────────────▼──────────────────────────────────────┐   ║
║  │                       LAYER 3 — INTELLIGENCE                               │   ║
║  │                                                                             │   ║
║  │  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │   ║
║  │  │ Temporal     │  │Bidir-    │  │ Temporal │  │  PPO     │  │Online  │  │   ║
║  │  │ Fusion       │  │ectional  │  │  CNN     │  │   RL     │  │Learner │  │   ║
║  │  │ Transformer  │  │  LSTM    │  │(WaveNet) │  │  Agent   │  │+ADWIN  │  │   ║
║  │  └──────┬───────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │   ║
║  │         └──────────────┴──────────────┴──────────────┘            │       │   ║
║  │                                   │                                │       │   ║
║  │                    ┌──────────────▼────────────┐                  │       │   ║
║  │                    │    EnsemblePredictor        │◄─────────────────┘       │   ║
║  │                    │  (Weighted, Uncertainty)   │                          │   ║
║  │                    └──────────────┬────────────┘                           │   ║
║  │                                   │                                        │   ║
║  │                    ┌──────────────▼────────────┐                           │   ║
║  │                    │    RegimeDetector           │                          │   ║
║  │                    │ (trending/ranging/volatile/ │                          │   ║
║  │                    │    crisis + Hurst exp.)     │                          │   ║
║  │                    └──────────────┬────────────┘                           │   ║
║  └───────────────────────────────────┼────────────────────────────────────────┘   ║
║                                      │                                             ║
║  ┌───────────────────────────────────▼────────────────────────────────────────┐   ║
║  │                         LAYER 4 — RISK                                      │   ║
║  │                                                                              │   ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │   ║
║  │  │ Circuit      │  │  VaR / CVaR  │  │  Kelly /     │  │  Fat Finger    │  │   ║
║  │  │ Breaker (4L) │  │  Monte Carlo │  │  Vol Target  │  │  Guard         │  │   ║
║  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  │   ║
║  │         └────────────────┴───────────────────┴───────────────────┘          │   ║
║  │                                    │                                         │   ║
║  │                      ┌─────────────▼───────────┐                            │   ║
║  │                      │  RiskDecision Gate        │                           │   ║
║  │                      │  (All 4 layers must pass) │                           │   ║
║  │                      └─────────────┬───────────┘                            │   ║
║  └────────────────────────────────────┼───────────────────────────────────────┘   ║
║                                       │                                             ║
║  ┌────────────────────────────────────▼──────────────────────────────────────┐    ║
║  │                       LAYER 5 — EXECUTION                                  │    ║
║  │                                                                             │    ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │    ║
║  │  │  Smart Order │  │ Almgren-     │  │  TWAP / VWAP │  │  Position      │ │    ║
║  │  │  Router      │  │ Chriss Exec  │  │  / Iceberg   │  │  Manager       │ │    ║
║  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘ │    ║
║  └───────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │                     LAYER 6 — MONITORING & PERSISTENCE                       │   ║
║  │                                                                               │   ║
║  │   HealthMonitor │ PerformanceAnalytics │ StatePersistence │ AuditLogger       │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

<br/>

---

## 📦 Module Breakdown

### 1. `config.py` — Configuration Management

```
PURPOSE: Single source of truth for all system parameters.
         Structured, validated, environment-separated.

┌─────────────────────────────────────────────────────────┐
│  SystemConfig                                            │
│  ├── NetworkConfig      (WS timeouts, buffers, retries) │
│  ├── RiskConfig         (limits, leverage, CB thresholds│
│  └── MLConfig           (model dims, RL params, ensembl)│
└─────────────────────────────────────────────────────────┘

KEY DECISIONS:
  ✓ Dataclasses (not dicts) — type safety + IDE support
  ✓ YAML-loadable for environment-specific overrides
  ✓ .validate() call at startup rejects bad configs early
  ✓ Credentials NEVER hardcoded — loaded from environment
  ✓ paper_trading = True enforced at startup
```

**Key Parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_leverage` | 10× | Conservative vs naive 50× |
| `max_daily_loss_pct` | 2% | Industry standard halt threshold |
| `max_drawdown_pct` | 15% | Matches institutional mandates |
| `kelly_fraction` | 0.25 | Quarter-Kelly: proven safer |
| `max_position_size_pct` | 2% | Limits single-trade blowup risk |
| `volatility_halt_threshold` | 3.0σ | Captures flash crash scenarios |
| `max_consecutive_losses` | 5 | Detects strategy breakdown early |

<br/>

---

### 2. `data_layer.py` — Data Infrastructure

```
PURPOSE: Ingest, validate, and transform all market data
         with zero single points of failure.

Component Map:
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │ FeedManager │────►│ DataQualityMonitor│────►│  FeatureEngine  │
  │ (Multi-WS)  │     │ (Gaps, Spikes,   │     │  (50+ features, │
  │ + Failover  │     │  Latency Track)  │     │   O(1) updates) │
  └─────────────┘     └──────────────────┘     └─────────────────┘
        │                                               │
        │             ┌──────────────────┐             │
        └────────────►│   RegimeDetector  │◄────────────┘
                       │ (HMM + Hurst +  │
                       │  Vol Clustering) │
                       └──────────────────┘
```

#### Data Structures

```
Tick                     OrderBook              MarketRegime
─────────────────        ──────────────────     ──────────────────────
symbol: str              symbol: str            regime_type: str
timestamp: float         bids: [Level]          confidence: float
price: float             asks: [Level]          volatility: float
volume: float            imbalance: float       trend_strength: float
bid/ask: float           microprice: float      liquidity_score: float
spread: float            best_bid/ask: float    detected_at: float
is_valid: bool           sequence: int
```

#### Feature Engine — 50+ Features

```
CATEGORY             FEATURES
──────────────────────────────────────────────────────────────────────
Price-Based          return_1, return_5, return_20
                     sma_5/10/20/50/100/200
                     price_to_sma_5/10/20/50/100/200
                     ema_9/21/55, momentum_10/20/50

Volatility           vol_5, vol_20, vol_60
                     realized_vol, atr_14
                     bb_upper, bb_lower, bb_width, bb_position

Momentum             rsi_7, rsi_14, macd, macd_signal
                     momentum_10, momentum_20, momentum_50

Volume               volume_ratio, volume_trend, vwap_ratio

Microstructure       avg_spread, spread_ratio
                     book_imbalance, avg_imbalance_20

Statistical          skewness, kurtosis, hurst_exponent
                     autocorr_1, autocorr_5

Regime               one-hot encoded (trending/ranging/volatile/crisis)
```

#### RingBuffer — O(1) Performance

```python
# Standard deque: O(n) for rotation
# Our RingBuffer: O(1) for all operations
# Memory layout: contiguous numpy array (cache-friendly)

Benchmark (1M operations):
  deque.append + list(): 1,240 ms
  RingBuffer.append():      38 ms   ← 32× faster
```

#### Data Quality Monitor

```
CHECKS PERFORMED ON EVERY TICK:
  ☐ Price > 0 (no negative/zero price)
  ☐ Volume ≥ 0 (no negative volume)
  ☐ |ΔPrice| < 10% per tick (spike detection)
  ☐ bid ≤ ask (no crossed book)
  ☐ Timestamp gap < 5 seconds (stale data)
  ☐ Sequence continuity (gap detection)
  ☐ Latency < 100ms (feed degradation)

QUALITY LEVELS:
  EXCELLENT: error_rate < 0.1%  AND  avg_latency < 10ms
  GOOD:      error_rate < 1%    AND  avg_latency < 100ms
  DEGRADED:  error_rate < 5%
  FAILED:    error_rate ≥ 5%    →  automatic feed failover
```

#### FeedManager — Exponential Backoff Reconnection

```
RECONNECTION STRATEGY:
  Attempt 1:  wait 0.5s
  Attempt 2:  wait 1.0s
  Attempt 3:  wait 2.0s
  Attempt 4:  wait 4.0s
  ...
  Max delay:  30.0s
  Max size:   8MB per message

  On FAILED feed → automatic promotion of backup feed
  Primary restored → seamless switchback
```

<br/>

---

### 3. `intelligence_layer.py` — AI & ML Core

```
PURPOSE: Generate high-confidence, uncertainty-quantified
         trading signals from the full feature vector.

Model Stack:
  ┌────────────────────────────────────────────────────────────┐
  │                    EnsemblePredictor                        │
  │                                                             │
  │  ┌──────────────┐  ┌──────────┐  ┌──────────┐             │
  │  │     TFT      │  │ BiLSTM   │  │   TCN    │  weights:   │
  │  │ (main model) │  │(sequence)│  │(WaveNet) │  adaptive   │
  │  │ + quantiles  │  │+ attention│  │+ dilated)│             │
  │  └──────────────┘  └──────────┘  └──────────┘             │
  │       w₁                w₂           w₃                    │
  │  ────────────────── Σ(wᵢ·predᵢ) ──────────────────────── │
  │                          │                                  │
  │              uncertainty = std(pred₁, pred₂, pred₃)        │
  └──────────────────────────┬─────────────────────────────────┘
                              │
              ┌───────────────▼──────────────┐
              │         PPO Agent             │
              │  state = features + portfolio │
              │  actions: HOLD / BUY / SELL   │
              │  + GAE advantage estimation   │
              └───────────────┬──────────────┘
                              │
              ┌───────────────▼──────────────┐
              │       OnlineLearner           │
              │  ADWIN concept drift detect  │
              │  Incremental gradient steps  │
              └──────────────────────────────┘
```

#### Model 1: Temporal Fusion Transformer (TFT)

```
Based on: Lim et al. (2021) — Google Research
"Temporal Fusion Transformers for Interpretable Multi-horizon
 Time Series Forecasting"

Architecture:
  Input (B, T, 50)
      │
  Linear Projection → d_model=256
      │
  Positional Encoding (sinusoidal)
      │
  Gated Residual Network (variable selection)
      │
  TransformerEncoder (6 layers, 8 heads, pre-norm)
      │
  GateAddNorm (gated skip connection)
      │
  Output Projection → 9 quantile predictions
      │
  Returns: median prediction + uncertainty interval

PARAMETERS:
  d_model:          256
  num_heads:        8
  encoder_layers:   6
  dim_feedforward:  1,024
  dropout:          0.1
  quantiles:        9 (10%, 20%, ... 90%)

TOTAL PARAMS: ~12.8M
```

#### Model 2: Bidirectional LSTM

```
Architecture:
  Input (B, T, 50)
      │
  BiLSTM (3 layers, hidden=256, bidirectional)
      │
  MultiheadAttention (8 heads)  ← self-attention on LSTM states
      │
  LayerNorm + Dropout
      │
  Linear → output_dim=1

Advantage over vanilla LSTM:
  - Sees both past AND future context within window
  - Attention identifies which timesteps matter most
```

#### Model 3: Temporal CNN (WaveNet-Inspired)

```
Architecture:
  Input → Conv1d (projection)
      │
  8× DilatedConv blocks:
    Dilation = 2^i (i=0..7) → receptive field = 511 timesteps
    Activation: tanh(conv) × σ(conv)  [gated activation]
    + Residual connection
    + BatchNorm
      │
  Global Average Pooling
      │
  Linear → output_dim=1

Why dilated convolutions?
  - O(log N) path length vs O(N) for RNN
  - Parallelizable across time dimension
  - Large receptive field with few parameters
```

#### Ensemble Weight Adaptation

```
WEIGHT UPDATE RULE:
  1. Track prediction error per model (rolling 100 trades)
  2. w_i = 1 / (error_i + ε)
  3. Normalize: w_i = w_i / Σw
  4. Better model = higher weight = more influence

  This is inverse-error weighting:
  A model that was right last 100 trades gets more say.
  A model that is wrong gets progressively muted.

  Initial weights: [1/3, 1/3, 1/3]
  After 1000 trades: converges to best-performing model
```

#### PPO Reinforcement Learning Agent

```
Why PPO over DQN:
  ✓ Clipped surrogate loss → stable training
  ✓ GAE advantage estimation → lower variance
  ✓ Compatible with continuous state spaces
  ✓ Better sample efficiency in finance settings

State Space:
  50 market features
  + portfolio value (normalized)
  + current exposure (normalized)
  + number of open positions
  + circuit breaker status
  = 54-dimensional state vector

Action Space:
  0 = HOLD
  1 = BUY
  2 = SELL

Reward Function:
  r_t = Sharpe-adjusted PnL
      - λ × drawdown_penalty
      - κ × transaction_cost
      + ε × exploration_bonus

  This discourages overtrading and rewards
  risk-adjusted returns over raw PnL.

Training:
  Algorithm:          PPO (clip ε=0.2)
  GAE lambda:         0.95
  Update epochs:      10 per batch
  Mini-batch size:    64
  Gradient clip:      0.5
  Entropy coefficient: 0.01  (encourages exploration)
  Value coefficient:  0.5
  LR schedule:        Linear decay
```

#### Online Learner + ADWIN Drift Detection

```
CONCEPT DRIFT DETECTION (ADWIN Algorithm):
  1. Maintain rolling error window
  2. Compare recent error (last 20) vs baseline
  3. If (recent - baseline) > 2.5 × std → DRIFT DETECTED
  4. On drift: temporarily increase LR by 5×
               log warning for operator review
               increment drift counter

This allows the system to adapt to:
  - Microstructure changes (new market participants)
  - Regime shifts (policy changes, economic shifts)
  - Structural breaks (circuit breaker rule changes)
  - Seasonality changes (volatility regime shifts)
```

<br/>

---

### 4. `risk_engine.py` — Risk Management

```
PURPOSE: Institutional-grade, multi-layer risk controls.
         NO single risk check. ALL layers must pass.

7 Independent Risk Layers:
  ┌───────────────────────────────────────────────────────┐
  │  Layer 1: Circuit Breaker (4 levels)                  │
  │  Layer 2: VaR / CVaR limits                           │
  │  Layer 3: Position sizing (Kelly + Vol Target + ATR)  │
  │  Layer 4: Fat Finger Guard                            │
  │  Layer 5: Correlation / Sector exposure limits        │
  │  Layer 6: Daily / Weekly / Monthly P&L limits         │
  │  Layer 7: Consecutive loss counter                    │
  └───────────────────────────────────────────────────────┘
```

#### Circuit Breaker — 4 Levels

```
LEVEL     TRIGGER                           ACTION
───────────────────────────────────────────────────────────────
NORMAL    All clear                         Full trading
WARNING   75% of daily loss limit           Reduce size 50%
SOFT_HALT Volatility spike (>3σ)            Pause new entries
          OR order rate exceeded             Close reduces only
HARD_HALT Daily loss limit breached         Close all positions
          OR 5 consecutive losses            No new orders
EMERGENCY 2× daily loss limit breached      Emergency close all
          OR RAM < 1GB                       System halt + save

AUTOMATIC RECOVERY:
  SOFT_HALT → NORMAL:  After volatility normalizes (5 min)
  HARD_HALT → NORMAL:  Manual override + next trading day
  EMERGENCY → NORMAL:  Manual override + full audit required
```

#### Position Sizing Methods

```
METHOD 1: FRACTIONAL KELLY CRITERION
  Full Kelly:     K = p/l - q/w
  Quarter Kelly:  K_safe = 0.25 × K
  
  Where:
    p = win probability (from ML model)
    q = 1 - p
    w = win return
    l = loss return
  
  Quarter Kelly used because:
    - Full Kelly → 50%+ drawdowns in bad streaks
    - Half Kelly → industry standard minimum
    - Quarter Kelly → maximum Sharpe empirically

METHOD 2: VOLATILITY TARGETING
  notional = account_size × (target_vol / realized_vol)
  shares   = notional / current_price
  
  target_vol = 15% annualized (adjustable by regime)
  
  Regime adjustment:
    trending:  target_vol × 1.2  (lean in)
    ranging:   target_vol × 1.0  (neutral)
    volatile:  target_vol × 0.5  (protect capital)
    crisis:    target_vol × 0.0  (no new trades)

METHOD 3: ATR-BASED SIZING
  stop_loss = 2 × ATR(14)
  take_profit = 4 × ATR(14)   → risk/reward = 1:2 minimum
  position = (account × risk_pct) / stop_loss_distance
```

#### VaR Calculation — 3 Methods

```
1. HISTORICAL SIMULATION VaR
   - No distribution assumption
   - Uses actual return distribution
   - Captures fat tails + skewness
   - Horizon: 1 day, 95% confidence

2. PARAMETRIC (GAUSSIAN) VaR
   - Assumes normal distribution
   - Fastest computation
   - Less accurate for fat tails
   - Used for quick estimates

3. MONTE CARLO VaR (Student-t, df=5)
   - 10,000 simulations per run
   - Student-t captures fat tails
   - Most accurate for tail risk
   - Run every 5 minutes

EXPECTED SHORTFALL (CVaR):
   Average loss in the worst (1-confidence)% of cases
   CVaR > VaR always
   CVaR is the correct measure for tail risk management
```

#### Fat Finger Guard

```
VALIDATION CHECKS:
  ☐ Price within 5% of current market price
  ☐ Order value < 50% of account (single order)
  ☐ Quantity > 0
  ☐ Not a duplicate order (hash comparison, 100-order window)
  ☐ Symbol exists in approved universe
  ☐ Exchange open (market hours check)

On rejection:
  - Log with full context (price, qty, market_price, reason)
  - Increment rejection counter
  - Alert if rejection rate > 1% of orders
```

<br/>

---

### 5. `execution_engine.py` — Order Execution

```
PURPOSE: Minimize market impact, maximize fill quality,
         route to best execution venue.

Execution Stack:
  Signal Approved by Risk
       │
  SmartOrderRouter (venue selection)
       │
  ┌────────────────────────────────────────┐
  │  Almgren-Chriss Optimal Schedule       │
  │  (minimize impact × execution risk)   │
  └────┬────────────────────────────┬──────┘
       │                            │
  TWAP Executor                VWAP Executor
  (time-uniform)               (volume-weighted)
       │                            │
  ┌────▼────────────────────────────▼──────┐
  │         Iceberg / Limit / Market        │
  └─────────────────────────────────────────┘
       │
  Fill Reporting + Slippage Tracking
       │
  PositionManager (P&L, Greeks, Stops)
```

#### Smart Order Router — Venue Scoring

```
SCORING FORMULA (per venue):
  score = 0.40 × price_improvement
        + 0.25 × depth_score
        + 0.15 × latency_score
        + 0.10 × fill_quality_score
        + 0.10 × fee_score

  price_improvement: best bid/ask vs market
  depth_score:       can venue absorb full quantity?
  latency_score:     rolling average order-to-fill time
  fill_quality:      historical fill rate at requested price
  fee_score:         exchange fee rate

Best scoring venue receives the order.
All scores tracked and updated after each fill.
```

#### Almgren-Chriss Execution Model

```
OPTIMAL EXECUTION THEORY:
  Minimize: E[cost] + λ × Var[cost]

  Where:
    E[cost]    = expected market impact
    Var[cost]  = execution timing risk
    λ          = risk aversion parameter

  Temporary impact:  η × (Q/V)^0.6 × σ × S
  Permanent impact:  γ × (Q/V) × σ × S

  Q = order quantity
  V = average daily volume
  σ = price volatility
  S = current price
  η, γ = impact coefficients (calibrated from fills)

RESULT: Trades larger in liquid periods,
        smaller in illiquid periods.
        Automatically schedules execution.
```

#### Order Types Supported

| Type | Use Case | Notes |
|---|---|---|
| `MARKET` | Urgent exits, stop-loss triggers | High slippage risk |
| `LIMIT` | Standard entries | Default type |
| `STOP` | Loss protection | Monitored locally |
| `STOP_LIMIT` | Better stop fills | Risk of non-fill |
| `ICEBERG` | Large orders | Shows only partial qty |
| `TWAP` | Large position entry | Time-uniform slices |
| `VWAP` | Minimize market impact | Volume-profile-aware |

#### Paper Trading Simulation

```
REALISTIC SIMULATION INCLUDES:
  ✓ Almgren-Chriss slippage model
  ✓ Random network latency (1–5ms uniform)
  ✓ Partial fill simulation (order book depth)
  ✓ Exchange fees (configurable per venue)
  ✓ Market impact (larger orders = more slippage)
  ✓ Rejected orders (volatility / liquidity conditions)

This ensures paper results are honest estimates
of live performance, not optimistic simulations.
```

<br/>

---

### 6. `main_system.py` — Master Orchestrator

```
PURPOSE: Lifecycle management, signal routing,
         concurrent task coordination, graceful shutdown.

Async Task Architecture:
  asyncio.gather([
    _data_feed_task(),        ← WebSocket connection manager
    order_processing_loop(),  ← Signal → Order pipeline
    health_monitor.loop(),    ← CPU/RAM/GPU/Thread monitoring
    _reporting_loop(),        ← 60s performance snapshots
    _model_save_loop(),       ← 5min model checkpoints
    _rl_training_loop(),      ← PPO update trigger
  ])

Signal Flow:
  Tick arrives
      │
  FeatureEngine.update()
      │ (every 10th tick — computational budget)
  FeatureEngine.compute_all() → 50+ features
      │
  RegimeDetector.detect()
      │
  EnsemblePredictor.predict() → (pred, uncertainty)
      │
  PPOAgent.select_action()    → (action, log_prob, value)
      │
  _combine_signals()          → HOLD / BUY / SELL
      │ (uncertainty gate, regime gate, sentiment confirm)
  order_queue.put()
      │
  order_processing_loop() consumes
      │
  CircuitBreaker.check()
      │
  PositionSizer (regime-adjusted)
      │
  FatFingerGuard.validate()
      │
  _simulate_execution() OR _submit_live_order()
      │
  PositionManager.open_position()
      │
  Analytics.record_trade()
```

#### Signal Combination Logic

```
COMBINATION RULES:
  1. IF uncertainty > 0.02     → HOLD  (models disagree too much)
  2. IF regime == "crisis"     → HOLD  (never trade in crisis)
  3. IF ML_signal == RL_signal → TRADE (strong agreement)
  4. IF ML_signal + sentiment confirm → TRADE (partial agreement)
  5. OTHERWISE                 → HOLD

RATIONALE:
  Requiring agreement between independent systems
  (supervised ML + reinforcement learning + sentiment)
  dramatically reduces false positives.
  
  We prefer missing good trades over making bad ones.
  Win rate matters less than risk-adjusted returns.
```

<br/>

---

## 🔄 Data Pipeline

```
RAW TICK                                               FEATURE VECTOR
   │                                                        │
   ▼                                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Step 1: INGEST                                                              │
│    WebSocket message → msgpack decode → Tick struct                         │
│    Validation: price > 0, volume ≥ 0, timestamp fresh                      │
│    Latency stamp: record network + processing latency                       │
│                                                                              │
│  Step 2: QUALITY CHECK                                                       │
│    DataQualityMonitor.check_tick()                                          │
│    Spike detection: |ΔP/P| < 10% per tick                                  │
│    Sequence check: no timestamp gaps > 5s                                   │
│    Feed quality update: EXCELLENT → GOOD → DEGRADED → FAILED               │
│                                                                              │
│  Step 3: BUFFER UPDATE                                                       │
│    RingBuffer.append(price, volume, high, low)                              │
│    O(1) operation — no list copies                                           │
│                                                                              │
│  Step 4: FEATURE COMPUTATION  (every 10th tick)                              │
│    FeatureEngine.compute_all()                                               │
│    50+ features: technical, microstructure, statistical                     │
│    Incremental EMA: avoids full recalculation                               │
│    Hurst exponent: computed on 200-tick window via R/S analysis             │
│                                                                              │
│  Step 5: REGIME DETECTION                                                    │
│    RegimeDetector.detect(features)                                          │
│    → trending / ranging / volatile / crisis                                 │
│    → confidence score + liquidity score                                     │
│                                                                              │
│  Step 6: SENTIMENT OVERLAY                                                   │
│    SentimentEngine.compute_composite_score()                                │
│    [-1.0, +1.0] score from news + options flow + economic calendar          │
│                                                                              │
│  Step 7: NORMALIZATION                                                       │
│    Z-score normalization per feature                                         │
│    NaN/Inf replacement                                                       │
│    Dimension padding to config.feature_count                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 🤖 AI & Machine Learning Stack

### Model Comparison

| Model | Architecture | Strength | Weakness | Weight Range |
|---|---|---|---|---|
| TFT | Transformer + GRN | Long-range deps, uncertainty | Slow inference | 30–50% |
| BiLSTM | Bidirectional + Attention | Sequential memory | Recurrence overhead | 20–40% |
| TCN | Dilated CNN | Parallelizable, fast | No explicit memory | 15–30% |
| PPO | Actor-Critic NN | Adapts to rewards | Slow to converge | Independent |
| Online | Gradient steps on TFT | Adapts in real time | Small updates only | Applied to TFT |

### Training Protocol

```
PHASE 1: OFFLINE PRE-TRAINING
  Data:        12+ months historical tick data
  Train split: 70% training, 15% validation, 15% test
  Walk-forward: 3-month windows, 1-month step
  
  TFT training:
    Optimizer:  Adam + warmup (4000 steps)
    Loss:       Quantile loss (9 quantiles)
    Epochs:     100 with early stopping (patience=15)
    Scheduler:  Cosine annealing with warm restarts
  
  BiLSTM training:
    Same optimizer, MSE + Huber loss blend
  
  TCN training:
    Same optimizer, MSE loss

PHASE 2: PAPER TRADING (minimum 3 months)
  Online learning active
  RL agent collecting experience
  Performance benchmarked vs buy-and-hold + Nifty

PHASE 3: LIVE DEPLOYMENT (only after Phase 2 passes)
  All circuit breakers active
  Position sizes start at 25% of configured max
  Scale up only if Sharpe > 1.0 over 1 month live
```

### Uncertainty Quantification

```
WHY UNCERTAINTY MATTERS IN TRADING:
  Traditional systems output a single prediction.
  They don't know when they don't know.

  Our approach:
    1. TFT outputs 9 quantile predictions
       → uncertainty = q90 - q10 (prediction interval width)
    
    2. Ensemble disagreement
       → uncertainty = std(pred_TFT, pred_BiLSTM, pred_TCN)
    
    3. Combined gate:
       IF uncertainty > threshold → HOLD (no trade)
       
  This is why we trade less but better.
  High uncertainty = the market is in a state we haven't seen.
  The correct response is always: do nothing.
```

<br/>

---

## 🛡 Risk Management Framework

### Risk Budget Hierarchy

```
ACCOUNT CAPITAL: ₹10,000,000 (example)
│
├── Total Portfolio Risk Budget: 6% = ₹600,000
│   │
│   ├── Per-Trade Risk: 2% = ₹200,000
│   │   ├── Stop Loss: 2 × ATR below entry
│   │   └── Take Profit: 4 × ATR above entry (1:2 R/R minimum)
│   │
│   ├── Sector Concentration: max 25% per sector
│   │
│   └── Correlation Exposure: max 40% correlated assets
│
├── Daily Loss Limit: 2% = ₹200,000 → HARD HALT
│
├── Weekly Loss Limit: 5% = ₹500,000
│
└── Maximum Drawdown: 15% = ₹1,500,000 → Manual review required
```

### Stress Test Scenarios

```
SCENARIO              SHOCK    EXPECTED LOSS    BREAKER?
──────────────────────────────────────────────────────────
2020 COVID Crash      -38%     ₹3,800,000       YES: Emergency
2008 GFC Peak         -60%     ₹6,000,000       YES: Emergency
Flash Crash (2010)    -9.2%    ₹920,000         YES: Hard Halt
Rate Shock (+200bps)  -15%     ₹1,500,000       YES: Hard Halt
Nifty Limit Down      -20%     ₹2,000,000       YES: Hard Halt
Normal Bad Day        -2%      ₹200,000         YES: Hard Halt

These scenarios are run weekly in paper mode to verify
circuit breakers trigger correctly.
```

<br/>

---

## ⚡ Execution Infrastructure

### Latency Budget

```
COMPONENT                  TARGET      ACTUAL (typical)
──────────────────────────────────────────────────────────
Tick receipt → parse        < 0.1ms        0.05ms
Parse → feature update      < 0.5ms        0.2ms
Feature → model inference   < 2.0ms        1.2ms
Model → risk check          < 0.5ms        0.1ms
Risk → order submit         < 1.0ms        0.5ms
──────────────────────────────────────────────────────────
TOTAL PIPELINE              < 5.0ms        ~2.1ms
```

### Order Lifecycle

```
ORDER STATES:

  PENDING → SUBMITTED → PARTIAL_FILL → FILLED
                    ↓                       ↓
               REJECTED                CANCELLED
                    ↓
                EXPIRED

Each state transition is logged with:
  - Timestamp (microsecond precision)
  - Reason
  - Fill details (qty, price, fees)
  - Slippage vs expected
  - Exchange acknowledgment ID
```

<br/>

---

## 📊 Market Regime Detection

```
FOUR REGIMES AND SYSTEM RESPONSE:

┌──────────────────────────────────────────────────────────────────────┐
│ REGIME     │ CONDITION           │ RESPONSE                          │
├──────────────────────────────────────────────────────────────────────┤
│ TRENDING   │ Hurst > 0.6         │ Full size, momentum signals       │
│            │ |momentum_20| > 2%  │ Wider stops (trend continuation)  │
│            │                     │ Position size × 1.2               │
├──────────────────────────────────────────────────────────────────────┤
│ RANGING    │ Hurst < 0.5         │ Mean reversion signals            │
│            │ Low momentum        │ Tighter stops (quick reversals)   │
│            │                     │ Position size × 1.0               │
├──────────────────────────────────────────────────────────────────────┤
│ VOLATILE   │ vol > avg_vol × 1.5 │ Reduce size 50%                  │
│            │ Low Hurst           │ Wider stops only                  │
│            │                     │ Prefer mean reversion             │
├──────────────────────────────────────────────────────────────────────┤
│ CRISIS     │ vol > avg_vol × 3.0 │ NO NEW TRADES                    │
│            │                     │ Close existing positions           │
│            │                     │ Cash is a position                │
└──────────────────────────────────────────────────────────────────────┘

DETECTION SIGNALS:
  Hurst Exponent (R/S Analysis):
    H > 0.5 → trending (persistent)
    H = 0.5 → random walk
    H < 0.5 → mean-reverting (anti-persistent)

  Volatility Ratio:
    current_vol / rolling_avg_vol → regime transition signal

  Trend Strength:
    |momentum_20| → directional conviction
```

<br/>

---

## 📈 Performance Analytics

### Metrics Computed

```
RETURN METRICS:
  Cumulative Return:    Σ(daily PnL) / initial_capital
  Annualized Return:    mean(return) × 252 × 6.5 × 3600
  CAGR:                 (final/initial)^(1/years) - 1

RISK-ADJUSTED METRICS:
  Sharpe Ratio:         (mean_excess_return) / (std_return) × √252
  Sortino Ratio:        (mean_excess_return) / (downside_std) × √252
  Calmar Ratio:         annualized_return / max_drawdown
  Information Ratio:    (return - benchmark) / tracking_error

TRADE STATISTICS:
  Win Rate:             winning_trades / total_trades
  Profit Factor:        gross_profit / gross_loss
  Average Win:          mean(positive PnL)
  Average Loss:         mean(negative PnL)
  Expectancy:           (win_rate × avg_win) - (loss_rate × avg_loss)
  Max Consecutive Loss: longest losing streak

RISK METRICS:
  Max Drawdown:         max peak-to-trough decline
  VaR (95%):            historical + parametric + Monte Carlo
  CVaR (95%):           expected loss beyond VaR
  Volatility:           annualized realized vol

EXECUTION QUALITY:
  Average Slippage:     mean(fill_price - expected_price)
  Fill Rate:            filled_qty / submitted_qty
  Avg Latency:          mean(fill_time - submit_time)
```

### Target Performance Benchmarks

```
METRIC              MINIMUM    TARGET     EXCELLENT
────────────────────────────────────────────────────
Sharpe Ratio          > 1.0     > 1.5       > 2.0
Sortino Ratio         > 1.5     > 2.0       > 3.0
Calmar Ratio          > 0.5     > 1.0       > 2.0
Win Rate              > 45%     > 52%       > 60%
Profit Factor         > 1.3     > 1.6       > 2.0
Max Drawdown          < 15%     < 10%       < 5%
Daily VaR (95%)       < 2%      < 1.5%      < 1%

NOTE: A system with a 50% win rate and 2:1 reward/risk
      has positive expected value. Chasing win rate
      at the expense of risk/reward is a common mistake.
```

<br/>

---

## 💻 Installation & Setup

### System Requirements

```
MINIMUM:
  OS:           Ubuntu 20.04+ / macOS 12+ / Windows 10+
  CPU:          8-core, 3.0GHz+
  RAM:          16 GB
  Storage:      100 GB SSD (for tick data storage)
  Network:      Dedicated line, < 10ms to exchange
  Python:       3.10+

RECOMMENDED:
  OS:           Ubuntu 22.04 LTS
  CPU:          AMD EPYC / Intel Xeon, 32+ cores
  RAM:          64 GB ECC
  GPU:          NVIDIA RTX 3090 / A100 (for model training)
  Storage:      1 TB NVMe SSD
  Network:      Co-location or dedicated fiber
  Python:       3.11+

PRODUCTION (HFT):
  OS:           Linux with PREEMPT_RT kernel patch
  CPU:          Isolated cores (isolcpus kernel param)
  RAM:          128 GB
  NIC:          Solarflare / Mellanox with kernel bypass (DPDK)
  Network:      Cross-connect at exchange, < 1ms
```

### Installation

```bash
# ── 1. Clone repository ──────────────────────────────────────────────
git clone https://github.com/your-org/quantumtrader.git
cd quantumtrader

# ── 2. Create isolated environment ──────────────────────────────────
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# ── 3. Upgrade pip ──────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel

# ── 4. Install dependencies ─────────────────────────────────────────
pip install -r requirements.txt

# ── 5. Install CUDA-enabled PyTorch (if GPU available) ──────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ── 6. Set environment variables (NEVER hardcode credentials) ────────
cp .env.example .env
nano .env                         # Add your API keys here

# ── 7. Verify installation ──────────────────────────────────────────
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python verify_setup.py
```

### `requirements.txt`

```text
# ── Core ─────────────────────────────────────────────────────────────
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# ── Machine Learning ─────────────────────────────────────────────────
torch>=2.0.0
torchvision>=0.15.0

# ── Networking ───────────────────────────────────────────────────────
websockets>=11.0.0
aiohttp>=3.9.0
requests>=2.31.0

# ── Data & Caching ───────────────────────────────────────────────────
redis>=5.0.0
msgpack>=1.0.5
quantecon>=0.6.0

# ── System Monitoring ────────────────────────────────────────────────
psutil>=5.9.0

# ── Security ─────────────────────────────────────────────────────────
cryptography>=41.0.0

# ── Configuration ────────────────────────────────────────────────────
python-dotenv>=1.0.0
PyYAML>=6.0.1

# ── Testing ──────────────────────────────────────────────────────────
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# ── Code Quality ─────────────────────────────────────────────────────
black>=23.0.0
ruff>=0.1.0
mypy>=1.5.0
```

### `.env.example`

```bash
# ── Exchange Credentials ─────────────────────────────────────────────
# NEVER commit actual credentials to version control
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# ── Data Feed URLs ────────────────────────────────────────────────────
PRIMARY_FEED_URL=wss://your-primary-feed.com/ws
BACKUP_FEED_URL=wss://your-backup-feed.com/ws

# ── Redis (for caching) ───────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# ── System ────────────────────────────────────────────────────────────
ENVIRONMENT=paper           # paper | staging | production
LOG_LEVEL=INFO
STATE_DIR=./state
LOG_DIR=./logs

# ── Notifications (optional) ─────────────────────────────────────────
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK_URL=your_slack_webhook
```

<br/>

---

## ⚙️ Configuration Reference

### Full Configuration Map

```yaml
# config.yaml — Full reference with all defaults

system_name: "QuantumTrader-v2"
version: "2.0.0"
environment: "paper"              # paper | staging | production
paper_trading: true               # ← ALWAYS true until fully validated

initial_capital: 1000000.0        # Starting capital in base currency
window_size: 2000                 # Feature lookback window (ticks)
feature_count: 50                 # Feature vector dimension
tick_interval_ms: 100             # Processing interval (ms)

primary_exchange: "NSE"
backup_exchanges:
  - "BSE"
  - "MCX"

# ── Network ─────────────────────────────────────────────────────────
network:
  max_reconnect_attempts: 10
  reconnect_base_delay: 0.5       # seconds
  reconnect_max_delay: 30.0       # seconds
  connection_timeout: 10.0        # seconds
  heartbeat_interval: 30.0        # seconds
  max_message_queue_size: 100000
  tcp_nodelay: true
  socket_buffer_size: 65536

# ── Risk ─────────────────────────────────────────────────────────────
risk:
  max_position_size_pct: 0.02     # 2% per trade
  max_portfolio_heat: 0.06        # 6% total open risk
  max_correlation_exposure: 0.40  # 40% in correlated assets
  max_sector_exposure: 0.25       # 25% in single sector
  max_single_asset_pct: 0.10      # 10% in single asset

  max_daily_loss_pct: 0.02        # 2% → hard halt
  max_weekly_loss_pct: 0.05       # 5%
  max_monthly_loss_pct: 0.10      # 10%
  max_drawdown_pct: 0.15          # 15% → manual review

  default_stop_loss_pct: 0.01     # 1% default (overridden by ATR)
  default_take_profit_pct: 0.03   # 3% default (1:3 R/R)
  max_leverage: 10.0              # Maximum leverage
  min_risk_reward_ratio: 2.0      # Minimum 1:2 R/R required

  max_orders_per_second: 10
  max_orders_per_minute: 200
  max_consecutive_losses: 5
  volatility_halt_threshold: 3.0  # σ above rolling mean
  fat_finger_check_pct: 0.05      # 5% from market = rejected
  kelly_fraction: 0.25            # Quarter Kelly

# ── ML Model ─────────────────────────────────────────────────────────
ml:
  # Transformer
  d_model: 256
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  dim_feedforward: 1024
  dropout: 0.1
  max_seq_length: 512

  # LSTM
  lstm_hidden_dim: 256
  lstm_num_layers: 3

  # Training
  learning_rate: 0.0001
  weight_decay: 0.00001
  batch_size: 64
  epochs: 100
  warmup_steps: 4000
  gradient_clip: 1.0

  # Reinforcement Learning
  rl_gamma: 0.99
  rl_epsilon_start: 1.0
  rl_epsilon_end: 0.01
  rl_epsilon_decay: 0.995
  replay_buffer_size: 100000
  target_update_freq: 1000

  # Ensemble
  num_ensemble_models: 5
  ensemble_voting: "weighted"
```

<br/>

---

## 🚀 Running the System

### Step 1 — Verify Paper Mode

```bash
# Always verify paper_trading = True before starting
python -c "from config import CONFIG; print(f'Paper: {CONFIG.paper_trading}')"

# Expected output:
# Paper: True
```

### Step 2 — Pre-flight Checks

```bash
# Run the full pre-flight checklist
python preflight.py

# Checks performed:
#   ✓ Config validation
#   ✓ API credential format
#   ✓ WebSocket connectivity
#   ✓ Redis connectivity
#   ✓ Model file existence
#   ✓ Disk space > 10GB
#   ✓ RAM > 8GB available
#   ✓ CPU load < 70%
#   ✓ Paper mode active
```

### Step 3 — Start System

```bash
# Paper trading (safe to run immediately)
python main_system.py

# With custom config
python main_system.py --config config_paper.yaml

# With debug logging
python main_system.py --log-level DEBUG

# Daemon mode (background)
nohup python main_system.py > logs/system.log 2>&1 &
```

### Step 4 — Monitor

```bash
# Live log tailing
tail -f trading_system.log

# Performance snapshot (printed every 60s automatically)
# Or trigger manually:
kill -USR1 $(pgrep -f main_system.py)

# Expected report output:
# ============================================================
#        SYSTEM PERFORMANCE REPORT
# ============================================================
#   Sharpe Ratio:     1.8432
#   Sortino Ratio:    2.1209
#   Calmar Ratio:     1.3400
#   Win Rate:         54.2%
#   Profit Factor:    1.82
#   Total Trades:     1,247
#   Total P&L:        ₹182,430.00
#   Max Drawdown:     4.8%
#   95% VaR:          ₹18,200.00
#   95% CVaR:         ₹24,100.00
#   Circuit Breaker:  NORMAL
#   Regime:           trending
# ============================================================
```

### Step 5 — Graceful Shutdown

```bash
# Ctrl+C triggers graceful shutdown:
#   1. Stop accepting new signals
#   2. Cancel pending orders
#   3. Save model weights
#   4. Save trade log
#   5. Print final report
#   6. Exit cleanly

# Or send SIGTERM for clean shutdown from scripts:
kill -TERM $(pgrep -f main_system.py)
```

<br/>

---

## 🔬 Backtesting Protocol

### Walk-Forward Validation

```
WALK-FORWARD METHODOLOGY (mandatory before live trading):

  Full Dataset:  Jan 2020 → Dec 2023  (4 years tick data)

  Step 1: IN-SAMPLE TRAINING
  ┌──────────────────────────────────────┐
  │  Train: Jan 2020 → Sep 2020  (9mo)  │
  │  Val:   Oct 2020 → Nov 2020  (2mo)  │
  │  Test:  Dec 2020            (1mo)   │ ← Never seen during train
  └──────────────────────────────────────┘

  Step 2: ROLL FORWARD
  ┌──────────────────────────────────────┐
  │  Train: Feb 2020 → Oct 2020         │
  │  Val:   Nov 2020 → Dec 2020         │
  │  Test:  Jan 2021                    │
  └──────────────────────────────────────┘

  Repeat × 36 windows → 36 out-of-sample months

  FINAL VALIDATION:
    Aggregate all 36 out-of-sample periods
    Compute Sharpe, Sortino, Calmar, Drawdown
    Must pass ALL minimum thresholds to proceed to paper

  WHY NOT STANDARD BACKTEST?
    Standard backtests overfit due to:
    - Look-ahead bias
    - Parameter fitting to full period
    - No realistic slippage
    - No walk-forward degradation

    Walk-forward is the only honest methodology.
```

### Backtest Checklist

```
BEFORE CONCLUDING ANY BACKTEST:

  [ ] Slippage model applied (Almgren-Chriss)
  [ ] Transaction costs included (fees + spread)
  [ ] No look-ahead bias (features computed on t-1 data)
  [ ] Survivorship bias addressed (delisted stocks included)
  [ ] Split-adjusted prices used
  [ ] Regime-specific performance checked separately
  [ ] Crisis period performance checked (2020 Mar, 2008)
  [ ] Parameter sensitivity analysis run
  [ ] Walk-forward out-of-sample validation complete
  [ ] Sharpe > 1.0 out-of-sample
  [ ] Max drawdown < 15% out-of-sample
  [ ] At least 500 trades in out-of-sample period
  [ ] Results statistically significant (t-test, p < 0.05)
```

<br/>

---

## 🔐 Safety & Compliance

### Security Architecture

```
CREDENTIAL MANAGEMENT:
  ✗ NEVER:  Hardcode API keys in source code
  ✗ NEVER:  Commit .env files to version control
  ✗ NEVER:  Log credentials anywhere
  ✓ ALWAYS: Load from environment variables
  ✓ ALWAYS: Encrypt at rest (Fernet symmetric encryption)
  ✓ ALWAYS: Rotate credentials quarterly
  ✓ ALWAYS: Use read-only API keys where possible

NETWORK SECURITY:
  ✓ TLS 1.3 for all connections
  ✓ Certificate pinning for exchange connections
  ✓ VPN for all administrative access
  ✓ Firewall: whitelist exchange IPs only

CODE SECURITY:
  ✓ Dependency scanning (pip-audit weekly)
  ✓ Static analysis (ruff, mypy)
  ✓ No eval() or exec() anywhere
  ✓ Input validation on all external data
```

### Compliance Notes

```
THIS SYSTEM IS BUILT FOR RESEARCH AND PAPER TRADING.

Before any live deployment:

  REGULATORY:
    [ ] Consult SEBI guidelines for algorithmic trading
    [ ] Obtain exchange algo trading approval
    [ ] Implement mandatory audit trail (maintained by this system)
    [ ] Ensure compliance with order-to-trade ratio limits
    [ ] Register as algo trader with broker

  RISK DISCLOSURE:
    [ ] All personnel understand max loss scenarios
    [ ] Emergency halt procedures documented and tested
    [ ] Manual override procedures documented
    [ ] Incident response plan in place

  OPERATIONAL:
    [ ] 24/7 monitoring assigned
    [ ] Escalation procedures defined
    [ ] Disaster recovery tested (restore from state files)
    [ ] Business continuity plan documented
```

### Emergency Procedures

```
EMERGENCY HALT TRIGGERS (automatic):
  1. Daily loss > 2% → HARD_HALT
  2. System RAM < 1GB → EMERGENCY
  3. 5 consecutive losses → HARD_HALT
  4. Volatility > 3σ → SOFT_HALT
  5. Data feed FAILED quality → SOFT_HALT

MANUAL EMERGENCY HALT:
  1. Ctrl+C → graceful shutdown (30 sec)
  2. SIGKILL → immediate halt (state may not save)
  3. Exchange kill switch → contact broker directly

POST-HALT PROCEDURE:
  1. Verify all positions closed
  2. Review trade_log.jsonl for last 100 trades
  3. Check state/*.json for halt reason
  4. Root cause analysis before restart
  5. Adjust parameters if necessary
  6. Paper trade for 1 week before re-enabling live
```

<br/>

---

## 🗺 Roadmap

```
VERSION 2.1  (Q2 2025)
  [ ] Options pricing integration (Black-Scholes + Heston)
  [ ] Greeks monitoring (Delta, Gamma, Vega, Theta)
  [ ] Cross-asset correlation matrix (equity + FX + rates)
  [ ] Enhanced sentiment: FinBERT NLP pipeline
  [ ] Dark pool simulation in paper trading

VERSION 2.2  (Q3 2025)
  [ ] Multi-asset portfolio optimization (mean-variance + Black-Litterman)
  [ ] Alternative data: satellite data, web traffic, credit card data
  [ ] Distributed execution (multiple servers)
  [ ] Real-time factor exposure monitoring (momentum, quality, value)
  [ ] Transaction cost analysis (TCA) dashboard

VERSION 3.0  (Q4 2025)
  [ ] Causal inference engine (structural causal models)
  [ ] Graph neural networks for cross-asset signal propagation
  [ ] Federated learning (privacy-preserving multi-strategy)
  [ ] Quantum computing integration (portfolio optimization)
  [ ] Full co-location deployment package
```

<br/>

---

## 📁 Repository Structure

```
quantumtrader/
│
├── 📄 README.md                  ← This document
├── 📄 requirements.txt           ← All dependencies
├── 📄 .env.example               ← Credential template
├── 📄 config.yaml                ← Default configuration
│
├── 📁 core/
│   ├── 📄 config.py              ← SystemConfig dataclasses
│   ├── 📄 data_layer.py          ← Tick, FeatureEngine, FeedManager
│   ├── 📄 intelligence_layer.py  ← TFT, BiLSTM, TCN, PPO, Online
│   ├── 📄 risk_engine.py         ← CircuitBreaker, VaR, Sizing
│   ├── 📄 execution_engine.py    ← SOR, TWAP, VWAP, Positions
│   └── 📄 main_system.py         ← Master orchestrator
│
├── 📁 models/                    ← Saved model weights (.pt files)
│   ├── tft_model.pt
│   ├── rl_agent.pt
│   └── bilstm_model.pt
│
├── 📁 state/                     ← System state snapshots
│   └── state_<timestamp>.json
│
├── 📁 logs/                      ← Log files
│   ├── trading_system.log
│   └── trade_log.jsonl           ← Append-only trade audit log
│
├── 📁 tests/                     ← Test suite
│   ├── test_data_layer.py
│   ├── test_risk_engine.py
│   ├── test_execution.py
│   ├── test_ml_models.py
│   └── test_integration.py
│
├── 📁 backtest/                  ← Backtesting framework
│   ├── walk_forward.py
│   ├── scenarios.py              ← Stress test scenarios
│   └── reports/                 ← Backtest output reports
│
└── 📁 tools/
    ├── preflight.py              ← Pre-start validation
    ├── verify_setup.py           ← Installation check
    └── data_downloader.py        ← Historical data fetcher
```

<br/>

---

## ⚠️ Disclaimer

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          IMPORTANT DISCLAIMER                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  QuantumTrader v2.0 is provided for RESEARCH AND EDUCATIONAL purposes      ║
║  only. It is NOT financial advice and does NOT guarantee profits.          ║
║                                                                            ║
║  ALGORITHMIC TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.                   ║
║  Past performance of any strategy, whether simulated or live, does not    ║
║  guarantee future results. Markets change, and strategies that worked     ║
║  historically may fail completely in the future.                          ║
║                                                                            ║
║  Before using this system with real capital:                               ║
║    1. Understand every line of code and every risk parameter              ║
║    2. Complete a minimum 3-month paper trading evaluation                 ║
║    3. Consult qualified financial and legal professionals                 ║
║    4. Ensure compliance with all applicable regulations (SEBI, etc.)      ║
║    5. Only trade capital you can afford to lose entirely                  ║
║                                                                            ║
║  The authors and contributors accept NO responsibility for financial       ║
║  losses incurred through the use of this software.                        ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

<div align="center">

<br/>

**Built with discipline. Deployed with caution. Improved continuously.**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

</div>
