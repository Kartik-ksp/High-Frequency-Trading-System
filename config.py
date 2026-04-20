"""
config.py - Centralized Configuration Management
Handles all system parameters with validation and environment separation
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
import json

logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Network and connectivity configuration"""
    max_reconnect_attempts: int = 10
    reconnect_base_delay: float = 0.5
    reconnect_max_delay: float = 30.0
    connection_timeout: float = 10.0
    heartbeat_interval: float = 30.0
    max_message_queue_size: int = 100_000
    tcp_nodelay: bool = True           # Minimize latency
    socket_buffer_size: int = 65536

@dataclass
class RiskConfig:
    """Comprehensive risk management parameters"""
    # Position limits
    max_position_size_pct: float = 0.02      # 2% per position
    max_portfolio_heat: float = 0.06          # 6% total risk
    max_correlation_exposure: float = 0.40   # Max correlated exposure
    max_sector_exposure: float = 0.25        # Max single sector
    max_single_asset_pct: float = 0.10       # Max 10% in one asset

    # Loss limits
    max_daily_loss_pct: float = 0.02         # 2% daily drawdown halt
    max_weekly_loss_pct: float = 0.05        # 5% weekly
    max_monthly_loss_pct: float = 0.10       # 10% monthly
    max_drawdown_pct: float = 0.15           # 15% max drawdown

    # Trade parameters
    default_stop_loss_pct: float = 0.01
    default_take_profit_pct: float = 0.03
    max_leverage: float = 10.0               # Conservative vs original 50
    min_risk_reward_ratio: float = 2.0

    # Circuit breakers
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 200
    max_consecutive_losses: int = 5
    volatility_halt_threshold: float = 3.0   # Std deviations
    fat_finger_check_pct: float = 0.05       # 5% from market price

    # Kelly criterion
    kelly_fraction: float = 0.25            # Quarter Kelly for safety

@dataclass
class MLConfig:
    """Machine learning model configuration"""
    # Transformer model
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 512

    # LSTM config
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 3

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    warmup_steps: int = 4000
    gradient_clip: float = 1.0

    # Reinforcement Learning
    rl_gamma: float = 0.99
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.01
    rl_epsilon_decay: float = 0.995
    replay_buffer_size: int = 100_000
    target_update_freq: int = 1000

    # Ensemble
    num_ensemble_models: int = 5
    ensemble_voting: str = "weighted"       # weighted, majority, stacking

@dataclass
class SystemConfig:
    """Master system configuration"""
    # Identity
    system_name: str = "QuantumTrader-v2"
    version: str = "2.0.0"
    environment: str = "production"          # production, staging, paper

    # Capital
    initial_capital: float = 1_000_000.0
    paper_trading: bool = True              # ALWAYS start with paper

    # Data
    window_size: int = 2000
    tick_interval_ms: int = 100
    feature_count: int = 50
    alternative_data_enabled: bool = True
    sentiment_analysis_enabled: bool = True

    # Infrastructure
    num_worker_threads: int = 8
    num_async_workers: int = 4
    gpu_enabled: bool = torch.cuda.is_available() if True else False
    memory_limit_gb: float = 16.0

    # Sub-configs
    network: NetworkConfig = field(default_factory=NetworkConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    # Exchanges/feeds
    primary_exchange: str = "NSE"
    backup_exchanges: List[str] = field(
        default_factory=lambda: ["BSE", "MCX"]
    )
    data_feeds: List[str] = field(
        default_factory=lambda: ["primary", "backup", "alternative"]
    )

    @classmethod
    def from_yaml(cls, path: str) -> 'SystemConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []

        if self.risk.max_leverage > 50:
            errors.append("Leverage exceeds safe limits")
        if self.risk.max_daily_loss_pct > 0.05:
            errors.append("Daily loss limit too high")
        if self.initial_capital <= 0:
            errors.append("Invalid capital amount")

        if errors:
            for e in errors:
                logger.error(f"Config validation error: {e}")
            return False
        return True

# Global config instance
CONFIG = SystemConfig()
