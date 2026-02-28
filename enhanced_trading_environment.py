#!/usr/bin/env python3
"""
Enhanced Trading Environment with Strategy Balancing Mechanisms
Adds explicit balancing mechanisms to encourage both long and short strategies
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import warnings
from risk_management import RiskManager, PositionSide

warnings.filterwarnings("ignore", category=DeprecationWarning)

class EnhancedTradingEnvironment(gym.Env):
    """
    Enhanced trading environment with strategy balancing mechanisms
    """
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018, episode_length=200, 
                 start_step=None, debug=False, enable_strategy_balancing=True):
        super(EnhancedTradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start_step = 0
        self.steps_in_episode = 0
        self.start_step = start_step
        self.debug = debug
        self.enable_strategy_balancing = enable_strategy_balancing

        # Enhanced risk management parameters
        self.max_position_size = 0.10  # Exactly one trade
        self.max_total_exposure = 0.40
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        self.max_episode_loss_pct = 0.05

        # P1-FIX: Initialize RiskManager for dynamic position sizing
        self.risk_manager = RiskManager(
            initial_capital=self.initial_balance,
            max_position_size=0.10,  # 10% per position
            max_total_exposure=0.40,  # 40% total exposure
            stop_loss_pct=0.08,  # 8% stop loss
            take_profit_pct=0.15,  # 15% take profit
            max_drawdown_limit=0.20  # 20% max drawdown
        )

        # Dynamic position sizing parameters
        self.volatility_window = 20
        self.min_position_size = 0.02
        self.max_dynamic_position_size = 0.10  # Match max_position_size
        
        # Adaptive stop-loss parameters
        self.adaptive_stop_loss_enabled = True
        self.atr_multiplier = 2.0
        self.min_stop_loss_pct = 0.03
        self.max_stop_loss_pct = 0.15
        
        # Margin trading parameters
        self.margin_requirement = 0.3
        self.maintenance_margin = 0.15
        self.liquidation_penalty = 0.01

        # Trailing stop-loss parameters
        self.trailing_stop_distance = 0.05
        self.trailing_stop_enabled = True

        # Trading parameters
        self.base_position_size = 0.10  # Match max_position_size for single trade
        self.hold_penalty = 0.00001
        self.inactivity_penalty = 0.0001
        self.termination_stop_loss_threshold = 0.95
        self.termination_profit_target = 1.50

        # Anti-overtrading parameters
        # V18-FIX: Aggressive overtrading prevention
        self.min_hold_steps = 10  # V18: Increased from 5 to 10 - force patience
        self.min_open_cooldown = 10  # V18: Increased from 5 to 10 - wait between trades
        self.hold_penalty_start = 30  # V18: Increased from 20 to 30 - give more time
        
        # V18-FIX: Overtrading penalty
        self.overtrading_threshold = 20  # Max trades per episode before penalty
        self.overtrading_penalty = -10.0  # V18: Strong penalty for excessive trading
        
        # V18-FIX: Position time quality tracking
        self.quick_close_threshold = 5  # Closing within 5 steps = bad
        self.quick_close_penalty = -5.0  # Penalty for closing too quickly

        # Reward parameters
        # V18-FIX: Balanced rewards for quality trading
        self.reward_clip_bounds = (-50.0, 50.0)

        # V18-FIX: Asymmetric rewards to encourage shorts
        self.long_reward_multiplier = 0.8  # V18: Reduced to balance long/short
        self.short_reward_multiplier = 1.5  # V18: Higher reward for shorts!
        
        self.action_diversity_penalty = 0.001
        self.min_episode_length_for_rewards = 1
        self.short_episode_penalty = 0.0

        # ============================================================
        # SMART REWARD SYSTEM v12.0 - ENHANCED PROFITABILITY
        # Key improvements from v11:
        # - Higher profit scale (25 → 30)
        # - Missed profit penalty (was: none)
        # - Correct hold reward (was: none)
        # - Overtrading penalty (was: none)
        # - Dynamic profit threshold
        # ============================================================

        # 1. HOLD BONUS - V12: Stronger reward for profitable positions
        self.hold_bonus_enabled = True
        self.hold_bonus_profit_threshold = 0.005  # Only if profit > 0.5%
        self.hold_bonus_per_step = 0.05  # Stronger reward per step
        self.hold_bonus_cap = 1.0

        # 2. FORCED CLOSE PENALTY - V12: Same as v11
        self.max_hold_steps_before_penalty = 20
        self.hold_penalty_per_step = 0.05

        # 3. CLOSE REWARD - V18-FIX: Усиленный reward для КАЧЕСТВЕННЫХ закрытий
        # Model earns BIG reward for PROFITABLE closes
        self.smart_close_enabled = True
        self.profit_close_bonus_scale = 50.0  # V18: Increased from 40.0 to 50.0 (+25%)
        self.loss_close_penalty_scale = 3.0  # V18: Increased from 2.5 to 3.0 (+20%)
        self.min_profit_for_bonus = 0.001  # V18: Reduced from 0.002 - reward even small profits
        
        # V18-FIX: Quick close penalty (prevent flip-flopping)
        self.quick_close_penalty_enabled = True
        self.missed_profit_penalty = 0.0  # V12.1: Removed penalty
        
        # V12.1: Correct hold reward - REDUCED to avoid over-conservative behavior
        self.correct_hold_reward = 0.02  # V12.1: Reduced from 0.1 to 0.02
        
        # 4. TIME PENALTY - V12.1: Same as v12
        self.time_penalty_enabled = True
        self.time_penalty_per_step = 0.005
        self.time_penalty_start = 15
        self.time_penalty_cap = 2.0
        
        # 5. EARLY CLOSE PENALTY - REMOVED
        self.early_close_penalty_enabled = False
        self.early_close_threshold_steps = 5
        self.early_close_penalty = 0.0
        
        # V12.1-FIX: Profit multiplier - SLIGHTLY REDUCED
        self.profit_multiplier_enabled = True
        self.profit_multiplier_threshold = 0.008  # V12.1: Reduced from 0.01 to 0.008 (easier to trigger)
        self.profit_multiplier_scale = 2.0  # V12.1: Reduced from 2.5 to 2.0
        
        # V13-FIX: Removed trading bonus (model found exploit)
        self.trading_encouragement_enabled = False  # V13: DISABLED
        self.trading_bonus_per_trade = 0.0
        self.min_trades_for_bonus = 5
        
        # V13-FIX: Stronger invalid action penalty
        self.invalid_action_penalty = 1.0  # V13: Increased from 0.5 to 1.0
        
        # V16-FIX: Action Masking
        self.action_masking_enabled = False  # Will be set by training script
        
        # Track position hold time for forced close penalty
        self.position_open_time = 0  # v7.0: Track how long position has been open

        # Strategy balancing parameters (V11-IMPROVED)
        self.direction_balance_target = 0.5  # Target 50% long, 50% short
        
        # V11-FIX: Усиленные параметры для баланса
        self.direction_balance_penalty_weight = 0.5  # V11-FIX: Увеличено с 0.25 до 0.5 (2x)
        self.max_direction_concentration = 0.60  # V11-FIX: Уменьшено с 0.65 до 0.60 (строже)
        self.direction_streak_limit = 3  # V11-FIX: Уменьшено с 4 до 3 (строже)
        self.exploration_bonus_weight = 0.25  # V11-FIX: Увеличено с 0.15 до 0.25
        
        # V11-FIX: Усиленные bonus за контр-тренд
        self.long_open_bonus = 0.5  # V11-FIX: Увеличено с 0.20 до 0.50 (2.5x)
        self.short_open_bonus = 0.5  # V11-FIX: Увеличено с 0.20 до 0.50 (2.5x)
        
        # ============================================================
        # HYBRID MODE v2.0 - RL LEARNS ENTRY, RULES HANDLE EXIT
        # When enabled, RL focuses on entry decisions only
        # Exits are handled automatically by fixed TP/SL rules
        # This separates learning: RL learns when to enter, rules handle exit
        # ============================================================
        # V13.1: ENABLED for better profitability
        self.hybrid_mode_enabled = True  # V13.1: ENABLED - RL learns entry only
        self.hybrid_take_profit_pct = 0.05  # 5% TP (reasonable profit target)
        self.hybrid_stop_loss_pct = 0.03  # 3% SL (controlled risk)
        self.hybrid_max_hold_steps = 50  # Max steps before forced close (give time to develop)
        self.hybrid_entry_reward_scale = 5.0  # Reward for successful entry (realized PnL)
        self.hybrid_allow_manual_close = False  # V13.1: DISABLED - only TP/SL exits
        self.hybrid_min_hold_for_tpsl = 3  # Min steps before TP/SL can trigger
        
        # Track hybrid mode exits for reward calculation
        self.hybrid_exit_reason = None  # 'tp', 'sl', 'time', 'manual', None
        self.hybrid_entry_unrealized_pnl = 0.0  # Track unrealized PnL at entry for reward

        # Action diversity tracking
        self.action_history = []
        self.max_action_history = 10

        # Strategy balancing tracking (NEW)
        self.long_trades = 0
        self.short_trades = 0
        self.direction_streak = 0
        self.last_direction = None
        self.direction_exploration_bonuses = {'long': 0, 'short': 0}

        # Use rolling window normalization instead of full dataset z-score
        price_col = 'close' if 'close' in df.columns else 'Close'
        window_size = min(100, len(df) // 10)
        self.price_rolling_mean = df[price_col].rolling(window=window_size, min_periods=1).mean().values.copy()
        self.price_rolling_std = df[price_col].rolling(window=window_size, min_periods=1).std().values.copy()
        self.price_rolling_std[self.price_rolling_std == 0] = 1

        if self.debug:
            print(f"Price normalization: rolling window {window_size}")
            print(f"Margin requirements: initial={self.margin_requirement*100:.0f}%, maintenance={self.maintenance_margin*100:.0f}%")
            print(f"Trailing stop-loss: {self.trailing_stop_distance*100:.1f}% distance")

        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Cover Short
        self.action_space = spaces.Discrete(5)

        # State: [balance_norm, position_norm, price_norm, can_close, steps_to_close_norm, unrealized_pnl, indicators...]
        # FIX v3: Replaced has_long/has_short (binary flags destroyed by VecNormalize) with:
        #   can_close         = 1.0 if position exists AND min_hold_steps passed (can close NOW)
        #   steps_to_close_norm = countdown 1→0 until close is allowed (0 when no position or can close)
        # position_norm sign already encodes direction: >0=long, <0=short
        # Indicators: RSI, BB_upper, BB_lower, ATR, short_trend, medium_trend, MFI,
        #             MACD_macd, MACD_signal, MACD_hist, Stoch_k, Stoch_d
        n_indicators = 12
        n_base_features = 6  # balance_norm, position_norm, price_norm, can_close, steps_to_close_norm, unrealized_pnl
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.5, -15.0, 0.0, 0.0, -1.0] + [-1.0] * n_indicators, dtype=np.float32),
            high=np.array([2.0, 2.5, 15.0, 1.0, 1.0, 1.0] + [1.0] * n_indicators, dtype=np.float32),
            dtype=np.float32
        )

        # Add state normalization parameters
        self.state_means = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0] + [0.0] * n_indicators, dtype=np.float32)
        self.state_stds = np.array([1.0, 1.0, 3.0, 0.5, 0.5, 0.5] + [0.5] * n_indicators, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset episode state variables
        if self.start_step is None:
            max_start = max(0, len(self.df) - self.episode_length)
            if max_start > 0:
                if seed is not None:
                    np.random.seed(seed)
                self.episode_start_step = np.random.randint(0, max_start + 1)
            else:
                self.episode_start_step = 0
        else:
            self.episode_start_step = min(self.start_step, max(0, len(self.df) - self.episode_length))
        
        self.current_step = self.episode_start_step
        self.steps_in_episode = 0
        
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.total_fees = 0.0
        self.portfolio_values = [float(self.initial_balance)]
        self.portfolio_history = [float(self.initial_balance)]

        # Margin trading tracking
        self.margin_locked = 0.0
        self.short_position_value = 0.0
        
        # Initialize with random start step to improve episode diversity
        if self.start_step is None:
            max_start = max(0, len(self.df) - self.episode_length)
            if max_start > 0:
                if seed is not None:
                    np.random.seed(seed)
                self.episode_start_step = np.random.randint(0, max_start + 1)
            else:
                self.episode_start_step = 0
        else:
            self.episode_start_step = min(self.start_step, max(0, len(self.df) - self.episode_length))
        
        # Initialize internal state for proper accounting
        self.short_opening_fees = 0.0
        self.balance_before_short = 0.0
        self.proceeds_from_short = 0.0
        
        # Proper margin accounting initialization
        self.cash_balance = self.initial_balance
        self.borrowed_assets = 0.0
        self.short_position_value = 0.0
        
        # Initialize with random start step for better episode diversity
        if self.start_step is None:
            max_start = max(0, len(self.df) - self.episode_length)
            if max_start > 0:
                if seed is not None:
                    np.random.seed(seed)
                self.episode_start_step = np.random.randint(0, max_start + 1)
            else:
                self.episode_start_step = 0
        else:
            self.episode_start_step = min(self.start_step, max(0, len(self.df) - self.episode_length))
        
        # Initialize episode metrics for proper accounting
        self.episode_start_balance = self.initial_balance
        self.cash = self.initial_balance
        self.equity = self.initial_balance
        self.liability = 0.0

        # Position tracking
        self.entry_price = 0.0
        self.entry_step = 0
        self.highest_price_since_entry = 0.0
        self.lowest_price_since_entry = float('inf')
        self.trailing_stop_loss = 0.0
        self.trailing_take_profit = 0.0

        # Track consecutive losses to penalize bad trading patterns
        self.consecutive_losses = 0

        # Track liquidations
        self.liquidation_count = 0

        # Track action history for diversity penalty
        self.action_history = []

        # Track trade statistics
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

        # Track inactivity
        self.steps_since_last_trade = 0
        
        # Track previous portfolio value for reward calculation
        self.prev_portfolio_value = float(self.initial_balance)
        
        # Track fees for current step
        self.fees_step = 0.0
        
        # Track short opening fees to avoid double-counting
        self.short_opening_fees = 0.0
        
        # Strategy balancing tracking (NEW)
        self.long_trades = 0
        self.short_trades = 0
        self.direction_streak = 0
        self.last_direction = None
        self.direction_exploration_bonuses = {'long': 0, 'short': 0}

        # Anti-overtrading state
        self.last_close_step = -999   # When last position was closed (-999 = no recent close)
        self.last_trade_hold_steps = 0  # How long was the last closed trade held
        
        # v8.0: Track open/close actions for close_rate calculation
        self.total_opens = 0   # Count of position opens (BUY_LONG, SELL_SHORT)
        self.total_closes = 0  # Count of position closes (SELL_LONG, COVER_SHORT)
        
        # v8.0: Track accumulated hold bonus for episode
        self.accumulated_hold_bonus = 0.0
        
        # V12: Track highest unrealized PnL for missed profit penalty
        self.highest_unrealized_pnl = 0.0
        
        # V12.1: Track episode trades for trading encouragement bonus
        self.episode_trades = 0
        
        # Initialize previous price for reward calculation
        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close', 500))
        self.prev_price = current_price
        
        # Ensure we don't start beyond data boundaries
        if self.current_step >= len(self.df):
            self.current_step = max(0, len(self.df) - 1)
            self.episode_start_step = self.current_step
        
        # Ensure episode_length doesn't exceed available data
        max_available_steps = len(self.df) - self.current_step
        if self.episode_length > max_available_steps:
            if self.debug:
                print(f"Warning: episode_length ({self.episode_length}) exceeds available data ({max_available_steps}). Adjusting.")
            # Don't adjust episode_length here, just ensure we don't go beyond data

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation with enhanced trend information"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]

        # Validate current price
        current_price = row.get('close', row.get('Close', 50000))
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            if self.debug:
                print(f"WARNING: Invalid price at step {self.current_step}: {current_price}")
            current_price = 50000  # Fallback price

        # Validate technical indicators
        rsi = row.get('RSI_15', 50)
        if np.isnan(rsi) or np.isinf(rsi) or rsi < 0 or rsi > 100:
            rsi = 50

        bb_upper = row.get('BB_15_upper', current_price)
        if np.isnan(bb_upper) or np.isinf(bb_upper) or bb_upper <= 0:
            bb_upper = current_price

        bb_lower = row.get('BB_15_lower', current_price)
        if np.isnan(bb_lower) or np.isinf(bb_lower) or bb_lower <= 0:
            bb_lower = current_price

        atr = row.get('ATR_15', 100)
        if np.isnan(atr) or np.isinf(atr) or atr <= 0:
            atr = 10

        mfi = row.get('MFI_15', 50)
        if np.isnan(mfi) or np.isinf(mfi) or mfi < 0 or mfi > 100:
            mfi = 50

        # MACD indicators for trend confirmation
        macd = row.get('MACD_default_macd', 0)
        if np.isnan(macd) or np.isinf(macd):
            macd = 0

        macd_signal = row.get('MACD_default_signal', 0)
        if np.isnan(macd_signal) or np.isinf(macd_signal):
            macd_signal = 0

        macd_hist = row.get('MACD_default_histogram', 0)
        if np.isnan(macd_hist) or np.isinf(macd_hist):
            macd_hist = 0

        # Stochastic oscillator for overbought/oversold
        stoch_k = row.get('Stochastic_slowk', 50)
        if np.isnan(stoch_k) or np.isinf(stoch_k) or stoch_k < 0 or stoch_k > 100:
            stoch_k = 50

        stoch_d = row.get('Stochastic_slowd', 50)
        if np.isnan(stoch_d) or np.isinf(stoch_d) or stoch_d < 0 or stoch_d > 100:
            stoch_d = 50

        # Normalize values
        balance_norm = self.balance / self.initial_balance - 1

        # Better position normalization: absolute position value relative to initial balance
        position_value = abs(self.position) * current_price
        position_norm = position_value / self.initial_balance if self.initial_balance > 0 else 0.0

        # Add direction indicator: positive for long, negative for short
        if self.position < 0:
            position_norm = -position_norm

        # Clip to reasonable range to prevent NN issues and ensure numerical stability
        position_norm = np.clip(position_norm, -2.0, 2.0)

        # Use rolling window normalization for better stationarity
        price_norm = 0
        if self.current_step < len(self.price_rolling_mean) and self.current_step < len(self.price_rolling_std):
            if not np.isnan(self.price_rolling_std[self.current_step]) and self.price_rolling_std[self.current_step] > 0:
                price_norm = (current_price - self.price_rolling_mean[self.current_step]) / self.price_rolling_std[self.current_step]
                # Clip extreme values to prevent neural network issues
                price_norm = np.clip(price_norm, -10, 10)
        # If normalization fails, use log return instead
        if price_norm == 0 and self.current_step > 0:
            prev_price = self.df.iloc[max(self.current_step-1, 0)].get('close', self.df.iloc[max(self.current_step-1, 0)].get('Close', current_price))
            if prev_price > 0 and not np.isnan(prev_price) and not np.isinf(prev_price):
                price_norm = np.log(current_price / prev_price) * 10  # Scale up for better learning

        # Calculate price trends (short and medium term)
        short_trend = 0  # 5-step trend
        medium_trend = 0  # 20-step trend
        
        if self.current_step >= 5:
            try:
                prev_price_5 = self.df.iloc[self.current_step - 5].get('close', 
                              self.df.iloc[self.current_step - 5].get('Close', current_price))
                if prev_price_5 > 0 and not np.isnan(prev_price_5):
                    short_trend = (current_price - prev_price_5) / prev_price_5
                    short_trend = np.clip(short_trend, -0.1, 0.1) * 10  # Scale and clip
            except:
                pass
        
        if self.current_step >= 20:
            try:
                prev_price_20 = self.df.iloc[self.current_step - 20].get('close', 
                               self.df.iloc[self.current_step - 20].get('Close', current_price))
                if prev_price_20 > 0 and not np.isnan(prev_price_20):
                    medium_trend = (current_price - prev_price_20) / prev_price_20
                    medium_trend = np.clip(medium_trend, -0.2, 0.2) * 5  # Scale and clip
            except:
                pass

        # Calculate unrealized PnL for current position
        unrealized_pnl = 0
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # Long position
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = np.clip(unrealized_pnl, -0.5, 0.5) * 2  # Scale for visibility

        # FIX v3: Replace has_long/has_short (binary flags destroyed by VecNormalize running stats)
        # with semantically meaningful continuous features:
        #   can_close         = 1.0 when position is open AND min_hold_steps elapsed → model knows it CAN close
        #   steps_to_close_norm = countdown ratio from 1.0 (just opened) → 0.0 (can close now)
        # position_norm SIGN already encodes direction: positive=long, negative=short
        steps_held_now = int(self.current_step - self.entry_step) if self.position != 0 else 0
        can_close = 1.0 if (self.position != 0 and steps_held_now >= self.min_hold_steps) else 0.0
        steps_to_close_norm = (
            max(0.0, (self.min_hold_steps - steps_held_now) / max(self.min_hold_steps, 1))
            if self.position != 0 else 0.0
        )

        # Technical indicators with validation - replaced OBV/AD with trends
        # Normalize MACD values - they can be positive or negative, scale by typical price
        macd_norm = macd / current_price * 100 if current_price > 0 else 0  # Scale to reasonable range
        macd_signal_norm = macd_signal / current_price * 100 if current_price > 0 else 0
        macd_hist_norm = macd_hist / current_price * 100 if current_price > 0 else 0
        
        indicators = [
            rsi / 100 - 0.5,  # RSI normalized to [-0.5, 0.5]
            (bb_upper / current_price - 1) if current_price > 0 else 0,  # BB upper as % deviation
            (bb_lower / current_price - 1) if current_price > 0 else 0,  # BB lower as % deviation
            atr / 1000,  # ATR normalized
            short_trend,  # 5-step price trend (replaced OBV)
            medium_trend,  # 20-step price trend (replaced AD)
            mfi / 100 - 0.5,  # MFI normalized to [-0.5, 0.5]
            # MACD indicators for trend confirmation
            np.clip(macd_norm, -1, 1),  # MACD line normalized
            np.clip(macd_signal_norm, -1, 1),  # MACD signal line normalized
            np.clip(macd_hist_norm, -1, 1),  # MACD histogram normalized
            # Stochastic oscillator for overbought/oversold
            stoch_k / 100 - 0.5,  # Stochastic K normalized to [-0.5, 0.5]
            stoch_d / 100 - 0.5,  # Stochastic D normalized to [-0.5, 0.5]
        ]

        # Build state: position_norm encodes direction (sign) + magnitude,
        # can_close tells model it CAN act on SELL_LONG/COVER_SHORT right now,
        # steps_to_close_norm is a countdown to when close becomes allowed.
        state = np.array([
            balance_norm,
            position_norm,          # >0 = long, <0 = short, 0 = flat
            price_norm,
            can_close,              # 1.0 = position open AND min_hold elapsed → can close NOW
            steps_to_close_norm,    # 1.0 = just opened, 0.0 = can close (or no position)
            unrealized_pnl
        ] + indicators, dtype=np.float32)
        # Protect against NaN and inf values that could break the neural network
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        # Additional safety: clip to observation space bounds to ensure numerical stability
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        return state

    def _update_strategy_balance_tracking(self, action):
        """Update strategy balance tracking based on action taken
        
        IMPORTANT: Only counts OPENING trades (Buy Long, Sell Short) for balance tracking.
        Closing trades (Sell Long, Cover Short) are NOT counted to avoid distorting balance.
        """
        if not self.enable_strategy_balancing:
            return

        # Only count OPENING trades for direction balance
        # Action 1 = Buy Long (opens long position)
        # Action 3 = Sell Short (opens short position)
        # Actions 2 and 4 are closing trades and should NOT affect balance tracking
        
        if action == 1:  # Buy Long - opening long position
            direction = 'long'
            self.long_trades += 1
        elif action == 3:  # Sell Short - opening short position
            direction = 'short'
            self.short_trades += 1
        else:
            # Hold, Sell Long (close), or Cover Short (close) - no direction change
            direction = 'hold'

        # Update streak tracking only for opening trades
        if direction != 'hold':
            if self.last_direction == direction:
                self.direction_streak += 1
            else:
                self.direction_streak = 1
            self.last_direction = direction

        # Update exploration bonuses based on current balance
        total_trades = self.long_trades + self.short_trades
        if total_trades > 0:
            long_ratio = self.long_trades / total_trades
            short_ratio = self.short_trades / total_trades
            
            # Give bonus for underrepresented direction (stronger incentive)
            if long_ratio < 0.35:  # Less than 35% long trades
                self.direction_exploration_bonuses['long'] += self.exploration_bonus_weight
            elif short_ratio < 0.35:  # Less than 35% short trades
                self.direction_exploration_bonuses['short'] += self.exploration_bonus_weight
            
            # Extra bonus for severely underrepresented direction
            if long_ratio < 0.20:  # Less than 20% long trades
                self.direction_exploration_bonuses['long'] += self.exploration_bonus_weight * 2
            elif short_ratio < 0.20:  # Less than 20% short trades
                self.direction_exploration_bonuses['short'] += self.exploration_bonus_weight * 2

    def _calculate_strategy_balance_reward(self, action):
        """Calculate reward component for strategy balancing
        
        Provides strong incentives for balanced trading:
        - Rewards opening positions in underrepresented directions
        - Penalizes extreme imbalances (e.g., 90% shorts)
        - Penalizes excessive consecutive same-direction trades
        """
        if not self.enable_strategy_balancing:
            return 0.0

        total_trades = self.long_trades + self.short_trades
        
        if total_trades == 0:
            # Small bonus for first trade to encourage any action
            if action in [1, 3]:  # Opening trade
                return 0.05
            return 0.0

        # Calculate direction ratios
        long_ratio = self.long_trades / total_trades
        short_ratio = self.short_trades / total_trades
        
        # Initialize reward
        balance_reward = 0.0

        # 1. DIRECTION-SPECIFIC OPENING BONUSES (strongest signal)
        # P1-FIX: Reduced multiplier from *2 to *0.1 to prevent balance reward from dominating PnL reward
        # Reward opening positions in underrepresented direction
        if action == 1:  # Buy Long (opening)
            if short_ratio > 0.6:  # Shorts dominate
                balance_reward += self.long_open_bonus * 0.1  # P1-FIX: Reduced from *2 to *0.1
            elif short_ratio > 0.5:
                balance_reward += self.long_open_bonus * 0.05  # P1-FIX: Reduced multiplier
        elif action == 3:  # Sell Short (opening)
            if long_ratio > 0.6:  # Longs dominate
                balance_reward += self.short_open_bonus * 0.1  # P1-FIX: Reduced from *2 to *0.1
            elif long_ratio > 0.5:
                balance_reward += self.short_open_bonus * 0.05  # P1-FIX: Reduced multiplier
        
        # 2. BALANCE DEVIATION PENALTY (applied to all actions)
        # Penalize extreme imbalances more heavily
        target_ratio = self.direction_balance_target
        imbalance = abs(long_ratio - target_ratio)
        
        # Progressive penalty - gets stronger as imbalance increases
        if imbalance > 0.4:  # More than 90% in one direction
            balance_reward -= self.direction_balance_penalty_weight * 3
        elif imbalance > 0.3:  # More than 80% in one direction
            balance_reward -= self.direction_balance_penalty_weight * 2
        elif imbalance > 0.2:  # More than 70% in one direction
            balance_reward -= self.direction_balance_penalty_weight
        
        # 3. STREAK PENALTY (prevent repetitive same-direction trading)
        if self.direction_streak > self.direction_streak_limit:
            streak_penalty = (self.direction_streak - self.direction_streak_limit) * 0.05
            balance_reward -= streak_penalty
        
        # 4. CONCENTRATION PENALTY (hard limit on direction concentration)
        if long_ratio > self.max_direction_concentration or short_ratio > self.max_direction_concentration:
            balance_reward -= 0.2  # Strong penalty for extreme concentration
        
        # 5. EXPLORATION BONUS (for taking underrepresented actions)
        if action == 1:  # Long action
            balance_reward += self.direction_exploration_bonuses.get('long', 0)
        elif action == 3:  # Short action
            balance_reward += self.direction_exploration_bonuses.get('short', 0)

        return balance_reward

    def _calculate_reward(self, current_price, action, action_performed=False, pnl_pct=None):
        """
        SMART REWARD SYSTEM v13.1 - HYBRID MODE
        
        Core principle: RL learns ENTRY, rules handle EXIT via TP/SL
        
        In hybrid mode:
        - Reward given when position is closed by TP/SL
        - Small reward for holding profitable positions
        - Penalty for invalid entry attempts
        """
        reward = 0.0
        
        # ========================================
        # HYBRID MODE: Check if position was closed by TP/SL
        # ========================================
        if self.hybrid_mode_enabled and self.hybrid_exit_reason is not None:
            # Position was closed by TP/SL - reward based on exit type
            pnl_pct = self.hybrid_entry_unrealized_pnl
            
            if self.hybrid_exit_reason == 'tp':
                # Take Profit hit - BIG reward
                reward += 10.0  # Base TP bonus
                reward += pnl_pct * 100 * 2.0  # Scaled PnL bonus
                if self.debug:
                    print(f"HYBRID TP: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
                    
            elif self.hybrid_exit_reason == 'sl':
                # Stop Loss hit - small penalty (acceptable loss)
                reward += -2.0  # Small penalty for SL
                if self.debug:
                    print(f"HYBRID SL: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
                    
            elif self.hybrid_exit_reason == 'time':
                # Time-based exit - neutral
                reward += pnl_pct * 100  # Just PnL
                if self.debug:
                    print(f"HYBRID TIME: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
            
            # Reset hybrid tracking
            self.hybrid_exit_reason = None
            self.hybrid_entry_unrealized_pnl = 0.0
            return reward
        
        # ========================================
        # NON-HYBRID MODE: Original reward logic
        # ========================================
        if not self.hybrid_mode_enabled:
            # Track highest unrealized PnL for missed profit penalty
            if not hasattr(self, 'highest_unrealized_pnl'):
                self.highest_unrealized_pnl = 0.0
            
            # HOLD BONUS + CORRECT HOLD REWARD
            if self.position != 0 and self.entry_price > 0:
                if self.position > 0:
                    unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                if unrealized_pnl_pct > self.highest_unrealized_pnl:
                    self.highest_unrealized_pnl = unrealized_pnl_pct
                
                if self.correct_hold_reward > 0 and unrealized_pnl_pct > 0:
                    reward += self.correct_hold_reward * unrealized_pnl_pct
                
                if self.hold_bonus_enabled and unrealized_pnl_pct > self.hold_bonus_profit_threshold:
                    if not hasattr(self, 'accumulated_hold_bonus'):
                        self.accumulated_hold_bonus = 0.0
                    
                    step_bonus = min(self.hold_bonus_per_step, 
                                    self.hold_bonus_cap - self.accumulated_hold_bonus)
                    if step_bonus > 0:
                        self.accumulated_hold_bonus += step_bonus
                        reward += step_bonus
            
            # CLOSE REWARD - V18 IMPROVED
            if self.smart_close_enabled and action in [2, 4] and action_performed and pnl_pct is not None:
                close_action_bonus = 5.0
                reward += close_action_bonus
                
                # V18: Track how long position was held
                steps_held = self.current_step - self.entry_step
                
                # V18: Quick close penalty (prevent flip-flopping)
                if self.quick_close_penalty_enabled and steps_held < self.quick_close_threshold:
                    reward += self.quick_close_penalty  # -5.0 penalty
                
                if pnl_pct > self.min_profit_for_bonus:
                    profit_bonus = pnl_pct * 100 * self.profit_close_bonus_scale
                    reward += profit_bonus
                    
                    # V18: Bonus for holding position longer (quality trade)
                    if steps_held >= 20:
                        hold_quality_bonus = min((steps_held - 20) * 0.1, 5.0)  # Up to +5.0
                        reward += hold_quality_bonus
                    
                    # V18: Extra bonus for shorts (encourage short trading)
                    if self.position < 0:
                        short_bonus = profit_bonus * 0.5  # +50% bonus for short profits
                        reward += short_bonus
                    
                    if self.profit_multiplier_enabled and pnl_pct >= self.profit_multiplier_threshold:
                        multiplier_bonus = profit_bonus * (self.profit_multiplier_scale - 1)
                        reward += multiplier_bonus
                    
                    if self.position > 0:
                        reward += pnl_pct * 100.0 * self.long_reward_multiplier
                    else:
                        reward += pnl_pct * 100.0 * self.short_reward_multiplier
                    
                elif pnl_pct > 0:
                    # Small profit - still reward
                    reward += pnl_pct * 100.0
                else:
                    # Loss - stronger penalty
                    loss_penalty = pnl_pct * 100.0 * self.loss_close_penalty_scale
                    reward += loss_penalty
                    
                    # V18: Extra penalty for quick losses (bad trade)
                    if steps_held < self.quick_close_threshold:
                        reward += -2.0  # Additional penalty

                if hasattr(self, 'accumulated_hold_bonus'):
                    self.accumulated_hold_bonus = 0.0
            
            # OPENING PENALTY - V18 IMPROVED
            if action in [1, 3] and action_performed:
                # V18: Base penalty for any entry (discourage overtrading)
                open_penalty = -1.0  # Increased from -0.5 to -1.0
                reward += open_penalty
                
                # V18: Extra penalty if we already traded too much this episode
                if hasattr(self, 'total_trades') and self.total_trades > self.overtrading_threshold:
                    reward += -2.0  # Additional penalty for excessive trading
                
                # V18: Bonus for short entries (encourage shorts)
                if action == 3:  # SELL_SHORT
                    reward += 1.0  # Small bonus to encourage shorts
            
            # FORCED CLOSE PENALTY
            if self.position != 0 and self.entry_price > 0:
                steps_held = self.current_step - self.entry_step
                if steps_held > self.max_hold_steps_before_penalty:
                    excess_steps = steps_held - self.max_hold_steps_before_penalty
                    hold_penalty = -self.hold_penalty_per_step * excess_steps
                    reward += hold_penalty
            
            # TIME PENALTY
            if self.time_penalty_enabled and self.position != 0 and self.entry_price > 0:
                steps_held = self.current_step - self.entry_step
                if steps_held > self.time_penalty_start:
                    if self.position > 0:
                        unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:
                        unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
                    
                    if unrealized_pnl_pct <= 0:
                        time_penalty = -self.time_penalty_per_step * (steps_held - self.time_penalty_start)
                        time_penalty = max(time_penalty, -self.time_penalty_cap)
                        reward += time_penalty
            
            # INVALID ACTION PENALTIES
            if action == 2 and not action_performed:
                if self.position <= 0:
                    reward += -self.invalid_action_penalty
            if action == 4 and not action_performed:
                if self.position >= 0:
                    reward += -self.invalid_action_penalty

            if action == 1 and not action_performed:
                if self.position > 0:
                    reward += -self.invalid_action_penalty
                elif self.position < 0:
                    reward += -0.5

            if action == 3 and not action_performed:
                if self.position < 0:
                    reward += -self.invalid_action_penalty
                elif self.position > 0:
                    reward += -0.5

            # STRATEGY BALANCE REWARD
            reward += self._calculate_strategy_balance_reward(action) * 0.1

        else:
            # ========================================
            # HYBRID MODE: Simplified reward - ENCOURAGE TRADING
            # ========================================
            
            # BIG reward for valid entry (encourage trading)
            if action in [1, 3] and action_performed:
                reward += 2.0  # Strong bonus for valid entry
            
            # Small penalty for invalid entry attempts
            if action == 1 and not action_performed:
                if self.position > 0:
                    reward += -0.5  # Already in long
                elif self.position < 0:
                    reward += -0.3  # In short
            
            if action == 3 and not action_performed:
                if self.position < 0:
                    reward += -0.5  # Already in short
                elif self.position > 0:
                    reward += -0.3  # In long
            
            # Very small penalty for excessive HOLD (encourage trying to enter)
            if action == 0 and self.position == 0:
                reward += -0.05  # Small penalty for not trying
        
        # Save previous price
        self.prev_price = current_price

        # Apply reward clipping
        reward = np.clip(reward, self.reward_clip_bounds[0], self.reward_clip_bounds[1])

        return reward

    def step(self, action):
        """Execute one step in environment with enhanced reward function"""
        try:
            # Check if episode should be done BEFORE executing action
            if self.current_step >= len(self.df):
                terminated = True
                truncated = False
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))
                final_portfolio = self.balance + self.margin_locked + self.position * current_price
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
                final_reward = portfolio_return * 100
                self.portfolio_values.append(final_portfolio)
                
                if self.debug:
                    print(f"Episode ended: current_step={self.current_step} >= len(df)={len(self.df)}")
                
                return self._get_state(), final_reward, terminated, truncated, {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_pnl': self.total_pnl,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'long_trades': self.long_trades,
                    'short_trades': self.short_trades,
                    'direction_balance_ratio': self.long_trades / max(self.short_trades, 1) if self.short_trades > 0 else float('inf')
                }

            # Get current market data
            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
            else:
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))

            # Calculate position sizes
            max_order_size_long, max_order_size_short, min_order_size = self._get_position_sizes()

            # Initialize reward and pnl
            action_performed = False
            pnl_pct = None

            # Add action to history for diversity tracking
            action_int = int(action.item()) if hasattr(action, 'item') else int(action)
            self.action_history.append(action_int)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)

            # Execute action FIRST, then update strategy tracking only if performed
            if action == 0:  # Hold
                self.steps_since_last_trade += 1
            
            # V16-FIX: Action Masking - prevent invalid actions
            elif self.action_masking_enabled:
                if action == 1:  # BUY_LONG
                    if self.position > 0:  # Already in long
                        action = 0  # Force HOLD
                        reward -= 0.5  # Penalty for invalid attempt
                    elif self.position < 0:  # In short, need to cover first
                        action = 0  # Force HOLD
                        reward -= 0.3
                    else:  # position == 0, valid entry
                        action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
                        if action_performed:
                            self._update_strategy_balance_tracking(action_int)
                            self.total_opens += 1
                
                elif action == 2:  # SELL_LONG
                    if self.position <= 0:  # No long position to close
                        action = 0  # Force HOLD
                        reward -= 1.0  # Strong penalty
                    else:  # Valid close
                        action_performed, pnl_pct = self._execute_sell_long(current_price)
                        if action_performed:
                            if pnl_pct is not None:
                                if pnl_pct > 0:
                                    self.consecutive_losses = 0
                                else:
                                    self.consecutive_losses += 1
                            self.total_closes += 1
                
                elif action == 3:  # SELL_SHORT
                    if self.position < 0:  # Already in short
                        action = 0  # Force HOLD
                        reward -= 0.5
                    elif self.position > 0:  # In long, need to sell first
                        action = 0  # Force HOLD
                        reward -= 0.3
                    else:  # position == 0, valid entry
                        action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
                        if action_performed:
                            self._update_strategy_balance_tracking(action_int)
                            self.total_opens += 1
                
                elif action == 4:  # COVER_SHORT
                    if self.position >= 0:  # No short position to cover
                        action = 0  # Force HOLD
                        reward -= 1.0  # Strong penalty
                    else:  # Valid close
                        action_performed, pnl_pct = self._execute_cover_short(current_price)
                        if action_performed:
                            if pnl_pct is not None:
                                if pnl_pct > 0:
                                    self.consecutive_losses = 0
                                else:
                                    self.consecutive_losses += 1
                            self.total_closes += 1
            
            # Original logic (no masking)
            elif action == 1:  # Buy Long
                action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
                if action_performed:
                    # Only count successfully opened positions in direction tracking
                    self._update_strategy_balance_tracking(action_int)
                    self.total_opens += 1  # v8.0: Track opens
            elif action == 2:  # Sell Long
                action_performed, pnl_pct = self._execute_sell_long(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1
                    self.total_closes += 1  # v8.0: Track closes
            elif action == 3:  # Sell Short
                action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
                if action_performed:
                    # Only count successfully opened positions in direction tracking
                    self._update_strategy_balance_tracking(action_int)
                    self.total_opens += 1  # v8.0: Track opens
            elif action == 4:  # Cover Short
                action_performed, pnl_pct = self._execute_cover_short(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1
                    self.total_closes += 1  # v8.0: Track closes

            # Check trailing stop-loss
            self._check_trailing_stop(current_price)
            
            # Calculate current portfolio value BEFORE hybrid check (needed for both TP/SL and reward)
            current_portfolio = self.balance + self.margin_locked + self.position * current_price
            
            # ============================================================
            # HYBRID MODE: Check TP/SL (if enabled)
            # This handles automatic exits so RL can focus on entry decisions
            # ============================================================
            if self.hybrid_mode_enabled:
                tp_triggered, exit_reason, exit_pnl = self._check_hybrid_tpsl(current_price)
                if tp_triggered:
                    # Position was closed by TP/SL - use hybrid reward
                    reward = self._calculate_hybrid_reward(current_price, action)
                    # Track trade stats (use existing counters)
                    self.total_trades += 1
                    if self.debug:
                        pnl_str = f"{exit_pnl*100:.2f}" if exit_pnl is not None else "0.00"
                        print(f"HYBRID EXIT: {exit_reason}, pnl={pnl_str}%")
                else:
                    # Calculate normal reward when no TP/SL triggered
                    reward = self._calculate_reward(current_price, action, action_performed, pnl_pct)
            else:
                # Original reward calculation when hybrid mode disabled
                reward = self._calculate_reward(current_price, action, action_performed, pnl_pct)
            
            # Reset fees_step for next step (after using it in reward calculation)
            self.fees_step = 0.0
            
            # Update previous portfolio value for next step
            self.prev_portfolio_value = current_portfolio

            # Increment step counters BEFORE checking termination
            self.current_step += 1
            self.steps_in_episode += 1
            
            # Add current portfolio value to history
            self.portfolio_history.append(current_portfolio)

            # Check termination conditions AFTER executing the step
            terminated = False
            truncated = False
            
            # First check: risk management conditions
            portfolio_return = (current_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            
            # P0-FIX: Check maximum episode loss condition (critical risk management)
            # Terminate episode if losses exceed max_episode_loss_pct to prevent catastrophic losses
            if portfolio_return <= -self.max_episode_loss_pct:
                terminated = True
                self.logger.warning(f"P0-FIX: Max episode loss reached: portfolio_return={portfolio_return:.4f}, "
                                   f"threshold={-self.max_episode_loss_pct:.4f}, total_value={current_portfolio:.2f}")
                if self.debug:
                    print(f"MAX EPISODE LOSS TRIGGERED: return={portfolio_return*100:.2f}%, threshold={-self.max_episode_loss_pct*100:.2f}%")
            
            # Second check: episode length limit
            if not terminated and self.steps_in_episode >= self.episode_length:
                truncated = True
                terminated = True
                if self.debug:
                    print(f"Episode truncated: steps_in_episode={self.steps_in_episode}, episode_length={self.episode_length}")
            
            # Third check: end of data
            if not terminated and self.current_step >= len(self.df):
                terminated = True
                if self.debug:
                    print(f"Episode terminated: end of data. current_step={self.current_step}, len(df)={len(self.df)}")
            
            # If episode ended, return final state
            if terminated:
                final_portfolio = current_portfolio
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
                
                # ========================================
                # EPISODE-LEVEL REWARD: Final PnL
                # FIX v3: Reduced scale from *100 to *10 and added clipping
                # Previous *100 caused rewards up to 16595 which destroyed learning signal
                # ========================================
                final_reward = portfolio_return * 10.0

                # Small penalty for no trading (to encourage exploration)
                if self.total_trades == 0:
                    final_reward -= 0.5

                # Penalty for leaving a position open at end of episode
                # P3-FIX: STRONG penalty - model MUST close positions before episode ends
                # -10.0 base penalty + -0.1 per step open
                # This makes it MUCH more costly to leave positions unclosed
                if self.position != 0:
                    steps_open = self.steps_in_episode - max(0, self.entry_step - self.episode_start_step)
                    unclosed_penalty = -(10.0 + 0.1 * steps_open)  # P3-FIX: Increased from -5.0 to -10.0
                    final_reward += unclosed_penalty
                    if self.debug:
                        print(f"Episode ended with open position! Penalty={unclosed_penalty:.2f} (steps_open={steps_open})")

                # FIX: Apply reward clipping to final reward as well
                final_reward = np.clip(final_reward, self.reward_clip_bounds[0], self.reward_clip_bounds[1])

                # Add strategy balance metrics to final info
                # v8.0: Calculate close_rate for tracking position closing behavior
                close_rate = self.total_closes / max(self.total_opens, 1) if self.total_opens > 0 else 0.0
                
                episode_info = {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_pnl': self.total_pnl,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'long_trades': self.long_trades,
                    'short_trades': self.short_trades,
                    'direction_balance_ratio': self.long_trades / max(self.short_trades, 1) if self.short_trades > 0 else float('inf'),
                    'exploration_bonus_applied': self.direction_exploration_bonuses,
                    'total_opens': self.total_opens,   # v8.0: Track opens
                    'total_closes': self.total_closes,  # v8.0: Track closes
                    'close_rate': close_rate,  # v8.0: closes / opens ratio
                    'episode': {
                        'r': final_reward,
                        'l': self.steps_in_episode
                    }
                }
                return self._get_state(), final_reward, terminated, truncated, episode_info

            # Calculate equity and position value for better tracking
            position_value = abs(self.position) * current_price if self.position != 0 else 0.0
            equity = current_portfolio
            cash = self.balance
            unrealized_pnl = 0.0
            if self.position > 0:  # Long
                unrealized_pnl = (current_price - self.entry_price) * self.position if self.entry_price > 0 else 0
            elif self.position < 0:  # Short
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position) if self.entry_price > 0 else 0
            
            # Prepare info dict with additional metrics
            info = {
                'portfolio_value': current_portfolio,
                'equity': equity,
                'cash': cash,
                'position': self.position,
                'position_value': position_value,
                'margin_locked': self.margin_locked,
                'current_price': current_price,
                'total_fees': self.total_fees,
                'fees_step': self.fees_step,
                'total_trades': self.total_trades,
                'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                'current_step': self.current_step,
                'action_taken': action,
                'pnl_pct': pnl_pct if pnl_pct is not None else 0,
                'action_performed': action_performed,
                'realized_pnl': self.total_pnl,
                'unrealized_pnl': unrealized_pnl,
                'long_trades': self.long_trades,
                'short_trades': self.short_trades,
                'direction_balance_ratio': self.long_trades / max(self.short_trades, 1) if self.short_trades > 0 else float('inf'),
                'exploration_bonus': self.direction_exploration_bonuses
            }

            # Debug prints only if enabled and limited to reduce I/O overhead
            if self.debug:
                if self.current_step < 10:
                    print(f"DEBUG: Step {self.current_step}, Action {action}, Performed {action_performed}, Position {self.position:.4f}, Equity {equity:.2f}, Reward {reward:.2f}")
                elif action_performed and self.current_step % 50 == 0:
                    pnl_str = f"{pnl_pct:.4f}" if pnl_pct is not None else "N/A"
                    print(f"TRADE: Step {self.current_step}, Action {action}, Position {self.position:.4f}, Equity {equity:.2f}, PnL% {pnl_str}")

            return self._get_state(), reward, terminated, truncated, info

        except Exception as e:
            if self.debug:
                print(f"ERROR: Exception in step() method: {e}")
                import traceback
                traceback.print_exc()
            return self._get_state(), 0, True, False, {}

    def render(self, mode='human'):
        """Render environment state"""
        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        portfolio_value = self.balance + self.margin_locked + self.position * current_price
        win_rate = self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0
        direction_ratio = self.long_trades / max(self.short_trades, 1) if self.short_trades > 0 else float('inf')
        
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.6f}, Portfolio: {portfolio_value:.2f}, Wins: {self.win_count}/{self.total_trades} ({win_rate:.2%}), Long/Short: {direction_ratio:.2f}")

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from portfolio values"""
        if len(self.portfolio_values) < 2:
            return 0.0

        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio from portfolio values"""
        if len(self.portfolio_values) < 2:
            return 0.0

        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

        # Annualize (assuming daily returns)
        return sharpe_ratio * np.sqrt(252)

    def _calculate_volatility_based_position_size(self, current_price):
        """Calculate position size based on market volatility"""
        if self.current_step < self.volatility_window:
            return self.base_position_size

        start_idx = max(0, self.current_step - self.volatility_window)
        recent_prices = self.df.iloc[start_idx:self.current_step + 1]
        
        price_col = 'close' if 'close' in recent_prices.columns else 'Close'
        prices = recent_prices[price_col].values
        
        if len(prices) < 2:
            return self.base_position_size
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.01
        
        avg_volatility = 0.02
        volatility_factor = avg_volatility / (volatility + 0.001)
        
        volatility_factor = np.clip(volatility_factor, 0.5, 2.0)
        
        dynamic_size = self.base_position_size * volatility_factor
        dynamic_size = np.clip(dynamic_size, self.min_position_size, self.max_dynamic_position_size)
        
        return dynamic_size

    def _get_position_sizes(self):
        """Calculate position sizes for orders with volatility adjustment"""
        dynamic_position_size = self._calculate_volatility_based_position_size(
            self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        )
        
        max_order_size_long = min(self.balance * dynamic_position_size, 
                                 self.balance * self.max_position_size,
                                 (self.initial_balance * self.max_total_exposure - abs(self.position * self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))) if self.position != 0 else self.initial_balance * self.max_total_exposure))
        
        max_order_size_short = min(self.balance * dynamic_position_size * 0.8,
                                   self.balance * self.max_position_size,
                                   (self.initial_balance * self.max_total_exposure - abs(self.position * self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))) if self.position != 0 else self.initial_balance * self.max_total_exposure))
        
        min_order_size = 5
        return max_order_size_long, max_order_size_short, min_order_size

    def _execute_buy_long(self, current_price, max_order_size_long, min_order_size):
        """Execute buy long action

        IMPORTANT:
        - If there's an open short position, do NOT open long (use COVER_SHORT first).
        - If there's already an open long position, do NOT pyramid (use SELL_LONG first).
        This forces the model to learn proper position management.
        
        P1-FIX: Uses RiskManager for dynamic position sizing based on risk.
        """
        # If we have a short position, refuse to open long
        if self.position < 0:
            return False

        # If we already have a long position, refuse to pyramid
        if self.position > 0:
            return False

        # Cooldown: enforce minimum pause between trades to prevent overtrading
        if self.current_step - self.last_close_step < self.min_open_cooldown:
            return False

        # P3-FIX: RiskManager returns size in BTC, not USD - causes invest_amount < min_order_size
        # Temporarily bypass RiskManager for LONG entries until fixed
        # risk_position_size = self.risk_manager.calculate_position_size(
        #     asset="BTC",
        #     entry_price=current_price,
        #     side=PositionSide.LONG
        # )
        
        # Use max_order_size_long directly (already risk-managed by environment)
        invest_amount = min(max_order_size_long, self.balance)
        
        # Normal buy long logic with risk-adjusted position size
        if invest_amount >= min_order_size and current_price > 0:
            fee = invest_amount * self.transaction_fee
            coins_bought = (invest_amount - fee) / current_price

            new_position_value = (self.position + coins_bought) * current_price
            if new_position_value > self.initial_balance * self.max_position_size:
                return False

            self.position += coins_bought
            self.balance -= invest_amount
            self.total_fees += fee
            self.fees_step += fee

            if self.balance < -1e-6 and not self.debug:
                pass

            if self.entry_price == 0:
                self.entry_price = current_price
                self.entry_step = self.current_step
                self.highest_price_since_entry = current_price
                self.lowest_price_since_entry = current_price
                self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)
                # v7.0: No accumulated hold bonus to reset (hold_bonus disabled)
            else:
                old_position = self.position - coins_bought
                self.entry_price = ((old_position * self.entry_price) + (coins_bought * current_price)) / self.position

            self.total_trades += 1
            self.steps_since_last_trade = 0

            return True
        return False

    def _execute_sell_long(self, current_price):
        """Execute sell long action"""
        if self.position > 0:
            # Minimum hold check: block premature close to avoid overtrading
            steps_held = self.current_step - self.entry_step
            if steps_held < self.min_hold_steps:
                return False, 0

            position_size = self.position
            revenue = position_size * current_price
            fee = revenue * self.transaction_fee
            pnl = revenue - (position_size * self.entry_price) - fee
            pnl_pct = pnl / (position_size * self.entry_price) if self.entry_price > 0 else 0

            self.balance += revenue - fee
            self.total_fees += fee
            self.fees_step += fee

            if self.balance < -1e-6 and not self.debug:
                pass

            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            # Track close timing for cooldown and reporting
            self.last_trade_hold_steps = steps_held
            self.last_close_step = self.current_step

            self.position = 0
            self.entry_price = 0
            self.entry_step = 0
            self.highest_price_since_entry = 0
            self.lowest_price_since_entry = float('inf')
            self.trailing_stop_loss = 0
            self.trailing_take_profit = 0

            self.steps_since_last_trade = 0

            return True, pnl_pct
        return False, 0

    def _execute_sell_short(self, current_price, max_order_size_short, min_order_size):
        """Execute sell short action

        CORRECTED LOGIC:
        - When opening short: we borrow coins and sell them
        - Proceeds from sale are blocked as collateral (margin_locked)
        - We also provide additional margin from our balance
        - Position becomes negative (amount of coins we owe)

        IMPORTANT:
        - If there's an open long position, do NOT open short (use SELL_LONG first).
        - If there's already an open short position, do NOT pyramid (use COVER_SHORT first).
        This forces the model to learn proper position management.
        
        P1-FIX: Uses RiskManager for dynamic position sizing based on risk.
        """
        # If we have a long position, refuse to open short
        if self.position > 0:
            return False

        # If we already have a short position, refuse to pyramid
        if self.position < 0:
            return False

        # Cooldown: enforce minimum pause between trades to prevent overtrading
        if self.current_step - self.last_close_step < self.min_open_cooldown:
            return False

        # P3-FIX: RiskManager returns size in BTC, not USD - causes short_amount < min_order_size
        # Temporarily bypass RiskManager for SHORT entries until fixed
        # risk_position_size = self.risk_manager.calculate_position_size(
        #     asset="BTC",
        #     entry_price=current_price,
        #     side=PositionSide.SHORT
        # )
        
        # Use max_order_size_short directly (already risk-managed by environment)
        short_amount = min(max_order_size_short, self.balance)
        
        # Normal sell short logic with risk-adjusted position size
        if short_amount >= min_order_size and current_price > 0:
            margin_required = short_amount * self.margin_requirement
            available_balance = self.balance - self.margin_locked

            if available_balance >= margin_required:
                coins_short = short_amount / current_price
                fee = coins_short * current_price * self.transaction_fee

                self.short_opening_fees = fee

                # CORRECTED:
                # - margin_locked includes: sale proceeds (short_amount) + our margin (margin_required)
                # - balance decreases by: our margin + fee
                # - This way portfolio value stays correct
                self.margin_locked += short_amount + margin_required  # Total collateral
                self.balance -= margin_required + fee  # Only deduct our margin and fee
                self.position -= coins_short
                self.total_fees += fee
                self.fees_step += fee

                if self.balance < -1e-6 and self.debug:
                    print(f"WARNING: Negative balance after Sell Short: {self.balance}")
                if self.margin_locked < -1e-6 and self.debug:
                    print(f"WARNING: Negative margin_locked after Sell Short: {self.margin_locked}")

                if self.entry_price == 0:
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    self.highest_price_since_entry = current_price
                    self.lowest_price_since_entry = current_price
                    self.trailing_stop_loss = current_price * (1 + self.trailing_stop_distance)
                    # v7.0: No accumulated hold bonus to reset (hold_bonus disabled)
                else:
                    old_position_size = abs(self.position + coins_short)
                    new_position_size = abs(self.position)
                    self.entry_price = ((old_position_size * self.entry_price) + (coins_short * current_price)) / new_position_size

                self.total_trades += 1
                self.steps_since_last_trade = 0

                return True
        return False

    def _execute_cover_short(self, current_price):
        """Execute cover short action
        
        CORRECTED LOGIC:
        - When closing short: we buy back the coins we owe
        - margin_locked includes: sale proceeds + our margin
        - We use sale proceeds to buy back coins
        - PnL = sale_proceeds - buy_cost - fees
        - We get back: our margin + PnL
        """
        if self.position < 0:
            # Minimum hold check: block premature close to avoid overtrading
            steps_held = self.current_step - self.entry_step
            if steps_held < self.min_hold_steps:
                return False, 0

            position_size = abs(self.position)

            # Calculate PnL correctly
            sale_proceeds = position_size * self.entry_price  # What we got when we sold
            buy_cost = position_size * current_price  # What we pay to buy back
            close_fee = buy_cost * self.transaction_fee
            
            # PnL = sale_proceeds - buy_cost - fees
            price_pnl = sale_proceeds - buy_cost
            pnl = price_pnl - close_fee
            pnl_pct = pnl / sale_proceeds if sale_proceeds > 0 else 0

            # Our margin was: sale_proceeds * margin_requirement
            our_margin = sale_proceeds * self.margin_requirement

            # Return our margin + PnL to balance
            self.balance += our_margin + pnl
            self.margin_locked = 0
            self.total_fees += close_fee
            self.fees_step += close_fee
            self.short_opening_fees = 0.0

            if self.balance < -1e-6 and self.debug:
                print(f"WARNING: Negative balance after Cover Short: {self.balance}")

            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            self.position = 0
            self.entry_price = 0
            self.entry_step = 0
            self.highest_price_since_entry = 0
            self.lowest_price_since_entry = float('inf')
            self.trailing_stop_loss = 0
            self.trailing_take_profit = 0

            self.steps_since_last_trade = 0

            return True, pnl_pct
        return False, 0

    def _calculate_adaptive_stop_loss(self, current_price):
        """Calculate adaptive stop loss based on ATR and market conditions"""
        if self.current_step < self.volatility_window:
            return self.stop_loss_pct

        start_idx = max(0, self.current_step - self.volatility_window)
        recent_data = self.df.iloc[start_idx:self.current_step + 1]
        
        atr_col = 'ATR_15' if 'ATR_15' in recent_data.columns else 'ATR'
        if atr_col in recent_data.columns:
            recent_atr = recent_data[atr_col].iloc[-1]
            current_price_val = current_price
            
            if recent_atr > 0 and current_price_val > 0:
                atr_stop_loss_pct = (recent_atr / current_price_val) * self.atr_multiplier
                adaptive_stop_loss = max(self.min_stop_loss_pct, min(atr_stop_loss_pct, self.max_stop_loss_pct))
                return adaptive_stop_loss
        
        return self.stop_loss_pct

    def _check_trailing_stop(self, current_price):
        """Check and execute trailing stop-loss if triggered"""
        if self.position == 0 or self.entry_price == 0:
            return

        if self.adaptive_stop_loss_enabled:
            adaptive_stop_pct = self._calculate_adaptive_stop_loss(current_price)
        else:
            adaptive_stop_pct = self.stop_loss_pct

        if self.position > 0:  # Long position
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_loss = self.highest_price_since_entry * (1 - adaptive_stop_pct)

            if current_price <= self.trailing_stop_loss:
                revenue = self.position * current_price
                fee = revenue * self.transaction_fee
                pnl = revenue - (self.position * self.entry_price) - fee
                
                self.balance += revenue - fee
                self.total_fees += fee

                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                if self.balance < -1e-6 and self.debug:
                    print(f"WARNING: Negative balance after trailing stop (long): {self.balance}")

                self.position = 0
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.lowest_price_since_entry = float('inf')
                
                self.trailing_stop_loss = 0
                self.trailing_take_profit = 0

        elif self.position < 0:  # Short position
            if current_price < self.lowest_price_since_entry:
                self.lowest_price_since_entry = current_price
                self.trailing_stop_loss = self.lowest_price_since_entry * (1 + adaptive_stop_pct)

            if current_price >= self.trailing_stop_loss:
                cover_cost = abs(self.position) * current_price
                close_fee = cover_cost * self.transaction_fee
                entry_value = abs(self.position) * self.entry_price if self.entry_price > 0 else 0
                price_pnl = entry_value - cover_cost
                pnl = price_pnl - close_fee

                self.balance = self.balance + self.margin_locked - cover_cost - close_fee
                self.margin_locked = 0
                self.total_fees += close_fee
                self.fees_step += close_fee
                self.short_opening_fees = 0.0

                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                if self.balance < -1e-6 and self.debug:
                    print(f"WARNING: Negative balance after trailing stop (short): {self.balance}")
                if self.margin_locked < -1e-6 and self.debug:
                    print(f"WARNING: Negative margin_locked after trailing stop (short): {self.margin_locked}")

                self.position = 0
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.lowest_price_since_entry = float('inf')
                
                self.trailing_stop_loss = 0
                self.trailing_take_profit = 0

    def _check_hybrid_tpsl(self, current_price):
        """
        HYBRID MODE: Check and execute fixed TP/SL for automatic exits.
        
        This is the ALTERNATIVE approach where:
        - RL focuses on entry decisions only
        - Fixed TP/SL rules handle exits automatically
        - This separates learning: RL learns when to enter, rules handle exit
        
        Returns:
            tuple: (exit_triggered: bool, exit_reason: str, pnl_pct: float or None)
        """
        if not self.hybrid_mode_enabled:
            return False, None, None
        
        if self.position == 0 or self.entry_price == 0:
            return False, None, None
        
        # Check minimum hold time before TP/SL can trigger
        steps_held = self.current_step - self.entry_step
        if steps_held < self.hybrid_min_hold_for_tpsl:
            return False, None, None
        
        # Calculate current unrealized PnL percentage
        if self.position > 0:  # Long position
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Check Take Profit
        if unrealized_pnl_pct >= self.hybrid_take_profit_pct:
            # Execute TP close
            exit_reason = 'tp'
            if self.debug:
                print(f"HYBRID TP TRIGGERED: pnl_pct={unrealized_pnl_pct*100:.2f}%, price={current_price}")
            
            # Close the position
            if self.position > 0:
                # Close long
                revenue = abs(self.position) * current_price
                fee = revenue * self.transaction_fee
                pnl = revenue - (abs(self.position) * self.entry_price) - fee
                self.balance += revenue - fee
                self.total_fees += fee
                self.fees_step += fee
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            else:
                # Close short
                cover_cost = abs(self.position) * current_price
                close_fee = cover_cost * self.transaction_fee
                entry_value = abs(self.position) * self.entry_price
                price_pnl = entry_value - cover_cost
                pnl = price_pnl - close_fee
                
                self.balance = self.balance + self.margin_locked - cover_cost - close_fee
                self.margin_locked = 0
                self.total_fees += close_fee
                self.fees_step += close_fee
                self.short_opening_fees = 0.0
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            
            # Reset position
            self.position = 0
            self.entry_price = 0
            self.hybrid_exit_reason = exit_reason
            self.hybrid_entry_unrealized_pnl = unrealized_pnl_pct
            
            return True, exit_reason, unrealized_pnl_pct
        
        # Check Stop Loss
        if unrealized_pnl_pct <= -self.hybrid_stop_loss_pct:
            # Execute SL close
            exit_reason = 'sl'
            if self.debug:
                print(f"HYBRID SL TRIGGERED: pnl_pct={unrealized_pnl_pct*100:.2f}%, price={current_price}")
            
            # Close the position
            if self.position > 0:
                # Close long
                revenue = abs(self.position) * current_price
                fee = revenue * self.transaction_fee
                pnl = revenue - (abs(self.position) * self.entry_price) - fee
                self.balance += revenue - fee
                self.total_fees += fee
                self.fees_step += fee
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            else:
                # Close short
                cover_cost = abs(self.position) * current_price
                close_fee = cover_cost * self.transaction_fee
                entry_value = abs(self.position) * self.entry_price
                price_pnl = entry_value - cover_cost
                pnl = price_pnl - close_fee
                
                self.balance = self.balance + self.margin_locked - cover_cost - close_fee
                self.margin_locked = 0
                self.total_fees += close_fee
                self.fees_step += close_fee
                self.short_opening_fees = 0.0
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            
            # Reset position
            self.position = 0
            self.entry_price = 0
            self.hybrid_exit_reason = exit_reason
            self.hybrid_entry_unrealized_pnl = unrealized_pnl_pct
            
            return True, exit_reason, unrealized_pnl_pct
        
        # Check max hold time (time-based exit)
        if steps_held >= self.hybrid_max_hold_steps:
            # Force close at market
            exit_reason = 'time'
            if self.debug:
                print(f"HYBRID TIME EXIT: steps_held={steps_held}, max={self.hybrid_max_hold_steps}")
            
            # Close the position
            if self.position > 0:
                revenue = abs(self.position) * current_price
                fee = revenue * self.transaction_fee
                pnl = revenue - (abs(self.position) * self.entry_price) - fee
                self.balance += revenue - fee
                self.total_fees += fee
                self.fees_step += fee
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            else:
                cover_cost = abs(self.position) * current_price
                close_fee = cover_cost * self.transaction_fee
                entry_value = abs(self.position) * self.entry_price
                price_pnl = entry_value - cover_cost
                pnl = price_pnl - close_fee
                
                self.balance = self.balance + self.margin_locked - cover_cost - close_fee
                self.margin_locked = 0
                self.total_fees += close_fee
                self.fees_step += close_fee
                self.short_opening_fees = 0.0
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            
            self.position = 0
            self.entry_price = 0
            self.hybrid_exit_reason = exit_reason
            self.hybrid_entry_unrealized_pnl = unrealized_pnl_pct
            
            return True, exit_reason, unrealized_pnl_pct
        
        return False, None, None

    def _calculate_hybrid_reward(self, current_price, action):
        """
        Calculate reward for HYBRID mode.
        
        In hybrid mode:
        - Entry reward: Based on the quality of entry (future realized PnL)
        - TP/SL exits are handled automatically - model gets reward for the realized PnL
        - RL policy focuses on learning WHEN to enter, not when to exit
        """
        if not self.hybrid_mode_enabled:
            return 0.0
        
        reward = 0.0
        
        # If position was just closed by hybrid TP/SL
        if self.hybrid_exit_reason is not None:
            pnl_pct = self.hybrid_entry_unrealized_pnl
            
            if pnl_pct > 0:
                # Profit: reward proportional to profit
                reward += pnl_pct * 100 * 3.0  # Scale up profits
            else:
                # Loss: penalty proportional to loss
                reward += pnl_pct * 100 * 1.5  # Scale up losses
            
            if self.debug:
                print(f"HYBRID REWARD: exit={self.hybrid_exit_reason}, pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
            
            # Reset hybrid tracking
            self.hybrid_exit_reason = None
            self.hybrid_entry_unrealized_pnl = 0.0
            
            return reward
        
        # If RL made a manual close (when allowed)
        if action in [2, 4]:
            # Use existing close reward logic but with hybrid scaling
            # This is optional - RL can still close manually if enabled
            pass
        
        # Small reward for holding good positions (potential future profit)
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
            # If in profit, small positive reward to encourage holding
            if unrealized_pnl > 0:
                reward += unrealized_pnl * 0.1  # Small holding bonus
            
            # If in loss and close to SL, small penalty
            if unrealized_pnl < -self.hybrid_stop_loss_pct * 0.5:
                reward -= 0.05  # Warning penalty
        
        return reward

# For backward compatibility
def create_env(df, initial_balance=10000, transaction_fee=0.0018):
    """Factory function to create environment instances"""
    return EnhancedTradingEnvironment(df, initial_balance, transaction_fee)

if __name__ == "__main__":
    print("Enhanced Trading Environment with Strategy Balancing")
    print("Features:")
    print("- Strategy balance tracking and rewards")
    print("- Directional exploration bonuses")
    print("- Streak limitation to prevent over-concentration")
    print("- Concentration penalties for extreme imbalances")
    print("- Enhanced monitoring metrics for strategy diversity")