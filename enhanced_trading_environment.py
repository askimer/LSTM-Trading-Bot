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
        self.max_position_size = 0.15
        self.max_total_exposure = 0.40
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        self.max_episode_loss_pct = 0.05
        
        # Dynamic position sizing parameters
        self.volatility_window = 20
        self.min_position_size = 0.02
        self.max_dynamic_position_size = 0.20
        
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
        self.base_position_size = 0.08
        self.hold_penalty = 0.00001
        self.inactivity_penalty = 0.0001
        self.termination_stop_loss_threshold = 0.95
        self.termination_profit_target = 1.50

        # Reward parameters
        self.reward_clip_bounds = (-50, 50)
        self.long_reward_multiplier = 1.0  # Equal reward for both directions
        self.short_reward_multiplier = 1.0  # Equal reward for both directions
        self.action_diversity_penalty = 0.001
        self.min_episode_length_for_rewards = 1
        self.short_episode_penalty = 0.0

        # Strategy balancing parameters (NEW)
        self.direction_balance_target = 0.5  # Target 50% long, 50% short
        self.direction_balance_penalty_weight = 0.25  # Weight for balance penalty (increased from 0.15)
        self.max_direction_concentration = 0.65  # Maximum 65% in one direction (reduced from 0.7)
        self.direction_streak_limit = 4  # Maximum consecutive same-direction trades (reduced from 5)
        self.exploration_bonus_weight = 0.15  # Bonus for exploring underrepresented directions (increased from 0.10)
        
        # Direction-specific reward multipliers (applied to opening positions)
        self.long_open_bonus = 0.20  # Bonus for opening long when shorts dominate (increased from 0.15)
        self.short_open_bonus = 0.20  # Bonus for opening short when longs dominate (increased from 0.15)

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
        self.price_rolling_mean = df[price_col].rolling(window=window_size, min_periods=1).mean().values
        self.price_rolling_std = df[price_col].rolling(window=window_size, min_periods=1).std().values
        self.price_rolling_std[self.price_rolling_std == 0] = 1

        if self.debug:
            print(f"Price normalization: rolling window {window_size}")
            print(f"Margin requirements: initial={self.margin_requirement*100:.0f}%, maintenance={self.maintenance_margin*100:.0f}%")
            print(f"Trailing stop-loss: {self.trailing_stop_distance*100:.1f}% distance")

        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Cover Short
        self.action_space = spaces.Discrete(5)

        # State: [balance_norm, position_norm, price_norm, indicators...]
        n_indicators = 7
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.5, -15.0] + [-1.0] * n_indicators, dtype=np.float32),
            high=np.array([2.0, 2.5, 15.0] + [1.0] * n_indicators, dtype=np.float32),
            dtype=np.float32
        )

        # Add state normalization parameters
        self.state_means = np.array([0.0, 0.0, 0.0] + [0.0] * n_indicators, dtype=np.float32)
        self.state_stds = np.array([1.0, 1.0, 3.0] + [0.5] * n_indicators, dtype=np.float32)

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
        """Get current state observation"""
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

        obv = row.get('OBV', 0)
        if np.isnan(obv) or np.isinf(obv):
            obv = 0

        ad = row.get('AD', 0)
        if np.isnan(ad) or np.isinf(ad):
            ad = 0

        mfi = row.get('MFI_15', 50)
        if np.isnan(mfi) or np.isinf(mfi) or mfi < 0 or mfi > 100:
            mfi = 50

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

        # Technical indicators with validation
        indicators = [
            rsi / 100 - 0.5,  # RSI normalized to [-0.5, 0.5]
            (bb_upper / current_price - 1) if current_price > 0 else 0,  # BB upper as % deviation
            (bb_lower / current_price - 1) if current_price > 0 else 0,  # BB lower as % deviation
            atr / 1000,  # ATR normalized
            obv / 1e10,  # OBV normalized
            ad / 1e10, # AD normalized
            mfi / 100 - 0.5  # MFI normalized to [-0.5, 0.5]
        ]

        state = np.array([balance_norm, position_norm, price_norm] + indicators, dtype=np.float32)
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
        # Reward opening positions in underrepresented direction
        if action == 1:  # Buy Long (opening)
            if short_ratio > 0.6:  # Shorts dominate
                balance_reward += self.long_open_bonus * 2  # Strong bonus for diversifying
            elif short_ratio > 0.5:
                balance_reward += self.long_open_bonus
        elif action == 3:  # Sell Short (opening)
            if long_ratio > 0.6:  # Longs dominate
                balance_reward += self.short_open_bonus * 2  # Strong bonus for diversifying
            elif long_ratio > 0.5:
                balance_reward += self.short_open_bonus
        
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
        Enhanced reward function with strategy balancing
        
        Key components:
        1. Base reward: portfolio change
        2. Trade reward: PnL from closed trades (with direction multipliers)
        3. Strategy balance reward: incentives for balanced trading
        4. Hold penalty: discourage excessive inactivity
        5. Market comparison: reward beating the market
        """
        # Calculate current equity (portfolio value)
        current_portfolio = self.balance + self.margin_locked + self.position * current_price

        # Base reward: portfolio change
        if self.prev_portfolio_value > 0:
            portfolio_return = (current_portfolio - self.prev_portfolio_value) / self.prev_portfolio_value
            base_reward = portfolio_return * 100
        else:
            base_reward = 0.0

        # Trade reward (if trade was performed) with direction multipliers
        trade_reward = 0
        if action_performed and pnl_pct is not None:
            # Apply direction-specific multipliers for profitable trades
            if pnl_pct > 0:
                # Check if this was a long or short trade
                if action == 2:  # Sell Long (closing long position)
                    trade_reward = pnl_pct * 100 * self.long_reward_multiplier
                elif action == 4:  # Cover Short (closing short position)
                    trade_reward = pnl_pct * 100 * self.short_reward_multiplier
                else:
                    trade_reward = pnl_pct * 100
            else:
                # Smaller penalty for losses to encourage risk-taking
                trade_reward = pnl_pct * 50

        # Action diversity reward
        action_diversity_reward = 0
        if len(self.action_history) >= 2:
            unique_actions = len(set(self.action_history))
            if unique_actions >= 3:
                action_diversity_reward = 0.1

        # Hold penalty
        hold_penalty = 0
        if action == 0:
            if self.steps_since_last_trade > 5:
                hold_penalty = -0.05

        # Market comparison reward
        market_comparison_reward = 0
        if hasattr(self, 'prev_price') and self.prev_price > 0:
            market_return = (current_price - self.prev_price) / self.prev_price
            excess_return = portfolio_return - market_return
            if excess_return > 0.001:
                market_comparison_reward = excess_return * 50

        # Episode length penalty
        episode_length_penalty = 0
        if self.steps_in_episode < self.min_episode_length_for_rewards:
            missing_steps = self.min_episode_length_for_rewards - self.steps_in_episode
            episode_length_penalty = -missing_steps * 0.2

        # Strategy balance reward (ENHANCED COMPONENT)
        strategy_balance_reward = self._calculate_strategy_balance_reward(action)

        # Combine all components
        reward = (base_reward + trade_reward + action_diversity_reward + 
                 hold_penalty + market_comparison_reward + 
                 episode_length_penalty + strategy_balance_reward)
        
        # Save previous price for next step
        self.prev_price = current_price

        # Clip reward for stability
        reward = np.clip(reward, -20, 20)

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

            # Update strategy balance tracking
            self._update_strategy_balance_tracking(action_int)

            # Execute action
            if action == 0:  # Hold
                self.steps_since_last_trade += 1
            elif action == 1:  # Buy Long
                action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
            elif action == 2:  # Sell Long
                action_performed, pnl_pct = self._execute_sell_long(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1
            elif action == 3:  # Sell Short
                action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
            elif action == 4:  # Cover Short
                action_performed, pnl_pct = self._execute_cover_short(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1

            # Check trailing stop-loss
            self._check_trailing_stop(current_price)
            
            # Calculate current portfolio value BEFORE reward calculation
            current_portfolio = self.balance + self.margin_locked + self.position * current_price

            # Calculate reward based on portfolio change
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
            
            # Check maximum episode loss condition (critical risk management)
            pass
            
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
                
                # Final reward based on overall performance (log return)
                if self.prev_portfolio_value > 0:
                    final_return = (final_portfolio - self.prev_portfolio_value) / self.prev_portfolio_value
                    final_reward = np.log(1 + final_return) * 200
                else:
                    final_reward = portfolio_return * 200
                
                # Bonus for overall performance
                if portfolio_return > 0.1:
                    final_reward += 15
                elif portfolio_return > 0.05:
                    final_reward += 8
                elif portfolio_return > 0.02:
                    final_reward += 3
                elif portfolio_return > 0:
                    final_reward += 1
                
                # Penalty for poor performance
                if portfolio_return < -0.1:
                    final_reward -= 15
                elif portfolio_return < -0.05:
                    final_reward -= 8
                elif portfolio_return < -0.02:
                    final_reward -= 3
                
                # Bonus for active trading
                if self.total_trades > 0:
                    trade_bonus = min(self.total_trades * 0.5, 5.0)
                    final_reward += trade_bonus
                    
                    # Additional bonus for good win rate
                    win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
                    if win_rate > 0.6:
                        final_reward += 3
                    elif win_rate > 0.5:
                        final_reward += 1
                
                # Penalty for no trading
                if self.total_trades == 0:
                    final_reward -= 5

                # Add strategy balance metrics to final info
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
        """Execute buy long action"""
        if max_order_size_long >= min_order_size and self.balance > max_order_size_long and current_price > 0:
            invest_amount = min(max_order_size_long, self.balance)
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
        """Execute sell short action"""
        if max_order_size_short >= min_order_size and self.balance > max_order_size_short and current_price > 0:
            short_amount = min(max_order_size_short, self.balance)
            margin_required = short_amount * self.margin_requirement
            available_balance = self.balance - self.margin_locked

            if available_balance >= margin_required:
                coins_short = short_amount / current_price
                fee = coins_short * current_price * self.transaction_fee
                
                self.short_opening_fees = fee

                self.margin_locked += margin_required
                self.balance -= margin_required
                self.balance += short_amount - fee
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
                else:
                    old_position_size = abs(self.position + coins_short)
                    new_position_size = abs(self.position)
                    self.entry_price = ((old_position_size * self.entry_price) + (coins_short * current_price)) / new_position_size

                self.total_trades += 1
                self.steps_since_last_trade = 0

                return True
        return False

    def _execute_cover_short(self, current_price):
        """Execute cover short action"""
        if self.position < 0:
            position_size = abs(self.position)

            price_pnl = (self.entry_price - current_price) * position_size
            
            open_fee = self.short_opening_fees
            close_fee = current_price * position_size * self.transaction_fee
            total_fee = close_fee
            
            pnl = price_pnl - close_fee
            pnl_pct = pnl / (self.entry_price * position_size) if self.entry_price > 0 else 0

            self.balance += self.margin_locked + pnl
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