#!/usr/bin/env python3
"""
Unified Trading Environment for Reinforcement Learning
Implements a standardized trading environment that can be used across training, paper trading, and live trading
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TradingEnvironment(gym.Env):
    """
    Standardized trading environment for RL agent
    """
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018, episode_length=1000):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.episode_length = episode_length
        self.current_step = 0

        # Risk management parameters
        self.max_position_size = 0.25  # Max 25% of balance per position
        self.max_total_exposure = 0.5  # Max 50% of balance in total exposure
        self.stop_loss_pct = 0.08  # Stop loss at 8%
        self.take_profit_pct = 0.15  # Take profit at 15%
        
        # Margin trading parameters
        self.margin_requirement = 0.3  # 30% initial margin for shorts
        self.maintenance_margin = 0.15  # 15% maintenance margin
        self.liquidation_penalty = 0.01  # 1% penalty on liquidation

        # Trailing stop-loss parameters
        self.trailing_stop_distance = 0.05  # 5% trailing distance
        self.trailing_stop_enabled = True

        # Trading parameters
        self.base_position_size = 0.1  # 10% of balance
        self.hold_penalty = 0.001  # Small penalty for holding
        self.inactivity_penalty = 0.002  # Penalty for not trading
        self.termination_stop_loss_threshold = 0.70  # Stop if balance drops to 70%
        self.termination_profit_target = 1.50  # Stop if balance reaches 150%
        
        # Reward parameters
        self.reward_clip_bounds = (-50, 50)  # Clip rewards to reasonable bounds
        self.long_reward_multiplier = 1.2  # Long positions get bonus
        self.short_reward_multiplier = 1.0  # Short positions get normal reward
        self.action_diversity_penalty = 0.01  # Penalty for repetitive actions

        # Action diversity tracking
        self.action_history = []  # Track last 10 actions
        self.max_action_history = 10

        # Use rolling window normalization instead of full dataset z-score
        price_col = 'close' if 'close' in df.columns else 'Close'
        # Calculate rolling mean and std for better stationarity
        window_size = min(100, len(df) // 10)  # 10% of data or 100, whichever is smaller
        self.price_rolling_mean = df[price_col].rolling(window=window_size, min_periods=1).mean().values
        self.price_rolling_std = df[price_col].rolling(window=window_size, min_periods=1).std().values
        self.price_rolling_std[self.price_rolling_std == 0] = 1  # Avoid division by zero

        print(f"Price normalization: rolling window {window_size}")
        print(f"Margin requirements: initial={self.margin_requirement*100:.0f}%, maintenance={self.maintenance_margin*100:.0f}%")
        print(f"Trailing stop-loss: {self.trailing_stop_distance*100:.1f}% distance")

        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Cover Short
        self.action_space = spaces.Discrete(5)

        # State: [balance_norm, position_norm, price_norm, indicators...]
        n_indicators = 7
        # Use reasonable bounds instead of -inf/inf to prevent NN issues
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.5, -15.0] + [-1.0] * n_indicators, dtype=np.float32),
            high=np.array([2.0, 2.5, 15.0] + [1.0] * n_indicators, dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset episode state variables
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = 0.0  # 0=no position, positive=long, negative=short
        self.total_fees = 0.0
        self.portfolio_values = [float(self.initial_balance)]

        # Margin trading tracking
        self.margin_locked = 0.0  # Amount of balance locked as margin for short positions

        # Position tracking
        self.entry_price = 0.0
        self.entry_step = 0  # Track when position was opened
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

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]

        # Validate current price
        current_price = row.get('close', row.get('Close', 50000))  # Support both naming conventions
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
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

        # Better position normalization: use market value of position relative to portfolio value
        portfolio_value = self.balance + self.margin_locked + self.position * current_price
        # Protect against division by zero or very small values that could cause NaN/inf
        if portfolio_value > 1e-6:  # Use small epsilon to avoid division by near-zero
            position_norm = (self.position * current_price) / portfolio_value
        else:
            position_norm = 0.0  # Safe fallback when portfolio is essentially zero

        # Clip to reasonable range to prevent NN issues and ensure numerical stability
        position_norm = np.clip(position_norm, -2.5, 2.5)

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
            ad / 1e10,  # AD normalized
            mfi / 100 - 0.5  # MFI normalized to [-0.5, 0.5]
        ]

        state = np.array([balance_norm, position_norm, price_norm] + indicators, dtype=np.float32)
        # Protect against NaN and inf values that could break the neural network
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        # Additional safety: clip to observation space bounds to ensure numerical stability
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        return state

    def _get_position_sizes(self):
        """Calculate position sizes for orders"""
        # Calculate max position sizes based on risk management
        max_order_size_long = min(self.balance * self.base_position_size, 
                                 self.balance * self.max_position_size,
                                 (self.initial_balance * self.max_total_exposure - abs(self.position * self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))) if self.position != 0 else self.initial_balance * self.max_total_exposure))
        
        max_order_size_short = min(self.balance * self.base_position_size * 0.8,  # Slightly more conservative for shorts
                                   self.balance * self.max_position_size,
                                   (self.initial_balance * self.max_total_exposure - abs(self.position * self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))) if self.position != 0 else self.initial_balance * self.max_total_exposure))
        
        min_order_size = 5  # Reduced from $10 to $5 minimum
        return max_order_size_long, max_order_size_short, min_order_size

    def _execute_buy_long(self, current_price, max_order_size_long, min_order_size):
        """Execute buy long action"""
        if max_order_size_long >= min_order_size and self.balance > max_order_size_long and current_price > 0:
            invest_amount = min(max_order_size_long, self.balance)
            fee = invest_amount * self.transaction_fee
            coins_bought = (invest_amount - fee) / current_price
            
            # Check if we exceed max position size
            new_position_value = (self.position + coins_bought) * current_price
            if new_position_value > self.initial_balance * self.max_position_size:
                return False  # Would exceed max position size

            self.position += coins_bought
            self.balance -= invest_amount
            self.total_fees += fee

            # Validate balance after operation
            if self.balance < -1e-6:
                print(f"WARNING: Negative balance after Buy Long: {self.balance}")

            # Update entry tracking
            if self.entry_price == 0:  # New position
                self.entry_price = current_price
                self.entry_step = self.current_step
                self.highest_price_since_entry = current_price
                self.lowest_price_since_entry = current_price
                self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)
            else:  # Adding to existing position
                old_position = self.position - coins_bought
                self.entry_price = ((old_position * self.entry_price) + (coins_bought * current_price)) / self.position

            # Update action tracking
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

            # Validate balance after operation
            if self.balance < -1e-6:
                print(f"WARNING: Negative balance after Sell Long: {self.balance}")

            # Update P&L tracking
            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            # Reset position tracking
            self.position = 0
            self.entry_price = 0
            self.entry_step = 0
            self.highest_price_since_entry = 0
            self.lowest_price_since_entry = float('inf')
            self.trailing_stop_loss = 0
            self.trailing_take_profit = 0

            # Update action tracking
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

                # Lock margin and receive proceeds
                self.margin_locked += margin_required
                self.balance -= margin_required
                self.balance += short_amount - fee
                self.position -= coins_short
                self.total_fees += fee

                # Validate balance and margin
                if self.balance < -1e-6:
                    print(f"WARNING: Negative balance after Sell Short: {self.balance}")
                if self.margin_locked < -1e-6:
                    print(f"WARNING: Negative margin_locked after Sell Short: {self.margin_locked}")

                # Update entry tracking
                if self.entry_price == 0:  # New position
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    self.highest_price_since_entry = current_price
                    self.lowest_price_since_entry = current_price
                    self.trailing_stop_loss = current_price * (1 + self.trailing_stop_distance)
                else:  # Adding to existing position
                    old_position_size = abs(self.position + coins_short)
                    new_position_size = abs(self.position)
                    self.entry_price = ((old_position_size * self.entry_price) + (coins_short * current_price)) / new_position_size

                # Update action tracking
                self.total_trades += 1
                self.steps_since_last_trade = 0

                return True
        return False

    def _execute_cover_short(self, current_price):
        """Execute cover short action"""
        if self.position < 0:
            position_size = abs(self.position)

            # Calculate PnL
            price_pnl = (self.entry_price - current_price) * position_size
            open_fee = self.entry_price * position_size * self.transaction_fee
            close_fee = current_price * position_size * self.transaction_fee
            total_fee = open_fee + close_fee
            pnl = price_pnl - total_fee
            pnl_pct = pnl / (self.entry_price * position_size) if self.entry_price > 0 else 0

            # Return margin + PnL
            self.balance += self.margin_locked + pnl
            self.margin_locked = 0
            self.total_fees += total_fee

            # Validate balance after operation
            if self.balance < -1e-6:
                print(f"WARNING: Negative balance after Cover Short: {self.balance}")

            # Update P&L tracking
            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            # Reset position tracking
            self.position = 0
            self.entry_price = 0
            self.entry_step = 0
            self.highest_price_since_entry = 0
            self.lowest_price_since_entry = float('inf')
            self.trailing_stop_loss = 0
            self.trailing_take_profit = 0

            # Update action tracking
            self.steps_since_last_trade = 0

            return True, pnl_pct
        return False, 0

    def _check_trailing_stop(self, current_price):
        """Check and execute trailing stop-loss if triggered"""
        if self.position == 0 or self.entry_price == 0:
            return

        if self.position > 0:  # Long position
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_loss = self.highest_price_since_entry * (1 - self.trailing_stop_distance)

            # Check trailing stop
            if current_price <= self.trailing_stop_loss:
                revenue = self.position * current_price
                fee = revenue * self.transaction_fee
                pnl = revenue - (self.position * self.entry_price) - fee
                
                self.balance += revenue - fee
                self.total_fees += fee

                # Update P&L tracking
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                # Validate balance
                if self.balance < -1e-6:
                    print(f"WARNING: Negative balance after trailing stop (long): {self.balance}")

                self.position = 0
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.lowest_price_since_entry = float('inf')
                
                # Reset trailing stops
                self.trailing_stop_loss = 0
                self.trailing_take_profit = 0

        elif self.position < 0:  # Short position
            if current_price < self.lowest_price_since_entry:
                self.lowest_price_since_entry = current_price
                self.trailing_stop_loss = self.lowest_price_since_entry * (1 + self.trailing_stop_distance)

            # Check trailing stop
            if current_price >= self.trailing_stop_loss:
                cover_cost = abs(self.position) * current_price
                fee = cover_cost * self.transaction_fee
                entry_value = abs(self.position) * self.entry_price if self.entry_price > 0 else 0
                pnl = entry_value - cover_cost - fee
                
                self.balance = self.balance + self.margin_locked - cover_cost - fee
                self.margin_locked = 0
                self.total_fees += fee

                # Update P&L tracking
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                # Validate balance and margin
                if self.balance < -1e-6:
                    print(f"WARNING: Negative balance after trailing stop (short): {self.balance}")
                if self.margin_locked < -1e-6:
                    print(f"WARNING: Negative margin_locked after trailing stop (short): {self.margin_locked}")

                self.position = 0
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.lowest_price_since_entry = float('inf')
                
                # Reset trailing stops
                self.trailing_stop_loss = 0
                self.trailing_take_profit = 0

    def _calculate_reward(self, current_price, action, pnl_pct=None):
        """Calculate reward based on multiple factors"""
        # Basic portfolio return
        current_portfolio = self.balance + self.margin_locked + self.position * current_price
        portfolio_return = (current_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Reward based on portfolio growth
        portfolio_reward = portfolio_return * 100  # Scale up for better learning signal
        
        # Risk-adjusted return using Sharpe-like ratio
        if len(self.portfolio_values) > 2:
            returns = np.diff(self.portfolio_values) / (np.array(self.portfolio_values[:-1]) + 1e-8)
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_like = np.mean(returns) / np.std(returns)
                risk_adjusted_reward = sharpe_like * 10  # Weighted risk-adjusted reward
            else:
                risk_adjusted_reward = 0
        else:
            risk_adjusted_reward = 0
        
        # Drawdown penalty
        if len(self.portfolio_values) > 1:
            peak = max(self.portfolio_values)
            drawdown = (peak - current_portfolio) / peak if peak > 0 else 0
            drawdown_penalty = -drawdown * 50  # Penalty proportional to drawdown
        else:
            drawdown_penalty = 0
        
        # Position-based reward
        position_reward = 0
        if abs(self.position) > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) if self.entry_price > 0 else 0
            position_reward = unrealized_pnl / self.initial_balance * 50  # Reward for profitable positions
        
        # Action-specific reward
        action_reward = 0
        if action == 0:  # Hold
            action_reward = -self.hold_penalty  # Small penalty for holding
            self.steps_since_last_trade += 1
        else:
            # Reward for taking action, especially for profitable trades
            if pnl_pct is not None and pnl_pct > 0:
                action_reward = abs(pnl_pct) * 50  # Higher reward for profitable trades
            else:
                action_reward = 0.1  # Base reward for taking action
        
        # Inactivity penalty
        inactivity_penalty = 0
        if self.steps_since_last_trade > 50:  # If no trade in 50 steps
            inactivity_penalty = -self.inactivity_penalty * (self.steps_since_last_trade // 50)
        
        # Action diversity reward/penalty
        diversity_score = 0
        if len(self.action_history) > 5:
            unique_actions = len(set(self.action_history[-5:]))  # Unique actions in last 5
            diversity_score = (unique_actions / 5.0) * 2  # Up to +2 for full diversity
        
        # Combine all reward components
        total_reward = (
            portfolio_reward * 0.3 +          # 30% weight to portfolio return
            risk_adjusted_reward * 0.2 +      # 20% weight to risk-adjusted return
            drawdown_penalty * 0.1 +          # 10% weight to drawdown penalty
            position_reward * 0.2 +           # 20% weight to position reward
            action_reward * 0.15 +            # 15% weight to action reward
            inactivity_penalty * 0.05 +       # 5% weight to inactivity penalty
            diversity_score * 0.1             # 10% weight to diversity
        )
        
        # Additional reward for profitable closed trades
        if pnl_pct is not None and pnl_pct > 0:
            total_reward += abs(pnl_pct) * 20  # Extra reward for profitable trades
        
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, *self.reward_clip_bounds)
        
        return total_reward

    def step(self, action):
        """Execute one step in environment with enhanced reward function"""
        try:
            # Check if episode is done
            if self.current_step >= len(self.df) - 1 or self.current_step >= self.episode_length - 1:
                terminated = True
                truncated = self.current_step >= self.episode_length - 1  # Truncated if max steps reached
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))
                final_portfolio = self.balance + self.margin_locked + self.position * current_price
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0

                # Final reward based on overall performance
                final_reward = portfolio_return * 100  # Scale up final return
                
                # Add bonus for good performance
                if portfolio_return > 0.1:  # 10% return
                    final_reward += 20
                elif portfolio_return > 0.05:  # 5% return
                    final_reward += 10
                elif portfolio_return > 0:  # Any positive return
                    final_reward += 5
                
                # Subtract penalty for poor performance
                if portfolio_return < -0.1:  # -10% return
                    final_reward -= 20
                elif portfolio_return < -0.05:  # -5% return
                    final_reward -= 10
                
                self.portfolio_values.append(final_portfolio)
                
                return self._get_state(), final_reward, terminated, truncated, {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_pnl': self.total_pnl,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown()
                }

            # Get current market data
            current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))

            # Calculate position sizes
            max_order_size_long, max_order_size_short, min_order_size = self._get_position_sizes()

            # Initialize reward and pnl
            reward = 0
            action_performed = False
            pnl_pct = None

            # Add action to history for diversity tracking
            self.action_history.append(action)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)

            # Execute action
            if action == 0:  # Hold
                reward = -self.hold_penalty  # Small penalty for holding
                self.steps_since_last_trade += 1
            elif action == 1:  # Buy Long
                action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
                if action_performed:
                    reward = 0.1  # Base reward for taking action
            elif action == 2:  # Sell Long
                action_performed, pnl_pct = self._execute_sell_long(current_price)
                if action_performed:
                    # Reward based on profit/loss of the trade
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            reward = pnl_pct * 50  # Higher reward for profitable trades
                            self.consecutive_losses = 0  # Reset loss counter
                        else:
                            reward = pnl_pct * 20  # Lower penalty for unprofitable trades
                            self.consecutive_losses += 1
                            reward -= self.consecutive_losses * 0.5  # Penalty for consecutive losses
            elif action == 3:  # Sell Short
                action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
                if action_performed:
                    reward = 0.1  # Base reward for taking action
            elif action == 4:  # Cover Short
                action_performed, pnl_pct = self._execute_cover_short(current_price)
                if action_performed:
                    # Reward based on profit/loss of the trade
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            reward = pnl_pct * 50  # Higher reward for profitable trades
                            self.consecutive_losses = 0  # Reset loss counter
                        else:
                            reward = pnl_pct * 20  # Lower penalty for unprofitable trades
                            self.consecutive_losses += 1
                            reward -= self.consecutive_losses * 0.5  # Penalty for consecutive losses

            # Check trailing stop-loss
            self._check_trailing_stop(current_price)

            # Calculate reward based on multiple factors
            reward = self._calculate_reward(current_price, action, pnl_pct)

            # Check termination conditions based on risk management
            current_portfolio = self.balance + self.margin_locked + self.position * current_price
            portfolio_return = (current_portfolio - self.initial_balance) / self.initial_balance

            terminated = False
            truncated = False

            # Check stop loss condition
            if current_portfolio < self.initial_balance * self.termination_stop_loss_threshold:  # -30% portfolio loss
                reward -= 10 # Penalty for early termination due to losses
                terminated = True
            # Check profit target condition
            elif current_portfolio > self.initial_balance * self.termination_profit_target:  # +50% portfolio gain
                reward += 20  # Bonus for reaching profit target
                terminated = True

            self.portfolio_values.append(current_portfolio)
            self.current_step += 1

            # Additional termination condition if we reach the end of the data
            if self.current_step >= len(self.df) - 1:
                terminated = True

            # Prepare info dict with additional metrics
            info = {
                'portfolio_value': current_portfolio,
                'balance': self.balance,
                'position': self.position,
                'margin_locked': self.margin_locked,
                'current_price': current_price,
                'total_fees': self.total_fees,
                'total_trades': self.total_trades,
                'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                'current_step': self.current_step,
                'action_taken': action,
                'pnl_pct': pnl_pct if pnl_pct is not None else 0
            }

            return self._get_state(), reward, terminated, truncated, info

        except Exception as e:
            print(f"ERROR: Exception in step() method: {e}")
            import traceback
            traceback.print_exc()
            return self._get_state(), 0, True, False, {}

    def render(self, mode='human'):
        """Render environment state"""
        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        # Include margin_locked in portfolio value
        portfolio_value = self.balance + self.margin_locked + self.position * current_price
        win_rate = self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.6f}, Portfolio: {portfolio_value:.2f}, Wins: {self.win_count}/{self.total_trades} ({win_rate:.2%})")

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

# For backward compatibility
def create_env(df, initial_balance=10000, transaction_fee=0.0018):
    """Factory function to create environment instances"""
    return TradingEnvironment(df, initial_balance, transaction_fee)
