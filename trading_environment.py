#!.venv/bin/ python3
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
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018, episode_length=200, start_step=None, debug=False):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start_step = 0  # Track where episode started
        self.steps_in_episode = 0  # Track steps since episode start
        self.start_step = start_step  # For random episode start
        self.debug = debug  # Control debug prints

        # Enhanced risk management parameters
        self.max_position_size = 0.15  # Max 15% of balance per position (reduced from 25%)
        self.max_total_exposure = 0.40  # Max 40% of balance in total exposure (reduced from 50%)
        self.stop_loss_pct = 0.08  # Stop loss at 8%
        self.take_profit_pct = 0.15  # Take profit at 15%
        self.max_episode_loss_pct = 0.05  # Maximum 5% loss per episode
        
        # Dynamic position sizing parameters
        self.volatility_window = 20  # Window for volatility calculation
        self.min_position_size = 0.02  # Minimum 2% position size
        self.max_dynamic_position_size = 0.20  # Maximum 20% dynamic position size
        
        # Adaptive stop-loss parameters
        self.adaptive_stop_loss_enabled = True
        self.atr_multiplier = 2.0  # ATR multiplier for stop loss
        self.min_stop_loss_pct = 0.03  # Minimum 3% stop loss
        self.max_stop_loss_pct = 0.15  # Maximum 15% stop loss
        
        # Margin trading parameters
        self.margin_requirement = 0.3  # 30% initial margin for shorts
        self.maintenance_margin = 0.15  # 15% maintenance margin
        self.liquidation_penalty = 0.01  # 1% penalty on liquidation

        # Trailing stop-loss parameters
        self.trailing_stop_distance = 0.05  # 5% trailing distance
        self.trailing_stop_enabled = True

        # Trading parameters
        self.base_position_size = 0.08  # Reduced to 8% of balance (dynamic sizing will adjust)
        self.hold_penalty = 0.00001  # Much smaller penalty for holding (was 0.0001)
        self.inactivity_penalty = 0.0001  # Much smaller penalty for not trading (was 0.001)
        
        # Directional balance tracking for long/short balancing
        self.long_count = 0
        self.short_count = 0
        self.balance_tracking_window = 100
        self.termination_stop_loss_threshold = 0.95  # Stop if balance drops to 95% (reduced from 70% for early loss control)
        self.termination_profit_target = 1.50  # Stop if balance reaches 150%

        # Reward parameters
        self.reward_clip_bounds = (-50, 50)  # Clip rewards to reasonable bounds (reduced from -500, 500)
        self.long_reward_multiplier = 1.2  # Long positions get bonus
        self.short_reward_multiplier = 1.0  # Short positions get normal reward
        self.action_diversity_penalty = 0.001  # Smaller penalty for repetitive actions (was 0.01)
        self.min_episode_length_for_rewards = 1  # Minimum episode length to get significant rewards (reduced to 1)
        self.short_episode_penalty = 0.0  # Penalty multiplier for short episodes (removed)

        # Action diversity tracking
        self.action_history = []  # Track last 10 actions
        self.max_action_history = 10

        # Use rolling window normalization instead of full dataset z-score
        price_col = 'close' if 'close' in df.columns else 'Close'
        # Calculate rolling mean and std for better stationarity
        window_size = min(100, len(df) // 10)  # 10% of data or 100, whichever is smaller
        self.price_rolling_mean = df[price_col].rolling(window=window_size, min_periods=1).mean().values.copy()
        self.price_rolling_std = df[price_col].rolling(window=window_size, min_periods=1).std().values.copy()
        self.price_rolling_std[self.price_rolling_std == 0] = 1  # Avoid division by zero

        if self.debug:
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

        # Add state normalization parameters
        self.state_means = np.array([0.0, 0.0, 0.0] + [0.0] * n_indicators, dtype=np.float32)
        self.state_stds = np.array([1.0, 1.0, 3.0] + [0.5] * n_indicators, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset episode state variables
        # Randomize start step if not specified and data is large enough
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
        self.position = 0.0  # 0=no position, positive=long, negative=short
        self.total_fees = 0.0
        self.portfolio_values = [float(self.initial_balance)]

        # Margin trading tracking
        self.margin_locked = 0.0  # Amount of balance locked as margin for short positions
        self.short_position_value = 0.0  # Value of short position (for accounting purposes)
        
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
        
        # Remove debug print statements to reduce I/O overhead
        # The following debug prints were removed:
        # print(f"Price normalization: rolling window {window_size}")
        # print(f"Margin requirements: initial={self.margin_requirement*100:.0f}%, maintenance={self.maintenance_margin*100:.0f}%")
        # print(f"Trailing stop-loss: {self.trailing_stop_distance*100:.1f}% distance")
        
        # Initialize internal state for proper accounting
        self.short_opening_fees = 0.0  # Track short opening fees to avoid double counting
        self.balance_before_short = 0.0  # Track balance before short position for proper accounting
        self.proceeds_from_short = 0.0  # Track proceeds from short sales
        
        # Proper margin accounting initialization
        self.cash_balance = self.initial_balance  # Pure cash balance (excluding margin effects)
        self.borrowed_assets = 0.0  # Track borrowed assets for short positions
        self.short_position_value = 0.0  # Track value of borrowed assets
        
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
        self.episode_start_balance = self.initial_balance  # Track initial balance for this episode
        self.cash = self.initial_balance  # Separate cash tracking
        self.equity = self.initial_balance  # Track total equity
        self.liability = 0.0  # Track liability for short positions

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
        
        # Track previous portfolio value for reward calculation
        self.prev_portfolio_value = float(self.initial_balance)
        
        # Track fees for current step
        self.fees_step = 0.0
        
        # Track short opening fees to avoid double-counting
        self.short_opening_fees = 0.0
        
        # Initialize portfolio history list
        self.portfolio_history = [float(self.initial_balance)]
        
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
        current_price = row.get('close', row.get('Close', 50000))  # Support both naming conventions
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
            ad / 1e10,  # AD normalized
            mfi / 100 - 0.5  # MFI normalized to [-0.5, 0.5]
        ]

        state = np.array([balance_norm, position_norm, price_norm] + indicators, dtype=np.float32)
        # Protect against NaN and inf values that could break the neural network
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        # Additional safety: clip to observation space bounds to ensure numerical stability
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        return state

    def _calculate_volatility_based_position_size(self, current_price):
        """Calculate position size based on market volatility"""
        if self.current_step < self.volatility_window:
            return self.base_position_size  # Use base size if not enough data
        
        # Get recent price data for volatility calculation
        start_idx = max(0, self.current_step - self.volatility_window)
        recent_prices = self.df.iloc[start_idx:self.current_step + 1]
        
        price_col = 'close' if 'close' in recent_prices.columns else 'Close'
        prices = recent_prices[price_col].values
        
        if len(prices) < 2:
            return self.base_position_size
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.01
        
        # Normalize volatility (higher volatility = smaller position size)
        avg_volatility = 0.02  # Historical average volatility assumption
        volatility_factor = avg_volatility / (volatility + 0.001)  # Add small epsilon to avoid division by zero
        
        # Clamp the factor to reasonable bounds
        volatility_factor = np.clip(volatility_factor, 0.5, 2.0)  # Between 50% and 200% of base size
        
        # Calculate dynamic position size
        dynamic_size = self.base_position_size * volatility_factor
        dynamic_size = np.clip(dynamic_size, self.min_position_size, self.max_dynamic_position_size)
        
        return dynamic_size

    def _get_position_sizes(self):
        """Calculate position sizes for orders with volatility adjustment"""
        # Calculate dynamic position size based on current market volatility
        dynamic_position_size = self._calculate_volatility_based_position_size(
            self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        )
        
        # Calculate max position sizes based on risk management with dynamic sizing
        max_order_size_long = min(self.balance * dynamic_position_size, 
                                 self.balance * self.max_position_size,
                                 (self.initial_balance * self.max_total_exposure - abs(self.position * self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))) if self.position != 0 else self.initial_balance * self.max_total_exposure))
        
        max_order_size_short = min(self.balance * dynamic_position_size * 0.8,  # Slightly more conservative for shorts
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
            self.fees_step += fee

            # Validate balance after operation
            if self.balance < -1e-6 and not self.debug:
                pass  # Silent validation unless debug mode

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
            self.fees_step += fee

            # Validate balance after operation
            if self.balance < -1e-6 and not self.debug:
                pass  # Silent validation unless debug mode

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
                
                # Store opening fee to avoid double-counting on cover
                self.short_opening_fees = fee

                # Lock margin and receive short proceeds (proceeds increase balance, but we have a liability)
                self.margin_locked += margin_required
                self.balance -= margin_required  # Lock margin from available balance
                self.balance += short_amount - fee  # Receive proceeds minus fee (proceeds are liability but temporarily increase balance)
                self.position -= coins_short  # Update position (negative for short)
                self.total_fees += fee
                self.fees_step += fee

                # Validate balance and margin
                if self.balance < -1e-6 and self.debug:
                    print(f"WARNING: Negative balance after Sell Short: {self.balance}")
                if self.margin_locked < -1e-6 and self.debug:
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
            # Price PnL: we sold at entry_price, buying back at current_price
            price_pnl = (self.entry_price - current_price) * position_size
            
            # Use stored opening fee instead of recalculating
            open_fee = self.short_opening_fees
            close_fee = current_price * position_size * self.transaction_fee
            total_fee = close_fee  # Opening fee already counted
            
            pnl = price_pnl - close_fee  # Opening fee already deducted from balance
            pnl_pct = pnl / (self.entry_price * position_size) if self.entry_price > 0 else 0

            # Return margin + PnL (opening fee was already deducted when opening)
            self.balance += self.margin_locked + pnl
            self.margin_locked = 0
            self.total_fees += close_fee  # Only count closing fee here
            self.fees_step += close_fee
            self.short_opening_fees = 0.0  # Reset

            # Validate balance after operation
            if self.balance < -1e-6 and self.debug:
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

    def _calculate_adaptive_stop_loss(self, current_price):
        """Calculate adaptive stop loss based on ATR and market conditions"""
        if self.current_step < self.volatility_window:
            # Use fixed stop loss if not enough data for ATR
            return self.stop_loss_pct

        # Get recent ATR values for adaptive calculation
        start_idx = max(0, self.current_step - self.volatility_window)
        recent_data = self.df.iloc[start_idx:self.current_step + 1]
        
        atr_col = 'ATR_15' if 'ATR_15' in recent_data.columns else 'ATR'
        if atr_col in recent_data.columns:
            recent_atr = recent_data[atr_col].iloc[-1]
            current_price_val = current_price
            
            if recent_atr > 0 and current_price_val > 0:
                # Calculate stop loss as ATR multiple
                atr_stop_loss_pct = (recent_atr / current_price_val) * self.atr_multiplier
                # Clamp between minimum and maximum stop loss percentages
                adaptive_stop_loss = max(self.min_stop_loss_pct, min(atr_stop_loss_pct, self.max_stop_loss_pct))
                return adaptive_stop_loss
        
        # Fallback to fixed stop loss
        return self.stop_loss_pct

    def _check_trailing_stop(self, current_price):
        """Check and execute trailing stop-loss if triggered"""
        if self.position == 0 or self.entry_price == 0:
            return

        # Calculate adaptive stop loss if enabled
        if self.adaptive_stop_loss_enabled:
            adaptive_stop_pct = self._calculate_adaptive_stop_loss(current_price)
        else:
            adaptive_stop_pct = self.stop_loss_pct

        if self.position > 0:  # Long position
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_loss = self.highest_price_since_entry * (1 - adaptive_stop_pct)  # Use adaptive stop loss

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
                if self.balance < -1e-6 and self.debug:
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
                self.trailing_stop_loss = self.lowest_price_since_entry * (1 + adaptive_stop_pct)  # Use adaptive stop loss

            # Check trailing stop
            if current_price >= self.trailing_stop_loss:
                cover_cost = abs(self.position) * current_price
                close_fee = cover_cost * self.transaction_fee
                # Opening fee was already counted when opening short
                entry_value = abs(self.position) * self.entry_price if self.entry_price > 0 else 0
                price_pnl = entry_value - cover_cost
                pnl = price_pnl - close_fee  # Opening fee already deducted
                
                self.balance = self.balance + self.margin_locked - cover_cost - close_fee
                self.margin_locked = 0
                self.total_fees += close_fee  # Only closing fee
                self.fees_step += close_fee
                self.short_opening_fees = 0.0  # Reset

                # Update P&L tracking
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                # Validate balance and margin
                if self.balance < -1e-6 and self.debug:
                    print(f"WARNING: Negative balance after trailing stop (short): {self.balance}")
                if self.margin_locked < -1e-6 and self.debug:
                    print(f"WARNING: Negative margin_locked after trailing stop (short): {self.margin_locked}")

                self.position = 0
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.lowest_price_since_entry = float('inf')
                
                # Reset trailing stops
                self.trailing_stop_loss = 0
                self.trailing_take_profit = 0

    def _calculate_reward(self, current_price, action, action_performed=False, pnl_pct=None):
        """
        Ultra-selective reward function - reward PATIENCE and QUALITY
        
        Key principle: Most trades should be HOLD, only trade when confident
        - Heavy penalty for unprofitable trades
        - Reward for HOLD when market is uncertain
        - Strong reward only for HIGH CONFIDENCE profitable trades
        """
        # Calculate current equity (portfolio value)
        current_portfolio = self.balance + self.margin_locked + self.position * current_price
        
        # Track trade outcomes
        if not hasattr(self, 'trade_outcomes'):
            self.trade_outcomes = []
            self.peak_portfolio = current_portfolio
            self.steps_without_trade = 0
        
        # Update peak
        if current_portfolio > self.peak_portfolio:
            self.peak_portfolio = current_portfolio
        
        # === 1. HOLD REWARD: Reward patience ===
        hold_reward = 0
        if action == 0:  # HOLD
            self.steps_without_trade += 1
            # Small reward for patience (avoiding bad trades)
            if self.steps_without_trade > 5:
                hold_reward = 0.02  # Small but consistent reward for waiting
        else:
            self.steps_without_trade = 0
        
        # === 2. TRADE QUALITY REWARD ===
        trade_reward = 0
        if action in [2, 4] and pnl_pct is not None:  # Closing positions
            if pnl_pct > 0.002:  # Profit > 0.2%
                # EXCELLENT trade - strong reward
                trade_reward = pnl_pct * 500  # 5x multiplier for good profits
                self.trade_outcomes.append(('excellent', pnl_pct))
            elif pnl_pct > 0:
                # OK trade - moderate reward
                trade_reward = pnl_pct * 200
                self.trade_outcomes.append(('ok', pnl_pct))
            elif pnl_pct > -0.002:
                # Small loss - small penalty
                trade_reward = pnl_pct * 150
                self.trade_outcomes.append(('small_loss', pnl_pct))
            else:
                # BAD trade - heavy penalty
                trade_reward = pnl_pct * 300  # 3x penalty for big losses
                self.trade_outcomes.append(('bad', pnl_pct))
            
            # Keep only recent trades
            if len(self.trade_outcomes) > 30:
                self.trade_outcomes.pop(0)
        
        # === 3. WIN RATE BONUS/PENALTY ===
        win_rate_reward = 0
        if len(self.trade_outcomes) >= 5:
            wins = sum(1 for t in self.trade_outcomes if t[0] in ['excellent', 'ok'])
            win_rate = wins / len(self.trade_outcomes)
            
            if win_rate >= 0.7:
                win_rate_reward = 2.0  # Big bonus for high win rate
            elif win_rate >= 0.5:
                win_rate_reward = 0.5  # Small bonus for decent win rate
            elif win_rate < 0.3:
                win_rate_reward = -2.0  # Heavy penalty for low win rate
        
        # === 4. DRAWDOWN PENALTY ===
        drawdown_penalty = 0
        if self.peak_portfolio > 0:
            drawdown = (self.peak_portfolio - current_portfolio) / self.peak_portfolio
            if drawdown > 0.01:
                drawdown_penalty = -drawdown * 30
        
        # === 5. PORTFOLIO GROWTH ===
        growth_reward = 0
        if current_portfolio > self.initial_balance:
            growth = (current_portfolio - self.initial_balance) / self.initial_balance
            growth_reward = growth * 30
        
        # === COMBINE ===
        reward = (
            hold_reward +          # Reward patience
            trade_reward +         # Trade quality
            win_rate_reward +      # Win rate bonus/penalty
            drawdown_penalty +     # Drawdown penalty
            growth_reward          # Overall growth
        )
        
        # Store for next iteration
        self.prev_price = current_price

        # Clip
        reward = np.clip(reward, -20, 20)

        return reward

    def step(self, action):
        """Execute one step in environment with enhanced reward function"""
        try:
            # Check if episode should be done BEFORE executing action
            # (but after checking if we can still access data)
            # Note: current_step is the index we're about to process
            if self.current_step >= len(self.df):
                # Already beyond data - return final state
                terminated = True
                truncated = False
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))
                final_portfolio = self.balance + self.margin_locked + self.position * current_price
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
                final_reward = portfolio_return * 100
                self.portfolio_values.append(final_portfolio)
                # Only print if debug mode is enabled
                if self.debug:
                    print(f"Episode ended: current_step={self.current_step} >= len(df)={len(self.df)}")
                return self._get_state(), final_reward, terminated, truncated, {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_pnl': self.total_pnl,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown()
                }

            # Get current market data (using current_step as index)
            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
            else:
                # Fallback - should not happen due to check above
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))

            # Calculate position sizes
            max_order_size_long, max_order_size_short, min_order_size = self._get_position_sizes()

            # Initialize reward and pnl
            action_performed = False
            pnl_pct = None

            # Add action to history for diversity tracking
            # Ensure action is a plain Python int to avoid hashable type issues
            action_int = int(action.item()) if hasattr(action, 'item') else int(action)
            self.action_history.append(action_int)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)

            # Execute action
            if action == 0:  # Hold
                self.steps_since_last_trade += 1
            elif action == 1:  # Buy Long
                action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
                if action_performed:
                    self.long_count += 1  # Track long actions
            elif action == 2:  # Sell Long
                action_performed, pnl_pct = self._execute_sell_long(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0  # Reset loss counter
                        else:
                            self.consecutive_losses += 1
            elif action == 3:  # Sell Short
                action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
                if action_performed:
                    self.short_count += 1  # Track short actions
            elif action == 4:  # Cover Short
                action_performed, pnl_pct = self._execute_cover_short(current_price)
                if action_performed:
                    if pnl_pct is not None:
                        if pnl_pct > 0:
                            self.consecutive_losses = 0  # Reset loss counter
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
            # Note: steps_in_episode was just incremented, so it's now the current step count
            terminated = False
            truncated = False
            
            # First check: risk management conditions (stop loss / profit target)
            portfolio_return = (current_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            
            # Check maximum episode loss condition (critical risk management)
            # Удалены все условия раннего завершения эпизода для увеличения длины эпизодов
            # Check profit target condition - removed to allow longer episodes
            # Check stop loss condition - removed to allow longer episodes
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
                # Используем ту же формулу что и в обычном reward для согласованности
                if self.prev_portfolio_value > 0:
                    final_return = (final_portfolio - self.prev_portfolio_value) / self.prev_portfolio_value
                    final_reward = np.log(1 + final_return) * 200  # Такое же масштабирование как в _calculate_reward
                else:
                    final_reward = portfolio_return * 200
                
                # Бонус за общую производительность эпизода
                if portfolio_return > 0.1:  # 10% return
                    final_reward += 15  # Увеличено с 5 до 15
                elif portfolio_return > 0.05:  # 5% return
                    final_reward += 8  # Увеличено с 2 до 8
                elif portfolio_return > 0.02:  # 2% return
                    final_reward += 3
                elif portfolio_return > 0:  # Any positive return
                    final_reward += 1
                
                # Штраф за плохую производительность
                if portfolio_return < -0.1:  # -10% return
                    final_reward -= 15  # Увеличено с 5 до 15
                elif portfolio_return < -0.05:  # -5% return
                    final_reward -= 8  # Увеличено с 2 до 8
                elif portfolio_return < -0.02:  # -2% return
                    final_reward -= 3
                
                # Бонус за активную торговлю в эпизоде
                if self.total_trades > 0:
                    trade_bonus = min(self.total_trades * 0.5, 5.0)  # До 5 бонусных очков за активную торговлю
                    final_reward += trade_bonus
                    
                    # Дополнительный бонус за хороший win rate
                    win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
                    if win_rate > 0.6:  # Win rate > 60%
                        final_reward += 3
                    elif win_rate > 0.5:  # Win rate > 50%
                        final_reward += 1
                
                # Штраф за отсутствие торговли
                if self.total_trades == 0:
                    final_reward -= 5  # Штраф за полное бездействие
                
                # Add episode info for TensorBoard tracking
                episode_info = {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_pnl': self.total_pnl,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_count / max(1, self.total_trades) if self.total_trades > 0 else 0,
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'episode': {
                        'r': final_reward,  # Episode reward for TensorBoard
                        'l': self.steps_in_episode  # Episode length
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
                'unrealized_pnl': unrealized_pnl
            }

            # Debug prints only if enabled and limited to reduce I/O overhead
            if self.debug:
                if self.current_step < 10:  # Only first 10 steps for debug
                    print(f"DEBUG: Step {self.current_step}, Action {action}, Performed {action_performed}, Position {self.position:.4f}, Equity {equity:.2f}, Reward {reward:.2f}")
                elif action_performed and self.current_step % 50 == 0:  # Print successful trades periodically
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
