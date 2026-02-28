#!/usr/bin/env python3
"""
Enhanced Trading Environment V19 - CRITICAL FIXES APPLIED
==========================================================
Fixes:
  P0 ✅ SHORT PnL% calculation fixed (uses margin_requirement)
  P1 ✅ Reward function simplified (removed conflicting components)
  P1 ✅ Weighted average price includes fees
  P2 ✅ Removed hold_quality_bonus (conflicted with time_penalty)
  P2 ✅ Removed quick_close_penalty (redundant with min_hold_steps)
  P2 ✅ Removed short_bonus (already have short_reward_multiplier)
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
    Enhanced trading environment with V19 critical fixes
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
        self.max_position_size = 0.10
        self.max_total_exposure = 0.40
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        self.max_episode_loss_pct = 0.05

        # Initialize RiskManager for dynamic position sizing
        self.risk_manager = RiskManager(
            initial_capital=self.initial_balance,
            max_position_size=0.10,
            max_total_exposure=0.40,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            max_drawdown_limit=0.20
        )

        # Dynamic position sizing parameters
        self.volatility_window = 20
        self.min_position_size = 0.02
        self.max_dynamic_position_size = 0.10

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
        self.base_position_size = 0.10
        self.hold_penalty = 0.00001
        self.inactivity_penalty = 0.0001
        self.termination_stop_loss_threshold = 0.95
        self.termination_profit_target = 1.50

        # Anti-overtrading parameters
        # V19.1-FIX: Reduced to allow trading
        self.min_hold_steps = 3  # V19.1: Reduced from 10 - allow quicker closes
        self.min_open_cooldown = 3  # V19.1: Reduced from 10 - faster re-entry
        self.hold_penalty_start = 30

        # Overtrading penalty
        self.overtrading_threshold = 20
        self.overtrading_penalty = -10.0

        # V19-FIX: Removed quick_close_penalty (redundant with min_hold_steps)
        # The min_hold_steps check already prevents premature closes

        # Reward parameters
        # V19-FIX: Increased clip bounds to preserve large trade signals
        self.reward_clip_bounds = (-200.0, 200.0)

        # V19-FIX: Balanced rewards - removed conflicting multipliers
        self.long_reward_multiplier = 1.0  # V19: Neutral (was 0.8)
        self.short_reward_multiplier = 1.0  # V19: Neutral (was 1.5)

        self.action_diversity_penalty = 0.001
        self.min_episode_length_for_rewards = 1
        self.short_episode_penalty = 0.0

        # HOLD BONUS
        self.hold_bonus_enabled = True
        self.hold_bonus_profit_threshold = 0.005
        self.hold_bonus_per_step = 0.05
        self.hold_bonus_cap = 1.0

        # FORCED CLOSE PENALTY
        self.max_hold_steps_before_penalty = 20
        self.hold_penalty_per_step = 0.05

        # CLOSE REWARD - V20 IMPROVED
        self.smart_close_enabled = True
        # V20-FIX: Increased profit scale for stronger incentive
        self.profit_close_bonus_scale = 60.0  # V19.3: 50.0 (+20%)
        # V20-FIX: Much higher loss penalty to discourage losing trades
        self.loss_close_penalty_scale = 10.0  # V19.3: 3.0 (+233%)
        self.min_profit_for_bonus = 0.001

        # V20-FIX: Bonus for profitable trades (any profit)
        self.profitable_trade_bonus = 5.0  # NEW: Fixed bonus for any profitable trade

        # V20-FIX: Penalty for consecutive losses (risk management)
        self.consecutive_loss_penalty = 2.0  # NEW: -2.0 per consecutive loss

        # V20-FIX: Bonus for holding profitable positions
        self.hold_profit_bonus_scale = 0.5  # NEW: 0.5x unrealized profit as reward

        # V19-FIX: Disabled quick_close_penalty (redundant)
        self.quick_close_penalty_enabled = False

        # Correct hold reward
        self.correct_hold_reward = 0.02

        # TIME PENALTY
        self.time_penalty_enabled = True
        self.time_penalty_per_step = 0.005
        self.time_penalty_start = 15
        self.time_penalty_cap = 2.0

        # EARLY CLOSE PENALTY - REMOVED
        self.early_close_penalty_enabled = False
        self.early_close_threshold_steps = 5
        self.early_close_penalty = 0.0

        # Profit multiplier
        self.profit_multiplier_enabled = True
        self.profit_multiplier_threshold = 0.008
        self.profit_multiplier_scale = 2.0

        # Trading bonus - REMOVED (model found exploit)
        self.trading_encouragement_enabled = False
        self.trading_bonus_per_trade = 0.0
        self.min_trades_for_bonus = 5

        # Invalid action penalty
        self.invalid_action_penalty = 1.0

        # Action Masking
        self.action_masking_enabled = False

        # Track position open time
        self.position_open_time = 0

        # Strategy balancing parameters
        self.direction_balance_target = 0.5
        self.direction_balance_penalty_weight = 0.5
        self.max_direction_concentration = 0.60
        self.direction_streak_limit = 3
        self.exploration_bonus_weight = 0.25

        # V19-FIX: Neutral bonuses (removed asymmetric rewards)
        self.long_open_bonus = 0.0
        self.short_open_bonus = 0.0

        # HYBRID MODE - V19.2-FIX: DISABLED (causes reward = 0)
        self.hybrid_mode_enabled = False  # V19.2: Use normal reward function
        self.hybrid_take_profit_pct = 0.05
        self.hybrid_stop_loss_pct = 0.03
        self.hybrid_max_hold_steps = 50
        self.hybrid_entry_reward_scale = 5.0
        self.hybrid_allow_manual_close = False
        self.hybrid_min_hold_for_tpsl = 3

        # Track hybrid mode exits
        self.hybrid_exit_reason = None
        self.hybrid_entry_unrealized_pnl = 0.0

        # Action diversity tracking
        self.action_history = []
        self.max_action_history = 10

        # Strategy balancing tracking
        self.long_trades = 0
        self.short_trades = 0
        self.direction_streak = 0
        self.last_direction = None
        self.direction_exploration_bonuses = {'long': 0, 'short': 0}

        # Use rolling window normalization
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

        # V19-FIX: Define observation space (22 features with technical indicators)
        # Features: price_norm, position_norm, pnl_pct, balance_norm, margin_locked_norm,
        # fees_norm, steps_since_trade_norm, steps_episode_norm, long_ratio, short_ratio,
        # direction_streak_norm, rsi_norm, bb_norm, atr_norm, short_trend, medium_trend,
        # mfi_norm, macd_norm, macd_signal_norm, macd_hist_norm, stoch_k_norm, stoch_d_norm
        n_indicators = 11  # rsi, bb, atr, short_trend, medium_trend, mfi, macd, macd_signal, macd_hist, stoch_k, stoch_d
        self.observation_space = spaces.Box(
            low=np.array([-15.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0] * n_indicators, dtype=np.float32),
            high=np.array([15.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * n_indicators, dtype=np.float32),
            dtype=np.float32
        )

    def _calculate_strategy_balance_reward(self, action):
        """Calculate reward for balanced long/short trading"""
        balance_reward = 0.0

        if self.long_trades + self.short_trades < 5:
            return 0.0

        long_ratio = self.long_trades / max(1, self.long_trades + self.short_trades)
        short_ratio = self.short_trades / max(1, self.long_trades + self.short_trades)

        target_ratio = self.direction_balance_target
        imbalance = abs(long_ratio - target_ratio)

        if imbalance > 0.4:
            balance_reward -= self.direction_balance_penalty_weight * 3
        elif imbalance > 0.3:
            balance_reward -= self.direction_balance_penalty_weight * 2
        elif imbalance > 0.2:
            balance_reward -= self.direction_balance_penalty_weight

        if self.direction_streak > self.direction_streak_limit:
            streak_penalty = (self.direction_streak - self.direction_streak_limit) * 0.05
            balance_reward -= streak_penalty

        if long_ratio > self.max_direction_concentration or short_ratio > self.max_direction_concentration:
            balance_reward -= 0.2

        if action == 1:
            balance_reward += self.direction_exploration_bonuses.get('long', 0)
        elif action == 3:
            balance_reward += self.direction_exploration_bonuses.get('short', 0)

        return balance_reward

    def _calculate_reward(self, current_price, action, action_performed=False, pnl_pct=None):
        """
        V19 SIMPLIFIED REWARD SYSTEM
        Fixes applied:
          ✅ Removed hold_quality_bonus (conflicted with time_penalty)
          ✅ Removed quick_close_penalty (redundant with min_hold_steps)
          ✅ Removed short_bonus (already have short_reward_multiplier)
          ✅ Neutral long/short reward multipliers (both 1.0)
        """
        reward = 0.0

        # ========================================
        # HYBRID MODE: Check if position was closed by TP/SL
        # ========================================
        if self.hybrid_mode_enabled and self.hybrid_exit_reason is not None:
            pnl_pct = self.hybrid_entry_unrealized_pnl

            if self.hybrid_exit_reason == 'tp':
                reward += 10.0
                reward += pnl_pct * 100 * 2.0
                if self.debug:
                    print(f"HYBRID TP: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")

            elif self.hybrid_exit_reason == 'sl':
                reward += -2.0
                if self.debug:
                    print(f"HYBRID SL: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")

            elif self.hybrid_exit_reason == 'time':
                reward += pnl_pct * 100
                if self.debug:
                    print(f"HYBRID TIME: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")

            self.hybrid_exit_reason = None
            self.hybrid_entry_unrealized_pnl = 0.0
            return reward

        # ========================================
        # NON-HYBRID MODE: Simplified reward logic
        # ========================================
        if not self.hybrid_mode_enabled:
            if not hasattr(self, 'highest_unrealized_pnl'):
                self.highest_unrealized_pnl = 0.0

            # HOLD BONUS + CORRECT HOLD REWARD - V20 IMPROVED
            if self.position != 0 and self.entry_price > 0:
                if self.position > 0:
                    unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

                if unrealized_pnl_pct > self.highest_unrealized_pnl:
                    self.highest_unrealized_pnl = unrealized_pnl_pct

                if self.correct_hold_reward > 0 and unrealized_pnl_pct > 0:
                    reward += self.correct_hold_reward * unrealized_pnl_pct

                # V20-FIX: Bonus for holding profitable positions (any profit)
                if unrealized_pnl_pct > 0:
                    hold_profit_bonus = unrealized_pnl_pct * 100 * self.hold_profit_bonus_scale
                    reward += hold_profit_bonus

                if self.hold_bonus_enabled and unrealized_pnl_pct > self.hold_bonus_profit_threshold:
                    if not hasattr(self, 'accumulated_hold_bonus'):
                        self.accumulated_hold_bonus = 0.0

                    step_bonus = min(self.hold_bonus_per_step,
                                    self.hold_bonus_cap - self.accumulated_hold_bonus)
                    if step_bonus > 0:
                        self.accumulated_hold_bonus += step_bonus
                        reward += step_bonus

            # CLOSE REWARD - V19 SIMPLIFIED
            if self.smart_close_enabled and action in [2, 4] and action_performed and pnl_pct is not None:
                # V19-FIX: Removed close_action_bonus (was causing overtrading)
                # V19-FIX: Removed quick_close_penalty (redundant with min_hold_steps)
                # V19-FIX: Removed hold_quality_bonus (conflicted with time_penalty)
                # V19-FIX: Removed short_bonus (already have short_reward_multiplier)

                if pnl_pct > self.min_profit_for_bonus:
                    profit_bonus = pnl_pct * 100 * self.profit_close_bonus_scale
                    reward += profit_bonus

                    # V19-FIX: Removed hold_quality_bonus
                    # V19-FIX: Removed short_bonus

                    if self.profit_multiplier_enabled and pnl_pct >= self.profit_multiplier_threshold:
                        multiplier_bonus = profit_bonus * (self.profit_multiplier_scale - 1)
                        reward += multiplier_bonus

                    # V19-FIX: Neutral multipliers (both 1.0)
                    if self.position > 0:
                        reward += pnl_pct * 100.0 * self.long_reward_multiplier
                    else:
                        reward += pnl_pct * 100.0 * self.short_reward_multiplier

                elif pnl_pct > 0:
                    # V20-FIX: Small profit - still reward with bonus
                    reward += pnl_pct * 100.0
                    # V20-FIX: Bonus for any profitable trade
                    reward += self.profitable_trade_bonus
                else:
                    # V20-FIX: Much higher penalty for losses
                    loss_penalty = pnl_pct * 100.0 * self.loss_close_penalty_scale
                    reward += loss_penalty
                    # V20-FIX: Extra penalty for consecutive losses
                    if hasattr(self, 'consecutive_losses') and self.consecutive_losses > 0:
                        streak_penalty = self.consecutive_losses * self.consecutive_loss_penalty
                        reward -= streak_penalty

                if hasattr(self, 'accumulated_hold_bonus'):
                    self.accumulated_hold_bonus = 0.0

            # OPENING PENALTY - V20-FIX: Further reduced
            if action in [1, 3] and action_performed:
                # V20-FIX: Reduced from -0.1 to -0.05 - minimal friction
                open_penalty = -0.05
                reward += open_penalty

                # Penalty for excessive trading (overtrading prevention)
                if hasattr(self, 'total_trades') and self.total_trades > self.overtrading_threshold:
                    reward += -2.0

                # V20-FIX: Neutral rewards (no asymmetric bonuses)

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
            if action in [1, 3] and action_performed:
                reward += 2.0

            if action == 1 and not action_performed:
                if self.position > 0:
                    reward += -0.5
                elif self.position < 0:
                    reward += -0.3

            if action == 3 and not action_performed:
                if self.position < 0:
                    reward += -0.5
                elif self.position > 0:
                    reward += -0.3

            if action == 0 and self.position == 0:
                reward += -0.05

        self.prev_price = current_price

        # V19-FIX: Increased clip bounds to preserve large trade signals
        reward = np.clip(reward, self.reward_clip_bounds[0], self.reward_clip_bounds[1])

        return reward

    # ========================================================================
    # V19 CRITICAL FIX: PnL EXECUTION METHODS
    # ========================================================================

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
            # V19-FIX: PnL% calculated relative to entry value (correct for LONG)
            pnl_pct = pnl / (position_size * self.entry_price) if self.entry_price > 0 else 0

            self.balance += revenue - fee
            self.total_fees += fee
            self.fees_step += fee

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

        V19-FIX: Corrected logic for opening short positions
        """
        # If we have a long position, refuse to open short
        if self.position > 0:
            return False

        # If we already have a short position, refuse to pyramid
        if self.position < 0:
            return False

        # Cooldown: enforce minimum pause between trades
        if self.current_step - self.last_close_step < self.min_open_cooldown:
            return False

        short_amount = min(max_order_size_short, self.balance)

        if short_amount >= min_order_size and current_price > 0:
            margin_required = short_amount * self.margin_requirement
            available_balance = self.balance - self.margin_locked

            if available_balance >= margin_required:
                coins_short = short_amount / current_price
                fee = coins_short * current_price * self.transaction_fee

                self.short_opening_fees = fee
                self.margin_locked += short_amount + margin_required
                self.balance -= margin_required + fee
                self.position -= coins_short
                self.total_fees += fee
                self.fees_step += fee

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
        """Execute cover short action

        V19-FIX: CORRECTED PnL% calculation for SHORT positions
        PnL% is now calculated relative to margin (not sale_proceeds)
        """
        if self.position < 0:
            # Minimum hold check
            steps_held = self.current_step - self.entry_step
            if steps_held < self.min_hold_steps:
                return False, 0

            position_size = abs(self.position)

            # Calculate PnL
            sale_proceeds = position_size * self.entry_price
            buy_cost = position_size * current_price
            close_fee = buy_cost * self.transaction_fee

            price_pnl = sale_proceeds - buy_cost
            pnl = price_pnl - close_fee

            # V19-FIX: PnL% relative to margin (not sale_proceeds)
            # This correctly reflects leveraged returns
            margin_used = sale_proceeds * self.margin_requirement
            pnl_pct = pnl / margin_used if margin_used > 0 else 0

            # Our margin was: sale_proceeds * margin_requirement
            our_margin = sale_proceeds * self.margin_requirement

            # Return our margin + PnL to balance
            self.balance += our_margin + pnl
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
            self.entry_step = 0
            self.highest_price_since_entry = 0
            self.lowest_price_since_entry = float('inf')
            self.trailing_stop_loss = 0
            self.trailing_take_profit = 0

            self.steps_since_last_trade = 0

            return True, pnl_pct
        return False, 0

    def _execute_buy_long(self, current_price, max_order_size_long, min_order_size):
        """Execute buy long action"""
        if self.position < 0:
            return False

        if self.position > 0:
            return False

        if self.current_step - self.last_close_step < self.min_open_cooldown:
            return False

        invest_amount = min(max_order_size_long, self.balance)

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

            if self.entry_price == 0:
                self.entry_price = current_price
                self.entry_step = self.current_step
                self.highest_price_since_entry = current_price
                self.lowest_price_since_entry = current_price
                self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)
            else:
                old_position = self.position - coins_bought
                # V19-FIX: Include fees in weighted average price calculation
                total_cost = (old_position * self.entry_price) + (coins_bought * current_price)
                total_fees_paid = coins_bought * current_price * self.transaction_fee
                self.entry_price = (total_cost + total_fees_paid) / self.position

            self.total_trades += 1
            self.steps_since_last_trade = 0

            return True
        return False

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed, options=options)

        self.current_step = self.start_step if self.start_step is not None else 0
        self.episode_start_step = self.current_step
        self.steps_in_episode = 0

        self.balance = self.initial_balance
        self.margin_locked = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.highest_price_since_entry = 0
        self.lowest_price_since_entry = float('inf')
        self.trailing_stop_loss = 0
        self.trailing_take_profit = 0

        self.total_fees = 0.0
        self.fees_step = 0.0
        self.short_opening_fees = 0.0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_opens = 0
        self.total_closes = 0

        self.long_trades = 0
        self.short_trades = 0
        self.direction_streak = 0
        self.last_direction = None
        self.direction_exploration_bonuses = {'long': 0, 'short': 0}

        self.action_history = []
        self.portfolio_history = [self.initial_balance]
        self.portfolio_values = [self.initial_balance]

        self.prev_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        self.prev_portfolio_value = self.initial_balance

        self.steps_since_last_trade = 0
        # V19-FIX: Allow trading from step 0 by setting last_close_step in the past
        self.last_close_step = -self.min_open_cooldown
        self.last_trade_hold_steps = 0

        self.hybrid_exit_reason = None
        self.hybrid_entry_unrealized_pnl = 0.0

        self.consecutive_losses = 0
        self.highest_unrealized_pnl = 0.0
        self.accumulated_hold_bonus = 0.0

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation - V19 with technical indicators"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]
        current_price = row.get('close', row.get('Close', 50000))
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            current_price = 50000

        # Technical indicators with validation
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

        macd = row.get('MACD_default_macd', 0)
        if np.isnan(macd) or np.isinf(macd):
            macd = 0

        macd_signal = row.get('MACD_default_signal', 0)
        if np.isnan(macd_signal) or np.isinf(macd_signal):
            macd_signal = 0

        macd_hist = row.get('MACD_default_histogram', 0)
        if np.isnan(macd_hist) or np.isinf(macd_hist):
            macd_hist = 0

        stoch_k = row.get('Stochastic_slowk', 50)
        if np.isnan(stoch_k) or np.isinf(stoch_k) or stoch_k < 0 or stoch_k > 100:
            stoch_k = 50

        stoch_d = row.get('Stochastic_slowd', 50)
        if np.isnan(stoch_d) or np.isinf(stoch_d) or stoch_d < 0 or stoch_d > 100:
            stoch_d = 50

        # Normalize values
        balance_norm = self.balance / self.initial_balance - 1
        position_value = abs(self.position) * current_price
        position_norm = position_value / self.initial_balance if self.initial_balance > 0 else 0.0
        if self.position < 0:
            position_norm = -position_norm
        position_norm = np.clip(position_norm, -2.0, 2.0)

        # Price normalization
        price_norm = 0
        if self.current_step < len(self.price_rolling_mean) and self.current_step < len(self.price_rolling_std):
            if self.price_rolling_std[self.current_step] > 0:
                price_norm = (current_price - self.price_rolling_mean[self.current_step]) / self.price_rolling_std[self.current_step]
                price_norm = np.clip(price_norm, -10, 10)

        # PnL percentage
        pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Time features
        steps_since_last_trade_norm = min(self.steps_since_last_trade / 100.0, 1.0)
        steps_in_episode_norm = self.steps_in_episode / self.episode_length

        # Strategy balance
        total_trades = max(1, self.long_trades + self.short_trades)
        long_ratio = self.long_trades / total_trades
        short_ratio = self.short_trades / total_trades
        direction_streak_norm = min(self.direction_streak / 10.0, 1.0)

        # Financial features
        margin_locked_norm = self.margin_locked / self.initial_balance
        fees_norm = self.total_fees / self.initial_balance

        # Indicator normalization
        rsi_norm = (rsi - 50) / 50
        bb_norm = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-8)
        bb_norm = np.clip(bb_norm, 0, 1)
        atr_norm = min(atr / current_price, 0.1)
        mfi_norm = (mfi - 50) / 50
        macd_norm = macd / current_price
        macd_signal_norm = macd_signal / current_price
        macd_hist_norm = macd_hist / current_price
        stoch_k_norm = (stoch_k - 50) / 50
        stoch_d_norm = (stoch_d - 50) / 50

        # Short and medium trend
        short_trend = row.get('short_trend', 0)
        medium_trend = row.get('medium_trend', 0)

        state = np.array([
            price_norm,
            position_norm,
            pnl_pct,
            balance_norm,
            margin_locked_norm,
            fees_norm,
            steps_since_last_trade_norm,
            steps_in_episode_norm,
            long_ratio,
            short_ratio,
            direction_streak_norm,
            rsi_norm,
            bb_norm,
            atr_norm,
            short_trend,
            medium_trend,
            mfi_norm,
            macd_norm,
            macd_signal_norm,
            macd_hist_norm,
            stoch_k_norm,
            stoch_d_norm,
        ], dtype=np.float32)

        return state

    def _update_strategy_balance_tracking(self, action):
        """Update strategy balance tracking after successful trade"""
        if action == 1:  # Long
            self.long_trades += 1
            if self.last_direction == 'short':
                self.direction_streak = 1
            elif self.last_direction == 'long':
                self.direction_streak += 1
            self.last_direction = 'long'
        elif action == 3:  # Short
            self.short_trades += 1
            if self.last_direction == 'long':
                self.direction_streak = 1
            elif self.last_direction == 'short':
                self.direction_streak += 1
            self.last_direction = 'short'

        # Update exploration bonuses
        total = self.long_trades + self.short_trades
        if total > 0:
            long_ratio = self.long_trades / total
            short_ratio = self.short_trades / total

            target = self.direction_balance_target
            if long_ratio < target:
                self.direction_exploration_bonuses['long'] = self.exploration_bonus_weight * (target - long_ratio)
            else:
                self.direction_exploration_bonuses['long'] = 0

            if short_ratio < (1 - target):
                self.direction_exploration_bonuses['short'] = self.exploration_bonus_weight * ((1 - target) - short_ratio)
            else:
                self.direction_exploration_bonuses['short'] = 0

    def _check_hybrid_tpsl(self, current_price):
        """Check and trigger TP/SL for hybrid mode"""
        if not self.hybrid_mode_enabled:
            return False, None, None

        if self.position == 0 or self.entry_price == 0:
            return False, None, None

        steps_held = self.current_step - self.entry_step
        if steps_held < self.hybrid_min_hold_for_tpsl:
            return False, None, None

        if self.position > 0:
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Check Take Profit
        if unrealized_pnl_pct >= self.hybrid_take_profit_pct:
            exit_reason = 'tp'
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

        # Check Stop Loss
        if unrealized_pnl_pct <= -self.hybrid_stop_loss_pct:
            exit_reason = 'sl'
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

        # Check max hold time
        if steps_held >= self.hybrid_max_hold_steps:
            exit_reason = 'time'
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
        """Calculate reward for hybrid mode"""
        if not self.hybrid_mode_enabled:
            return 0.0

        reward = 0.0

        if self.hybrid_exit_reason is not None:
            pnl_pct = self.hybrid_entry_unrealized_pnl

            if self.hybrid_exit_reason == 'tp':
                reward = 10.0 + pnl_pct * 100 * 2.0
            elif self.hybrid_exit_reason == 'sl':
                reward = -2.0
            elif self.hybrid_exit_reason == 'time':
                reward = pnl_pct * 100

            self.hybrid_exit_reason = None
            self.hybrid_entry_unrealized_pnl = 0.0

        return reward

    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 2:
            return 0.0

        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio * np.sqrt(252)

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0

        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    def _calculate_volatility_based_position_size(self, current_price):
        """Calculate position size based on volatility"""
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
        return np.clip(dynamic_size, self.min_position_size, self.max_dynamic_position_size)

    def _get_position_sizes(self):
        """Calculate position sizes"""
        dynamic_position_size = self._calculate_volatility_based_position_size(
            self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        )

        max_order_size_long = min(
            self.balance * dynamic_position_size,
            self.balance * self.max_position_size,
            self.initial_balance * self.max_total_exposure
        )

        max_order_size_short = min(
            self.balance * dynamic_position_size * 0.8,
            self.balance * self.max_position_size,
            self.initial_balance * self.max_total_exposure
        )

        min_order_size = 5
        return max_order_size_long, max_order_size_short, min_order_size

    def step(self, action):
        """Execute one step in the environment"""
        try:
            if self.current_step >= len(self.df):
                terminated = True
                truncated = False
                current_price = self.df.iloc[-1].get('close', self.df.iloc[-1].get('Close'))
                final_portfolio = self.balance + self.margin_locked + self.position * current_price
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
                final_reward = portfolio_return * 10.0
                self.portfolio_values.append(final_portfolio)

                return self._get_state(), final_reward, terminated, truncated, {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_trades': self.total_trades,
                    'long_trades': self.long_trades,
                    'short_trades': self.short_trades,
                }

            current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
            max_order_size_long, max_order_size_short, min_order_size = self._get_position_sizes()

            action_performed = False
            pnl_pct = None

            if action == 0:
                self.steps_since_last_trade += 1
            elif action == 1:
                action_performed = self._execute_buy_long(current_price, max_order_size_long, min_order_size)
                if action_performed:
                    self._update_strategy_balance_tracking(1)
                    self.total_opens += 1
            elif action == 2:
                action_performed, pnl_pct = self._execute_sell_long(current_price)
                if action_performed:
                    self.total_closes += 1
            elif action == 3:
                action_performed = self._execute_sell_short(current_price, max_order_size_short, min_order_size)
                if action_performed:
                    self._update_strategy_balance_tracking(3)
                    self.total_opens += 1
            elif action == 4:
                action_performed, pnl_pct = self._execute_cover_short(current_price)
                if action_performed:
                    self.total_closes += 1

            current_portfolio = self.balance + self.margin_locked + self.position * current_price

            if self.hybrid_mode_enabled:
                tp_triggered, exit_reason, exit_pnl = self._check_hybrid_tpsl(current_price)
                if tp_triggered:
                    reward = self._calculate_hybrid_reward(current_price, action)
                    self.total_trades += 1
                else:
                    reward = self._calculate_reward(current_price, action, action_performed, pnl_pct)
            else:
                reward = self._calculate_reward(current_price, action, action_performed, pnl_pct)

            self.fees_step = 0.0
            self.prev_portfolio_value = current_portfolio

            self.current_step += 1
            self.steps_in_episode += 1
            self.portfolio_history.append(current_portfolio)

            terminated = False
            truncated = False

            portfolio_return = (current_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0

            if portfolio_return <= -self.max_episode_loss_pct:
                terminated = True

            if not terminated and self.steps_in_episode >= self.episode_length:
                truncated = True
                terminated = True

            if not terminated and self.current_step >= len(self.df):
                terminated = True

            if terminated:
                final_portfolio = current_portfolio
                portfolio_return = (final_portfolio - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
                final_reward = portfolio_return * 10.0

                if self.position != 0:
                    steps_open = self.steps_in_episode - max(0, self.entry_step - self.episode_start_step)
                    unclosed_penalty = -(10.0 + 0.1 * steps_open)
                    final_reward += unclosed_penalty

                final_reward = np.clip(final_reward, self.reward_clip_bounds[0], self.reward_clip_bounds[1])

                return self._get_state(), final_reward, terminated, truncated, {
                    'portfolio_value': final_portfolio,
                    'total_return': portfolio_return,
                    'total_trades': self.total_trades,
                    'long_trades': self.long_trades,
                    'short_trades': self.short_trades,
                }

            info = {
                'portfolio_value': current_portfolio,
                'total_trades': self.total_trades,
                'action_performed': action_performed,
                'pnl_pct': pnl_pct if pnl_pct is not None else 0,
                'long_trades': self.long_trades,
                'short_trades': self.short_trades,
            }

            return self._get_state(), reward, terminated, truncated, info

        except Exception as e:
            if self.debug:
                print(f"ERROR in step(): {e}")
            return self._get_state(), 0, True, False, {}
