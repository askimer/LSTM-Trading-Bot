#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent Training
Uses PPO algorithm to learn optimal trading strategy
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

class RLTrainingProgressCallback(BaseCallback):
    """Custom callback to show detailed RL training progress"""

    def __init__(self, total_timesteps, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.start_time = None
        self.last_eval_time = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        """Called at the beginning of training"""
        self.start_time = time.time()
        print("🚀 Starting RL Training")
        print("=" * 60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Evaluation frequency: {self.eval_freq:,} timesteps")
        print()

    def _on_step(self):
        """Called at each step"""
        # Calculate progress
        current_timestep = self.num_timesteps
        progress = current_timestep / self.total_timesteps

        # Show progress every 1000 timesteps or at key milestones
        if current_timestep % 1000 == 0 or progress >= 1.0:
            elapsed = time.time() - self.start_time
            timesteps_per_sec = current_timestep / elapsed if elapsed > 0 else 0
            eta_seconds = (self.total_timesteps - current_timestep) / timesteps_per_sec if timesteps_per_sec > 0 else 0

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.1f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}min"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            # Progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            print(f"📈 Timestep {current_timestep:6,}/{self.total_timesteps:6,} | [{bar}] {progress*100:5.1f}% | TPS: {timesteps_per_sec:6.0f} | ETA: {eta_str}")

        # Show evaluation results
        if self.eval_freq > 0 and current_timestep % self.eval_freq == 0 and current_timestep > 0:
            if hasattr(self, 'last_mean_reward'):
                eval_elapsed = time.time() - (self.last_eval_time or self.start_time)
                print(f"🎯 Evaluation at {current_timestep:,} timesteps:")
                print(f"   Mean Reward: {self.last_mean_reward:.2f}")
                print(f"   Mean Episode Length: {self.last_mean_length:.1f}")
                print(f"   Evaluation Time: {eval_elapsed:.1f}s")
                print()

        return True

    def _on_training_end(self):
        """Called at the end of training"""
        total_time = time.time() - self.start_time
        print()
        print("✅ RL Training Completed!")
        print("=" * 60)
        print(f"Total training time: {total_time:.1f}s")
        print(f"Average timesteps/second: {self.total_timesteps/total_time:.1f}")
        print()

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agent
    """
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.current_step = 0

        # Margin trading parameters
        self.margin_requirement = 0.5  # 50% initial margin for shorts
        self.maintenance_margin = 0.25  # 25% maintenance margin
        self.liquidation_penalty = 0.02  # 2% penalty on liquidation

        # Trailing stop-loss parameters
        self.trailing_stop_distance = 0.03  # 3% trailing distance
        self.trailing_stop_enabled = True

        # Calculate dynamic price normalization based on data
        price_col = 'close' if 'close' in df.columns else 'Close'
        self.price_mean = df[price_col].mean()
        self.price_std = df[price_col].std()
        print(f"Price normalization: mean={self.price_mean:.2f}, std={self.price_std:.2f}")
        print(f"Margin requirements: initial={self.margin_requirement*100:.0f}%, maintenance={self.maintenance_margin*100:.0f}%")
        print(f"Trailing stop-loss: {self.trailing_stop_distance*100:.1f}% distance")

        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Buy Short
        self.action_space = spaces.Discrete(5)

        # State: [balance_norm, position_norm, price_norm, indicators...]
        n_indicators = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 + n_indicators,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, positive=long, negative=short
        self.total_fees = 0
        self.portfolio_values = [self.initial_balance]

        # Trailing stop-loss tracking
        self.entry_price = 0
        self.highest_price_since_entry = 0
        self.lowest_price_since_entry = float('inf')
        self.trailing_stop_loss = 0
        self.trailing_take_profit = 0

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]

        # Normalize values
        balance_norm = self.balance / self.initial_balance - 1
        # Normalize position: positive for long, negative for short
        # Use abs() to handle both long and short positions
        if self.balance > 0:
            position_norm = self.position / (self.balance * 0.1)
        else:
            position_norm = 0
        current_price = row.get('close', row.get('Close', self.price_mean))  # Support both naming conventions
        price_norm = (current_price - self.price_mean) / self.price_std  # Dynamic z-score normalization

        # Technical indicators (removed MACD as it's not calculated in feature_engineer.py)
        indicators = [
            row.get('RSI_15', 50) / 100 - 0.5,  # RSI normalized to [-0.5, 0.5]
            (row.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,  # BB upper as % deviation
            (row.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,  # BB lower as % deviation
            row.get('ATR_15', 100) / 1000,  # ATR normalized
            row.get('OBV', 0) / 1e10,  # OBV normalized
            row.get('AD', 0) / 1e10,  # AD normalized
            row.get('MFI_15', 50) / 100 - 0.5  # MFI normalized to [-0.5, 0.5]
        ]

        state = np.array([balance_norm, position_norm, price_norm] + indicators, dtype=np.float32)
        return state

    def step(self, action):
        """Execute one step in environment"""
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            reward = 0
            return self._get_state(), reward, terminated, truncated, {}

        # Support both 'close' and 'Close' column names
        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        next_price = self.df.iloc[self.current_step + 1].get('close', self.df.iloc[self.current_step + 1].get('Close'))

        reward = 0
        terminated = False
        truncated = False

        # Check for margin call / liquidation before executing action
        current_portfolio = self.balance + self.position * current_price
        
        # Only check margin for short positions (long positions don't use margin in this model)
        if self.position < 0:  # Short position
            position_value = abs(self.position) * current_price
            margin_used = position_value * self.margin_requirement
            
            # Check maintenance margin: equity must be >= position_value * maintenance_margin
            # Equity = portfolio value - margin_used (simplified)
            # For shorts: if portfolio drops too much, we need to liquidate
            required_equity = position_value * self.maintenance_margin
            
            if current_portfolio < required_equity:
                # Liquidation triggered for short position
                print(f"💥 LIQUIDATION: Portfolio ${current_portfolio:.2f} < Required ${required_equity:.2f}")

                # Force close short position with penalty
                cost = abs(self.position) * current_price * (1 + self.liquidation_penalty)
                # If balance insufficient, use all available balance (partial liquidation)
                if self.balance >= cost:
                    self.balance -= cost + (cost * self.transaction_fee)
                    self.total_fees += cost * self.transaction_fee
                    self.position = 0
                else:
                    # Partial liquidation: close as much as possible
                    max_cover = self.balance / (current_price * (1 + self.liquidation_penalty + self.transaction_fee))
                    if max_cover > 0:
                        cost_partial = max_cover * current_price * (1 + self.liquidation_penalty)
                        self.balance -= cost_partial + (cost_partial * self.transaction_fee)
                        self.total_fees += cost_partial * self.transaction_fee
                        self.position += max_cover  # Reduce short position
                        print(f"⚠️ Partial liquidation: closed {max_cover:.6f} of {abs(self.position):.6f} short position")

                reward -= 50  # Heavy penalty for liquidation
                print(f"🔥 Position liquidated with penalty")
        
        # For long positions, check if portfolio value drops too low (simple stop-loss)
        elif self.position > 0:
            if current_portfolio < self.initial_balance * 0.3:  # 70% loss
                # Force close long position (stop-loss)
                revenue = self.position * current_price * (1 - self.liquidation_penalty)
                self.balance += revenue - (revenue * self.transaction_fee)
                self.total_fees += revenue * self.transaction_fee
                self.position = 0
                reward -= 30  # Penalty for stop-loss
                print(f"🛑 Stop-loss triggered: Long position closed")

        # Update trailing stops for existing positions
        if self.trailing_stop_enabled and self.position != 0:
            if self.position > 0:  # Long position
                # Update highest price since entry
                if current_price > self.highest_price_since_entry:
                    self.highest_price_since_entry = current_price
                    # Update trailing stop-loss (3% below highest price)
                    self.trailing_stop_loss = self.highest_price_since_entry * (1 - self.trailing_stop_distance)

                # Check trailing stop-loss
                if current_price <= self.trailing_stop_loss:
                    # Trailing stop triggered - close long position
                    revenue = self.position * current_price
                    fee = revenue * self.transaction_fee
                    self.balance += revenue - fee
                    self.total_fees += fee

                    pnl = (current_price - self.entry_price) * self.position - fee
                    print(f"🎯 Trailing stop triggered: Long closed at ${current_price:.2f}, PnL: ${pnl:.2f}")

                    self.position = 0
                    self.entry_price = 0
                    self.highest_price_since_entry = 0
                    self.trailing_stop_loss = 0

                    reward += 2  # Small bonus for disciplined exit
                    # Skip normal action processing
                    self.portfolio_values.append(self.balance + self.position * current_price)
                    self.current_step += 1
                    return self._get_state(), reward, terminated, truncated, {}

            elif self.position < 0:  # Short position
                # Update lowest price since entry
                if current_price < self.lowest_price_since_entry:
                    self.lowest_price_since_entry = current_price
                    # Update trailing stop-loss (3% above lowest price for shorts)
                    self.trailing_stop_loss = self.lowest_price_since_entry * (1 + self.trailing_stop_distance)

                # Check trailing stop-loss for shorts
                if current_price >= self.trailing_stop_loss:
                    # Trailing stop triggered - close short position
                    cost = abs(self.position) * current_price
                    fee = cost * self.transaction_fee
                    self.balance -= cost + fee
                    self.total_fees += fee

                    pnl = (self.entry_price - current_price) * abs(self.position) - fee
                    print(f"🎯 Trailing stop triggered: Short closed at ${current_price:.2f}, PnL: ${pnl:.2f}")

                    self.position = 0
                    self.entry_price = 0
                    self.lowest_price_since_entry = float('inf')
                    self.trailing_stop_loss = 0

                    reward += 2  # Small bonus for disciplined exit
                    # Skip normal action processing
                    self.portfolio_values.append(self.balance + self.position * current_price)
                    self.current_step += 1
                    return self._get_state(), reward, terminated, truncated, {}

        # Execute action
        if action == 1:  # Buy Long - открыть/добавить длинную позицию
            if self.position < 0:
                # Convert short to long: close short first, then open long
                cover_amount = abs(self.position)
                cost = cover_amount * current_price
                fee = cost * self.transaction_fee

                # Check if we can afford to close short AND open long
                if self.balance >= cost + fee:
                    # Estimate balance after closing short
                    estimated_balance = self.balance - cost - fee
                    # Check if we can open long position after closing short
                    min_investment = current_price * (1 + self.transaction_fee) + 10

                    if estimated_balance >= min_investment:
                        # Close short - calculate PnL for conversion bonus
                        pnl_short = (self.entry_price - current_price) * cover_amount - fee
                        self.position = 0
                        self.balance -= cost + fee
                        self.total_fees += fee

                        # Reset trailing stops
                        self.entry_price = 0
                        self.lowest_price_since_entry = float('inf')
                        self.trailing_stop_loss = 0

                        # Now open long position
                        invest_amount = min(self.balance * 0.1, self.balance - 100)
                        if invest_amount > 10:
                            fee_long = invest_amount * self.transaction_fee
                            coins_bought = (invest_amount - fee_long) / current_price

                            self.position += coins_bought
                            self.balance -= invest_amount
                            self.total_fees += fee_long

                            # Initialize trailing stops for new long position
                            self.entry_price = current_price
                            self.highest_price_since_entry = current_price
                            self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)

                            # Bonus for successful conversion
                            conversion_bonus = 5 + (pnl_short * 0.1 if pnl_short > 0 else 0)
                            reward += conversion_bonus
                            print(f"🔄💰 Converted short to long: PnL ${pnl_short:.2f}, Bonus +{conversion_bonus:.1f}")
                        else:
                            # Short closed but can't open long - bonus for closing profitable short
                            if pnl_short > 0:
                                reward += 3
                                print(f"🔄✅ Closed profitable short: PnL ${pnl_short:.2f}")
                            else:
                                reward -= 0.5  # Smaller penalty for partial conversion
                    else:
                        # Can't afford conversion - don't close short
                        reward -= 1  # Penalty for failed conversion
                else:
                    reward -= 1  # Penalty for failed conversion

            elif self.position >= 0:
                # Normal long buy
                if self.balance > current_price * (1 + self.transaction_fee):
                    invest_amount = min(self.balance * 0.1, self.balance - 100)
                    fee = invest_amount * self.transaction_fee
                    coins_bought = (invest_amount - fee) / current_price

                    # Initialize trailing stops for new position
                    if self.position == 0:  # First long position
                        self.entry_price = current_price
                        self.highest_price_since_entry = current_price
                        self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)

                    self.position += coins_bought
                    self.balance -= invest_amount
                    self.total_fees += fee

                    reward -= 0.01

        elif action == 2:  # Sell Long - закрыть часть длинной позиции
            if self.position > 0:
                sell_amount = min(self.position * 0.5, self.position)
                revenue = sell_amount * current_price
                fee = revenue * self.transaction_fee

                self.position -= sell_amount
                self.balance += revenue - fee
                self.total_fees += fee

                reward -= 0.01

        elif action == 3:  # Sell Short - открыть короткую позицию
            if self.position > 0:
                # Convert long to short: close long first, then open short
                sell_amount = self.position
                revenue = sell_amount * current_price
                fee = revenue * self.transaction_fee

                # Calculate PnL for conversion bonus
                pnl_long = (current_price - self.entry_price) * sell_amount - fee

                self.position = 0  # Close long
                self.balance += revenue - fee
                self.total_fees += fee

                # Reset trailing stops
                self.entry_price = 0
                self.highest_price_since_entry = 0
                self.trailing_stop_loss = 0

                # Now open short position
                short_value = min(self.balance * 0.1, self.balance - 100)
                margin_required = short_value * self.margin_requirement
                fee_short = short_value * self.transaction_fee

                if self.balance > margin_required + fee_short:
                    coins_short = (short_value - fee_short) / current_price
                    self.position -= coins_short
                    self.balance += short_value - fee_short
                    self.total_fees += fee_short

                    # Initialize trailing stops for new short position
                    self.entry_price = current_price
                    self.lowest_price_since_entry = current_price
                    self.trailing_stop_loss = current_price * (1 + self.trailing_stop_distance)

                    # Bonus for successful conversion
                    conversion_bonus = 5 + (pnl_long * 0.1 if pnl_long > 0 else 0)
                    reward += conversion_bonus
                    print(f"🔄💰 Converted long to short: PnL ${pnl_long:.2f}, Bonus +{conversion_bonus:.1f}")
                else:
                    # Long closed but can't open short - bonus for closing profitable long
                    if pnl_long > 0:
                        reward += 3
                        print(f"🔄✅ Closed profitable long: PnL ${pnl_long:.2f}")
                    else:
                        reward -= 0.5  # Smaller penalty for partial conversion

            elif self.position <= 0:
                # Normal short sell
                short_value = min(self.balance * 0.1, self.balance - 100)
                margin_required = short_value * self.margin_requirement
                fee = short_value * self.transaction_fee

                if self.balance > margin_required + fee:
                    coins_short = (short_value - fee) / current_price
                    self.position -= coins_short
                    self.balance += short_value - fee
                    self.total_fees += fee

                    # Initialize trailing stops for new position
                    if self.position == -coins_short:  # First short position
                        self.entry_price = current_price
                        self.lowest_price_since_entry = current_price
                        self.trailing_stop_loss = current_price * (1 + self.trailing_stop_distance)

                    reward -= 0.01

        elif action == 4:  # Buy Short - закрыть короткую позицию (cover short)
            if self.position < 0:
                cover_amount = min(abs(self.position) * 0.5, abs(self.position))
                cost = cover_amount * current_price
                fee = cost * self.transaction_fee

                if self.balance >= cost + fee:
                    self.position += cover_amount
                    self.balance -= cost + fee
                    self.total_fees += fee

                    reward -= 0.01

        # Calculate reward based on portfolio change with risk adjustment
        current_portfolio = self.balance + self.position * current_price
        next_portfolio = self.balance + self.position * next_price

        portfolio_change = (next_portfolio - current_portfolio) / current_portfolio if current_portfolio > 0 else 0

        # Risk-adjusted reward: portfolio return minus volatility penalty
        # Calculate rolling volatility (simplified Sharpe-like component)
        if len(self.portfolio_values) >= 10:
            recent_portfolio_values = self.portfolio_values[-10:]
            returns = np.diff(recent_portfolio_values) / recent_portfolio_values[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            # Penalize high volatility (risk-adjusted component)
            risk_penalty = volatility * 50  # Scale volatility penalty
        else:
            risk_penalty = 0

        # Main reward: portfolio change scaled appropriately
        # Scale by 10000 to make rewards more meaningful (0.1% change = 10 reward)
        # This is important because minute-level price changes are very small
        reward += (portfolio_change * 10000) - risk_penalty

        # Reward for good trading decisions (updated for short positions)
        price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0

        # Long position rewards
        if action == 1 and price_change_pct > 0:
            reward += abs(price_change_pct) * 1000  # Bonus for correct long buy
        if action == 1 and price_change_pct < 0:
            reward -= abs(price_change_pct) * 500   # Penalty for wrong long buy
        if action == 2 and price_change_pct < 0:
            reward += abs(price_change_pct) * 1000  # Bonus for correct long sell
        if action == 2 and price_change_pct > 0:
            reward -= abs(price_change_pct) * 500   # Penalty for wrong long sell

        # Short position rewards
        if action == 3 and price_change_pct < 0:
            reward += abs(price_change_pct) * 1000  # Bonus for correct short sell
        if action == 3 and price_change_pct > 0:
            reward -= abs(price_change_pct) * 500   # Penalty for wrong short sell
        if action == 4 and price_change_pct > 0:
            reward += abs(price_change_pct) * 1000  # Bonus for correct short cover
        if action == 4 and price_change_pct < 0:
            reward -= abs(price_change_pct) * 500   # Penalty for wrong short cover

        # Penalty for holding too long without action
        if action == 0:
            if self.position != 0:  # Have position - small penalty
                reward -= 0.005
            else:  # No position - larger penalty to encourage action
                reward -= 0.02

        # Large penalty for going negative
        if current_portfolio < self.initial_balance * 0.5:
            reward -= 10

        self.portfolio_values.append(current_portfolio)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            terminated = True

        return self._get_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render environment state"""
        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        portfolio_value = self.balance + self.position * current_price
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.6f}, Portfolio: {portfolio_value:.2f}")

def train_rl_agent(data_path, total_timesteps=100000, eval_freq=10000):
    """
    Train RL agent using PPO
    
    Recommendations for timesteps:
    - Quick testing: 50,000 - 100,000 steps
    - Development: 200,000 - 500,000 steps  
    - Production: 1,000,000 - 5,000,000 steps
    
    Note: More timesteps generally lead to better performance, but diminishing returns
    after 1-2M steps for most trading tasks. The optimal depends on:
    - Complexity of the trading strategy
    - Amount of training data
    - Market conditions variability
    """

    print("Loading training data...")
    df = pd.read_csv(data_path)

    # Validate data
    print(f"Loaded {len(df)} rows of data")
    if df.empty:
        raise ValueError(f"No data found in {data_path}")

    # Check for required columns
    required_columns = ['close', 'ATR_15', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'OBV', 'AD', 'MFI_15']
    missing_columns = [col for col in required_columns if col not in df.columns and f'{col}_15' not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Clean data
    df = df.dropna()
    print(f"After dropping NaN: {len(df)} rows")

    if df.empty:
        raise ValueError("All data was removed after dropping NaN values")

    # Check minimum data requirements for RL training
    if len(df) < 1000:
        print(f"WARNING: Limited data for RL training ({len(df)} rows). Consider using more historical data.")

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} features")

    print("Creating RL environment...")
    env = DummyVecEnv([lambda: TradingEnvironment(df)])

    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./rl_tensorboard/"
    )

    # Evaluation callback - use subset of data for faster evaluation
    # Use last 10% of data for evaluation (much faster than full dataset)
    # This prevents the massive slowdown you experienced (TPS dropped from 180 to 12)
    eval_df = df.tail(max(1000, len(df) // 10)).reset_index(drop=True)
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnvironment(eval_df), "./rl_logs/")])
    
    # Adjust eval frequency based on total timesteps
    # For longer training, evaluate less frequently to avoid slowdowns
    if total_timesteps > 500000:
        actual_eval_freq = max(eval_freq, total_timesteps // 20)  # Max 20 evaluations
    else:
        actual_eval_freq = eval_freq
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./rl_models/",
        log_path="./rl_logs/",
        eval_freq=actual_eval_freq,
        n_eval_episodes=1,  # Only 1 episode for faster evaluation
        deterministic=True,
        render=False
    )

    # Custom progress callback
    progress_callback = RLTrainingProgressCallback(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq
    )

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback]
    )

    print("Saving final model...")
    model.save("ppo_trading_agent")

    return model

def evaluate_agent(model, data_path, n_episodes=5):
    """Evaluate trained agent"""
    print("Evaluating agent...")

    df = pd.read_csv(data_path)
    df = df.dropna()

    env = TradingEnvironment(df)

    episode_rewards = []
    episode_portfolio_values = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        portfolio_history = [env.initial_balance]

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            current_price = df.iloc[min(env.current_step, len(df)-1)].get('close', df.iloc[min(env.current_step, len(df)-1)].get('Close'))
            portfolio_history.append(env.balance + env.position * current_price)

        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(portfolio_history)

        final_portfolio = portfolio_history[-1]
        profit = (final_portfolio - env.initial_balance) / env.initial_balance * 100
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Final Portfolio=${final_portfolio:.2f}, Profit={profit:.2f}%")

    # Plot average portfolio value
    avg_portfolio = np.mean(episode_portfolio_values, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_portfolio)
    plt.title('RL Agent Portfolio Value Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig('rl_portfolio_performance.png')
    plt.show()

    return episode_rewards, episode_portfolio_values

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument("--data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to training data")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps (default: 500000. For production: 1M-5M recommended)")
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate trained agent")

    args = parser.parse_args()

    if args.eval:
        print("Loading trained model...")
        model = PPO.load("ppo_trading_agent")
        evaluate_agent(model, args.data)
    else:
        model = train_rl_agent(args.data, args.timesteps)
        print("Training completed. Use --eval to evaluate the agent.")
