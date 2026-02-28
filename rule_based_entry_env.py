#!/usr/bin/env python3
"""
Rule-Based Entry + RL Exit Trading Environment
Entry decisions based on technical indicators (RSI, MACD)
Exit decisions made by RL model
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import warnings
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore", category=DeprecationWarning)

class RuleBasedEntryEnv(gym.Env):
    """
    Trading environment with:
    - RULE-BASED ENTRY: RSI + MACD signals
    - RL EXIT: Model decides when to close position
    """
    
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018, 
                 episode_length=300, debug=False):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.episode_length = episode_length
        self.debug = debug
        
        # Entry rules parameters - RELAXED for more signals
        self.rsi_oversold = 40  # Buy when RSI < 40 (was 30)
        self.rsi_overbought = 60  # Sell when RSI > 60 (was 70)
        self.macd_threshold = 0  # MACD histogram threshold
        
        # TP/SL for safety
        self.take_profit_pct = 0.05  # 5%
        self.stop_loss_pct = 0.03    # 3%
        self.max_hold_steps = 50     # Force close after 50 steps
        
        # Actions: 0=HOLD, 1=CLOSE_LONG, 2=CLOSE_SHORT
        # Entry is automatic based on rules
        self.action_space = spaces.Discrete(3)
        
        # State: [balance_norm, position_norm, price_norm, unrealized_pnl, indicators...]
        n_indicators = 12
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.5, -15.0, -1.0] + [-1.0] * n_indicators, dtype=np.float32),
            high=np.array([2.0, 2.5, 15.0, 1.0] + [1.0] * n_indicators, dtype=np.float32),
            dtype=np.float32
        )
        
        # Price normalization
        price_col = 'close' if 'close' in df.columns else 'Close'
        window_size = min(100, len(df) // 10)
        self.price_rolling_mean = df[price_col].rolling(window=window_size, min_periods=1).mean().values.copy()
        self.price_rolling_std = df[price_col].rolling(window=window_size, min_periods=1).std().values.copy()
        self.price_rolling_std[self.price_rolling_std == 0] = 1
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Random start
        max_start = max(0, len(self.df) - self.episode_length)
        if max_start > 0 and seed is not None:
            np.random.seed(seed)
            self.episode_start_step = np.random.randint(0, max_start + 1)
        else:
            self.episode_start_step = 0
        
        self.current_step = self.episode_start_step
        self.steps_in_episode = 0
        
        self.balance = float(self.initial_balance)
        self.position = 0.0  # >0 long, <0 short, 0 flat
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_fees = 0.0
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        
        # Track episode metrics
        self.portfolio_values = [float(self.initial_balance)]
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state observation"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)
        
        row = self.df.iloc[self.current_step]
        current_price = row.get('close', row.get('Close', 50000))
        
        # Validate price
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            current_price = 50000
        
        # Balance normalization
        balance_norm = self.balance / self.initial_balance - 1
        
        # Position normalization
        position_value = abs(self.position) * current_price
        position_norm = position_value / self.initial_balance if self.initial_balance > 0 else 0.0
        if self.position < 0:
            position_norm = -position_norm
        position_norm = np.clip(position_norm, -2.0, 2.0)
        
        # Price normalization
        price_norm = 0
        if self.current_step < len(self.price_rolling_mean):
            if self.price_rolling_std[self.current_step] > 0:
                price_norm = (current_price - self.price_rolling_mean[self.current_step]) / self.price_rolling_std[self.current_step]
                price_norm = np.clip(price_norm, -10, 10)
        
        # Unrealized PnL
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = np.clip(unrealized_pnl, -0.5, 0.5) * 2
        
        # Technical indicators
        rsi = row.get('RSI_15', 50)
        rsi = np.clip(rsi / 100 - 0.5, -0.5, 0.5)
        
        bb_upper = row.get('BB_15_upper', current_price)
        bb_lower = row.get('BB_15_lower', current_price)
        
        atr = row.get('ATR_15', 100)
        mfi = row.get('MFI_15', 50)
        
        macd = row.get('MACD_default_macd', 0)
        macd_signal = row.get('MACD_default_signal', 0)
        macd_hist = row.get('MACD_default_histogram', 0)
        
        stoch_k = row.get('Stochastic_slowk', 50)
        stoch_d = row.get('Stochastic_slowd', 50)
        
        # Price trends
        short_trend = 0
        medium_trend = 0
        if self.current_step >= 5:
            prev_5 = self.df.iloc[self.current_step - 5].get('close', current_price)
            if prev_5 > 0:
                short_trend = np.clip((current_price - prev_5) / prev_5 * 10, -0.1, 0.1)
        
        if self.current_step >= 20:
            prev_20 = self.df.iloc[self.current_step - 20].get('close', current_price)
            if prev_20 > 0:
                medium_trend = np.clip((current_price - prev_20) / prev_20 * 5, -0.2, 0.2)
        
        indicators = [
            rsi,
            np.clip((bb_upper / current_price - 1), -0.1, 0.1),
            np.clip((bb_lower / current_price - 1), -0.1, 0.1),
            np.clip(atr / 1000, 0, 0.1),
            short_trend,
            medium_trend,
            np.clip(mfi / 100 - 0.5, -0.5, 0.5),
            np.clip(macd / current_price * 100, -1, 1),
            np.clip(macd_signal / current_price * 100, -1, 1),
            np.clip(macd_hist / current_price * 100, -1, 1),
            np.clip(stoch_k / 100 - 0.5, -0.5, 0.5),
            np.clip(stoch_d / 100 - 0.5, -0.5, 0.5),
        ]
        
        state = np.array([balance_norm, position_norm, price_norm, unrealized_pnl] + indicators, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        
        return state
    
    def _check_entry_signal(self, current_price):
        """Check if entry conditions are met - RELAXED"""
        row = self.df.iloc[self.current_step]
        
        rsi = row.get('RSI_15', 50)
        macd_hist = row.get('MACD_default_histogram', 0)
        
        # LONG entry: RSI < 40 (oversold) + MACD neutral or bullish
        long_signal = (rsi < self.rsi_oversold) and (macd_hist >= self.macd_threshold)
        
        # SHORT entry: RSI > 60 (overbought) + MACD neutral or bearish
        short_signal = (rsi > self.rsi_overbought) and (macd_hist <= self.macd_threshold)
        
        return long_signal, short_signal
    
    def _execute_entry(self, current_price, is_long):
        """Execute entry based on rules"""
        if self.position != 0:
            return False  # Already in position
        
        # Calculate position size (10% of balance)
        invest_amount = self.balance * 0.10
        fee = invest_amount * self.transaction_fee
        
        if is_long:
            coins = (invest_amount - fee) / current_price
            self.position = coins
            self.balance -= invest_amount
        else:
            # Short selling
            margin_required = invest_amount * 0.3
            if self.balance >= margin_required:
                coins = invest_amount / current_price
                self.position = -coins
                self.balance -= margin_required + fee
            else:
                return False
        
        self.entry_price = current_price
        self.entry_step = self.current_step
        self.total_fees += fee
        
        return True
    
    def _execute_exit(self, current_price, pnl_pct):
        """Execute exit based on RL action"""
        if self.position == 0:
            return False, 0
        
        position_size = abs(self.position)
        
        if self.position > 0:  # Close long
            revenue = position_size * current_price
            fee = revenue * self.transaction_fee
            pnl = revenue - (position_size * self.entry_price) - fee
            self.balance += revenue - fee
        else:  # Close short
            sale_proceeds = position_size * self.entry_price
            buy_cost = position_size * current_price
            fee = buy_cost * self.transaction_fee
            pnl = sale_proceeds - buy_cost - fee
            self.balance += sale_proceeds * 0.3 + pnl  # Return margin + PnL
        
        self.total_pnl += pnl
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        self.total_trades += 1
        self.total_fees += fee
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        
        return True, pnl_pct
    
    def step(self, action):
        """Execute one step"""
        if self.current_step >= len(self.df):
            return self._get_state(), 0, True, False, {}
        
        current_price = self.df.iloc[self.current_step].get('close', 
                            self.df.iloc[self.current_step].get('Close', 50000))
        
        reward = 0.0
        action_performed = False
        pnl_pct = 0.0
        
        # ========================================
        # 1. CHECK ENTRY SIGNALS (Rule-Based)
        # ========================================
        if self.position == 0:
            long_signal, short_signal = self._check_entry_signal(current_price)
            
            if long_signal:
                if self._execute_entry(current_price, is_long=True):
                    reward += 0.1  # Small reward for valid entry
                    if self.debug:
                        print(f"ENTRY LONG: price={current_price:.2f}, RSI={self.df.iloc[self.current_step].get('RSI_15', 0):.1f}")
            
            elif short_signal:
                if self._execute_entry(current_price, is_long=False):
                    reward += 0.1
                    if self.debug:
                        print(f"ENTRY SHORT: price={current_price:.2f}, RSI={self.df.iloc[self.current_step].get('RSI_15', 0):.1f}")
        
        # ========================================
        # 2. CHECK TP/SL (Safety)
        # ========================================
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
            # Take Profit
            if unrealized_pnl >= self.take_profit_pct:
                pnl_pct = unrealized_pnl
                action_performed, _ = self._execute_exit(current_price, pnl_pct)
                reward += 10.0 + pnl_pct * 100 * 2  # Big reward for TP
                if self.debug:
                    print(f"TP HIT: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
            
            # Stop Loss
            elif unrealized_pnl <= -self.stop_loss_pct:
                pnl_pct = unrealized_pnl
                action_performed, _ = self._execute_exit(current_price, pnl_pct)
                reward += -2.0  # Small penalty for SL (acceptable loss)
                if self.debug:
                    print(f"SL HIT: pnl={pnl_pct*100:.2f}%, reward={reward:.2f}")
            
            # Max hold steps
            elif self.current_step - self.entry_step >= self.max_hold_steps:
                pnl_pct = unrealized_pnl
                action_performed, _ = self._execute_exit(current_price, pnl_pct)
                reward += pnl_pct * 100  # Neutral exit
                if self.debug:
                    print(f"TIME EXIT: pnl={pnl_pct*100:.2f}%")
        
        # ========================================
        # 3. RL EXIT DECISION (when in position)
        # ========================================
        if self.position != 0 and self.entry_price > 0:
            # Check if TP/SL already triggered
            if self.position == 0:
                pass  # Already closed
            elif action == 1 and self.position > 0:  # Close long
                if self.current_step - self.entry_step >= 3:  # Min hold 3 steps
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                    pnl_pct = unrealized_pnl
                    action_performed, _ = self._execute_exit(current_price, pnl_pct)
                    reward += 5.0 + pnl_pct * 100 * 2  # Reward for closing
                    if self.debug:
                        print(f"RL CLOSE LONG: pnl={pnl_pct*100:.2f}%")
            
            elif action == 2 and self.position < 0:  # Close short
                if self.current_step - self.entry_step >= 3:
                    unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                    pnl_pct = unrealized_pnl
                    action_performed, _ = self._execute_exit(current_price, pnl_pct)
                    reward += 5.0 + pnl_pct * 100 * 2
                    if self.debug:
                        print(f"RL CLOSE SHORT: pnl={pnl_pct*100:.2f}%")
            
            # HOLD - small penalty to encourage closing
            elif action == 0:
                reward += -0.01
        
        # ========================================
        # 4. INVALID ACTION PENALTY
        # ========================================
        if self.position == 0 and action in [1, 2]:
            reward += -0.5  # Trying to close when flat
        
        # Update portfolio
        self.portfolio_values.append(self.balance)
        self.steps_in_episode += 1
        self.current_step += 1
        
        # Check termination
        terminated = False
        truncated = self.steps_in_episode >= self.episode_length
        
        info = {
            'portfolio_value': self.balance,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'pnl_pct': pnl_pct,
            'action_performed': action_performed
        }
        
        return self._get_state(), reward, terminated, truncated, info


def train_rule_based_entry():
    """Train RL model with rule-based entry"""
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    
    print("=" * 70)
    print("🚀 RULE-BASED ENTRY + RL EXIT TRAINING")
    print("=" * 70)
    
    # Load data
    DATA_PATH = './btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
    SAVE_PATH = './rl_checkpoints_v15_rule_entry'
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")
    
    # Create environment
    def make_env(seed=0):
        def _init():
            return RuleBasedEntryEnv(
                df=df,
                initial_balance=10000,
                transaction_fee=0.0018,
                episode_length=300,
                debug=False
            )
        return _init
    
    print("Creating environments...")
    env = DummyVecEnv([make_env(i) for i in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher exploration
        verbose=1,
        tensorboard_log='./logs_v15/'
    )
    
    # Callback
    class SaveBestCallback(BaseCallback):
        def __init__(self, save_freq, save_path):
            super().__init__()
            self.save_freq = save_freq
            self.save_path = save_path
            self.best_reward = -np.inf
            
        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                if 'rewards' in self.locals:
                    mean_reward = np.mean(self.locals['rewards'])
                    if mean_reward > self.best_reward:
                        self.best_reward = mean_reward
                        path = f"{self.save_path}/ppo_v15_best"
                        print(f"\n💾 Saving BEST (reward={mean_reward:.2f})...")
                        self.model.save(path)
                    
                    path = f"{self.save_path}/ppo_v15_{self.num_timesteps}_steps"
                    self.model.save(path)
                    print(f"💾 Checkpoint: {self.num_timesteps:,}")
            return True
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    callback = SaveBestCallback(100000, SAVE_PATH)
    
    # Train
    print("\n📚 STARTING TRAINING...\n")
    print("-" * 70)
    
    model.learn(
        total_timesteps=500000,
        callback=callback,
        tb_log_name='ppo_v15_rule_entry',
        progress_bar=False
    )
    
    # Save final
    model.save(f"{SAVE_PATH}/ppo_v15_final")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nBest model: {SAVE_PATH}/ppo_v15_best.zip")
    print(f"Best reward: {callback.best_reward:.2f}")
    
    return model


if __name__ == "__main__":
    train_rule_based_entry()
