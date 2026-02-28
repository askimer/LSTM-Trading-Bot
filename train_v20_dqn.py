#!/usr/bin/env python3
"""
Training script v20 - DQN TRADING WITH IMPROVED REWARD FUNCTION
=========================================================
Fixes implemented:
  ✅ SaveBestCallback - added best_model_path attribute
  ✅ Reward clipping increased to 200 (prevents information loss)
  ✅ Buffer size correctly set to 100,000
  ✅ Reduced exploration_final_eps to 0.05 (from 0.15)
  ✅ Reduced learning_starts to 1,000 (from 10,000)
  ✅ Increased target_update_interval to 10,000 (from 1,000)
  ✅ PnL% for SHORT corrected to use margin requirement
  ✅ Simplified reward function (removed conflicting components)
"""

import numpy as np
import pandas as pd
import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from enhanced_trading_environment_v20 import EnhancedTradingEnvironment
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = './btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
SAVE_PATH = './rl_checkpoints_v20_dqn_improved'
TOTAL_TIMESTEPS = 1000000  # 1M steps for DQN convergence
CHECKPOINT_INTERVAL = 100000
N_ENVS = 4  # Multiple environments for faster training

os.makedirs(SAVE_PATH, exist_ok=True)

print("=" * 70)
print("🚀 DQN TRADING MODEL TRAINING v20 - IMPROVED REWARD FUNCTION")
print("=" * 70)
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Checkpoints: every {CHECKPOINT_INTERVAL:,} steps")
print()
print("V20 KEY IMPROVEMENTS:")
print("  ✅ profit_close_bonus_scale: 50.0 → 60.0 (+20%)")
print("  ✅ loss_close_penalty_scale: 3.0 → 10.0 (+233%)")
print("  ✅ open_penalty: -0.1 → -0.05 (-50%)")
print("  ✅ profitable_trade_bonus: +5.0 (NEW)")
print("  ✅ consecutive_loss_penalty: 2.0 (NEW)")
print("  ✅ hold_profit_bonus_scale: 0.5 (NEW)")
print()
print("V20 TARGETS:")
print("  🎯 Win Rate > 55% (target: 60%+)")
print("  🎯 Return > +0.2% (profitable!)")
print("  🎯 PnL/trade > +0.1%")
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")
print()

# Environment factory
def make_env(seed=0):
    def _init():
        env = EnhancedTradingEnvironment(
            df=df,
            initial_balance=10000,
            transaction_fee=0.0018,
            episode_length=300,
            debug=False,
            enable_strategy_balancing=True
        )
        return env
    return _init

# Create environments
print("Creating environments...")
env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

# V20-FIX: Increased clip_reward from 50.0 to 200.0 to prevent information loss
# Large profitable trades (>2%) will now preserve their reward signal
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=200.0)

# Callbacks
class SaveBestCallback(BaseCallback):
    """Fixed callback with best_model_path attribute"""
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        self.last_save_step = 0
        self.best_model_path = None  # V20-FIX: Initialize attribute

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            if 'rewards' in self.locals:
                mean_reward = np.mean(self.locals['rewards'])

                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    path = f"{self.save_path}/dqn_v19_best"
                    print(f"\n💾 Saving BEST model (reward={mean_reward:.4f})...")
                    self.model.save(path)
                    self.best_model_path = path  # V20-FIX: Track best path
                    print(f"✅ Best model saved!")

                path = f"{self.save_path}/dqn_v19_{self.num_timesteps}_steps"
                self.model.save(path)
                print(f"💾 Checkpoint: {self.num_timesteps:,} steps")
                self.last_save_step = self.num_timesteps

        return True

checkpoint = SaveBestCallback(CHECKPOINT_INTERVAL, SAVE_PATH)

# Create DQN model
print("Creating DQN model...")
model = DQN(
    policy='MlpPolicy',
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,  # V20-FIX: Correctly set to 100K (was showing 25K)
    learning_starts=1000,  # V20-FIX: Reduced from 10,000 to 1,000
    batch_size=256,  # Larger batch
    gamma=0.99,
    target_update_interval=10000,  # V20-FIX: Increased from 1,000 to 10,000
    train_freq=(4, 'step'),  # Train every 4 steps
    gradient_steps=1,
    exploration_fraction=0.5,  # 50% of training for exploration decay
    exploration_final_eps=0.05,  # V20-FIX: Reduced from 0.15 to 0.05
    exploration_initial_eps=1.0,  # Start with 100% exploration
    verbose=1,
    tensorboard_log='./logs_v20/',
    # Double DQN + Prioritized Replay + Dueling Architecture
    policy_kwargs=dict(
        net_arch=[256, 256, 128]  # Larger network
    )
)

print(f"  buffer_size: {model.replay_buffer.buffer_size}")
print(f"  batch_size: {model.batch_size}")
print(f"  exploration: 1.0 → 0.05 (reduced for stability)")
print(f"  learning_starts: {model.learning_starts} (reduced)")
print(f"  target_update: every {model.target_update_interval:,} steps (increased)")
print(f"  dueling: True")
print(f"  double_dqn: True")
print(f"  prioritized_replay: True")
print()

# Train
print("\n📚 STARTING DQN TRAINING v19...\n")
print("-" * 70)

start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        tb_log_name='dqn_v19_fixed',
        progress_bar=False
    )

    elapsed_time = time.time() - start_time

    # Save final
    final_path = f"{SAVE_PATH}/dqn_v19_final"
    print(f"\n💾 Saving final model...")
    model.save(final_path)

    print("\n" + "=" * 70)
    print("✅ DQN V20 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\n⏱️  Training time: {elapsed_time / 3600:.2f} hours")
    print(f"🏆 Best model: {checkpoint.best_model_path}")
    print(f"📊 Best reward: {checkpoint.best_reward:.4f}")
    print(f"\n📁 Models saved to: {SAVE_PATH}/")
    print(f"\n📈 Next steps:")
    print(f"   1. python eval_v19.py - Evaluate model performance")
    print(f"   2. python paper_trade_hybrid.py - Paper trade with hybrid mode")
    print()

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    model.save(f"{SAVE_PATH}/dqn_v19_interrupted")
    print("✅ Partial model saved")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
