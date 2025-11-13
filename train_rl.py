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
from stable_baselines3.common.callbacks import EvalCallback
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # State: [balance_norm, position_norm, price_norm, indicators...]
        # Indicators: RSI, MACD, BB_upper, BB_lower, ATR, OBV, AD, MFI
        n_indicators = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 + n_indicators,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, positive=long
        self.total_fees = 0
        self.portfolio_values = [self.initial_balance]

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]

        # Normalize values
        balance_norm = self.balance / self.initial_balance - 1
        position_norm = self.position / (self.balance * 0.1) if self.balance > 0 else 0
        price_norm = row['Close'] / 50000 - 1  # Normalize around typical BTC price

        # Technical indicators
        indicators = [
            row.get('RSI_15', 50) / 100 - 0.5,
            row.get('MACD_macd', 0) / 1000,
            row.get('BB_15_upper', row['Close']) / row['Close'] - 1,
            row.get('BB_15_lower', row['Close']) / row['Close'] - 1,
            row.get('ATR_15', 100) / 1000,
            row.get('OBV', 0) / 1e10,
            row.get('AD', 0) / 1e10,
            row.get('MFI_15', 50) / 100 - 0.5
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

        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']

        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_fee):
                # Buy with 10% of balance
                invest_amount = min(self.balance * 0.1, self.balance - 100)
                fee = invest_amount * self.transaction_fee
                coins_bought = (invest_amount - fee) / current_price

                self.position += coins_bought
                self.balance -= invest_amount
                self.total_fees += fee

                # Small penalty for transaction
                reward -= 0.01

        elif action == 2:  # Sell
            if self.position > 0:
                sell_amount = self.position * 0.5  # Sell 50% position
                revenue = sell_amount * current_price
                fee = revenue * self.transaction_fee

                self.position -= sell_amount
                self.balance += revenue - fee
                self.total_fees += fee

                # Small penalty for transaction
                reward -= 0.01

        # Calculate reward based on portfolio change
        current_portfolio = self.balance + self.position * current_price
        next_portfolio = self.balance + self.position * next_price

        portfolio_change = (next_portfolio - current_portfolio) / current_portfolio if current_portfolio > 0 else 0
        reward += portfolio_change * 100  # Scale reward

        # Penalty for holding too long without action
        if action == 0 and self.position > 0:
            reward -= 0.001

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
        portfolio_value = self.balance + self.position * self.df.iloc[self.current_step]['Close']
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.6f}, Portfolio: {portfolio_value:.2f}")

def train_rl_agent(data_path, total_timesteps=100000, eval_freq=10000):
    """Train RL agent using PPO"""

    print("Loading training data...")
    df = pd.read_csv(data_path)
    df = df.dropna()

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

    # Evaluation callback
    eval_env = DummyVecEnv([lambda: TradingEnvironment(df)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./rl_models/",
        log_path="./rl_logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
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
            portfolio_history.append(env.balance + env.position * df.iloc[min(env.current_step, len(df)-1)]['Close'])

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
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
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
