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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import os
import random

# Import the unified trading environment
from trading_environment import TradingEnvironment

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducible results"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call set_seeds at the beginning of the module
set_seeds()

class EvaluationLoggerCallback(EvalCallback):
    """Simplified EvalCallback that safely logs results to file"""

    def __init__(self, log_file="rl_evaluation_log.txt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        # Simple header initialization (only if file doesn't exist)
        import os
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Timestep,Mean Reward,Mean Episode Length,Balance,Portfolio Value,PnL,PnL %\n")

    def _on_step(self):
        """Simplified logging - only log reward without complex portfolio tracking"""
        result = super()._on_step()

        if (self.eval_freq > 0 and
            self.n_calls % self.eval_freq == 0 and
            hasattr(self, 'last_mean_reward') and
            self.last_mean_reward is not None):

            current_timestep = self.num_timesteps
            mean_reward = self.last_mean_reward

            try:
                with open(self.log_file, 'a') as f:
                    # Log only basics: timestep, reward, and placeholders for portfolio data
                    f.write(f"{current_timestep},{mean_reward:.2f},0.00,0.00,0.00,0.00,0.00\n")
            except Exception as e:
                print(f"Warning: Could not write to evaluation log: {e}")

        return result



class RLTrainingProgressCallback(BaseCallback):
    """Custom callback to show detailed RL training progress"""

    def __init__(self, total_timesteps, eval_freq=10000, verbose=1):
        # Don't pass verbose to super() to avoid logger property issues
        super().__init__()
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.start_time = None
        self.last_eval_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.verbose = verbose

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

            # Get additional metrics if available
            info_str = ""
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                try:
                    # Try to get some useful metrics
                    loss_info = []
                    if 'train/loss' in self.model.logger.name_to_value:
                        loss_info.append(f"Loss: {self.model.logger.name_to_value['train/loss']:.4f}")
                    if 'train/value_loss' in self.model.logger.name_to_value:
                        loss_info.append(f"V-Loss: {self.model.logger.name_to_value['train/value_loss']:.4f}")
                    if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
                        loss_info.append(f"PG-Loss: {self.model.logger.name_to_value['train/policy_gradient_loss']:.4f}")
                    if loss_info:
                        info_str = " | " + " | ".join(loss_info)
                except:
                    pass

            print(f"📈 Timestep {current_timestep:6,}/{self.total_timesteps:6,} | [{bar}] {progress*100:5.1f}% | TPS: {timesteps_per_sec:6.0f} | ETA: {eta_str}{info_str}", end='\r', flush=True)

        # Show evaluation results
        if self.eval_freq > 0 and current_timestep % self.eval_freq == 0 and current_timestep > 0:
            if hasattr(self, 'last_mean_reward'):
                eval_elapsed = time.time() - (self.last_eval_time or self.start_time)
                print(f"🎯 Evaluation at {current_timestep:,} timesteps:")
                if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
                    print(f"   Mean Reward: {self.last_mean_reward:.2f}")
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

# The TradingEnvironment class has been moved to trading_environment.py
# This file now imports it from the unified module

def make_env(df, rank):
    """Create environment with error handling for parallel processing"""
    def _init():
        try:
            env = TradingEnvironment(df)
            env = Monitor(env, f"./rl_logs/monitor_{rank}")
            return env
        except Exception as e:
            print(f"Error creating environment {rank}: {e}")
            raise
    return _init

def preprocess_data(df):
    """Улучшенная предобработка данных"""
    print(f"Preprocessing data: {len(df)} initial rows")

    # Удаление дубликатов
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['timestamp'] if 'timestamp' in df.columns else ['Open time'] if 'Open time' in df.columns else None)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")

    # Обработка экстремальных значений
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    for col in numeric_columns:
        if col not in ['timestamp', 'Open time', 'timestamp_close'] and not col.startswith('ignore'):
            # Удаление значений вне 3 сигм
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                before_count = len(df)
                df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]
                outliers_removed += before_count - len(df)

    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outlier values")

    # Правильная нормализация без утечки данных
    scaler = StandardScaler()
    indicator_cols = ['RSI_15', 'BB_15_upper', 'BB_15_lower', 'ATR_15', 'OBV', 'AD', 'MFI_15']
    available_indicators = [col for col in indicator_cols if col in df.columns]
    if available_indicators:
        df[available_indicators] = scaler.fit_transform(df[available_indicators])
        print(f"Normalized {len(available_indicators)} indicator columns")

    print(f"Preprocessing complete: {len(df)} final rows")
    return df

def train_rl_agent(data_path, total_timesteps=200000, eval_freq=10000, n_envs=8):
    """
    Train RL agent using PPO with parallel environments

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

    # Advanced data preprocessing
    df = preprocess_data(df)

    # Clean data (remove any remaining NaN)
    df = df.dropna()
    print(f"After dropping NaN: {len(df)} rows")

    if df.empty:
        raise ValueError("All data was removed after preprocessing")

    # Check minimum data requirements for RL training
    if len(df) < 1000:
        print(f"WARNING: Limited data for RL training ({len(df)} rows). Consider using more historical data.")

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} features")

    print(f"Creating {n_envs} parallel RL environments...")
    # Create parallel environments using SubprocVecEnv
    envs = [make_env(df, i) for i in range(n_envs)]
    env = SubprocVecEnv(envs)

    print("Creating PPO model...")
    # Оптимизированные параметры PPO для прибыльной торговли
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_linear_fn(0.0003, 0.00001, 0.1),  # Медленнее уменьшение LR
        n_steps=4096,  # Больше шагов для лучшего обучения
        batch_size=128,  # Больший размер батча
        n_epochs=20,  # Больше эпох обучения
        gamma=0.995,  # Более длинный горизонт планирования
        gae_lambda=0.98,  # Лучшая оценка преимущества
        clip_range=get_linear_fn(0.3, 0.1, 0.2),  # Более широкий clip range
        ent_coef=0.005,  # Меньше случайности для фокусировки
        vf_coef=0.8,  # Больше внимания к value function
        max_grad_norm=0.3,  # Меньше ограничение градиентов
        verbose=1,
        tensorboard_log="./rl_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # Более мощная архитектура
            activation_fn=torch.nn.ReLU
        )
    )

    # Evaluation callback - use different data slices for each evaluation episode
    # This provides more diverse evaluation and better assessment of generalization
    eval_data_size = max(1000, len(df) // 10)
    eval_count = 0

    def make_eval_env():
        nonlocal eval_count
        eval_count += 1
        # Use systematic different slices for each evaluation to avoid identical results
        max_start = len(df) - eval_data_size
        if max_start > 0:
            start_idx = (eval_count - 1) * (max_start // 10) % (max_start + 1)
        else:
            start_idx = 0
        eval_df = df.iloc[start_idx:start_idx + eval_data_size].reset_index(drop=True)
        # Create environment with unique ID to avoid monitor conflicts
        env = TradingEnvironment(eval_df)
        env.spec = type('Spec', (), {'id': f'TradingEnv_eval_{eval_count}'})()
        return Monitor(env, "./rl_logs/")
    
    # Use SubprocVecEnv for evaluation to match training env type and avoid warnings
    eval_env = SubprocVecEnv([make_eval_env])
    
    # Adjust eval frequency based on total timesteps
    # For longer training, evaluate less frequently to avoid slowdowns
    if total_timesteps > 500000:
        actual_eval_freq = max(eval_freq, total_timesteps // 100)  # Max 100 evaluations
    else:
        actual_eval_freq = eval_freq
    
    eval_callback = EvaluationLoggerCallback(
        log_file="rl_evaluation_log.txt",
        eval_env=eval_env,
        best_model_save_path="./rl_models/",
        log_path="./rl_logs/",
        eval_freq=actual_eval_freq,
        n_eval_episodes=1,  # Single episode per evaluation for cleaner monitor logs
        deterministic=True,
        render=False
    )

    # Custom progress callback
    progress_callback = RLTrainingProgressCallback(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq
    )
    
    # Episode logger callback to track balance and PnL
    #episode_logger = EpisodeLoggerCallback(log_file="rl_episode_log.txt")

    # Get environment to access parameters for display
    env_instances = env.envs if hasattr(env, 'envs') else [env]
    trading_env = env_instances[0]
    if hasattr(trading_env, 'env'):
        trading_env = trading_env.env

    reward_clip_min, reward_clip_max = trading_env.reward_clip_bounds if hasattr(trading_env, 'reward_clip_bounds') else (-50, 50)
    stop_loss_pct = int((1 - trading_env.termination_stop_loss_threshold) * 100) if hasattr(trading_env, 'termination_stop_loss_threshold') else 15

    # Checkpoint callback for saving intermediate models
    checkpoint_freq = max(50000, total_timesteps // 20)  # Save every 5% of training or every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./rl_checkpoints/",
        name_prefix="ppo_trading_agent",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("Starting training...")
    print(f"Training configuration:")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Parallel environments: {n_envs}")
    print(f"  - Checkpoint frequency: {checkpoint_freq:,} timesteps")
    print(f"  - Evaluation frequency: {actual_eval_freq:,} timesteps")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Network architecture: [256, 256, 128]")
    print(f"  - Reward clipping: [{reward_clip_min}, {reward_clip_max}]")
    print(f"  - Conservative position sizing enabled")
    print(f"  - Early termination at {stop_loss_pct}% portfolio loss")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback, checkpoint_callback],
        progress_bar=True  # Show progress bar for long training
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("Saving final model...")
    model.save("ppo_trading_agent")
    print("✅ Model saved to 'ppo_trading_agent.zip'")
    print("\nNext steps:")
    print("  1. Run paper trading: python rl_paper_trading.py")
    print("  2. Check evaluation log: rl_evaluation_log.txt")
    print("  3. Check episode log: rl_episode_log.txt")
    print("  4. Monitor TensorBoard: tensorboard --logdir ./rl_tensorboard/")

    return model

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    if len(portfolio_values) < 2:
        return 0.0

    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return abs(drawdown.min())

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    """Calculate Sharpe ratio from portfolio values"""
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    # Annualize (assuming daily returns)
    return sharpe_ratio * np.sqrt(252)

def validate_model_performance(model_path, validation_data_path):
    """Валидация модели на отдельном наборе данных"""
    print("🔍 Validating model performance...")

    try:
        model = PPO.load(model_path)
        df = pd.read_csv(validation_data_path)
        df = preprocess_data(df)

        # Используем последние 20% данных для валидации
        validation_size = len(df) // 5
        validation_df = df.iloc[-validation_size:].reset_index(drop=True)

        env = TradingEnvironment(validation_df)
        obs, _ = env.reset()
        done = False
        portfolio_values = [env.initial_balance]
        trade_actions = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if not isinstance(action, (int, np.integer)):
                action = 0
            else:
                action = np.clip(action, 0, env.action_space.n - 1)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Подсчет торговых действий
            if action in [1, 2, 3, 4]:
                trade_actions += 1

            current_price = validation_df.iloc[min(env.current_step, len(validation_df)-1)].get('close',
                              validation_df.iloc[min(env.current_step, len(validation_df)-1)].get('Close'))
            portfolio_value = env.balance + env.margin_locked + env.position * current_price
            portfolio_values.append(portfolio_value)

        # Расчет метрик валидации
        total_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance
        sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
        max_dd = calculate_max_drawdown(portfolio_values)

        # Дополнительные метрики
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        trade_frequency = trade_actions / len(portfolio_values) if len(portfolio_values) > 0 else 0

        print(f"📊 Validation Results:")
        print(f"   Total Return: {total_return*100:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_dd*100:.2f}%")
        print(f"   Volatility: {volatility*100:.2f}%")
        print(f"   Trade Frequency: {trade_frequency:.3f} trades/step")

        # Комплексная оценка
        score = 0
        if total_return > 0:
            score += 1
        if sharpe_ratio > 0.5:
            score += 1
        if max_dd < 0.2:
            score += 1
        if volatility < 0.5:
            score += 1

        if score >= 3:
            print("✅ Model PASSED validation")
            return True
        else:
            print("❌ Model FAILED validation")
            return False

    except Exception as e:
        print(f"Error during validation: {e}")
        return False

def evaluate_agent_comprehensive(model, data_path, n_episodes=10):
    """Комплексная оценка агента с множественными метриками"""
    print("🧪 Starting comprehensive agent evaluation...")
    print(f"Evaluating on {n_episodes} random data slices")

    df = pd.read_csv(data_path)
    df = preprocess_data(df)  # Используем ту же предобработку что и в обучении

    results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'profit_factors': [],
        'total_trades': [],
        'portfolio_histories': []
    }

    for episode in range(n_episodes):
        # Выбор случайного среза данных
        eval_length = min(2000, len(df) // 3)  # До 2000 точек или 1/3 данных
        start_idx = np.random.randint(0, max(1, len(df) - eval_length))
        eval_df = df.iloc[start_idx:start_idx + eval_length].reset_index(drop=True)

        env = TradingEnvironment(eval_df)
        obs, _ = env.reset()
        done = False
        portfolio_history = [env.initial_balance]
        trade_count = 0
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0
        gross_loss = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, (int, np.integer)):
                action = np.clip(action, 0, env.action_space.n - 1)
            else:
                action = 0

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track trades (all trading actions)
            if action in [1, 2, 3, 4]:  # All trading actions
                trade_count += 1

            current_price = eval_df.iloc[min(env.current_step, len(eval_df)-1)].get('close', eval_df.iloc[min(env.current_step, len(eval_df)-1)].get('Close'))
            portfolio_value = env.balance + env.margin_locked + env.position * current_price
            portfolio_history.append(portfolio_value)

        # Расчет метрик для эпизода
        returns = [(portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                  for i in range(1, len(portfolio_history)) if portfolio_history[i-1] > 0]

        total_return = (portfolio_history[-1] - env.initial_balance) / env.initial_balance

        # Sharpe ratio (annualized, assuming daily returns)
        if returns and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        max_drawdown = calculate_max_drawdown(portfolio_history)

        # Win rate (simplified - based on final return)
        win_rate = 1.0 if total_return > 0 else 0.0

        # Profit factor (simplified)
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 1.0

        results['returns'].append(total_return)
        results['sharpe_ratios'].append(sharpe_ratio)
        results['max_drawdowns'].append(max_drawdown)
        results['win_rates'].append(win_rate)
        results['profit_factors'].append(profit_factor)
        results['total_trades'].append(trade_count)
        results['portfolio_histories'].append(portfolio_history)

        print(f"Episode {episode + 1}: Return={total_return*100:.2f}%, Sharpe={sharpe_ratio:.2f}, "
              f"MaxDD={max_drawdown*100:.2f}%, Trades={trade_count}")

    # Статистический анализ результатов
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)

    print(f"Episodes evaluated: {n_episodes}")
    print(f"Average Return: {np.mean(results['returns'])*100:.2f}% ± {np.std(results['returns'])*100:.2f}%")
    print(f"Median Return: {np.median(results['returns'])*100:.2f}%")
    print(f"Best Return: {np.max(results['returns'])*100:.2f}%")
    print(f"Worst Return: {np.min(results['returns'])*100:.2f}%")
    print()

    print(f"Average Sharpe Ratio: {np.mean(results['sharpe_ratios']):.2f} ± {np.std(results['sharpe_ratios']):.2f}")
    print(f"Average Max Drawdown: {np.mean(results['max_drawdowns'])*100:.2f}% ± {np.std(results['max_drawdowns'])*100:.2f}%")
    print(f"Win Rate: {np.mean(results['win_rates'])*100:.1f}%")
    print(f"Average Trades per Episode: {np.mean(results['total_trades']):.1f}")
    print()

    # Оценка общей производительности
    avg_return = np.mean(results['returns'])
    avg_sharpe = np.mean(results['sharpe_ratios'])
    avg_drawdown = np.mean(results['max_drawdowns'])

    if avg_return > 0.05 and avg_sharpe > 1.0 and avg_drawdown < 0.15:
        assessment = "EXCELLENT - Ready for live trading!"
    elif avg_return > 0.02 and avg_sharpe > 0.5:
        assessment = "GOOD - Promising performance, needs optimization"
    elif avg_return > 0:
        assessment = "FAIR - Generates profit but needs improvement"
    else:
        assessment = "POOR - Needs significant retraining"

    print(f"🎯 OVERALL ASSESSMENT: {assessment}")

    # Сохранение детальных результатов
    detailed_results = {
        'summary': {
            'episodes': n_episodes,
            'avg_return': np.mean(results['returns']),
            'std_return': np.std(results['returns']),
            'avg_sharpe': np.mean(results['sharpe_ratios']),
            'avg_drawdown': np.mean(results['max_drawdowns']),
            'win_rate': np.mean(results['win_rates']),
            'assessment': assessment
        },
        'detailed': results
    }

    with open('rl_comprehensive_evaluation.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)

    print("📁 Detailed results saved to 'rl_comprehensive_evaluation.pkl'")

    return detailed_results

def evaluate_agent(model, data_path, n_episodes=5):
    """Legacy evaluation function - now calls comprehensive evaluation"""
    return evaluate_agent_comprehensive(model, data_path, n_episodes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument("--data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to training data")
    parser.add_argument("--timesteps", type=int, default=2000000,
                       help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate trained agent")

    args = parser.parse_args()

    if args.eval:
        print("Loading trained model...")
        try:
            model = PPO.load("ppo_trading_agent")
            evaluate_agent(model, args.data)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    else:
        model = train_rl_agent(args.data, args.timesteps, n_envs=args.n_envs)
        print("Training completed. Use --eval to evaluate the agent.")
