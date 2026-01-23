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
from tqdm import tqdm

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

class MetricsTrackingWrapper(gym.Wrapper):
    """Wrapper that tracks portfolio metrics from info dictionary"""
    def __init__(self, env):
        super().__init__(env)
        self.last_info = {}
        self.episode_metrics = {
            'initial_balance': None,
            'final_balance': None,
            'final_portfolio_value': None,
            'total_return': None,
            'total_pnl': None,
            'pnl_pct': None
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Store initial balance from environment
        if hasattr(self.env, 'initial_balance'):
            self.episode_metrics['initial_balance'] = float(self.env.initial_balance)
        elif hasattr(self.env, 'balance'):
            self.episode_metrics['initial_balance'] = float(self.env.balance)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store info for later access
        self.last_info = info.copy() if isinstance(info, dict) else {}
        
        # Update episode metrics from info
        if isinstance(info, dict):
            if 'portfolio_value' in info:
                self.episode_metrics['final_portfolio_value'] = float(info['portfolio_value'])
            if 'cash' in info:
                self.episode_metrics['final_balance'] = float(info['cash'])
            if 'total_return' in info:
                self.episode_metrics['total_return'] = float(info['total_return'])
            if 'total_pnl' in info:
                self.episode_metrics['total_pnl'] = float(info['total_pnl'])
            
            # Calculate PnL percentage if we have initial balance
            if self.episode_metrics['initial_balance'] is not None:
                if self.episode_metrics['final_portfolio_value'] is not None:
                    final_value = self.episode_metrics['final_portfolio_value']
                    initial_value = self.episode_metrics['initial_balance']
                    if initial_value > 0:
                        self.episode_metrics['pnl_pct'] = ((final_value - initial_value) / initial_value) * 100
                        self.episode_metrics['total_pnl'] = final_value - initial_value
        
        # If episode ended, ensure we have final metrics
        if terminated or truncated:
            if hasattr(self.env, 'balance'):
                self.episode_metrics['final_balance'] = float(self.env.balance)
            if hasattr(self.env, 'portfolio_values') and len(self.env.portfolio_values) > 0:
                self.episode_metrics['final_portfolio_value'] = float(self.env.portfolio_values[-1])
                if self.episode_metrics['initial_balance'] is not None:
                    initial = self.episode_metrics['initial_balance']
                    final = self.episode_metrics['final_portfolio_value']
                    if initial > 0:
                        self.episode_metrics['pnl_pct'] = ((final - initial) / initial) * 100
                        self.episode_metrics['total_pnl'] = final - initial
        
        return obs, reward, terminated, truncated, info


class EvaluationLoggerCallback(EvalCallback):
    """Enhanced EvalCallback that logs comprehensive portfolio metrics"""

    def __init__(self, log_file="rl_evaluation_log.txt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        # Ensure header exists in log file
        import os
        header = "Timestep,Mean Reward,Mean Episode Length,Balance,Portfolio Value,PnL,PnL %\n"
        if not os.path.exists(self.log_file):
            # Create new file with header
            with open(self.log_file, 'w') as f:
                f.write(header)
        else:
            # Check if file is empty or missing header
            try:
                with open(self.log_file, 'r') as f:
                    first_line = f.readline()
                    if not first_line or not first_line.startswith("Timestep"):
                        # File exists but missing header, prepend it
                        content = f.read()
                        with open(self.log_file, 'w') as fw:
                            fw.write(header)
                            if content:
                                fw.write(content)
            except Exception:
                # If read fails, create new file
                with open(self.log_file, 'w') as f:
                    f.write(header)
        # Store last evaluation metrics
        self.last_eval_metrics = {}
        # Store evaluation environment reference for later access
        self._eval_env_ref = None

    def _on_step(self):
        """Enhanced logging with actual portfolio metrics"""
        result = super()._on_step()

        if (self.eval_freq > 0 and
            self.n_calls % self.eval_freq == 0 and
            hasattr(self, 'last_mean_reward') and
            self.last_mean_reward is not None):

            current_timestep = self.num_timesteps
            mean_reward = self.last_mean_reward

            # Get episode length from evaluation results
            episode_length = 0.0
            # Try multiple ways to get episode length
            if hasattr(self, 'last_episode_lengths') and self.last_episode_lengths is not None and len(self.last_episode_lengths) > 0:
                episode_length = np.mean(self.last_episode_lengths)
            elif hasattr(self, 'logger') and hasattr(self.logger, 'name_to_value'):
                # Try to get from logger if available
                if 'eval/mean_ep_length' in self.logger.name_to_value:
                    episode_length = self.logger.name_to_value['eval/mean_ep_length']
            # If still 0, try to estimate from stored metrics
            if episode_length == 0.0 and 'episode_length' in self.last_eval_metrics:
                episode_length = self.last_eval_metrics['episode_length']

            # Try to get portfolio metrics from stored evaluation metrics or environment
            balance = self.last_eval_metrics.get('balance', 0.0)
            portfolio_value = self.last_eval_metrics.get('portfolio_value', 0.0)
            pnl = self.last_eval_metrics.get('pnl', 0.0)
            pnl_pct = self.last_eval_metrics.get('pnl_pct', 0.0)

            # If metrics not stored, try to get from environment or estimate from reward
            if portfolio_value == 0.0:
                # First try to get from stored metrics (set during evaluation)
                if 'portfolio_value' in self.last_eval_metrics:
                    portfolio_value = self.last_eval_metrics['portfolio_value']
                    balance = self.last_eval_metrics.get('balance', portfolio_value)
                    pnl = self.last_eval_metrics.get('pnl', 0.0)
                    pnl_pct = self.last_eval_metrics.get('pnl_pct', 0.0)
                else:
                    # Try to get from environment directly
                    try:
                        if hasattr(self, 'eval_env') and self.eval_env is not None:
                            env = self.eval_env
                            # Handle vectorized environments - get the underlying TradingEnvironment
                            actual_env = env
                            
                            # If using SubprocVecEnv, access the first environment
                            if hasattr(env, 'envs') and len(env.envs) > 0:
                                wrapped_env = env.envs[0]
                                actual_env = wrapped_env
                            else:
                                # Direct access to environment
                                actual_env = env
                            
                            # Unwrap Monitor and MetricsTrackingWrapper to get to TradingEnvironment
                            while hasattr(actual_env, 'env'):
                                actual_env = actual_env.env
                                # Check if it's our wrapper and access the underlying TradingEnvironment
                                if isinstance(actual_env, MetricsTrackingWrapper):
                                    actual_env = actual_env.env
                                    break  # We've reached the TradingEnvironment
                            
                            # Now access the actual TradingEnvironment properties
                            if hasattr(actual_env, 'balance'):
                                balance = float(actual_env.balance)
                            if hasattr(actual_env, 'portfolio_values') and len(actual_env.portfolio_values) > 0:
                                portfolio_value = float(actual_env.portfolio_values[-1])
                                if len(actual_env.portfolio_values) > 1:
                                    initial_value = float(actual_env.portfolio_values[0])
                                    if initial_value > 0:
                                        pnl = portfolio_value - initial_value
                                        pnl_pct = (pnl / initial_value) * 100
                            elif hasattr(actual_env, 'initial_balance'):
                                portfolio_value = float(actual_env.initial_balance)
                                balance = portfolio_value
                    except Exception as e:
                        # If we can't get portfolio data, estimate from reward
                        # Reward is roughly log(1 + return) * 100, so return ~ exp(reward/100) - 1
                        if mean_reward != 0:
                            try:
                                if hasattr(self, 'eval_env') and self.eval_env is not None:
                                    env = self.eval_env
                                    # Handle vectorized environments
                                    if hasattr(env, 'envs') and len(env.envs) > 0:
                                        wrapped_env = env.envs[0]
                                        actual_env = wrapped_env
                                    else:
                                        actual_env = env
                                    
                                    # Unwrap to get TradingEnvironment
                                    while hasattr(actual_env, 'env'):
                                        actual_env = actual_env.env
                                        if isinstance(actual_env, MetricsTrackingWrapper):
                                            actual_env = actual_env.env
                                            break
                                    
                                    if hasattr(actual_env, 'initial_balance'):
                                        initial_balance = float(actual_env.initial_balance)
                                        # Estimate: reward = log(1 + return) * 100
                                        estimated_return = np.exp(mean_reward / 100.0) - 1.0
                                        portfolio_value = initial_balance * (1 + estimated_return)
                                        pnl = portfolio_value - initial_balance
                                        pnl_pct = estimated_return * 100
                                        balance = portfolio_value
                            except Exception:
                                pass

            # Debug information
            if hasattr(self, 'eval_env') and self.eval_env is not None:
                try:
                    env = self.eval_env
                    if hasattr(env, 'envs') and len(env.envs) > 0:
                        wrapped_env = env.envs[0]
                        # Unwrap to get actual TradingEnvironment
                        actual_env = wrapped_env
                        while hasattr(actual_env, 'env'):
                            actual_env = actual_env.env
                            if isinstance(actual_env, MetricsTrackingWrapper):
                                actual_env = actual_env.env
                                break
                        
                        # Print debug info
                        if hasattr(actual_env, 'portfolio_values') and len(actual_env.portfolio_values) > 0:
                            print(f"DEBUG: Portfolio values: {actual_env.portfolio_values}")
                            print(f"DEBUG: Balance: {getattr(actual_env, 'balance', 'Not found')}")
                        else:
                            print(f"DEBUG: No portfolio_values found in actual_env")
                            print(f"DEBUG: Available attributes: {[attr for attr in dir(actual_env) if not attr.startswith('_')]}")
                except Exception as debug_e:
                    print(f"DEBUG: Error getting env info: {debug_e}")

            try:
                with open(self.log_file, 'a') as f:
                    # Log comprehensive metrics
                    f.write(f"{current_timestep},{mean_reward:.2f},{episode_length:.2f},{balance:.2f},{portfolio_value:.2f},{pnl:.2f},{pnl_pct:.4f}\n")
                    f.flush()  # Ensure data is written immediately
            except Exception as e:
                print(f"Warning: Could not write to evaluation log: {e}")

        return result
    
    def _on_rollout_end(self):
        """Capture portfolio metrics during evaluation"""
        # Try to capture metrics from eval environment during evaluation
        # This is called after each rollout, including evaluation rollouts
        try:
            if hasattr(self, 'eval_env') and self.eval_env is not None:
                env = self.eval_env
                # Handle different types of environments
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    # For vectorized environments, get the first environment
                    wrapped_env = env.envs[0]
                    
                    # Unwrap Monitor and MetricsTrackingWrapper to get to TradingEnvironment
                    actual_env = wrapped_env
                    while hasattr(actual_env, 'env'):
                        actual_env = actual_env.env
                        if isinstance(actual_env, MetricsTrackingWrapper):
                            actual_env = actual_env.env
                            break  # We've reached the TradingEnvironment
                else:
                    # Direct environment access
                    actual_env = env
                    # Unwrap if needed
                    while hasattr(actual_env, 'env'):
                        actual_env = actual_env.env
                        if isinstance(actual_env, MetricsTrackingWrapper):
                            actual_env = actual_env.env
                            break  # We've reached the TradingEnvironment
                
                if hasattr(actual_env, 'balance'):
                    self.last_eval_metrics['balance'] = float(actual_env.balance)
                if hasattr(actual_env, 'portfolio_values') and len(actual_env.portfolio_values) > 0:
                    portfolio_value = float(actual_env.portfolio_values[-1])
                    self.last_eval_metrics['portfolio_value'] = portfolio_value
                    if len(actual_env.portfolio_values) > 1:
                        initial_value = float(actual_env.portfolio_values[0])
                        if initial_value > 0:
                            pnl = portfolio_value - initial_value
                            self.last_eval_metrics['pnl'] = pnl
                            self.last_eval_metrics['pnl_pct'] = (pnl / initial_value) * 100
                elif hasattr(actual_env, 'initial_balance'):
                    # If no portfolio_values, use initial_balance as fallback
                    self.last_eval_metrics['balance'] = float(actual_env.initial_balance)
                    self.last_eval_metrics['portfolio_value'] = float(actual_env.initial_balance)
        except Exception as e:
            # Silently fail - metrics will be estimated from reward if needed
            print(f"DEBUG: Error in _on_rollout_end: {e}")
            pass
        
        return super()._on_rollout_end()
    
    def _on_evaluation_end(self):
        """Called after evaluation completes - capture final metrics here and write to log"""
        # This is the best place to capture metrics as evaluation just finished
        try:
            # Store episode length if available
            if hasattr(self, 'last_episode_lengths') and self.last_episode_lengths is not None and len(self.last_episode_lengths) > 0:
                self.last_eval_metrics['episode_length'] = np.mean(self.last_episode_lengths)
            
            # Try to get metrics from the evaluation environment
            if hasattr(self, 'eval_env') and self.eval_env is not None:
                env = self.eval_env
                # Handle different types of environments
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    # For vectorized environments, get the first environment
                    wrapped_env = env.envs[0]
                    
                    # Unwrap Monitor and MetricsTrackingWrapper to get to TradingEnvironment
                    actual_env = wrapped_env
                    while hasattr(actual_env, 'env'):
                        actual_env = actual_env.env
                        if isinstance(actual_env, MetricsTrackingWrapper):
                            actual_env = actual_env.env
                            break  # We've reached the TradingEnvironment
                else:
                    # Direct environment access
                    actual_env = env
                    # Unwrap if needed
                    while hasattr(actual_env, 'env'):
                        actual_env = actual_env.env
                        if isinstance(actual_env, MetricsTrackingWrapper):
                            actual_env = actual_env.env
                            break  # We've reached the TradingEnvironment
                
                # Try to get metrics from the environment's latest info
                if hasattr(actual_env, 'balance'):
                    self.last_eval_metrics['balance'] = float(actual_env.balance)
                if hasattr(actual_env, 'portfolio_values') and len(actual_env.portfolio_values) > 0:
                    portfolio_value = float(actual_env.portfolio_values[-1])
                    self.last_eval_metrics['portfolio_value'] = portfolio_value
                    if len(actual_env.portfolio_values) > 1:
                        initial_value = float(actual_env.portfolio_values[0])
                        if initial_value > 0:
                            pnl = portfolio_value - initial_value
                            self.last_eval_metrics['pnl'] = pnl
                            self.last_eval_metrics['pnl_pct'] = (pnl / initial_value) * 100
                    else:
                        self.last_eval_metrics['pnl'] = 0.0
                        self.last_eval_metrics['pnl_pct'] = 0.0
                elif hasattr(actual_env, 'initial_balance'):
                    self.last_eval_metrics['balance'] = float(actual_env.initial_balance)
                    self.last_eval_metrics['portfolio_value'] = float(actual_env.initial_balance)
                    self.last_eval_metrics['pnl'] = 0.0
                    self.last_eval_metrics['pnl_pct'] = 0.0
            
            # Write to log file after evaluation completes
            # This ensures we log even if _on_step() conditions aren't met
            if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
                current_timestep = self.num_timesteps
                mean_reward = self.last_mean_reward
                
                # Get episode length
                episode_length = self.last_eval_metrics.get('episode_length', 0.0)
                if episode_length == 0.0 and hasattr(self, 'last_episode_lengths') and self.last_episode_lengths is not None and len(self.last_episode_lengths) > 0:
                    episode_length = np.mean(self.last_episode_lengths)
                
                # Get portfolio metrics
                balance = self.last_eval_metrics.get('balance', 0.0)
                portfolio_value = self.last_eval_metrics.get('portfolio_value', 0.0)
                pnl = self.last_eval_metrics.get('pnl', 0.0)
                pnl_pct = self.last_eval_metrics.get('pnl_pct', 0.0)
                
                # If portfolio metrics are still 0, try to estimate from reward
                if portfolio_value == 0.0 and hasattr(self, 'eval_env') and self.eval_env is not None:
                    try:
                        env = self.eval_env
                        if hasattr(env, 'envs') and len(env.envs) > 0:
                            wrapped_env = env.envs[0]
                            # Unwrap Monitor and MetricsTrackingWrapper to get to TradingEnvironment
                            actual_env = wrapped_env
                            while hasattr(actual_env, 'env'):
                                actual_env = actual_env.env
                                if isinstance(actual_env, MetricsTrackingWrapper):
                                    actual_env = actual_env.env
                                    break
                        
                            if hasattr(actual_env, 'initial_balance'):
                                initial_balance = float(actual_env.initial_balance)
                                # Estimate: reward = log(1 + return) * 100
                                estimated_return = np.exp(mean_reward / 100.0) - 1.0
                                portfolio_value = initial_balance * (1 + estimated_return)
                                pnl = portfolio_value - initial_balance
                                pnl_pct = estimated_return * 100
                                balance = portfolio_value
                    except Exception:
                        pass
                
                # Write to log file
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(f"{current_timestep},{mean_reward:.2f},{episode_length:.2f},{balance:.2f},{portfolio_value:.2f},{pnl:.2f},{pnl_pct:.4f}\n")
                        f.flush()  # Ensure data is written immediately
                except Exception as e:
                    print(f"Warning: Could not write to evaluation log: {e}")
        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Error in _on_evaluation_end: {e}")
        
        return super()._on_evaluation_end()



class RLTrainingProgressCallback(BaseCallback):
    """Custom callback to show detailed RL training progress using tqdm"""

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
        self.pbar = None

    def _on_training_start(self):
        """Called at the beginning of training"""
        self.start_time = time.time()
        print("🚀 Starting RL Training")
        print("=" * 60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Evaluation frequency: {self.eval_freq:,} timesteps")
        print()

        # Initialize tqdm progress bar
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="📈 RL Training",
            unit="steps",
            unit_scale=True,
            ncols=120,
            bar_format='{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

    def _on_step(self):
        """Called at each step"""
        current_timestep = self.num_timesteps

        # Get additional metrics for postfix
        postfix_info = {}
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            try:
                if 'train/loss' in self.model.logger.name_to_value:
                    postfix_info['loss'] = f"{self.model.logger.name_to_value['train/loss']:.4f}"
                if 'train/value_loss' in self.model.logger.name_to_value:
                    postfix_info['v_loss'] = f"{self.model.logger.name_to_value['train/value_loss']:.4f}"
                if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
                    postfix_info['pg_loss'] = f"{self.model.logger.name_to_value['train/policy_gradient_loss']:.4f}"
            except:
                pass

        # Update progress bar
        if self.pbar:
            self.pbar.set_postfix(postfix_info)
            self.pbar.update(current_timestep - self.pbar.n)

        # Show evaluation results
        if self.eval_freq > 0 and current_timestep % self.eval_freq == 0 and current_timestep > 0:
            if hasattr(self, 'last_mean_reward'):
                eval_elapsed = time.time() - (self.last_eval_time or self.start_time)
                print(f"\n🎯 Evaluation at {current_timestep:,} timesteps:")
                if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
                    print(f"   Mean Reward: {self.last_mean_reward:.2f}")
                print(f"   Evaluation Time: {eval_elapsed:.1f}s")

        return True

    def _on_training_end(self):
        """Called at the end of training"""
        if self.pbar:
            self.pbar.close()

        total_time = time.time() - self.start_time
        print()
        print("✅ RL Training Completed!")
        print("=" * 60)
        print(f"Total training time: {total_time:.1f}s")
        print(f"Average timesteps/second: {self.total_timesteps/total_time:.1f}")
        print()

# The TradingEnvironment class has been moved to trading_environment.py
# This file now imports it from the unified module

def make_env(df, rank, debug=False):
    """Create environment with error handling for parallel processing"""
    def _init():
        try:
            env = TradingEnvironment(df, debug=debug)
            env = Monitor(env, f"./rl_logs/monitor_{rank}")
            return env
        except Exception as e:
            print(f"Error creating environment {rank}: {e}")
            raise
    return _init

def preprocess_data(df, scaler=None, fit_scaler=True):
    """
    Предобработка данных: очистка от дубликатов и выбросов
    НЕ нормализует индикаторы - нормализация происходит в среде

    Args:
        df: DataFrame для обработки
        scaler: Не используется (оставлен для совместимости)
        fit_scaler: Не используется (оставлен для совместимости)

    Returns:
        df: Обработанный DataFrame
        scaler: None (нормализация происходит в среде)
    """
    print(f"Preprocessing data: {len(df)} initial rows")

    # Удаление дубликатов
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['timestamp'] if 'timestamp' in df.columns else ['Open time'] if 'Open time' in df.columns else None)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")

    # Обработка экстремальных значений (только на train данных, не на val/test)
    # Для val/test используем те же пороги что были на train
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    if fit_scaler:  # Только на train данных
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

    # УБРАНА нормализация индикаторов - теперь она происходит только в среде
    # Это устраняет двойную нормализацию и несоответствия между обучением и оценкой

    print(f"Preprocessing complete: {len(df)} final rows")
    return df, None

def train_rl_agent(data_path, total_timesteps=200000, eval_freq=10000, n_envs=8, train_test_split=0.8):
    """
    Train RL agent using PPO with parallel environments

    БЕЗ утечки данных: данные разделяются на train/test по времени,
    скейлер обучается только на train данных.

    Recommendations for timesteps:
    - Quick testing: 50,000 - 100,000 steps
    - Development: 200,000 - 500,000 steps
    - Production: 1,000,000 - 5,000,000 steps

    Note: More timesteps generally lead to better performance, but diminishing returns
    after 1-2M steps for most trading tasks. The optimal depends on:
    - Complexity of the trading strategy
    - Amount of training data
    - Market conditions variability

    Args:
        data_path: Path to CSV data file
        total_timesteps: Total training timesteps
        eval_freq: Frequency of evaluation (default: 10000 steps)
        n_envs: Number of parallel environments
        train_test_split: Train/test split ratio
    """
    """
    Train RL agent using PPO with parallel environments
    
    БЕЗ утечки данных: данные разделяются на train/test по времени,
    скейлер обучается только на train данных.

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

    # Разделение данных на train/test по времени (walk-forward split)
    split_idx = int(len(df) * train_test_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Data split: {len(train_df)} train rows, {len(test_df)} test rows")

    # Предобработка train данных (с обучением скейлера)
    train_df, scaler = preprocess_data(train_df, fit_scaler=True)
    
    # Предобработка test данных (с применением обученного скейлера)
    test_df, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)

    # Clean data (remove any remaining NaN)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    print(f"After dropping NaN: {len(train_df)} train rows, {len(test_df)} test rows")

    if train_df.empty:
        raise ValueError("All training data was removed after preprocessing")
    
    if test_df.empty:
        print("WARNING: All test data was removed after preprocessing. Using train data for evaluation.")
        test_df = train_df.copy()

    # Check minimum data requirements for RL training
    if len(train_df) < 1000:
        print(f"WARNING: Limited data for RL training ({len(train_df)} rows). Consider using more historical data.")

    print(f"Final dataset: {len(train_df)} train rows, {len(test_df)} test rows, {len(train_df.columns)} features")
    
    # Используем train данные для обучения
    df = train_df

    print(f"Creating {n_envs} parallel RL environments...")
    # Create parallel environments using SubprocVecEnv (debug=False for performance)
    envs = [make_env(df, i, debug=False) for i in range(n_envs)]
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
        ent_coef=0.5,  # Увеличено с 0.2 до 0.5 для большей exploration
        vf_coef=0.8,  # Больше внимания к value function
        max_grad_norm=0.3,  # Меньше ограничение градиентов
        verbose=0,  # Отключаем вывод stable_baselines3
        tensorboard_log="./rl_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # Более мощная архитектура
            activation_fn=torch.nn.ReLU,
            ortho_init=False,  # Отключаем ортогональную инициализацию для лучшей регуляризации
        )
    )

    # Enhanced TensorBoard callback for comprehensive monitoring
    class EnhancedTensorBoardCallback(BaseCallback):
        """Enhanced callback for comprehensive TensorBoard logging"""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_wins = []
            self.learning_rates = []
            self.gradient_norms = []
            self.entropy_values = []
            self.value_losses = []
            self.policy_losses = []

        def _on_training_start(self):
            """Initialize comprehensive logging"""
            print("📊 Enhanced TensorBoard logging enabled")
            print("   Tracking: rewards, lengths, learning rate, gradients, entropy, losses")

        def _on_step(self):
            """Log comprehensive training metrics"""
            # Calculate progress percentage
            current_timestep = self.num_timesteps
            total_timesteps = self.locals.get('total_timesteps', 50000)
            progress = min(current_timestep / total_timesteps, 1.0)
            self.logger.record('train/progress', progress * 100)  # Log as percentage

            # Log learning rate
            if hasattr(self.model, 'learning_rate'):
                current_lr = self.model.learning_rate
                if isinstance(current_lr, (int, float)):
                    self.learning_rates.append(current_lr)
                    self.logger.record('train/learning_rate', current_lr)

            # Log gradient norms
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                if 'train/grad_norm' in self.model.logger.name_to_value:
                    grad_norm = self.model.logger.name_to_value['train/grad_norm']
                    self.gradient_norms.append(grad_norm)
                    self.logger.record('train/gradient_norm', grad_norm)

                # Log entropy
                if 'train/entropy_loss' in self.model.logger.name_to_value:
                    entropy = self.model.logger.name_to_value['train/entropy_loss']
                    self.entropy_values.append(entropy)
                    self.logger.record('train/entropy', entropy)

                # Log value and policy losses
                if 'train/value_loss' in self.model.logger.name_to_value:
                    v_loss = self.model.logger.name_to_value['train/value_loss']
                    self.value_losses.append(v_loss)
                    self.logger.record('train/value_loss', v_loss)

                if 'train/policy_loss' in self.model.logger.name_to_value:
                    p_loss = self.model.logger.name_to_value['train/policy_loss']
                    self.policy_losses.append(p_loss)
                    self.logger.record('train/policy_loss', p_loss)

            return True

        def _on_rollout_end(self):
            """Log episode statistics"""
            # Use more reliable method to get episode info from Monitor logs
            # Instead of accessing locals which may not be reliable with VecEnv
            try:
                # Get episode info from infos if available
                if 'infos' in self.locals:
                    infos = self.locals['infos']
                    for info in infos:
                        if isinstance(info, dict) and 'episode' in info:
                            ep_info = info['episode']
                            if 'r' in ep_info:
                                ep_reward = ep_info['r']
                                ep_length = ep_info.get('l', 0)
                                self.episode_rewards.append(ep_reward)
                                self.episode_lengths.append(ep_length)
                                self.logger.record('rollout/ep_rew', ep_reward)
                                self.logger.record('rollout/ep_len', ep_length)
                                win = 1 if ep_reward > 0 else 0
                                self.episode_wins.append(win)
                                self.logger.record('rollout/win_rate', win)
                                
                                # Also log portfolio metrics if available
                                if 'portfolio_value' in info:
                                    self.logger.record('rollout/portfolio_value', info['portfolio_value'])
                                if 'total_return' in info:
                                    self.logger.record('rollout/total_return', info['total_return'])
                                if 'total_trades' in info:
                                    self.logger.record('rollout/total_trades', info['total_trades'])
            except Exception as e:
                # Fallback: silently skip if we can't get episode info
                # Monitor will log this separately anyway
                pass

        def _on_training_end(self):
            """Log final statistics"""
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards)
                max_reward = np.max(self.episode_rewards)
                min_reward = np.min(self.episode_rewards)
                std_reward = np.std(self.episode_rewards)

                self.logger.record('train/avg_reward', avg_reward)
                self.logger.record('train/max_reward', max_reward)
                self.logger.record('train/min_reward', min_reward)
                self.logger.record('train/reward_std', std_reward)

            if self.episode_lengths:
                avg_length = np.mean(self.episode_lengths)
                self.logger.record('train/avg_length', avg_length)

            if self.episode_wins and len(self.episode_wins) > 0:
                win_rate = np.mean(self.episode_wins)
                self.logger.record('train/win_rate', win_rate)

            print(f"\n📊 TensorBoard Summary:")
            print(f"   Episodes: {len(self.episode_rewards)}")
            print(f"   Avg Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"   Avg Length: {np.mean(self.episode_lengths):.1f}")
            print(f"   Win Rate: {np.mean(self.episode_wins)*100:.1f}%")
            print(f"   Learning Rate: {np.mean(self.learning_rates):.6f}")
            print(f"   Gradient Norm: {np.mean(self.gradient_norms):.4f}")
            print(f"   Entropy: {np.mean(self.entropy_values):.4f}")

    # Add enhanced TensorBoard callback
    tensorboard_callback = EnhancedTensorBoardCallback()

    # Evaluation callback - use different data slices for each evaluation episode
    # This provides more diverse evaluation and better assessment of generalization
    eval_data_size = max(1000, len(df) // 10)
    eval_count = 0

    def make_eval_env():
        nonlocal eval_count, test_df
        eval_count += 1
        # Use test data for evaluation (different from training data)
        # Use systematic different slices for each evaluation to avoid identical results
        eval_df_to_use = test_df if len(test_df) >= eval_data_size else df
        max_start = len(eval_df_to_use) - eval_data_size
        if max_start > 0:
            start_idx = (eval_count - 1) * (max_start // 10) % (max_start + 1)
        else:
            start_idx = 0
        eval_df = eval_df_to_use.iloc[start_idx:start_idx + eval_data_size].reset_index(drop=True)
        # Create environment with unique ID to avoid monitor conflicts (debug=False for performance)
        env = TradingEnvironment(eval_df, debug=False)
        # Wrap with MetricsTrackingWrapper to capture metrics from info dict
        env = MetricsTrackingWrapper(env)
        return Monitor(env, "./rl_logs/")
    
    # Use SubprocVecEnv for evaluation to match training env type and avoid warnings
    # Using single process for eval is fine - it matches the type requirement
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
        callback=[progress_callback, eval_callback, checkpoint_callback, tensorboard_callback],
        progress_bar=False  # Disabled to avoid conflicts with custom progress bar
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
        # For validation, we should use a pre-trained scaler or fit on validation data separately
        # For simplicity, fit on validation data (but ideally should use train scaler)
        df, _ = preprocess_data(df, fit_scaler=True)

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
    
    # Diagnostic: Check model policy
    try:
        import torch
        # Create dummy observation to check policy output
        dummy_obs = np.zeros((1, model.observation_space.shape[0]), dtype=np.float32)
        obs_tensor = torch.tensor(dummy_obs, dtype=torch.float32)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            action_probs = dist.distribution.probs[0].detach().cpu().numpy()
            print(f"📊 Model diagnostic - Action probabilities on zero observation:")
            action_names = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
            for name, prob in zip(action_names, action_probs):
                print(f"   {name}: {prob:.4f}")
            if action_probs[0] > 0.9:
                print("   ⚠️  WARNING: Model strongly prefers HOLD action. May need retraining.")
    except Exception as e:
        print(f"   Could not perform model diagnostic: {e}")

    df = pd.read_csv(data_path)
    # Используем ту же предобработку что и для обучения (без нормализации индикаторов)
    df, _ = preprocess_data(df, fit_scaler=True)
    print(f"Evaluation preprocessing complete: {len(df)} final rows")

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
        current_step = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, (int, np.integer)):
                action = np.clip(action, 0, env.action_space.n - 1)
            else:
                action = 0

            # Debug: print actions and action probabilities for first episode
            if episode == 0 and current_step < 10:  # Only for first episode, first 10 steps
                # Get action probabilities for diagnostics
                try:
                    import torch
                    obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
                    with torch.no_grad():
                        dist = model.policy.get_distribution(obs_tensor)
                        action_probs = dist.distribution.probs[0].detach().cpu().numpy()
                        action_names = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
                        prob_str = ", ".join([f"{name}={prob:.3f}" for name, prob in zip(action_names, action_probs)])
                        print(f"DEBUG: Step {current_step}, Action {action} ({action_names[action]}), Position {env.position:.6f}, Balance {env.balance:.2f}")
                        print(f"DEBUG: Action probabilities: {prob_str}")
                except Exception as e:
                    print(f"DEBUG: Step {current_step}, Action {action}, Position {env.position:.6f}, Balance {env.balance:.2f}")
                    print(f"DEBUG: Could not get action probabilities: {e}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track action distribution
            action_counts[action] = action_counts.get(action, 0) + 1

            # Track trades (all trading actions)
            if action in [1, 2, 3, 4]:  # All trading actions
                trade_count += 1
                if episode == 0 and trade_count <= 5:  # Debug first few trades
                    print(f"DEBUG: Trade {trade_count}, Action {action}, Success: {info.get('action_taken', action) == action}")

            current_price = eval_df.iloc[min(env.current_step, len(eval_df)-1)].get('close', eval_df.iloc[min(env.current_step, len(eval_df)-1)].get('Close'))
            portfolio_value = env.balance + env.margin_locked + env.position * current_price
            portfolio_history.append(portfolio_value)
            current_step += 1

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

        # Print action distribution for first episode
        if episode == 0:
            action_names = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
            total_actions = sum(action_counts.values())
            action_dist = ", ".join([f"{name}={count}({count/total_actions*100:.1f}%)" 
                                     for name, count in zip(action_names, [action_counts.get(i, 0) for i in range(5)])])
            print(f"Episode {episode + 1}: Return={total_return*100:.2f}%, Sharpe={sharpe_ratio:.2f}, "
                  f"MaxDD={max_drawdown*100:.2f}%, Trades={trade_count}")
            print(f"  Action distribution: {action_dist}")
        else:
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
    parser.add_argument("--timesteps", type=int, default=500000,
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
