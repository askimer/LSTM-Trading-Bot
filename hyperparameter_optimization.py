#!/usr/bin/env python3
"""
Hyperparameter Optimization for RL Trading Agent using Optuna
"""

import optuna
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch
import torch.nn as nn
import os
from datetime import datetime
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the unified trading environment
from trading_environment import TradingEnvironment


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    if len(portfolio_values) < 2:
        return 0.0

    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return abs(drawdown.min())


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    """
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.01, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.7, 0.9)
    
    # Neural network architecture
    net_arch_pi = []
    net_arch_vf = []
    
    # Number of hidden layers
    n_layers_pi = trial.suggest_int("n_layers_pi", 1, 3)
    n_layers_vf = trial.suggest_int("n_layers_vf", 1, 3)
    
    # Hidden layer sizes
    for i in range(n_layers_pi):
        net_arch_pi.append(trial.suggest_int(f"pi_layer_{i}_units", 64, 512))
    
    for i in range(n_layers_vf):
        net_arch_vf.append(trial.suggest_int(f"vf_layer_{i}_units", 64, 512))

    # Load and prepare data
    data_path = "btc_usdt_data/full_btc_usdt_data_feature_engineered.csv"
    if not os.path.exists(data_path):
        # If the main data file doesn't exist, try to find any CSV file with BTC data
        import glob
        csv_files = glob.glob("btc_usdt_data/*.csv")
        if csv_files:
            data_path = csv_files[0]
        else:
            raise FileNotFoundError(f"No BTC data file found at {data_path}")
    
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    # Use a subset of data for faster optimization
    subset_size = min(5000, len(df) // 2)  # Use at most half the data
    df_subset = df.tail(subset_size).reset_index(drop=True)
    
    print(f"Using {subset_size} rows for hyperparameter optimization")
    
    # Create environment
    env = make_vec_env(lambda: TradingEnvironment(df_subset, initial_balance=10000), n_envs=1)
    
    # Create model with suggested hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs={
            "net_arch": {"pi": net_arch_pi, "vf": net_arch_vf},
            "activation_fn": nn.ReLU,
        },
        verbose=0,
        tensorboard_log="./rl_tensorboard_optimization/",
    )
    
    # Train the model
    try:
        model.learn(total_timesteps=min(50000, len(df_subset) * 2))  # Limit training steps for optimization
    except Exception as e:
        print(f"Error during training: {e}")
        return -1000  # Return a very low value to indicate failure
    
    # Evaluate the model
    eval_env = TradingEnvironment(df_subset, initial_balance=10000)
    obs, _ = eval_env.reset()
    
    portfolio_values = [eval_env.initial_balance]
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < len(df_subset) - 1:
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = int(action.item())
        else:
            action = int(action)
        
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        current_price = df_subset.iloc[min(eval_env.current_step, len(df_subset)-1)].get('close', 
                               df_subset.iloc[min(eval_env.current_step, len(df_subset)-1)].get('Close'))
        portfolio_value = eval_env.balance + eval_env.margin_locked + eval_env.position * current_price
        portfolio_values.append(portfolio_value)
        
        step_count += 1
    
    # Calculate evaluation metrics
    final_portfolio = portfolio_values[-1]
    total_return = (final_portfolio - eval_env.initial_balance) / eval_env.initial_balance
    
    # Calculate Sharpe ratio
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = calculate_sharpe_ratio(returns)
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate profit factor
    if len(portfolio_values) > 1:
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        if sum(abs(r) for r in negative_returns) != 0:
            profit_factor = sum(positive_returns) / abs(sum(negative_returns))
        else:
            profit_factor = float('inf') if sum(positive_returns) > 0 else 1.0
    else:
        profit_factor = 1.0
    
    # Composite score combining multiple metrics
    # Weights can be adjusted based on importance
    score = (
        total_return * 0.4 +  # 40% weight to total return
        sharpe_ratio * 0.3 +  # 30% weight to risk-adjusted return
        (1 - max_drawdown) * 0.2 +  # 20% weight to drawdown (lower is better)
        min(profit_factor / 10.0, 1.0) * 0.1  # 10% weight to profit factor (capped at 1.0)
    )
    
    print(f"Trial {trial.number}: Score={score:.4f}, Return={total_return:.4f}, Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.4f}")
    
    return score


def run_optimization(n_trials=50, timeout=3600):
    """
    Run hyperparameter optimization using Optuna
    
    Args:
        n_trials: Number of trials to run
        timeout: Timeout in seconds
    """
    print("🚀 Starting hyperparameter optimization...")
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout} seconds")
    
    # Create study
    study = optuna.create_study(direction="maximize")
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print("\n🎉 Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = f"hyperparameter_study_{timestamp}.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to {study_path}")
    
    return study


def get_best_hyperparameters(study_path=None):
    """
    Get the best hyperparameters from a saved study or run new optimization
    
    Args:
        study_path: Path to saved study, if None will run new optimization
    """
    if study_path and os.path.exists(study_path):
        import pickle
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"Loaded study from {study_path}")
    else:
        # Run new optimization
        study = run_optimization(n_trials=20, timeout=1800)  # 20 trials, 30 minutes
    
    return study.best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for RL Trading Agent")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Optimization timeout in seconds")
    parser.add_argument("--load_study", type=str, help="Load existing study from file")
    
    args = parser.parse_args()
    
    if args.load_study:
        best_params = get_best_hyperparameters(args.load_study)
        print("\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        run_optimization(n_trials=args.n_trials, timeout=args.timeout)
