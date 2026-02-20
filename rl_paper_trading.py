#!/usr/bin/env python3
"""
RL Paper Trading Module
Tests trained RL agent on historical data without real transactions
"""

import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pickle
import time
from datetime import datetime

# Import the unified TradingEnvironment
from enhanced_trading_environment import EnhancedTradingEnvironment as TradingEnvironment

def calculate_sharpe_ratio(returns, risk_free_rate=0.000006811):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()

def run_rl_paper_trading(model_path, data_path, initial_balance=10000, n_episodes=5, use_random_slices=False):
    """
    Run RL agent on historical data for paper trading simulation

    Args:
        model_path: Path to trained RL model (.zip)
        data_path: Path to historical data CSV
        initial_balance: Starting balance in USD
        n_episodes: Number of episodes to run (for slice testing)
        use_random_slices: If True, run on random slices like training evaluation
    """
    print("🚀 Starting RL Paper Trading Simulation")
    print("=" * 50)
    print(f"Mode: {'Random slices evaluation' if use_random_slices else 'Full data simulation'}")
    if use_random_slices:
        print(f"Episodes: {n_episodes}")

    # Load model
    print(f"Loading RL model from {model_path}...")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Load data
    print(f"Loading historical data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        df = df.dropna()
        print(f"✅ Loaded {len(df)} rows of data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

    # Support both 'close' and 'Close' column names
    if 'close' not in df.columns and 'Close' in df.columns:
        df['close'] = df['Close']

    # Check required columns
    required_columns = ['close', 'ATR_15', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'OBV', 'AD', 'MFI_15']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return None

    if use_random_slices:
        # Run multiple episodes on random slices (like training evaluation)
        eval_data_size = max(10000, len(df) // 10)  # 10% of data or 10k, whichever larger
        max_start = len(df) - eval_data_size

        episode_results = []
        episode_summaries = []

        print(f"\nRunning {n_episodes} episodes on random data slices...")
        print(f"Slice size: {eval_data_size} rows")
        print("-" * 50)

        for episode in range(n_episodes):
            # Use random slice for each episode
            start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
            eval_df = df.iloc[start_idx:start_idx + eval_data_size].reset_index(drop=True)

            print(f"Episode {episode + 1}: data slice {start_idx}-{start_idx + eval_data_size}")

            # Initialize environment with same parameters as training
            env = TradingEnvironment(eval_df, initial_balance=initial_balance, transaction_fee=0.0005)

            # Run single episode
            result = _run_single_episode(env, model, initial_balance, eval_df, episode_number=episode+1, verbose=False)
            episode_results.append(result)

        # Aggregate results
        all_returns = [r['total_return'] for r in episode_results]
        all_sharpe = [r['sharpe_ratio'] for r in episode_results]
        all_drawdowns = [r['max_drawdown'] for r in episode_results]

        print("\n" + "=" * 60)
        print("🎯 AGGREGATED RESULTS ACROSS EPISODES")
        print("=" * 60)
        print(f"Episodes run: {n_episodes}")
        print(f"Avg return: {np.mean(all_returns):.2f}% ± {np.std(all_returns):.2f}%")
        print(f"Avg Sharpe ratio: {np.mean(all_sharpe):.4f}")

        best_idx = np.argmax(all_returns)
        print(f"Best episode: {best_idx + 1} ({all_returns[best_idx]:.2f}%)")
        worst_idx = np.argmin(all_returns)
        print(f"Worst episode: {worst_idx + 1} ({all_returns[worst_idx]:.2f}%)")

        print(f"Sharpe ratio: Avg ± Std: {np.mean(all_sharpe):.4f} ± {np.std(all_sharpe):.4f}")
        print(f"Max drawdown: Avg ± Std: {np.mean(all_drawdowns)*100:.2f}% ± {np.std(all_drawdowns)*100:.2f}%")

        # Overall assessment
        avg_return = np.mean(all_returns)
        if avg_return > 0:
            print(f"\n✅ MODEL ASSESSMENT: PROFITABLE (Avg return: +{avg_return:.2f}%)")
        elif avg_return > -20:
            print(f"\n⚠️ MODEL ASSESSMENT: MODERATE LOSSES (Avg return: {avg_return:.2f}%) - Consider retraining")
        else:
            print(f"\n❌ MODEL ASSESSMENT: SIGNIFICANT LOSSES (Avg return: {avg_return:.2f}%) - Model needs major fixes")

        # Save aggregated results
        aggregated_results = {
            'episode_results': episode_results,
            'aggregated': {
                'avg_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'avg_sharpe': np.mean(all_sharpe),
                'avg_drawdown': np.mean(all_drawdowns),
                'assessment': 'profitable' if avg_return > 0 else 'unprofitable'
            }
        }

        with open('rl_paper_trading_aggregated_results.pkl', 'wb') as f:
            pickle.dump(aggregated_results, f)

        print("\n📊 Aggregated results saved to 'rl_paper_trading_aggregated_results.pkl'")
        return aggregated_results

    else:
        # Original full data simulation
        print("Initializing trading environment...")
        # Ensure same parameters as in train_rl.py
        env = TradingEnvironment(df, initial_balance=initial_balance, transaction_fee=0.0005)

        # Run full data simulation
        results = _run_single_episode(env, model, initial_balance, df, episode_number=1, verbose=True)
        return results

def _correct_action(action, env_position, env):
    """
    Correct action based on current position to prevent invalid trades
    NOTE: Only correct truly invalid actions, let the environment handle the rest
    """
    original_action = action
    
    # If trying to sell long but no long position, hold instead
    if action == 2 and env_position <= 0:
        return 0  # Hold
    
    # If trying to cover short but no short position, hold instead
    if action == 4 and env_position >= 0:
        return 0  # Hold
    
    # NOTE: Removed blocking of repeated buys/shorts - let environment handle position limits
    # The environment has its own position size limits
    
    # Allow actions that are valid for the current position
    return action

def _run_single_episode(env, model, initial_balance, df, episode_number=1, verbose=True):
    """Run a single episode"""
    # Trading simulation
    trades = []
    portfolio_history = []

    state, _ = env.reset()
    done = False
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution
    
    # Track consecutive same actions to detect churning
    consecutive_actions = []
    max_consecutive_same = 0

    while not done and step_count < len(df) - 1:  # Safety limit
        step_count += 1

        # Get current price
        current_price = df.iloc[min(env.current_step, len(df)-1)].get('close', df.iloc[min(env.current_step, len(df)-1)].get('Close'))

        # Get action from RL model
        # Use stochastic sampling to allow exploration of all actions
        # Deterministic mode always picks the highest probability action (biased to SELL_SHORT)
        action, _ = model.predict(state, deterministic=False)
        action = int(action)  # Ensure action is integer
        
        # Validate action is within bounds
        if action < 0 or action >= env.action_space.n:
            if verbose and step_count <= 10:
                print(f"  ⚠️ Invalid action {action}, using Hold (0)")
            action = 0  # Default to Hold if action is invalid

        # Apply intelligent action correction
        corrected_action = _correct_action(action, env.position, env)
        if corrected_action != action and verbose and step_count <= 20:
            print(f"  🛡️ Action corrected: {action} → {corrected_action} (position: {env.position:.6f})")
        action = corrected_action

        # Log consecutive actions to detect churning
        if consecutive_actions and consecutive_actions[-1] == action:
            consecutive_count = 1
            for i in range(len(consecutive_actions) - 2, -1, -1):
                if consecutive_actions[i] == action:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count > max_consecutive_same:
                max_consecutive_same = consecutive_count
        consecutive_actions.append(action)

        action_counts[action] += 1

        # Enhanced debugging for first few steps
        if step_count <= 10 and verbose:
            position_type = "NONE"
            if env.position > 0:
                position_type = "LONG"
            elif env.position < 0:
                position_type = "SHORT"
            
            print(f"  Step {step_count}:")
            print(f"    State[0:3]: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}]")
            print(f"    Position: {env.position:.6f} ({position_type})")
            print(f"    Balance: ${env.balance:.2f}")
            print(f"    Price: ${current_price:.2f}")
            print(f"    Action: {action}")
            print(f"    Portfolio: ${env.balance + env.margin_locked + env.position * current_price:.2f}")

        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update state
        state = next_state

        # Record portfolio state - calculate correctly for margin trading
        # Portfolio = Available Balance + Margin Locked + Position_Value
        # Position value should account for short positions (negative for shorts)
        current_price = df.iloc[min(env.current_step, len(df)-1)].get('close', df.iloc[min(env.current_step, len(df)-1)].get('Close'))
        portfolio_value = env.balance + env.margin_locked + env.position * current_price
        portfolio_history.append({
            'step': step_count,
            'price': current_price,
            'balance': env.balance,
            'position': env.position,
            'portfolio_value': portfolio_value,
            'action': action
        })

        # Progress logging for single episodes
        if verbose and step_count % 1000 == 0:
            pnl = portfolio_value - initial_balance
            pct_change = (pnl / initial_balance) * 100
            print(f"  Step {step_count}: Portfolio=${portfolio_value:.2f} (PnL: {pct_change:+.2f}%)")

        # Early stop for very long losing episodes - more reasonable threshold
        if portfolio_value < initial_balance * 0.85 and step_count > 100:  # 15% loss threshold like training
            if verbose:
                print(f"Early stop: Portfolio < 85% initial balance at step {step_count}")
            break

    # Calculate episode metrics
    final_portfolio = portfolio_history[-1]['portfolio_value'] if portfolio_history else initial_balance
    total_pnl = final_portfolio - initial_balance
    total_return = (total_pnl / initial_balance) * 100

    # Sharpe and drawdown
    if len(portfolio_history) > 1:
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        excess_returns = returns - 0.000006811  # Daily risk-free rate approximation
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        max_drawdown = calculate_max_drawdown(portfolio_values)
    else:
        sharpe_ratio = 0
        max_drawdown = 0

    # Print episode summary
    print(f"\n📊 Episode {episode_number} Summary:")
    print(f"  Steps: {step_count}")
    print(f"  Return: {total_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Final Portfolio: ${final_portfolio:.2f}")
    print(f"  Action Distribution: {action_counts}")
    print(f"  Max Consecutive Same Actions: {max_consecutive_same}")

    results = {
        'initial_balance': initial_balance,
        'final_portfolio': final_portfolio,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'steps': step_count,
        'action_counts': action_counts,
        'portfolio_history': portfolio_history,
        'max_consecutive_same': max_consecutive_same
    }

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Paper Trading Simulation")
    parser.add_argument("--model", default="ppo_trading_agent",
                       help="Path to trained RL model (without .zip extension)")
    parser.add_argument("--data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to historical data")
    parser.add_argument("--balance", type=float, default=10000,
                       help="Initial balance")
    parser.add_argument("--slices", action="store_true",
                       help="Run evaluation on random data slices (recommended for validation)")
    parser.add_argument("--n_episodes", type=int, default=5,
                       help="Number of episodes for slice evaluation (default: 5)")

    args = parser.parse_args()

    results = run_rl_paper_trading(args.model, args.data, args.balance, use_random_slices=args.slices, n_episodes=args.n_episodes)

    if results:
        print("\n✅ RL Paper trading simulation completed successfully!")
    else:
        print("\n❌ RL Paper trading simulation failed!")
