#!/usr/bin/env python3
"""
Debug script to analyze model behavior and identify issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import json
import os

from trading_environment import TradingEnvironment

def debug_model_behavior(model_path, test_data_path):
    """Debug the model's behavior step by step"""
    print("🔍 Debugging model behavior...")
    
    # Load model and data
    print("Loading model...")
    model = PPO.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    print("Loading test data...")
    test_data = pd.read_csv(test_data_path)
    print(f"✅ Test data loaded with {len(test_data)} rows")
    
    # Create environment with debug enabled
    env = TradingEnvironment(test_data.head(500), episode_length=100, debug=True)  # Small sample for debugging
    env = Monitor(env)
    
    print("\n" + "="*60)
    print(".DEBUGGING SINGLE EPISODE")
    print("="*60)
    
    # Run one episode with detailed logging
    obs, info = env.reset()
    print(f"Initial balance: {env.unwrapped.initial_balance}")
    print(f"Initial portfolio: {info.get('portfolio_value', 'N/A')}")
    print(f"Initial position: {env.unwrapped.position}")
    print(f"Initial state shape: {obs.shape}")
    
    done = False
    step_count = 0
    actions_taken = []
    balances = []
    positions = []
    portfolio_values = []
    rewards = []
    
    while not done and step_count < 50:  # Limit for debugging
        print(f"\n--- Step {step_count} ---")
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        print(f"Model action: {action}")
        
        # Get action probabilities for debugging
        from stable_baselines3.common.distributions import kl_divergence
        import torch
        obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        dist = model.policy.get_distribution(obs_tensor)
        action_probs = dist.distribution.probs[0].detach().numpy()
        action_names = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
        print(f"Action probabilities: {dict(zip(action_names, action_probs))}")
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Balance: {env.unwrapped.balance}")
        print(f"Position: {env.unwrapped.position}")
        print(f"Portfolio value: {info.get('portfolio_value', 'N/A')}")
        print(f"Current price: {info.get('current_price', 'N/A')}")
        print(f"Total trades: {info.get('total_trades', 'N/A')}")
        print(f"Action performed: {info.get('action_performed', 'N/A')}")
        
        # Store values for analysis
        # Ensure action is a plain Python int to avoid pandas issues
        action_int = int(action.item()) if hasattr(action, 'item') else int(action)
        actions_taken.append(action_int)
        balances.append(env.unwrapped.balance)
        positions.append(env.unwrapped.position)
        portfolio_values.append(info.get('portfolio_value', env.unwrapped.balance))
        rewards.append(reward)
        
        done = terminated or truncated
        step_count += 1
        
        if step_count >= 50:  # Safety break
            print("Breaking after 50 steps for debugging")
            break
    
    print(f"\nFinal balance: {env.unwrapped.balance}")
    print(f"Final position: {env.unwrapped.position}")
    print(f"Final portfolio: {info.get('portfolio_value', 'N/A')}")
    print(f"Episode terminated: {terminated}, truncated: {truncated}")
    
    # Analyze the results
    print("\n" + "="*60)
    print("📊 BEHAVIOR ANALYSIS")
    print("="*60)
    
    print(f"Steps taken: {step_count}")
    print(f"Actions distribution: {dict(pd.Series(actions_taken).value_counts())}")
    print(f"Final return: {(portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]:.4f}")
    print(f"Balance change: {balances[-1] - balances[0]:.2f}")
    
    # Check for common issues
    if abs(balances[-1] - balances[0]) > balances[0] * 0.9:  # Lost more than 90%
        print("❌ CRITICAL: Massive balance loss detected!")
    
    if len(set(actions_taken)) == 1:  # Same action all the time
        print("❌ CRITICAL: Model is stuck repeating the same action!")
    
    if all(abs(r) < 1e-6 for r in rewards):  # All zero rewards
        print("❌ CRITICAL: All rewards are zero!")
    
    if all(pos == 0 for pos in positions):  # Never took positions
        print("❌ CRITICAL: Model never opened any positions!")
    
    # Plot the behavior
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Balance over time
    axes[0, 0].plot(balances)
    axes[0, 0].set_title('Balance Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Balance')
    axes[0, 0].grid(True)
    
    # Position over time
    axes[0, 1].plot(positions)
    axes[0, 1].set_title('Position Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Position Size')
    axes[0, 1].grid(True)
    
    # Portfolio value over time
    axes[1, 0].plot(portfolio_values)
    axes[1, 0].set_title('Portfolio Value Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Portfolio Value')
    axes[1, 0].grid(True)
    
    # Rewards over time
    axes[1, 1].plot(rewards)
    axes[1, 1].set_title('Rewards Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('debug_behavior.png', dpi=300, bbox_inches='tight')
    print("📊 Behavior plots saved to debug_behavior.png")
    
    # Detailed action analysis
    print("\n" + "="*60)
    print("📋 DETAILED ACTION ANALYSIS")
    print("="*60)
    
    action_names = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
    for action_code in range(5):
        action_indices = [i for i, a in enumerate(actions_taken) if a == action_code]
        if action_indices:
            print(f"{action_names[action_code]} ({action_code}): {len(action_indices)} times at steps {action_indices[:10]}{'...' if len(action_indices) > 10 else ''}")
        else:
            print(f"{action_names[action_code]} ({action_code}): 0 times")
    
    # Check environment state
    print("\n" + "="*60)
    print("🔧 ENVIRONMENT STATE CHECK")
    print("="*60)
    
    env_state = env.unwrapped
    print(f"Current step: {env_state.current_step}")
    print(f"Steps in episode: {env_state.steps_in_episode}")
    print(f"Total fees: {env_state.total_fees}")
    print(f"Total trades: {env_state.total_trades}")
    print(f"Win count: {env_state.win_count}")
    print(f"Loss count: {env_state.loss_count}")
    print(f"Total P&L: {env_state.total_pnl}")
    print(f"Margin locked: {env_state.margin_locked}")
    print(f"Short opening fees: {env_state.short_opening_fees}")
    
    return {
        'actions_taken': actions_taken,
        'balances': balances,
        'positions': positions,
        'portfolio_values': portfolio_values,
        'rewards': rewards,
        'final_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    }

def test_environment_standalone():
    """Test the environment without model to check basic functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING ENVIRONMENT STANDALONE")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 40000 + np.random.randn(100).cumsum() * 100,
        'high': 40000 + np.random.randn(100).cumsum() * 100 + 50,
        'low': 40000 + np.random.randn(100).cumsum() * 100 - 50,
        'close': 40000 + np.random.randn(100).cumsum() * 100,
        'volume': np.random.randint(100, 1000, 100),
    })
    
    # Add technical indicators
    df['RSI_15'] = 50 + np.random.randn(100) * 10
    df['BB_15_upper'] = df['close'] * 1.02
    df['BB_15_lower'] = df['close'] * 0.98
    df['ATR_15'] = 100 + np.random.randn(100) * 10
    df['OBV'] = np.random.randn(100).cumsum() * 1000
    df['AD'] = np.random.randn(100).cumsum() * 500
    df['MFI_15'] = 50 + np.random.randn(100) * 5
    
    # Test environment
    env = TradingEnvironment(df, episode_length=50, debug=True)
    obs, info = env.reset()
    
    print(f"Initial balance: {env.initial_balance}")
    print(f"Initial state shape: {obs.shape}")
    
    # Test manual actions
    test_actions = [1, 0, 2, 0, 3, 0, 4, 0]  # Buy, Hold, Sell, Hold, Short, Hold, Cover, Hold
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Manual Test Step {i} - Action {action} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Balance: {env.balance}")
        print(f"Position: {env.position}")
        print(f"Portfolio: {info.get('portfolio_value', env.balance)}")
        print(f"Total trades: {info.get('total_trades', 0)}")
        
        if terminated or truncated:
            print(f"Episode ended - terminated: {terminated}, truncated: {truncated}")
            break
    
    print(f"\nFinal standalone test - Balance: {env.balance}, Position: {env.position}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug model behavior")
    parser.add_argument("--model", default="ppo_trading_agent.zip", help="Path to model")
    parser.add_argument("--test-data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv", 
                       help="Path to test data")
    
    args = parser.parse_args()
    
    # Run environment test first
    test_environment_standalone()
    
    # Run model behavior debug
    results = debug_model_behavior(args.model, args.test_data)
    
    print("\n" + "="*60)
    print("🎯 DEBUGGING COMPLETE")
    print("="*60)
    print("Check debug_behavior.png for visual analysis")
    print(f"Final return: {results['final_return']:.4f} ({results['final_return']*100:.2f}%)")