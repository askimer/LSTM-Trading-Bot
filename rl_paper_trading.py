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

# Import the latest TradingEnvironment from train_rl.py
from train_rl import TradingEnvironment

def calculate_sharpe_ratio(returns, risk_free_rate=0.000006811):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()

def run_rl_paper_trading(model_path, data_path, initial_balance=10000):
    """
    Run RL agent on historical data for paper trading simulation

    Args:
        model_path: Path to trained RL model (.zip)
        data_path: Path to historical data CSV
        initial_balance: Starting balance in USD
    """
    print("🚀 Starting RL Paper Trading Simulation")
    print("=" * 50)

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

    # Check required columns
    required_columns = ['close', 'ATR_15', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'OBV', 'AD', 'MFI_15']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return None

    # Initialize environment with same parameters as training
    print("Initializing trading environment...")
    # Ensure same parameters as in train_rl.py
    env = TradingEnvironment(df, initial_balance=initial_balance, transaction_fee=0.0018)

    # Trading simulation
    print("Starting trading simulation...")
    print("-" * 50)

    trades = []
    portfolio_history = []
    
    # Use environment's state instead of separate variables
    # This ensures synchronization with the environment

    state, _ = env.reset()
    done = False
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution

    while not done:
        step_count += 1

        # Save previous state BEFORE executing action
        prev_position = env.position
        prev_balance = env.balance
        prev_entry_price = env.entry_price

        # Get action from RL model
        action, _ = model.predict(state, deterministic=True)
        action = int(action)  # Ensure action is integer
        
        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long (close long), 3=Sell Short (open short), 4=Buy Short (close short)
        # Action 2 (Sell Long) requires an existing long position
        # Action 3 (Sell Short) is for selling without position (opening short)
        # WORKAROUND: If model incorrectly chooses action 2 without position, convert to action 3
        # This is a temporary fix for a poorly trained model - should retrain with proper penalties
        
        original_action = action
        if action == 2 and env.position == 0:
            # Model incorrectly chose "Sell Long" without position
            # Convert to action 3 (Sell Short) which is correct for selling without position
            action = 3
            if step_count <= 10:
                print(f"  ⚠️ Model error corrected: Action 2→3 (Sell Long→Sell Short) at step {step_count}")
                print(f"     Note: Action 2 = close long position, Action 3 = open short position (sell without position)")
        
        # Also handle action 4 (Buy Short/cover) when no short position exists
        if action == 4 and env.position >= 0:
            # Model incorrectly chose "Buy Short" (cover) without short position
            # Convert to action 0 (Hold) as there's nothing to cover
            if step_count <= 10:
                print(f"  ⚠️ Model error corrected: Action 4→0 (Buy Short→Hold) at step {step_count} (no short to cover)")
            action = 0
        
        action_counts[original_action] += 1  # Track original action for statistics
        
        # Debug: Print state and action every 1000 steps
        if step_count % 1000 == 0 and step_count <= 2000:
            print(f"  Debug Step {step_count}: state[0:3]={state[0:3]}, position={env.position:.6f}, action={action}")

        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Get current market data
        current_price = df.iloc[min(env.current_step, len(df)-1)].get('close', df.iloc[min(env.current_step, len(df)-1)].get('Close'))

        # Get new state AFTER executing action
        new_position = env.position
        new_balance = env.balance
        new_entry_price = env.entry_price
        
        # Detect if a trade occurred by checking position or balance changes
        position_changed = abs(new_position - prev_position) > 1e-8
        balance_changed = abs(new_balance - prev_balance) > 0.01
        
        # Track trades that occurred in the environment
        if position_changed or balance_changed:
            # Determine trade type based on action and position change
            if action == 1 and new_position > prev_position:  # Buy Long
                btc_amount = new_position - prev_position
                invest_amount = prev_balance - new_balance
                fee = abs(invest_amount) * 0.0018 if invest_amount > 0 else 0
                
                trades.append({
                    'step': step_count,
                    'type': 'BUY_LONG',
                    'price': current_price,
                    'amount': btc_amount,
                    'value': abs(invest_amount),
                    'fee': fee
                })
                
            elif action == 2 and new_position < prev_position and prev_position > 0:  # Sell Long
                btc_amount = prev_position - new_position
                revenue = new_balance - prev_balance
                fee = revenue * 0.0018
                pnl = (current_price - prev_entry_price) * btc_amount - fee if prev_entry_price > 0 else 0
                
                trades.append({
                    'step': step_count,
                    'type': 'SELL_LONG',
                    'price': current_price,
                    'amount': btc_amount,
                    'value': revenue,
                    'fee': fee,
                    'pnl': pnl
                })
                
            elif action == 3 and new_position < prev_position:  # Sell Short
                btc_amount = abs(new_position - prev_position)
                short_value = new_balance - prev_balance
                fee = short_value * 0.0018
                
                trades.append({
                    'step': step_count,
                    'type': 'SELL_SHORT',
                    'price': current_price,
                    'amount': btc_amount,
                    'value': short_value,
                    'fee': fee
                })
                
            elif action == 4 and new_position > prev_position and prev_position < 0:  # Buy Short (cover)
                btc_amount = new_position - prev_position
                cost = prev_balance - new_balance
                fee = cost * 0.0018
                pnl = (prev_entry_price - current_price) * btc_amount - fee if prev_entry_price > 0 else 0
                
                trades.append({
                    'step': step_count,
                    'type': 'BUY_SHORT',
                    'price': current_price,
                    'amount': btc_amount,
                    'value': cost,
                    'fee': fee,
                    'pnl': pnl
                })
            
            # Also check for trailing stop-loss trades (handled by environment)
            # These might close positions even if action was 0 (Hold)
            if action == 0 and position_changed:
                if prev_position > 0 and new_position == 0:  # Long closed
                    trades.append({
                        'step': step_count,
                        'type': 'TRAILING_STOP_LONG',
                        'price': current_price,
                        'amount': prev_position,
                        'value': new_balance - prev_balance,
                        'fee': (new_balance - prev_balance) * 0.0018,
                        'pnl': (current_price - prev_entry_price) * prev_position - (new_balance - prev_balance) * 0.0018 if prev_entry_price > 0 else 0
                    })
                elif prev_position < 0 and new_position == 0:  # Short closed
                    trades.append({
                        'step': step_count,
                        'type': 'TRAILING_STOP_SHORT',
                        'price': current_price,
                        'amount': abs(prev_position),
                        'value': prev_balance - new_balance,
                        'fee': (prev_balance - new_balance) * 0.0018,
                        'pnl': (prev_entry_price - current_price) * abs(prev_position) - (prev_balance - new_balance) * 0.0018 if prev_entry_price > 0 else 0
                    })

        # Record portfolio state (use environment's state)
        portfolio_value = env.balance + (env.position * current_price)
        portfolio_history.append({
            'step': step_count,
            'price': current_price,
            'balance': env.balance,
            'position': env.position,
            'portfolio_value': portfolio_value,
            'action': action
        })

        # Progress logging with action distribution
        if step_count % 1000 == 0:
            pnl = portfolio_value - initial_balance
            # Show both original model actions and actual executed actions
            action_dist = f"H:{action_counts[0]} B:{action_counts[1]} S:{action_counts[2]} SS:{action_counts[3]} BS:{action_counts[4]}"
            print(f"Step {step_count:5d} | Portfolio: ${portfolio_value:10.2f} | P&L: ${pnl:8.2f} | Trades: {len(trades):3d} | Model Actions: {action_dist}")

        state = next_state

    # Calculate final metrics
    final_portfolio = portfolio_history[-1]['portfolio_value']
    total_pnl = final_portfolio - initial_balance
    total_return = (total_pnl / initial_balance) * 100

    # Calculate Sharpe ratio
    if len(portfolio_history) > 1:
        portfolio_values = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = calculate_sharpe_ratio(returns)
    else:
        sharpe_ratio = 0

    # Calculate max drawdown
    portfolio_values = [p['portfolio_value'] for p in portfolio_history]
    max_drawdown = calculate_max_drawdown(portfolio_values)

    # Results
    print("\n" + "=" * 60)
    print("🎯 RL PAPER TRADING RESULTS")
    print("=" * 60)
    print(f"Initial Balance:     ${initial_balance:,.2f}")
    print(f"Final Portfolio:     ${final_portfolio:,.2f}")
    print(f"Total P&L:           ${total_pnl:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:.4f}")
    print(f"Max Drawdown:        {max_drawdown:.4f}")
    print(f"Total Trades:        {len(trades)}")
    print(f"Buy Long Trades:     {len([t for t in trades if t['type'] == 'BUY_LONG'])}")
    print(f"Sell Long Trades:    {len([t for t in trades if t['type'] == 'SELL_LONG'])}")
    print(f"Sell Short Trades:   {len([t for t in trades if t['type'] == 'SELL_SHORT'])}")
    print(f"Buy Short Trades:    {len([t for t in trades if t['type'] == 'BUY_SHORT'])}")
    print(f"Stop-Loss Trades:    {len([t for t in trades if 'STOP-LOSS' in t.get('type', '')])}")
    print(f"Take-Profit Trades:  {len([t for t in trades if 'TAKE-PROFIT' in t.get('type', '')])}")
    print(f"Total Fees:          ${env.total_fees:.2f}")
    print(f"Final Position:      {env.position:.6f} BTC")
    print(f"Final Balance:       ${env.balance:.2f}")
    print(f"\nModel Action Distribution (original model choices):")
    print(f"  Hold (0):          {action_counts[0]:,} ({action_counts[0]/step_count*100:.1f}%)")
    print(f"  Buy Long (1):      {action_counts[1]:,} ({action_counts[1]/step_count*100:.1f}%)")
    print(f"  Sell Long (2):     {action_counts[2]:,} ({action_counts[2]/step_count*100:.1f}%)")
    print(f"  Sell Short (3):    {action_counts[3]:,} ({action_counts[3]/step_count*100:.1f}%)")
    print(f"  Buy Short (4):     {action_counts[4]:,} ({action_counts[4]/step_count*100:.1f}%)")
    if action_counts[2] > 0 and len([t for t in trades if t['type'] == 'SELL_SHORT']) > 0:
        print(f"\n⚠️  Note: Model incorrectly chose 'Sell Long' (action 2) without position.")
        print(f"   These were automatically converted to 'Sell Short' (action 3) to enable trading.")
        print(f"   Recommendation: Retrain the model with proper action penalties.")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Portfolio value over time
    plt.subplot(2, 2, 1)
    steps = [p['step'] for p in portfolio_history]
    portfolio_values = [p['portfolio_value'] for p in portfolio_history]
    plt.plot(steps, portfolio_values, label='RL Agent')
    plt.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)

    # Actions over time
    plt.subplot(2, 2, 2)
    actions = [p['action'] for p in portfolio_history]
    plt.plot(steps, actions, 'o-', markersize=2)
    plt.title('RL Actions Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Action (0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Buy Short)')
    plt.yticks([0, 1, 2, 3, 4], ['Hold', 'Buy Long', 'Sell Long', 'Sell Short', 'Buy Short'])
    plt.grid(True)

    # Price vs Portfolio
    plt.subplot(2, 2, 3)
    prices = [p['price'] for p in portfolio_history]
    plt.plot(steps, prices, label='BTC Price', alpha=0.7)
    plt.plot(steps, portfolio_values, label='Portfolio Value', linewidth=2)
    plt.title('Price vs Portfolio Value')
    plt.xlabel('Steps')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)

    # Trade markers
    plt.subplot(2, 2, 4)
    plt.plot(steps, prices, label='BTC Price')

    # Long trades
    buy_long_trades = [t for t in trades if t['type'] == 'BUY_LONG']
    sell_long_trades = [t for t in trades if t['type'] == 'SELL_LONG']

    if buy_long_trades:
        bl_steps = [t['step'] for t in buy_long_trades]
        bl_prices = [t['price'] for t in buy_long_trades]
        plt.scatter(bl_steps, bl_prices, color='green', marker='^', s=50, label='Buy Long')

    if sell_long_trades:
        sl_steps = [t['step'] for t in sell_long_trades]
        sl_prices = [t['price'] for t in sell_long_trades]
        plt.scatter(sl_steps, sl_prices, color='red', marker='v', s=50, label='Sell Long')

    # Short trades
    sell_short_trades = [t for t in trades if t['type'] == 'SELL_SHORT']
    buy_short_trades = [t for t in trades if t['type'] == 'BUY_SHORT']

    if sell_short_trades:
        ss_steps = [t['step'] for t in sell_short_trades]
        ss_prices = [t['price'] for t in sell_short_trades]
        plt.scatter(ss_steps, ss_prices, color='orange', marker='1', s=50, label='Sell Short')

    if buy_short_trades:
        bs_steps = [t['step'] for t in buy_short_trades]
        bs_prices = [t['price'] for t in buy_short_trades]
        plt.scatter(bs_steps, bs_prices, color='purple', marker='2', s=50, label='Buy Short')

    # Risk management markers
    stop_loss_trades = [t for t in trades if 'STOP-LOSS' in t.get('type', '')]
    take_profit_trades = [t for t in trades if 'TAKE-PROFIT' in t.get('type', '')]

    if stop_loss_trades:
        sl_steps = [t['step'] for t in stop_loss_trades]
        sl_prices = [t['price'] for t in stop_loss_trades]
        plt.scatter(sl_steps, sl_prices, color='darkred', marker='X', s=80, label='Stop-Loss')

    if take_profit_trades:
        tp_steps = [t['step'] for t in take_profit_trades]
        tp_prices = [t['price'] for t in take_profit_trades]
        plt.scatter(tp_steps, tp_prices, color='darkgreen', marker='P', s=80, label='Take-Profit')

    plt.title('Trading Actions on Price Chart')
    plt.xlabel('Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rl_paper_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    results = {
        'summary': {
            'initial_balance': initial_balance,
            'final_portfolio': final_portfolio,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'total_fees': env.total_fees
        },
        'trades': trades,
        'portfolio_history': portfolio_history
    }

    with open('rl_paper_trading_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n📊 Detailed results saved to 'rl_paper_trading_results.pkl'")
    print("📈 Charts saved to 'rl_paper_trading_results.png'")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Paper Trading Simulation")
    parser.add_argument("--model", default="ppo_trading_agent.zip",
                       help="Path to trained RL model")
    parser.add_argument("--data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to historical data")
    parser.add_argument("--balance", type=float, default=10000,
                       help="Initial balance")

    args = parser.parse_args()

    results = run_rl_paper_trading(args.model, args.data, args.balance)

    if results:
        print("\n✅ RL Paper trading simulation completed successfully!")
    else:
        print("\n❌ RL Paper trading simulation failed!")
