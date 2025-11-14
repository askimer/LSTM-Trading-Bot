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

# Import TradingEnvironment from train_rl.py
class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agent (same as in train_rl.py)
    """
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.current_step = 0

        # Calculate dynamic price normalization based on data
        price_col = 'close' if 'close' in df.columns else 'Close'
        self.price_mean = df[price_col].mean()
        self.price_std = df[price_col].std()

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

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
        self.position = 0
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
        current_price = row.get('close', row.get('Close', self.price_mean))
        price_norm = (current_price - self.price_mean) / self.price_std

        # Technical indicators
        indicators = [
            row.get('RSI_15', 50) / 100 - 0.5,
            (row.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,
            (row.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,
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

        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        next_price = self.df.iloc[self.current_step + 1].get('close', self.df.iloc[self.current_step + 1].get('Close'))

        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_fee):
                invest_amount = min(self.balance * 0.1, self.balance - 100)
                fee = invest_amount * self.transaction_fee
                coins_bought = (invest_amount - fee) / current_price

                self.position += coins_bought
                self.balance -= invest_amount
                self.total_fees += fee
                reward -= 0.01

        elif action == 2:  # Sell
            if self.position > 0:
                sell_amount = self.position * 0.5
                revenue = sell_amount * current_price
                fee = revenue * self.transaction_fee

                self.position -= sell_amount
                self.balance += revenue - fee
                self.total_fees += fee
                reward -= 0.01

        # Calculate reward
        current_portfolio = self.balance + self.position * current_price
        next_portfolio = self.balance + self.position * next_price
        portfolio_change = (next_portfolio - current_portfolio) / current_portfolio if current_portfolio > 0 else 0

        if len(self.portfolio_values) >= 10:
            recent_portfolio_values = self.portfolio_values[-10:]
            returns = np.diff(recent_portfolio_values) / recent_portfolio_values[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            risk_penalty = volatility * 50
        else:
            risk_penalty = 0

        reward += (portfolio_change * 10000) - risk_penalty

        # Additional reward components
        price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0

        if action == 1 and price_change_pct > 0:
            reward += abs(price_change_pct) * 1000
        if action == 1 and price_change_pct < 0:
            reward -= abs(price_change_pct) * 500
        if action == 2 and price_change_pct < 0:
            reward += abs(price_change_pct) * 1000
        if action == 2 and price_change_pct > 0:
            reward -= abs(price_change_pct) * 500

        if action == 0 and self.position > 0:
            reward -= 0.01

        if current_portfolio < self.initial_balance * 0.5:
            reward -= 10

        self.portfolio_values.append(current_portfolio)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            terminated = True

        return self._get_state(), reward, terminated, truncated, {}

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

    # Initialize environment
    print("Initializing trading environment...")
    env = TradingEnvironment(df, initial_balance=initial_balance)

    # Trading simulation
    print("Starting trading simulation...")
    print("-" * 50)

    trades = []
    portfolio_history = []
    balance = initial_balance
    position = 0
    total_fees = 0
    entry_price = 0

    # Trailing stop-loss and take-profit tracking
    trailing_stop_loss = 0
    trailing_take_profit = 0
    highest_price_since_entry = 0
    lowest_price_since_entry = float('inf')
    stop_loss_pct = 0.05  # 5% stop loss
    take_profit_pct = 0.10  # 10% take profit
    trailing_stop_distance = 0.03  # 3% trailing distance

    state, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        step_count += 1

        # Get action from RL model
        action, _ = model.predict(state, deterministic=True)

        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Get current market data
        current_price = df.iloc[min(env.current_step, len(df)-1)].get('close', df.iloc[min(env.current_step, len(df)-1)].get('Close'))

        # Update trailing levels if we have a position
        if position > 0:
            # Update highest price since entry
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price

                # Update trailing stop-loss (3% below highest price)
                trailing_stop_loss = highest_price_since_entry * (1 - trailing_stop_distance)

                # Update trailing take-profit (10% above entry, but trails with price)
                trailing_take_profit = max(trailing_take_profit,
                                         entry_price * (1 + take_profit_pct),
                                         highest_price_since_entry * 0.95)  # At least 5% profit

        # Check for stop-loss or take-profit triggers
        stop_loss_triggered = False
        take_profit_triggered = False

        if position > 0:
            # Check stop-loss
            if current_price <= trailing_stop_loss:
                stop_loss_triggered = True
                print(f"🛑 STOP-LOSS triggered at ${current_price:.2f} (trailing: ${trailing_stop_loss:.2f})")

            # Check take-profit
            elif current_price >= trailing_take_profit:
                take_profit_triggered = True
                print(f"💰 TAKE-PROFIT triggered at ${current_price:.2f} (trailing: ${trailing_take_profit:.2f})")

        # Force sell if stop-loss or take-profit triggered
        if stop_loss_triggered or take_profit_triggered:
            if position > 0:
                revenue = position * current_price
                fee = revenue * 0.0018
                revenue_after_fee = revenue - fee

                pnl = (current_price - entry_price) * position - fee

                balance += revenue_after_fee
                total_fees += fee

                trigger_type = "STOP-LOSS" if stop_loss_triggered else "TAKE-PROFIT"

                trades.append({
                    'step': step_count,
                    'type': f'FORCE_SELL_{trigger_type}',
                    'price': current_price,
                    'amount': position,
                    'value': revenue_after_fee,
                    'fee': fee,
                    'pnl': pnl,
                    'trigger_price': trailing_stop_loss if stop_loss_triggered else trailing_take_profit
                })

                position = 0
                entry_price = 0
                trailing_stop_loss = 0
                trailing_take_profit = 0
                highest_price_since_entry = 0
                lowest_price_since_entry = float('inf')

                trade_executed = True
                continue  # Skip normal RL action this step

        # Execute trade logic (simplified version)
        trade_executed = False

        if action == 1:  # Buy
            if balance > current_price * 1.002:  # Account for fee
                invest_amount = min(balance * 0.1, balance)  # Max 10% of balance
                if invest_amount > 10:  # Minimum trade
                    fee = invest_amount * 0.0018
                    invest_after_fee = invest_amount - fee
                    btc_amount = invest_after_fee / current_price

                    position += btc_amount
                    balance -= invest_amount
                    total_fees += fee
                    entry_price = current_price

                    trades.append({
                        'step': step_count,
                        'type': 'BUY',
                        'price': current_price,
                        'amount': btc_amount,
                        'value': invest_amount,
                        'fee': fee
                    })
                    trade_executed = True

        elif action == 2:  # Sell
            if position > 0:
                sell_amount = min(position * 0.5, position)  # Sell max 50% position
                if sell_amount * current_price > 10:  # Minimum trade value
                    revenue = sell_amount * current_price
                    fee = revenue * 0.0018
                    revenue_after_fee = revenue - fee

                    position -= sell_amount
                    balance += revenue_after_fee
                    total_fees += fee

                    # Calculate P&L
                    pnl = (current_price - entry_price) * sell_amount - fee

                    trades.append({
                        'step': step_count,
                        'type': 'SELL',
                        'price': current_price,
                        'amount': sell_amount,
                        'value': revenue_after_fee,
                        'fee': fee,
                        'pnl': pnl
                    })
                    trade_executed = True

        # Record portfolio state
        portfolio_value = balance + (position * current_price)
        portfolio_history.append({
            'step': step_count,
            'price': current_price,
            'balance': balance,
            'position': position,
            'portfolio_value': portfolio_value,
            'action': action
        })

        # Progress logging
        if step_count % 1000 == 0:
            pnl = portfolio_value - initial_balance
            print(f"Step {step_count:5d} | Portfolio: ${portfolio_value:10.2f} | P&L: ${pnl:8.2f} | Trades: {len(trades):3d}")

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
    print(f"Buy Trades:          {len([t for t in trades if t['type'] == 'BUY'])}")
    print(f"Sell Trades:         {len([t for t in trades if t['type'] == 'SELL'])}")
    print(f"Total Fees:          ${total_fees:.2f}")
    print(f"Final Position:      {position:.6f} BTC")
    print(f"Final Balance:       ${balance:.2f}")

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
    plt.ylabel('Action (0=Hold, 1=Buy, 2=Sell)')
    plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
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
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']

    if buy_trades:
        buy_steps = [t['step'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        plt.scatter(buy_steps, buy_prices, color='green', marker='^', s=50, label='Buy')

    if sell_trades:
        sell_steps = [t['step'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        plt.scatter(sell_steps, sell_prices, color='red', marker='v', s=50, label='Sell')

    # Add stop-loss and take-profit markers
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
            'total_fees': total_fees
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
    parser.add_argument("--data", default="paper_trade_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to historical data")
    parser.add_argument("--balance", type=float, default=10000,
                       help="Initial balance")

    args = parser.parse_args()

    results = run_rl_paper_trading(args.model, args.data, args.balance)

    if results:
        print("\n✅ RL Paper trading simulation completed successfully!")
    else:
        print("\n❌ RL Paper trading simulation failed!")
