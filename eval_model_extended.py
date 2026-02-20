#!/usr/bin/env python3
"""
Extended model evaluation script with comprehensive metrics.
Collects: Total Return, Win Rate, Sharpe Ratio, Max Drawdown, Long/Short Ratio, Profit Factor
"""
import numpy as np
import pandas as pd
import subprocess
import sys
from stable_baselines3 import PPO
from enhanced_trading_environment import EnhancedTradingEnvironment

# Load model
print("Loading model...")
model = PPO.load('ppo_trading_agent.zip')

# Load data
data_file = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
print(f"Loading data from {data_file}...")

# Fast read: get header + last 5000 lines via shell for more data
result = subprocess.run(['tail', '-n', '5001', data_file], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

with open(data_file, 'r') as f:
    header = f.readline().strip()

import io
csv_data = header + '\n' + '\n'.join(lines)
df = pd.read_csv(io.StringIO(csv_data)).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded rows: {len(df)}")

# Check required columns
required = ['close','RSI_15','BB_15_upper','BB_15_lower','ATR_15','MFI_15',
            'MACD_default_macd','MACD_default_signal','MACD_default_histogram',
            'Stochastic_slowk','Stochastic_slowd']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"MISSING COLUMNS: {missing}")
    sys.exit(1)

# Constants
ACTION_NAMES = {0:'HOLD', 1:'BUY_LONG', 2:'SELL_LONG', 3:'SELL_SHORT', 4:'COVER_SHORT'}
INITIAL_BALANCE = 10000
NUM_EPISODES = 10
EPISODE_LENGTH = 500

# Metrics storage
all_returns = []
all_win_rates = []
all_sharpe_ratios = []
all_max_drawdowns = []
all_long_trades = []
all_short_trades = []
all_total_trades = []
all_profit_factors = []
all_avg_win = []
all_avg_loss = []
total_actions = {i: 0 for i in range(5)}
all_pnl_per_trade = []

print(f"\nRunning {NUM_EPISODES} episodes x {EPISODE_LENGTH} steps...")
print("=" * 70)

for ep in range(NUM_EPISODES):
    env = EnhancedTradingEnvironment(
        df=df, 
        initial_balance=INITIAL_BALANCE,
        transaction_fee=0.0018, 
        episode_length=EPISODE_LENGTH, 
        debug=False, 
        enable_strategy_balancing=True
    )
    state, _ = env.reset()
    done = False
    step = 0
    
    # Track portfolio values for Sharpe and Drawdown
    portfolio_values = [INITIAL_BALANCE]
    daily_returns = []
    
    # Track actions
    actions_count = {i: 0 for i in range(5)}
    
    # Track trades
    trade_pnls = []
    pos_open = False
    entry_value = 0
    
    while not done and step < EPISODE_LENGTH:
        step += 1
        action, _ = model.predict(state, deterministic=False)
        action = int(action)
        actions_count[action] += 1
        total_actions[action] += 1
        
        prev_balance = env.balance + env.margin_locked
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Calculate current portfolio value
        cur_price = df.iloc[min(env.current_step, len(df)-1)]['close']
        pv = env.balance + env.margin_locked + env.position * cur_price
        portfolio_values.append(pv)
        
        # Track daily returns
        if len(portfolio_values) > 1:
            ret = (portfolio_values[-1] / portfolio_values[-2]) - 1
            daily_returns.append(ret)
        
        # Track trade PnL
        if info.get('action_performed'):
            if action in [1, 3] and not pos_open:
                pos_open = True
                entry_value = pv
            elif action in [2, 4] and pos_open:
                pos_open = False
                trade_pnl = (pv - entry_value) / entry_value
                trade_pnls.append(trade_pnl)
                all_pnl_per_trade.append(trade_pnl)
    
    # Calculate episode metrics
    final_pv = portfolio_values[-1]
    total_return = (final_pv / INITIAL_BALANCE - 1) * 100
    all_returns.append(total_return)
    
    # Win rate
    win_rate = env.win_count / max(env.total_trades, 1) * 100
    all_win_rates.append(win_rate)
    
    # Sharpe Ratio (annualized, assuming 1-minute bars)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(525600)  # minutes per year
    else:
        sharpe = 0
    all_sharpe_ratios.append(sharpe)
    
    # Max Drawdown
    peak = portfolio_values[0]
    max_dd = 0
    for pv in portfolio_values:
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak * 100
        if dd > max_dd:
            max_dd = dd
    all_max_drawdowns.append(max_dd)
    
    # Long/Short trades
    all_long_trades.append(env.long_trades)
    all_short_trades.append(env.short_trades)
    all_total_trades.append(env.total_trades)
    
    # Profit Factor
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
    all_profit_factors.append(profit_factor)
    
    # Average win/loss
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    all_avg_win.append(avg_win)
    all_avg_loss.append(avg_loss)
    
    print(f"Ep{ep+1:2d}: Return={total_return:+6.2f}%  WinRate={win_rate:5.1f}%  "
          f"Sharpe={sharpe:6.2f}  MaxDD={max_dd:5.2f}%  "
          f"Trades={env.total_trades}(L:{env.long_trades} S:{env.short_trades})")

# Calculate summary statistics
print("\n" + "=" * 70)
print("SUMMARY METRICS")
print("=" * 70)

avg_return = np.mean(all_returns)
avg_win_rate = np.mean(all_win_rates)
avg_sharpe = np.mean(all_sharpe_ratios)
avg_max_dd = np.mean(all_max_drawdowns)
total_long = sum(all_long_trades)
total_short = sum(all_short_trades)
total_trades = sum(all_total_trades)
avg_profit_factor = np.mean([pf for pf in all_profit_factors if pf != float('inf')])

# Long/Short ratio
if total_long + total_short > 0:
    long_ratio = total_long / (total_long + total_short)
else:
    long_ratio = 0.5

print(f"\n### Метрики производительности")
print(f"| Metric | Value | Target | Status |")
print(f"|--------|-------|--------|--------|")
print(f"| Total Return | {avg_return:+.2f}% | >0% | {'PASS' if avg_return > 0 else 'FAIL'} |")
print(f"| Win Rate | {avg_win_rate:.1f}% | >55% | {'PASS' if avg_win_rate > 55 else 'FAIL'} |")
print(f"| Sharpe Ratio | {avg_sharpe:.2f} | >1.0 | {'PASS' if avg_sharpe > 1.0 else 'FAIL'} |")
print(f"| Max Drawdown | {avg_max_dd:.2f}% | <20% | {'PASS' if avg_max_dd < 20 else 'FAIL'} |")
print(f"| Long/Short Ratio | {long_ratio:.2f} | 0.35-0.65 | {'PASS' if 0.35 <= long_ratio <= 0.65 else 'FAIL'} |")

print(f"\n### Распределение действий")
print(f"| Action | Count | Percentage |")
print(f"|--------|-------|------------|")
grand_total = sum(total_actions.values())
for a in range(5):
    pct = total_actions[a] / grand_total * 100 if grand_total > 0 else 0
    print(f"| {ACTION_NAMES[a]} | {total_actions[a]} | {pct:.1f}% |")

print(f"\n### Статистика сделок")
print(f"- Всего сделок: {total_trades}")
print(f"- Лонг сделок: {total_long} ({total_long/max(total_trades,1)*100:.1f}%)")
print(f"- Шорт сделок: {total_short} ({total_short/max(total_trades,1)*100:.1f}%)")

# Calculate win/loss stats from all trades
if all_pnl_per_trade:
    wins = [p for p in all_pnl_per_trade if p > 0]
    losses = [p for p in all_pnl_per_trade if p < 0]
    print(f"- Прибыльных: {len(wins)} ({len(wins)/len(all_pnl_per_trade)*100:.1f}%)")
    print(f"- Убыточных: {len(losses)} ({len(losses)/len(all_pnl_per_trade)*100:.1f}%)")
    if wins:
        print(f"- Средняя прибыль: {np.mean(wins)*100:.2f}%")
    if losses:
        print(f"- Средний убыток: {np.mean(losses)*100:.2f}%")
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"- Profit Factor: {pf:.2f}")

print(f"\n### Выявленные проблемы при тестировании")
issues = []
if avg_return <= 0:
    issues.append(f"[CRITICAL] Отрицательная доходность: {avg_return:.2f}%")
if avg_win_rate < 55:
    issues.append(f"[WARNING] Низкий Win Rate: {avg_win_rate:.1f}% (цель >55%)")
if avg_sharpe < 1.0:
    issues.append(f"[WARNING] Низкий Sharpe Ratio: {avg_sharpe:.2f} (цель >1.0)")
if avg_max_dd > 20:
    issues.append(f"[WARNING] Высокий Max Drawdown: {avg_max_dd:.2f}% (лимит <20%)")
if not (0.35 <= long_ratio <= 0.65):
    issues.append(f"[WARNING] Дисбаланс Long/Short: {long_ratio:.2f} (цель 0.35-0.65)")

if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("Критических проблем не обнаружено.")

print("\n" + "=" * 70)
print("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ЭПИЗОДАМ")
print("=" * 70)
print(f"Return: min={min(all_returns):.2f}%, max={max(all_returns):.2f}%, std={np.std(all_returns):.2f}%")
print(f"Win Rate: min={min(all_win_rates):.1f}%, max={max(all_win_rates):.1f}%, std={np.std(all_win_rates):.1f}%")
print(f"Sharpe: min={min(all_sharpe_ratios):.2f}, max={max(all_sharpe_ratios):.2f}, std={np.std(all_sharpe_ratios):.2f}")
print(f"Max DD: min={min(all_max_drawdowns):.2f}%, max={max(all_max_drawdowns):.2f}%, std={np.std(all_max_drawdowns):.2f}%")
