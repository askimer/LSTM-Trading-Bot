#!/usr/bin/env python3
"""
Paper Trading Test for v14 Hybrid Model
Tests hybrid mode (TP=5% / SL=3%) on historical data
"""

import numpy as np
import pandas as pd
import subprocess
from stable_baselines3 import PPO
from enhanced_trading_environment import EnhancedTradingEnvironment

# Configuration
MODEL_PATH = 'rl_checkpoints_v14_hybrid/ppo_v14_hybrid_best.zip'
DATA_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
INITIAL_BALANCE = 10000
N_EPISODES = 10

print("=" * 70)
print("📊 PAPER TRADING TEST - V14 HYBRID MODE")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Hybrid Mode: TP=+5% / SL=-3%")
print(f"Initial Balance: ${INITIAL_BALANCE:,}")
print(f"Episodes: {N_EPISODES}")
print()

# Load data
print("Loading data...")
result = subprocess.run(['tail', '-n', '3001', DATA_FILE], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

with open(DATA_FILE, 'r') as f:
    header = f.readline().strip()

import io
csv_data = header + '\n' + '\n'.join(lines)
df = pd.read_csv(io.StringIO(csv_data)).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded {len(df):,} rows")
print()

# Load model
print("Loading hybrid model...")
model = PPO.load(MODEL_PATH)
print("✅ Model loaded")
print()

# Run episodes
results = []
all_exits = {'tp': 0, 'sl': 0, 'time': 0, 'manual': 0}

print("=" * 70)
print("RUNNING EPISODES...")
print("=" * 70)

for ep in range(N_EPISODES):
    env = EnhancedTradingEnvironment(
        df=df,
        initial_balance=INITIAL_BALANCE,
        transaction_fee=0.0018,
        episode_length=300,
        debug=False,
        enable_strategy_balancing=True
    )
    state, _ = env.reset()
    done = False
    step = 0
    
    trades = []
    tp_count = sl_count = time_count = 0
    
    while not done and step < 300:
        step += 1
        action, _ = model.predict(state, deterministic=True)
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track hybrid exits
        if info.get('hybrid_exit_reason'):
            exit_reason = info['hybrid_exit_reason']
            pnl = info.get('pnl_pct', 0)
            trades.append({'exit': exit_reason, 'pnl': pnl})
            if exit_reason == 'tp':
                tp_count += 1
            elif exit_reason == 'sl':
                sl_count += 1
            elif exit_reason == 'time':
                time_count += 1
    
    # Calculate final portfolio value
    cur_price = df.iloc[min(env.current_step, len(df)-1)]['close']
    pv = env.balance + env.margin_locked + env.position * cur_price
    ret = (pv / INITIAL_BALANCE - 1) * 100
    
    results.append({
        'episode': ep + 1,
        'return': ret,
        'trades': trades,
        'tp': tp_count,
        'sl': sl_count,
        'time': time_count,
        'total_trades': len(trades)
    })
    
    win_rate = tp_count / max(len(trades), 1) * 100
    print(f"\nEp{ep+1}: return={ret:+.2f}%  trades={len(trades)} "
          f"(TP:{tp_count} SL:{sl_count} Time:{time_count})  "
          f"win_rate={win_rate:.0f}%")
    
    if trades:
        pnl_list = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnl_list) * 100
        best_pnl = np.max(pnl_list) * 100
        worst_pnl = np.min(pnl_list) * 100
        print(f"      PnL: avg={avg_pnl:+.2f}%  best={best_pnl:+.2f}%  worst={worst_pnl:+.2f}%")
    
    # Update totals
    for t in trades:
        if t['exit'] in all_exits:
            all_exits[t['exit']] += 1

# Summary
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

avg_return = np.mean([r['return'] for r in results])
std_return = np.std([r['return'] for r in results])
total_trades = sum(r['total_trades'] for r in results)

# Calculate overall win rate
total_tp = all_exits['tp']
total_sl = all_exits['sl']
total_time = all_exits['time']
win_rate = total_tp / max(total_trades, 1) * 100

# Calculate PnL statistics
all_pnl = [t['pnl'] for r in results for t in r['trades']]
if all_pnl:
    avg_pnl = np.mean(all_pnl) * 100
    best_pnl = np.max(all_pnl) * 100
    worst_pnl = np.min(all_pnl) * 100
    profitable = sum(1 for p in all_pnl if p > 0)
    profit_factor = profitable / len(all_pnl) * 100
else:
    avg_pnl = best_pnl = worst_pnl = profit_factor = 0

print(f"""
Performance Metrics:
  Average Return:     {avg_return:+.2f}% ± {std_return:.2f}%
  Total Trades:       {total_trades}
  Win Rate (TP%):     {win_rate:.1f}%
  Avg PnL/Trade:      {avg_pnl:+.2f}%
  Best Trade:         {best_pnl:+.2f}%
  Worst Trade:        {worst_pnl:+.2f}%
  Profit Factor:      {profit_factor:.1f}%
""")

print(f"""
Exit Distribution:
  Take Profit (5%):   {total_tp} ({total_tp/max(total_trades,1)*100:.1f}%)
  Stop Loss (3%):     {total_sl} ({total_sl/max(total_trades,1)*100:.1f}%)
  Time Exit (50stp):  {total_time} ({total_time/max(total_trades,1)*100:.1f}%)
""")

# Assessment
print("=" * 70)
print("ASSESSMENT")
print("=" * 70)

if avg_return > 0:
    print("✅ MODEL IS PROFITABLE!")
elif avg_return > -0.3:
    print("⚠️  Small loss - acceptable")
else:
    print("❌ Model is losing money")

if win_rate > 40:
    print("✅ Good win rate (>40%)")
elif win_rate > 25:
    print("⚠️  Moderate win rate (25-40%)")
else:
    print("❌ Low win rate (<25%)")

if total_tp > total_sl * 1.5:
    print("✅ More TP hits than SL (good risk/reward)")
else:
    print("⚠️  Too many SL hits")

print("\n" + "=" * 70)
print("✅ HYBRID MODE PAPER TRADING TEST COMPLETED")
print("=" * 70)
