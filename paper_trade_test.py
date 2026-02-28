#!/usr/bin/env python3
"""
Paper Trading Test for v11 Best Model (250K steps)
Tests the best performing model on historical data
"""

import numpy as np
import pandas as pd
import subprocess
from stable_baselines3 import PPO
from enhanced_trading_environment import EnhancedTradingEnvironment

# Configuration
MODEL_PATH = 'rl_checkpoints_profitable/ppo_profitable_250000_steps.zip'
DATA_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
INITIAL_BALANCE = 10000
N_EPISODES = 10  # More episodes for better statistics

print("=" * 70)
print("📊 PAPER TRADING TEST - V11 BEST MODEL (250K STEPS)")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
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
print("Loading model...")
model = PPO.load(MODEL_PATH)
print("✅ Model loaded successfully")
print()

# Run episodes
ACTION_NAMES = {0: 'HOLD', 1: 'BUY_LONG', 2: 'SELL_LONG', 3: 'SELL_SHORT', 4: 'COVER_SHORT'}
results = []

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
    
    ac = {i: 0 for i in range(5)}
    pc = {i: 0 for i in range(5)}
    cycles = 0
    pos_open = False
    pnl_list = []
    
    while not done and step < 300:
        step += 1
        action, _ = model.predict(state, deterministic=True)
        action = int(action)
        ac[action] += 1
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if info.get('action_performed'):
            pc[action] += 1
            if action in [1, 3] and not pos_open:
                pos_open = True
            elif action in [2, 4] and pos_open:
                cycles += 1
                pos_open = False
                pnl_list.append(info.get('pnl_pct', 0))
    
    # Calculate final portfolio value
    cur_price = df.iloc[min(env.current_step, len(df)-1)]['close']
    pv = env.balance + env.margin_locked + env.position * cur_price
    ret = (pv / INITIAL_BALANCE - 1) * 100
    wr = env.win_count / max(env.total_trades, 1) * 100 if env.total_trades > 0 else 0
    
    results.append({
        'episode': ep + 1,
        'return': ret,
        'trades': env.total_trades,
        'long_trades': env.long_trades,
        'short_trades': env.short_trades,
        'win_rate': wr,
        'cycles': cycles,
        'unclosed': 1 if pos_open else 0,
        'pnl_list': pnl_list
    })
    
    print(f"\nEp{ep+1}: return={ret:+.2f}%  trades={env.total_trades} "
          f"(L:{env.long_trades} S:{env.short_trades})  win={wr:.0f}%  "
          f"cycles={cycles}  unclosed={1 if pos_open else 0}")
    
    if pnl_list:
        arr = np.array(pnl_list)
        print(f"      PnL/trade: avg={arr.mean()*100:.2f}%  "
              f"best={arr.max()*100:.2f}%  worst={arr.min()*100:.2f}%")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

avg_return = np.mean([r['return'] for r in results])
std_return = np.std([r['return'] for r in results])
avg_trades = np.mean([r['trades'] for r in results])
avg_win_rate = np.mean([r['win_rate'] for r in results])
avg_cycles = np.mean([r['cycles'] for r in results])
total_pnl = sum([p for r in results for p in r['pnl_list']])

# Calculate all PnL values
all_pnl = [p for r in results for p in r['pnl_list']]
if all_pnl:
    avg_pnl = np.mean(all_pnl)
    best_pnl = np.max(all_pnl)
    worst_pnl = np.min(all_pnl)
    profitable_trades = sum(1 for p in all_pnl if p > 0)
    trade_win_rate = profitable_trades / len(all_pnl) * 100 if all_pnl else 0
else:
    avg_pnl = best_pnl = worst_pnl = 0
    trade_win_rate = 0

# Calculate action distribution
total_actions = sum(sum(ac.values()) for ac in [r.get('ac', {i:0 for i in range(5)}) for r in results])
# Recalculate from results
action_counts = {i: 0 for i in range(5)}
for r in results:
    # We didn't save action counts, estimate from trades
    pass

print(f"""
Performance Metrics:
  Average Return:     {avg_return:+.2f}% ± {std_return:.2f}%
  Total PnL:          {total_pnl*100:+.2f}%
  Total Trades:       {sum(r['trades'] for r in results)}
  Win Rate (trades):  {trade_win_rate:.1f}%
  Avg Win Rate (ep):  {avg_win_rate:.1f}%
  Avg Trades/Ep:      {avg_trades:.1f}
  Avg Cycles/Ep:      {avg_cycles:.1f}
  Unclosed Positions: {sum(r['unclosed'] for r in results)}/{N_EPISODES}
""")

if all_pnl:
    print(f"""
PnL Statistics:
  Avg PnL/Trade:      {avg_pnl*100:+.2f}%
  Best Trade:         {best_pnl*100:+.2f}%
  Worst Trade:        {worst_pnl*100:+.2f}%
  Profitable Trades:  {profitable_trades}/{len(all_pnl)} ({trade_win_rate:.1f}%)
""")

# Long/Short breakdown
total_long = sum(r['long_trades'] for r in results)
total_short = sum(r['short_trades'] for r in results)
long_ratio = total_long / (total_long + total_short) * 100 if (total_long + total_short) > 0 else 0

print(f"""
Position Balance:
  Long Trades:        {total_long} ({long_ratio:.0f}%)
  Short Trades:       {total_short} ({100-long_ratio:.0f}%)
""")

# Final assessment
print("=" * 70)
print("ASSESSMENT")
print("=" * 70)

if avg_return > 0:
    print("✅ MODEL IS PROFITABLE!")
elif avg_return > -0.5:
    print("⚠️  Small loss - acceptable for further testing")
else:
    print("❌ Model is losing money - needs improvement")

if trade_win_rate > 40:
    print("✅ Good win rate (>40%)")
elif trade_win_rate > 30:
    print("⚠️  Moderate win rate (30-40%)")
else:
    print("❌ Low win rate (<30%)")

if avg_trades > 10 and avg_trades < 30:
    print("✅ Good trading frequency")
elif avg_trades < 5:
    print("⚠️  Too conservative (few trades)")
else:
    print("⚠️  Possible overtrading")

print("\n" + "=" * 70)
print("✅ PAPER TRADING TEST COMPLETED")
print("=" * 70)
