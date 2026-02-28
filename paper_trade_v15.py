#!/usr/bin/env python3
"""
Paper Trading Test for v15 Rule-Based Entry + RL Exit
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from rule_based_entry_env import RuleBasedEntryEnv

# Configuration
MODEL_PATH = 'rl_checkpoints_v15_rule_entry/ppo_v15_best.zip'
DATA_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
INITIAL_BALANCE = 10000
N_EPISODES = 10

print("=" * 70)
print("📊 PAPER TRADING TEST - V15 RULE-BASED ENTRY + RL EXIT")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Entry: RSI + MACD rules")
print(f"Exit: RL model")
print(f"Initial Balance: ${INITIAL_BALANCE:,}")
print(f"Episodes: {N_EPISODES}")
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_FILE).tail(3001).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded {len(df):,} rows")
print()

# Load model
print("Loading model...")
model = PPO.load(MODEL_PATH)
print("✅ Model loaded")
print()

# Run episodes
results = []
all_trades = []

print("=" * 70)
print("RUNNING EPISODES...")
print("=" * 70)

for ep in range(N_EPISODES):
    env = RuleBasedEntryEnv(
        df=df,
        initial_balance=INITIAL_BALANCE,
        transaction_fee=0.0018,
        episode_length=300,
        debug=False
    )
    state, _ = env.reset()
    done = False
    step = 0
    
    trades = []
    
    while not done and step < 300:
        step += 1
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if info.get('action_performed') and info.get('pnl_pct', 0) != 0:
            trades.append({
                'pnl': info['pnl_pct'],
                'win': info['pnl_pct'] > 0
            })
    
    # Calculate final portfolio value
    cur_price = df.iloc[min(env.current_step, len(df)-1)]['close']
    pv = env.balance + env.position * cur_price if env.position != 0 else env.balance
    ret = (pv / INITIAL_BALANCE - 1) * 100
    wr = env.win_count / max(env.total_trades, 1) * 100 if env.total_trades > 0 else 0
    
    results.append({
        'episode': ep + 1,
        'return': ret,
        'trades': env.total_trades,
        'win_rate': wr
    })
    
    all_trades.extend(trades)
    
    print(f"\nEp{ep+1}: return={ret:+.2f}%  trades={env.total_trades}  win={wr:.0f}%")
    
    if trades:
        pnl_list = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnl_list) * 100
        best_pnl = np.max(pnl_list) * 100
        worst_pnl = np.min(pnl_list) * 100
        print(f"      PnL: avg={avg_pnl:+.2f}%  best={best_pnl:+.2f}%  worst={worst_pnl:+.2f}%")

# Summary
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

avg_return = np.mean([r['return'] for r in results])
std_return = np.std([r['return'] for r in results])
total_trades = sum(r['trades'] for r in results)

if all_trades:
    win_trades = sum(1 for t in all_trades if t['win'])
    win_rate = win_trades / len(all_trades) * 100
    avg_pnl = np.mean([t['pnl'] for t in all_trades]) * 100
    best_pnl = np.max([t['pnl'] for t in all_trades]) * 100
    worst_pnl = np.min([t['pnl'] for t in all_trades]) * 100
else:
    win_rate = avg_pnl = best_pnl = worst_pnl = 0

print(f"""
Performance Metrics:
  Average Return:     {avg_return:+.2f}% ± {std_return:.2f}%
  Total Trades:       {total_trades}
  Win Rate:           {win_rate:.1f}%
  Avg PnL/Trade:      {avg_pnl:+.2f}%
  Best Trade:         {best_pnl:+.2f}%
  Worst Trade:        {worst_pnl:+.2f}%
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

if total_trades > 0:
    print("✅ Model is trading!")
else:
    print("❌ Model is NOT trading")

print("\n" + "=" * 70)
print("✅ PAPER TRADING TEST COMPLETED")
print("=" * 70)
