#!/usr/bin/env python3
"""
Evaluation script v19 - DQN MODEL WITH CRITICAL FIXES
Evaluates the v19 model trained with:
  ✅ Fixed reward clipping (200.0)
  ✅ Correct SHORT PnL% calculation
  ✅ Reduced exploration (0.05)
  ✅ Simplified reward function
"""

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from enhanced_trading_environment_v19 import EnhancedTradingEnvironment

MODEL_PATH = 'rl_checkpoints_v19_dqn_fixed/dqn_v19_best.zip'
DATA_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
INITIAL_BALANCE = 10000

print("=" * 70)
print("📊 V19 DQN MODEL EVALUATION - CRITICAL FIXES")
print("=" * 70)
print()
print("V19 IMPROVEMENTS:")
print("  ✅ Reward clipping: 200.0 (preserves large trade signals)")
print("  ✅ SHORT PnL%: corrected calculation")
print("  ✅ Exploration: 0.05 (more stable)")
print("  ✅ Reward function: simplified (no conflicts)")
print()

# Load data
df = pd.read_csv(DATA_FILE).tail(3001).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded {len(df):,} rows")

# Load model
print("\nLoading DQN model...")
model = DQN.load(MODEL_PATH)
print("✅ Model loaded")

# Create environment
env = EnhancedTradingEnvironment(
    df=df,
    initial_balance=INITIAL_BALANCE,
    transaction_fee=0.0018,
    episode_length=300,
    debug=False,
    enable_strategy_balancing=True
)

# Run episodes
ACTION_NAMES = {0: 'HOLD', 1: 'BUY_LONG', 2: 'SELL_LONG', 3: 'SELL_SHORT', 4: 'COVER_SHORT'}
results = []

print("\n" + "=" * 70)
print("RUNNING EVALUATION (4 episodes)...")
print("=" * 70)

for ep in range(4):
    obs, _ = env.reset()
    done = False
    ac = {i: 0 for i in range(5)}
    pnl_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        ac[int(action)] += 1

        if info.get('action_performed') and info.get('pnl_pct', 0) != 0:
            pnl_list.append(info['pnl_pct'])

    # Calculate return
    ret = (env.balance / INITIAL_BALANCE - 1) * 100
    wr = env.win_count / max(env.total_trades, 1) * 100

    results.append({
        'episode': ep + 1,
        'return': ret,
        'trades': env.total_trades,
        'long_trades': env.long_trades,
        'short_trades': env.short_trades,
        'win_rate': wr,
        'pnl_list': pnl_list,
        'actions': ac.copy()
    })

    print(f"\nEp{ep+1}: return={ret:+.2f}%  trades={env.total_trades} "
          f"(L:{env.long_trades} S:{env.short_trades})  win={wr:.0f}%")

    total = sum(ac.values())
    for a in range(5):
        pct = ac[a] / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {ACTION_NAMES[a]:12s} {ac[a]:3d}({pct:4.1f}%) {bar}")

    if pnl_list:
        arr = np.array(pnl_list)
        print(f"  PnL/trade: avg={arr.mean()*100:.2f}%  best={arr.max()*100:.2f}%  worst={arr.min()*100:.2f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

avg_return = np.mean([r['return'] for r in results])
total_trades = sum(r['trades'] for r in results)
total_long = sum(r['long_trades'] for r in results)
total_short = sum(r['short_trades'] for r in results)

all_pnl = [p for r in results for p in r['pnl_list']]
if all_pnl:
    avg_pnl = np.mean(all_pnl) * 100
    best_pnl = np.max(all_pnl) * 100
    worst_pnl = np.min(all_pnl) * 100
    win_count = sum(1 for p in all_pnl if p > 0)
    win_rate = win_count / len(all_pnl) * 100
else:
    avg_pnl = best_pnl = worst_pnl = win_rate = 0

avg_wr = np.mean([r['win_rate'] for r in results])

print(f"""
📊 PERFORMANCE METRICS:
├─ Average Return:     {avg_return:+.2f}%
├─ Average Win Rate:   {avg_wr:.1f}%
├─ Total Trades:       {total_trades} ({total_trades/4:.1f} per episode)
│  ├─ Long:            {total_long} ({total_long/max(total_trades,1)*100:.0f}%)
│  └─ Short:           {total_short} ({total_short/max(total_trades,1)*100:.0f}%)
├─ PnL per Trade:
│  ├─ Average:         {avg_pnl:+.2f}%
│  ├─ Best:            {best_pnl:+.2f}%
│  └─ Worst:           {worst_pnl:+.2f}%
└─ Win Rate (overall): {win_rate:.1f}%

🎯 V19 TARGETS:
├─ Return/episode:     >-0.3% (v17 was -1.10%)
├─ Win Rate:           30-40% (v17 was 12.5%)
├─ Trades/episode:     40-60 (v17 was 120 - overtrading)
└─ Short %:            20-40% (v17 was 0%)
""")

# Check if targets met
targets_met = []
targets_met.append(("Return > -0.3%", avg_return > -0.3))
targets_met.append(("Win Rate > 30%", win_rate > 30))
targets_met.append(("Trades < 60", total_trades/4 < 60))
targets_met.append(("Short % > 20%", total_short/max(total_trades,1)*100 > 20))

print("✅ TARGETS MET:")
for name, met in targets_met:
    status = "✅" if met else "❌"
    print(f"  {status} {name}")

if all(met for _, met in targets_met):
    print("\n🎉 ALL V19 TARGETS ACHIEVED!")
else:
    print("\n⚠️  Some targets not met - further tuning needed")

print("\n" + "=" * 70)
