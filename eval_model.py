#!/usr/bin/env python3
"""Quick model evaluation script — reads last N rows from CSV for speed."""
import numpy as np
import pandas as pd
import subprocess
import sys
from stable_baselines3 import PPO
from enhanced_trading_environment import EnhancedTradingEnvironment

model = PPO.load('ppo_trading_agent.zip')

data_file = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'

# Fast read: get header + last 3000 lines via shell
result = subprocess.run(['tail', '-n', '3001', data_file], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

# Get column names from original file header
with open(data_file, 'r') as f:
    header = f.readline().strip()

import io
csv_data = header + '\n' + '\n'.join(lines)
df = pd.read_csv(io.StringIO(csv_data)).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded rows: {len(df)}")

required = ['close','RSI_15','BB_15_upper','BB_15_lower','ATR_15','MFI_15',
            'MACD_default_macd','MACD_default_signal','MACD_default_histogram',
            'Stochastic_slowk','Stochastic_slowd']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"MISSING COLUMNS: {missing}")
    sys.exit(1)

ACTION_NAMES = {0:'HOLD', 1:'BUY_LONG', 2:'SELL_LONG', 3:'SELL_SHORT', 4:'COVER_SHORT'}
INITIAL_BALANCE = 10000
total_ac = {i: 0 for i in range(5)}
all_ret = []; all_cycles = []; all_unclosed = []

for ep in range(4):
    env = EnhancedTradingEnvironment(df=df, initial_balance=INITIAL_BALANCE,
        transaction_fee=0.0018, episode_length=300, debug=False, enable_strategy_balancing=True)
    state, _ = env.reset()
    done = False; step = 0
    ac = {i: 0 for i in range(5)}; pc = {i: 0 for i in range(5)}
    cycles = 0; pos_open = False; pnl_list = []

    while not done and step < 300:
        step += 1
        action, _ = model.predict(state, deterministic=False)
        action = int(action); ac[action] += 1
        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if info.get('action_performed'):
            pc[action] += 1
            if action in [1, 3] and not pos_open:
                pos_open = True
            elif action in [2, 4] and pos_open:
                cycles += 1; pos_open = False
                pnl_list.append(info.get('pnl_pct', 0))

    for i in range(5):
        total_ac[i] += ac[i]

    cur_price = df.iloc[min(env.current_step, len(df)-1)]['close']
    pv = env.balance + env.margin_locked + env.position * cur_price
    ret = (pv / INITIAL_BALANCE - 1) * 100
    all_ret.append(ret); all_cycles.append(cycles)
    all_unclosed.append(1 if pos_open else 0)

    total = sum(ac.values())
    wr = env.win_count / max(env.total_trades, 1) * 100
    print(f"\nEp{ep+1}: return={ret:+.2f}%  trades={env.total_trades}"
          f"(L:{env.long_trades} S:{env.short_trades})  win={wr:.0f}%"
          f"  cycles={cycles}  unclosed={1 if pos_open else 0}")
    for a in range(5):
        pct = ac[a] / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        ex_pct = pc[a] / ac[a] * 100 if ac[a] > 0 else 0
        print(f"  {ACTION_NAMES[a]:12s} {ac[a]:3d}({pct:4.1f}%) {bar:<20s} exec={pc[a]}({ex_pct:.0f}%)")
    if pnl_list:
        arr = np.array(pnl_list)
        print(f"  PnL/trade: avg={arr.mean()*100:.2f}%  best={arr.max()*100:.2f}%  worst={arr.min()*100:.2f}%")

grand = sum(total_ac.values())
buy_l = total_ac[1]; sell_l = total_ac[2]; sell_s = total_ac[3]; cov_s = total_ac[4]
avg_ret = np.mean(all_ret); avg_c = np.mean(all_cycles); avg_u = np.mean(all_unclosed)

print("\n" + "=" * 60)
print("SUMMARY (4 ep x 300 steps)")
print("=" * 60)
print(f"Avg return: {avg_ret:+.2f}%  |  Avg cycles: {avg_c:.1f}  |  Avg unclosed: {avg_u:.1f}")
print("\nOverall action distribution:")
for a in range(5):
    pct = total_ac[a] / grand * 100 if grand > 0 else 0
    bar = '█' * int(pct / 2)
    print(f"  {ACTION_NAMES[a]:12s} {total_ac[a]:4d} ({pct:4.1f}%) {bar}")

ratio_l = sell_l / max(buy_l, 1)
ratio_s = cov_s / max(sell_s, 1)
long_sh = buy_l / max(buy_l + sell_s, 1) * 100

print("\nDIAGNOSIS:")
print(f"  SELL_LONG/BUY_LONG    = {ratio_l:.2f}  "
      f"{'✅ Лонги закрываются нормально' if ratio_l>=0.5 else '⚠️  Лонги закрываются, но редко' if ratio_l>=0.3 else '❌ Модель почти не закрывает лонги'}")
print(f"  COVER_SHORT/SELL_SHORT= {ratio_s:.2f}  "
      f"{'✅ Шорты закрываются нормально' if ratio_s>=0.5 else '⚠️  Шорты закрываются, но редко' if ratio_s>=0.3 else '❌ Модель почти не закрывает шорты'}")
print(f"  Long/Short баланс:    {long_sh:.0f}%/{100-long_sh:.0f}%  "
      f"{'✅ Хороший баланс' if 30<=long_sh<=70 else '⚠️  Небольшой дисбаланс' if 20<=long_sh<=80 else '❌ Сильный перекос'}")
print(f"  Полных циклов:        {avg_c:.1f}/ep  "
      f"{'✅' if avg_c>=3 else '⚠️  Мало' if avg_c>=1 else '❌ Нет'}")
print(f"  Незакрытых позиций:   {avg_u:.1f}/ep  "
      f"{'✅' if avg_u<0.5 else '⚠️'}")

if ratio_l >= 0.5 and ratio_s >= 0.5 and avg_c >= 2:
    print("\n🎉 Модель научилась закрывать позиции! Распределение действий разумное.")
elif ratio_l >= 0.3 or avg_c >= 1:
    print("\n⚠️  Улучшение есть, но закрытие позиций пока нестабильное. Рекомендуется продолжить обучение.")
else:
    print("\n❌ Проблема не решена. Модель по-прежнему не закрывает позиции.")
