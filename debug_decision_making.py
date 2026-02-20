#!/usr/bin/env python3
"""
Debug script to analyze model decision making process.
Shows what data the model sees and why it makes certain decisions.
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

result = subprocess.run(['tail', '-n', '2001', data_file], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

with open(data_file, 'r') as f:
    header = f.readline().strip()

import io
csv_data = header + '\n' + '\n'.join(lines)
df = pd.read_csv(io.StringIO(csv_data)).dropna()
if 'close' not in df.columns and 'Close' in df.columns:
    df['close'] = df['Close']
print(f"Loaded rows: {len(df)}")

# Constants
ACTION_NAMES = {0:'HOLD', 1:'BUY_LONG', 2:'SELL_LONG', 3:'SELL_SHORT', 4:'COVER_SHORT'}
# State vector (18 elements) based on enhanced_trading_environment.py
STATE_NAMES = [
    'balance_norm',       # [0] balance / initial_balance - 1
    'position_norm',      # [1] position_value / initial_balance (+ for long, - for short)
    'price_norm',         # [2] Rolling z-score of price
    'can_close',          # [3] 1.0 if position open AND min_hold elapsed
    'steps_to_close',     # [4] countdown ratio to when close becomes allowed
    'unrealized_pnl',     # [5] current unrealized PnL
    'rsi_norm',           # [6] RSI_15 / 100 - 0.5
    'bb_upper_norm',      # [7] BB_15_upper / price - 1
    'bb_lower_norm',      # [8] BB_15_lower / price - 1
    'atr_norm',           # [9] ATR_15 / 1000
    'short_trend',        # [10] 5-step price trend
    'medium_trend',       # [11] 20-step price trend
    'mfi_norm',           # [12] MFI_15 / 100 - 0.5
    'macd_norm',          # [13] MACD line normalized
    'macd_signal_norm',   # [14] MACD signal line normalized
    'macd_hist_norm',     # [15] MACD histogram normalized
    'stoch_k_norm',       # [16] Stochastic K normalized
    'stoch_d_norm',       # [17] Stochastic D normalized
]
INITIAL_BALANCE = 10000

# Create environment
env = EnhancedTradingEnvironment(
    df=df, 
    initial_balance=INITIAL_BALANCE,
    transaction_fee=0.0018, 
    episode_length=100, 
    debug=False, 
    enable_strategy_balancing=True
)

state, _ = env.reset()
state_dim = len(state)
print(f"\nState vector dimension: {state_dim}")
print(f"Expected state names: {len(STATE_NAMES)}")

print("\n" + "=" * 100)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРИНЯТИЯ РЕШЕНИЙ")
print("=" * 100)

# Track decisions
decisions_log = []
losing_trades = []
winning_trades = []

# Run episode with detailed logging
done = False
step = 0
prev_pv = INITIAL_BALANCE
entry_price = None
entry_step = None
position_type = None  # 'long' or 'short'

print("\n### Первые 30 шагов с детальным логированием:")
print("-" * 100)

while not done and step < 100:
    step += 1
    
    # Get action probabilities
    action, _ = model.predict(state, deterministic=False)
    action = int(action)
    
    # Get action probabilities from policy using torch
    import torch
    obs_tensor = torch.FloatTensor(state.reshape(1, -1))
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        action_probs = distribution.distribution.probs.numpy()[0]
    
    # Get current market data
    cur_idx = min(env.current_step, len(df)-1)
    cur_price = df.iloc[cur_idx]['close']
    cur_rsi = df.iloc[cur_idx].get('RSI_15', 50)
    cur_mfi = df.iloc[cur_idx].get('MFI_15', 50)
    
    # Calculate portfolio value
    pv = env.balance + env.margin_locked + env.position * cur_price
    pv_change = (pv - prev_pv) / prev_pv * 100
    
    # Log decision
    decision = {
        'step': step,
        'state': state.copy(),
        'action': action,
        'action_probs': action_probs.copy(),
        'price': cur_price,
        'rsi': cur_rsi,
        'mfi': cur_mfi,
        'position': env.position,
        'balance': env.balance,
        'pv': pv,
        'pv_change': pv_change
    }
    decisions_log.append(decision)
    
    # Print detailed info for first 30 steps
    if step <= 30:
        print(f"\n--- Step {step} ---")
        print(f"Price: ${cur_price:.2f}  RSI: {cur_rsi:.1f}  MFI: {cur_mfi:.1f}")
        print(f"Position: {env.position:.6f}  Balance: ${env.balance:.2f}  PV: ${pv:.2f} ({pv_change:+.3f}%)")
        print(f"State vector:")
        for i, (name, val) in enumerate(zip(STATE_NAMES, state)):
            if i < len(STATE_NAMES):
                print(f"  [{i}] {name:15s}: {val:+.4f}")
        print(f"Action probabilities:")
        for a in range(5):
            marker = " <--" if a == action else ""
            print(f"  {ACTION_NAMES[a]:12s}: {action_probs[a]*100:5.1f}%{marker}")
        print(f"Chosen action: {ACTION_NAMES[action]}")
    
    # Execute action
    prev_state = state.copy()
    prev_position = env.position
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Track trade outcomes
    if info.get('action_performed'):
        if action in [1, 3]:  # Open position
            entry_price = cur_price
            entry_step = step
            position_type = 'long' if action == 1 else 'short'
        elif action in [2, 4] and entry_price is not None:  # Close position
            if position_type == 'long':
                pnl_pct = (cur_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - cur_price) / entry_price * 100
            
            trade_info = {
                'entry_step': entry_step,
                'exit_step': step,
                'entry_price': entry_price,
                'exit_price': cur_price,
                'type': position_type,
                'pnl_pct': pnl_pct,
                'holding_time': step - entry_step
            }
            
            if pnl_pct > 0:
                winning_trades.append(trade_info)
            else:
                losing_trades.append(trade_info)
            
            entry_price = None
            position_type = None
    
    prev_pv = pv

print("\n" + "=" * 100)
print("АНАЛИЗ УБЫТОЧНЫХ СДЕЛОК")
print("=" * 100)

if losing_trades:
    print(f"\nВсего убыточных сделок: {len(losing_trades)}")
    print("\nПримеры убыточных сделок:")
    for i, trade in enumerate(losing_trades[:10]):
        print(f"\n{i+1}. {trade['type'].upper()} сделка:")
        print(f"   Вход: шаг {trade['entry_step']}, цена ${trade['entry_price']:.2f}")
        print(f"   Выход: шаг {trade['exit_step']}, цена ${trade['exit_price']:.2f}")
        print(f"   Время удержания: {trade['holding_time']} шагов")
        print(f"   PnL: {trade['pnl_pct']:+.3f}%")
        
        # Analyze what model saw at entry
        entry_decision = decisions_log[trade['entry_step']-1]
        print(f"   Состояние при входе:")
        print(f"     RSI: {entry_decision['rsi']:.1f}, MFI: {entry_decision['mfi']:.1f}")
        print(f"     Вероятности действий: ", end="")
        for a in range(5):
            print(f"{ACTION_NAMES[a]}={entry_decision['action_probs'][a]*100:.0f}% ", end="")
        print()
else:
    print("Убыточных сделок не найдено в первых 100 шагах")

print("\n" + "=" * 100)
print("АНАЛИЗ ПРИБЫЛЬНЫХ СДЕЛОК")
print("=" * 100)

if winning_trades:
    print(f"\nВсего прибыльных сделок: {len(winning_trades)}")
    for i, trade in enumerate(winning_trades[:5]):
        print(f"\n{i+1}. {trade['type'].upper()} сделка:")
        print(f"   Вход: шаг {trade['entry_step']}, цена ${trade['entry_price']:.2f}")
        print(f"   Выход: шаг {trade['exit_step']}, цена ${trade['exit_price']:.2f}")
        print(f"   PnL: {trade['pnl_pct']:+.3f}%")
else:
    print("Прибыльных сделок не найдено в первых 100 шагах")

print("\n" + "=" * 100)
print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ВЕРОЯТНОСТЕЙ")
print("=" * 100)

# Analyze action probability patterns
all_probs = np.array([d['action_probs'] for d in decisions_log])
print("\nСредние вероятности действий:")
for a in range(5):
    mean_prob = np.mean(all_probs[:, a]) * 100
    std_prob = np.std(all_probs[:, a]) * 100
    print(f"  {ACTION_NAMES[a]:12s}: {mean_prob:5.1f}% ± {std_prob:.1f}%")

# Analyze when model is confident vs uncertain
max_probs = np.max(all_probs, axis=1)
print(f"\nУверенность модели (макс. вероятность):")
print(f"  Средняя: {np.mean(max_probs)*100:.1f}%")
print(f"  Мин: {np.min(max_probs)*100:.1f}%")
print(f"  Макс: {np.max(max_probs)*100:.1f}%")

# Analyze state values
print("\n" + "=" * 100)
print("АНАЛИЗ ВХОДНЫХ ДАННЫХ (STATE)")
print("=" * 100)

all_states = np.array([d['state'] for d in decisions_log])
print("\nСтатистика по компонентам state:")
for i, name in enumerate(STATE_NAMES):
    if i < all_states.shape[1]:
        vals = all_states[:, i]
        print(f"  {name:15s}: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):+.4f}, max={np.max(vals):+.4f}")

# Check for anomalies
print("\n" + "=" * 100)
print("ПРОВЕРКА НА АНОМАЛИИ")
print("=" * 100)

# Check for NaN or Inf
nan_count = np.sum(np.isnan(all_states))
inf_count = np.sum(np.isinf(all_states))
print(f"NaN значений в state: {nan_count}")
print(f"Inf значений в state: {inf_count}")

# Check for extreme values
for i, name in enumerate(STATE_NAMES):
    if i < all_states.shape[1]:
        vals = all_states[:, i]
        extreme_count = np.sum(np.abs(vals) > 10)
        if extreme_count > 0:
            print(f"⚠️  {name}: {extreme_count} экстремальных значений (|x| > 10)")

# Analyze correlation between state and actions
print("\n" + "=" * 100)
print("КОРРЕЛЯЦИЯ STATE -> ACTION")
print("=" * 100)

actions = np.array([d['action'] for d in decisions_log])
print("\nСредние значения state для каждого действия:")
for a in range(5):
    mask = actions == a
    if np.sum(mask) > 0:
        print(f"\n{ACTION_NAMES[a]} (n={np.sum(mask)}):")
        for i, name in enumerate(STATE_NAMES[:6]):  # First 6 most important
            if i < all_states.shape[1]:
                mean_val = np.mean(all_states[mask, i])
                print(f"  {name:15s}: {mean_val:+.4f}")

print("\n" + "=" * 100)
print("ВЫВОДЫ")
print("=" * 100)

# Calculate key insights
avg_holding = np.mean([t['holding_time'] for t in losing_trades + winning_trades]) if (losing_trades + winning_trades) else 0
win_rate = len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1) * 100

print(f"""
1. Win Rate в первых 100 шагах: {win_rate:.1f}%
2. Среднее время удержания позиции: {avg_holding:.1f} шагов
3. Убыточных сделок: {len(losing_trades)}, Прибыльных: {len(winning_trades)}
4. Модель распределяет вероятности относительно равномерно (нет сильной уверенности)
5. Средняя уверенность модели: {np.mean(max_probs)*100:.1f}%

ВОЗМОЖНЫЕ ПРИЧИНЫ НИЗКОГО WIN RATE:
- Модель не научилась различать хорошие и плохие точки входа
- Слишком короткое время удержания позиций
- Недостаточная дифференциация между действиями (вероятности близки)
- Возможно, reward shaping не стимулирует правильное поведение
""")
