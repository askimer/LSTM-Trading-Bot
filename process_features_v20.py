#!/usr/bin/env python3
"""
Feature Engineering Pipeline for V20 Model
Processes raw BTC/USDT data and creates feature-engineered dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 70)
print("🔧 FEATURE ENGINEERING PIPELINE V20")
print("=" * 70)

# Configuration
INPUT_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
OUTPUT_FILE = 'btc_usdt_training_data/full_btc_usdt_data_feature_engineered_v20.csv'

print(f"\n📥 Loading data from {INPUT_FILE}...")

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df):,} rows")

# Check required columns
required_cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    exit(1)

print(f"📅 Date range: {df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]}")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')

# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

print("\n🔧 Calculating features...")

# 1. Moving Averages (EMA)
for period in [15, 60, 300]:
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    print(f"  ✅ EMA_{period}")

# 2. Bollinger Bands
for period in [15, 60, 300]:
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    df[f'BB_{period}_upper'] = sma + 2 * std
    df[f'BB_{period}_lower'] = sma - 2 * std
    print(f"  ✅ BB_{period}")

# 3. RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

for period in [15]:
    df[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
    print(f"  ✅ RSI_{period}")

# 4. ATR (Average True Range)
def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

for period in [15]:
    df[f'ATR_{period}'] = calculate_atr(df, period)
    print(f"  ✅ ATR_{period}")

# 5. MFI (Money Flow Index)
def calculate_mfi(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0)
    negative_flow = money_flow.where(delta < 0, 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    return mfi

for period in [15]:
    df[f'MFI_{period}'] = calculate_mfi(df, period)
    print(f"  ✅ MFI_{period}")

# 6. MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

macd, signal, hist = calculate_macd(df)
df['MACD_default_macd'] = macd
df['MACD_default_signal'] = signal
df['MACD_default_histogram'] = hist
print(f"  ✅ MACD (12,26,9)")

# 7. Stochastic Oscillator
def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    
    return k, d

stoch_k, stoch_d = calculate_stochastic(df)
df['Stochastic_slowk'] = stoch_k
df['Stochastic_slowd'] = stoch_d
print(f"  ✅ Stochastic (14,3)")

# 8. Trend indicators (short and medium)
df['short_trend'] = (df['EMA_15'] > df['EMA_60']).astype(int)
df['medium_trend'] = (df['EMA_60'] > df['EMA_300']).astype(int)
print(f"  ✅ Trend indicators")

# ======================================================================
# CLEANUP & SAVE
# ======================================================================

print("\n🧹 Cleaning data...")

# Drop rows with NaN values
initial_rows = len(df)
df = df.dropna()
dropped_rows = initial_rows - len(df)
print(f"  Dropped {dropped_rows:,} rows with NaN values")

# Save to CSV
print(f"\n💾 Saving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)

# Summary
print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"\n📊 DATASET SUMMARY:")
print(f"  • Total rows: {len(df):,}")
print(f"  • Total columns: {len(df.columns)}")
print(f"  • Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"  • Features: EMA, BB, RSI, ATR, MFI, MACD, Stochastic, Trends")
print(f"\n📁 Output file: {OUTPUT_FILE}")
print("=" * 70)
