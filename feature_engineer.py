import pandas as pd
import os
import ta

from enum import Enum

def calculate_ema(data, window):
    closed = data['Close']
    return closed.ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(data, window):
    closed = data['Close']
    sma = closed.rolling(window=window).mean()
    std = closed.rolling(window=window).std()
    bollinger_upper = sma + (std * 2)
    bollinger_lower = sma - (std * 2)
    return bollinger_upper, bollinger_lower

def calculate_rsi(data, window):
    closed = data['Close']
    delta = closed.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def calculate_macd(data, window_slow=26, window_fast=12, signal=9):
    ema_fast = calculate_ema(data, window_fast)
    ema_slow = calculate_ema(data, window_slow)
    macd = ema_fast - ema_slow
    signal_line = macd.rolling(window=signal).mean()
    return macd, signal_line

def calculate_OBV(data):
    close = data['Close']
    volume = data['Volume']
    obv_indicator = ta.volume.OnBalanceVolumeIndicator(close, volume)
    return obv_indicator.on_balance_volume()

def calculate_ATR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=window)
    return atr_indicator.average_true_range()

def calculate_ADX(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=window)
    return adx_indicator.adx()

def calculate_stochastic(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    stoch_indicator = ta.momentum.StochasticOscillator(high, low, close)
    slowk = stoch_indicator.stoch()
    slowd = stoch_indicator.stoch_signal()
    return slowk, slowd

def calculate_AD(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    ad_indicator = ta.volume.AccDistIndexIndicator(high, low, close, volume)
    return ad_indicator.acc_dist_index()

def calculate_StdDev(data, window):
    closed = data['Close']
    return closed.rolling(window=window).std()

def calculate_LinearReg(data, window):
    closed = data['Close']
    linreg_indicator = ta.trend.LinearRegression(close=closed, window=window)
    return linreg_indicator.linear_regression()

def calculate_MFI(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    mfi_indicator = ta.volume.MFIIndicator(high, low, close, volume, window=window)
    return mfi_indicator.money_flow_index()

def calculate_MOM(data, window):
    closed = data['Close']
    mom_indicator = ta.momentum.ROCIndicator(close=closed, window=window)
    return mom_indicator.roc()

def calculate_ULTOSC(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    ultosc_indicator = ta.momentum.UltimateOscillator(high, low, close)
    return ultosc_indicator.ultimate_oscillator()

def calculate_WillR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    willr_indicator = ta.momentum.WilliamsRIndicator(high, low, close, lbp=window)
    return willr_indicator.williams_r()

def calculate_NATR(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']
    natr_indicator = ta.volatility.NATR(high, low, close, window=window)
    return natr_indicator.natr()

def calculate_TRANGE(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    trange_indicator = ta.volatility.TrueRange(high, low, close)
    return trange_indicator.true_range()

def calculate_WCLPRICE(data, _):
    high = data['High']
    low = data['Low']
    close = data['Close']
    return (high + low + 2 * close) / 4

def calculate_HT_DCPERIOD(data, _):
    close = data['Close']
    ht_dcperiod_indicator = ta.cycle.HilbertTransformDominantCyclePeriod(close)
    return ht_dcperiod_indicator.hilbert_transform_dominant_cycle_period()

def calculate_BETA(data, window):
    high = data['High']
    low = data['Low']
    beta_indicator = ta.momentum.BetaIndicator(high, low, window=window)
    return beta_indicator.beta()

def calculate_VAR(data, window):
    close = data['Close']
    return close.rolling(window=window).var()

def process_chunk(chunk, overlap, indicators, window_size):
    chunk_combined = pd.concat([overlap, chunk])

    for indicator_name, indicator_func_param in indicators.items():
        indicator_func, params = indicator_func_param

        if indicator_name.startswith('BB_'):
            upper, lower = indicator_func(chunk_combined, params)
            chunk_combined[f'{indicator_name}_upper'] = upper
            chunk_combined[f'{indicator_name}_lower'] = lower
        elif indicator_name.startswith('MACD_'):
            macd, signal = indicator_func(chunk_combined, *params)
            chunk_combined[f'{indicator_name}_macd'] = macd
            chunk_combined[f'{indicator_name}_signal'] = signal
        elif indicator_name == 'OBV':
            chunk_combined[indicator_name] = indicator_func(chunk_combined)
        elif indicator_name.startswith('ATR_') or indicator_name.startswith('ADX_'):
            chunk_combined[indicator_name] = indicator_func(chunk_combined, window_size)
        elif indicator_name == 'Stochastic':
            slowk, slowd = indicator_func(chunk_combined)
            chunk_combined[f'{indicator_name}_slowk'] = slowk
            chunk_combined[f'{indicator_name}_slowd'] = slowd
        else:
            chunk_combined[indicator_name] = indicator_func(chunk_combined, window_size)



    new_overlap = chunk_combined.iloc[-window_size:]
    return chunk_combined.iloc[:-window_size], new_overlap

# Initializations
chunk_size = 10 ** 6
window_size = 50
overlap = pd.DataFrame()

class TimeWindows(Enum):
    super_short = 15
    short = 60
    long = 300

indicators = {
    'EMA_15': (calculate_ema, (TimeWindows.super_short.value)),
    'EMA_60': (calculate_ema, (TimeWindows.short.value)),
    'EMA_300': (calculate_ema, (TimeWindows.long.value)),
    'BB_15': (calculate_bollinger_bands, (TimeWindows.super_short.value)),
    'BB_60': (calculate_bollinger_bands, (TimeWindows.short.value)),
    'BB_300': (calculate_bollinger_bands, (TimeWindows.long.value)),

    # Momentum Indicators
    'RSI_15': (calculate_rsi, (TimeWindows.super_short.value)),
    'RSI_60': (calculate_rsi, (TimeWindows.short.value)),
    'RSI_300': (calculate_rsi, (TimeWindows.long.value)),
    'ULTOSC': (calculate_ULTOSC, ()),

    # Volume Indicators
    'OBV': (calculate_OBV, ()),
    'AD': (calculate_AD, ()),

    # Volatility Indicators
    'ATR_15': (calculate_ATR, (TimeWindows.super_short.value)),
    'ATR_60': (calculate_ATR, (TimeWindows.short.value)),

    # Price Transform
    'WCLPRICE': (calculate_WCLPRICE, ()),



    # Statistical Indicators
    'VAR_15': (calculate_VAR, (TimeWindows.super_short.value)),
    'VAR_60': (calculate_VAR, (TimeWindows.short.value)),
    'VAR_300': (calculate_VAR, (TimeWindows.long.value)),

    # Market Strength Indicators
    'MFI_15': (calculate_MFI, (TimeWindows.super_short.value)),
    'MFI_60': (calculate_MFI, (TimeWindows.short.value)),
    'MFI_300': (calculate_MFI, (TimeWindows.long.value)),
}

output_file = './btc_usdt_data/full_btc_usdt_data_feature_engineered.csv'

dtypes = {
    'Open time': 'int64',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'float64',
    'Quote asset volume': 'float64',
    'Number of trades': 'int64',
    'Taker buy base asset volume': 'float64',
    'Taker buy quote asset volume': 'float64'
}

i=0
for chunk in pd.read_csv('./btc_usdt_data/full_btc_usdt_data_cleaned.csv', chunksize=chunk_size, dtype=dtypes, low_memory=False):
    processed_chunk, overlap = process_chunk(chunk, overlap, indicators, window_size)

    processed_chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    i+=len(processed_chunk)
    print(f'Processed chunk, {i} rows saved to {output_file}')
