#!.venv/bin/ python3
"""
RL Live Trading Module
Performs virtual/paper trading on live market data using trained RL agent
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import ccxt
import ta
from dotenv import load_dotenv
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# Import the enhanced trading environment (same as used in training)
from enhanced_trading_environment import EnhancedTradingEnvironment

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding for Windows console compatibility
import sys
import io

# Wrap stdout/stderr with UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_live_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The TradingEnvironment class has been moved to trading_environment.py
# This file now imports it from the unified module

class RLLiveTradingBot:
    """
    Live trading bot that uses trained RL model for virtual trading on real market data
    """

    def __init__(self, model_path, symbol='BTC-USDT', test_mode=True, initial_balance=10000):
        """
        Initialize the RL trading bot

        Args:
            model_path: Path to trained RL model (.zip)
            symbol: Trading symbol (default: BTC-USDT)
            test_mode: If True, only virtual trades (no real transactions)
            initial_balance: Virtual balance in USDT (MUST match training initial_balance=10000)
        """
        self.symbol = symbol
        self.test_mode = test_mode
        self.initial_balance = initial_balance
        self.training_initial_balance = 10000  # Must match the value used during training
        
        # CRITICAL: Warn if initial_balance doesn't match training
        if initial_balance != 10000:
            logger.warning(f"WARNING: initial_balance={initial_balance} differs from training value (10000). "
                         f"This will cause state normalization mismatch and unpredictable behavior!")

        # Load RL model
        print(f"Loading RL model from {model_path}...")
        try:
            self.model = PPO.load(model_path)
            print("✅ RL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            raise

        # Initialize BingX API (for getting market data)
        self.api_key = os.getenv('BINGX_API_KEY')
        self.secret_key = os.getenv('BINGX_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("BINGX_API_KEY and BINGX_SECRET_KEY must be set in .env file")

        # Initialize CCXT client for BingX
        self.exchange = ccxt.bingx({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })

        # Virtual portfolio
        self.balance = initial_balance
        self.position = 0  # BTC position (positive = long, negative = short)
        self.entry_price = 0
        self.total_fees = 0
        self.cumulative_pnl = 0.0  # Track cumulative P&L between trades

        # Separate tracking for long and short positions
        self.long_position = 0  # Positive BTC amount for long positions
        self.short_position = 0  # Positive BTC amount for short positions
        self.long_entry_price = 0
        self.short_entry_price = 0

        # Margin trading tracking (similar to trading_environment)
        self.margin_locked = 0.0  # Amount of balance locked as margin for short positions
        self.short_opening_fees = 0.0  # Track short opening fees to avoid double counting
        self.proceeds_from_short = 0.0  # Track proceeds from short sales
        self.margin_requirement = 0.3  # 30% initial margin requirement for shorts
        self.maintenance_margin = 0.15  # 15% maintenance margin
        self.cash_balance = initial_balance  # Pure cash balance (excluding margin effects)
        self.borrowed_assets = 0.0  # Track borrowed assets for short positions
        self.liability = 0.0  # Track liability for short positions

        # Trailing stop-loss and take-profit tracking
        self.trailing_stop_loss = 0
        self.trailing_take_profit = 0
        self.highest_price_since_entry = 0
        self.lowest_price_since_entry = float('inf')
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.trailing_stop_distance = 0.03  # 3% trailing distance

        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.price_history = []

        # Technical indicators history
        self.indicators_history = []

        # CSV logging for trade results
        self.trade_log_file = 'trade_results.csv'
        self.init_csv_logging()

        # Market data buffer (need enough data for indicators)
        self.market_data_buffer = []

        # Trading rate limiting (increased for real market conditions)
        self.last_trade_time = 0
        self.min_trade_interval = 300  # Minimum 5 minutes between trades for real market

        # Create EnhancedTradingEnvironment instance for consistent state calculation
        # This ensures the live trading uses the exact same environment as training
        self.live_env = None  # Will be initialized when we have market data
        self.env_df = None  # DataFrame for environment

        logger.info(f"Initialized RL Live Trading Bot for {symbol}")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Initial balance: {self.initial_balance} USDT")
        logger.info(f"Training initial balance: {self.training_initial_balance} USDT (for state normalization)")
        logger.info(f"Using EnhancedTradingEnvironment for state calculation (same as training)")

    def init_csv_logging(self):
        """Initialize CSV logging for trade results"""
        import csv
        import os
        
        # Check if file exists, if not create with headers
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'type', 'price', 'amount', 'value', 'fee', 
                    'pnl', 'cumulative_pnl', 'balance_after', 'position_after',
                    'long_position', 'short_position', 'trigger_price'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                logger.info(f"Created CSV log file: {self.trade_log_file}")

    def log_trade_to_csv(self, trade_data):
        """Log trade data to CSV file"""
        import csv
        
        # Prepare row data, ensuring all required fields are present
        row = {
            'timestamp': trade_data.get('timestamp', ''),
            'type': trade_data.get('type', ''),
            'price': trade_data.get('price', 0),
            'amount': trade_data.get('amount', 0),
            'value': trade_data.get('value', 0),
            'fee': trade_data.get('fee', 0),
            'pnl': trade_data.get('pnl', 0),
            'cumulative_pnl': trade_data.get('cumulative_pnl', 0),
            'balance_after': trade_data.get('balance_after', 0),
            'position_after': trade_data.get('position_after', 0),
            'long_position': trade_data.get('long_position', 0),
            'short_position': trade_data.get('short_position', 0),
            'trigger_price': trade_data.get('trigger_price', 0)
        }
        
        # Check if file exists and write header if it's a new file
        file_exists = os.path.exists(self.trade_log_file)
        with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'type', 'price', 'amount', 'value', 'fee', 
                'pnl', 'cumulative_pnl', 'balance_after', 'position_after',
                'long_position', 'short_position', 'trigger_price'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def calculate_fee(self, amount, fee_rate=0.0005):
        """Расчет комиссий для фьючерсов BingX (тейкер: 0.05%, мейкер: 0.02%)"""
        # Для бессрочных фьючерсов: 0.02% (мейкер) / 0.05% (тейкер)
        # Для стандартных фьючерсов: 0.045% (фиксированная, списывается при закрытии)
        # По умолчанию используем ставку тейкера (0.05%), так как бот использует рыночные ордера
        return amount * fee_rate

    def _fetch_from_binance(self, symbol_ccxt, timeframe='1m', limit=350):
        """Fallback: Get data from Binance public API"""
        try:
            binance = ccxt.binance({'enableRateLimit': True})
            klines = binance.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, limit=limit)
            if klines:
                logger.info("Successfully fetched data from Binance fallback")
                return klines
        except Exception as e:
            logger.warning(f"Binance fallback failed: {e}")
        return None

    def _process_klines_to_df(self, klines):
        """Convert klines to DataFrame with proper format"""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=1)
        df['quote_volume'] = df['volume'] * df['close']
        df['trades'] = 0
        df['taker_buy_volume'] = df['volume'] * 0.5
        df['taker_buy_quote_volume'] = df['taker_buy_volume'] * df['close']
        df['ignore'] = 0
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        return df

    def get_market_data(self):
        """Get current market data with fallback to Binance public API"""
        symbol_ccxt = self.symbol.replace('-', '/')
        
        # Try BingX first
        try:
            klines = self.exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=350)
            if klines:
                logger.debug("Successfully fetched data from BingX")
                return self._process_klines_to_df(klines)
        except Exception as e:
            logger.warning(f"BingX fetch failed: {e}")
        
        # Fallback 1: Try Binance public API
        logger.info("Trying Binance public API fallback...")
        klines = self._fetch_from_binance(symbol_ccxt)
        if klines:
            return self._process_klines_to_df(klines)
        
        # Fallback 2: Try alternative BingX endpoint
        try:
            logger.info("Trying alternative BingX endpoint...")
            binance_exchange = ccxt.bingx({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            klines = binance_exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=350)
            if klines:
                logger.info("Successfully fetched data from BingX spot")
                return self._process_klines_to_df(klines)
        except Exception as e:
            logger.warning(f"BingX spot fallback failed: {e}")
        
        # Fallback 3: Use cached data if available
        if len(self.market_data_buffer) >= 300:
            logger.warning("Using cached market data due to API failures")
            return self.market_data_buffer.copy()
        
        logger.error("All market data sources failed")
        return None

    def calculate_indicators(self, df):
        """Calculate technical indicators matching feature_engineer.py"""
        try:
            # Validate minimum data requirements
            min_candles = len(df)

            # Define minimum requirements for different indicator types
            requirements = {
                'critical': 15,    # Basic indicators (RSI_15, ATR_15, BB_15, etc.)
                'moderate': 60,    # Medium-term indicators (RSI_60, ATR_60, BB_60, etc.)
                'long_term': 300   # Long-term indicators (EMA_300, BB_300, RSI_300, etc.)
            }

            # Check if we have minimum viable data
            if min_candles < requirements['critical']:
                logger.error(f"CRITICAL: Insufficient data for basic indicators. Need at least {requirements['critical']} candles, got {min_candles}")
                return None

            # Log warnings for different data levels
            if min_candles < requirements['moderate']:
                logger.warning(f"WARNING: Limited data ({min_candles} candles). Medium-term indicators may be inaccurate.")
            if min_candles < requirements['long_term']:
                logger.warning(f"WARNING: Insufficient data for long-term indicators ({min_candles} < {requirements['long_term']} candles). EMA_300, BB_300, RSI_300, VAR_300, MFI_300 will be less reliable.")

            # Use all available data (need at least 300 for some indicators)
            data = df.copy()

            # Basic price data
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            # Calculate indicators (same as feature_engineer.py)
            indicators = {}

            # Helper function to safely get last value
            def safe_last_value(series, name):
                try:
                    value = series.iloc[-1]
                    if pd.isna(value) or np.isinf(value):
                        logger.warning(f"Invalid value for {name}: {value}")
                        return 0.0
                    return float(value)
                except (IndexError, TypeError) as e:
                    logger.warning(f"Could not calculate {name}: {e}")
                    return 0.0

            # EMA indicators
            indicators['EMA_15'] = safe_last_value(close.ewm(span=15, adjust=False).mean(), 'EMA_15')
            indicators['EMA_60'] = safe_last_value(close.ewm(span=60, adjust=False).mean(), 'EMA_60')
            indicators['EMA_300'] = safe_last_value(close.ewm(span=300, adjust=False).mean(), 'EMA_300')

            # Bollinger Bands
            for window in [15, 60, 300]:
                if len(close) >= window:
                    sma = close.rolling(window=window).mean()
                    std = close.rolling(window=window).std()
                    indicators[f'BB_{window}_upper'] = safe_last_value(sma + 2*std, f'BB_{window}_upper')
                    indicators[f'BB_{window}_lower'] = safe_last_value(sma - 2*std, f'BB_{window}_lower')
                else:
                    logger.warning(f"Insufficient data for BB_{window} (need {window}, got {len(close)})")
                    indicators[f'BB_{window}_upper'] = close.iloc[-1]
                    indicators[f'BB_{window}_lower'] = close.iloc[-1]

            # RSI indicators
            for window in [15, 60, 300]:
                if len(close) >= window + 1:
                    delta = close.diff()
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    ma_up = up.rolling(window=window).mean()
                    ma_down = down.rolling(window=window).mean()
                    rsi = 100 - (100 / (1 + ma_up / ma_down))
                    indicators[f'RSI_{window}'] = safe_last_value(rsi, f'RSI_{window}')
                else:
                    logger.warning(f"Insufficient data for RSI_{window} (need {window+1}, got {len(close)})")
                    indicators[f'RSI_{window}'] = 50.0

            # Ultimate Oscillator
            if len(data) >= 28:
                try:
                    indicators['ULTOSC'] = safe_last_value(
                        ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator(),
                        'ULTOSC'
                    )
                except Exception as e:
                    logger.warning(f"Error calculating ULTOSC: {e}")
                    indicators['ULTOSC'] = 50.0
            else:
                logger.warning(f"Insufficient data for ULTOSC (need 28, got {len(data)})")
                indicators['ULTOSC'] = 50.0

            # Volume indicators
            try:
                indicators['OBV'] = safe_last_value(
                    ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume(),
                    'OBV'
                )
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
                indicators['OBV'] = 0.0

            try:
                indicators['AD'] = safe_last_value(
                    ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index(),
                    'AD'
                )
            except Exception as e:
                logger.warning(f"Error calculating AD: {e}")
                indicators['AD'] = 0.0

            # ATR indicators
            for window in [15, 60]:
                if len(data) >= window + 1:
                    try:
                        indicators[f'ATR_{window}'] = safe_last_value(
                            ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range(),
                            f'ATR_{window}'
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating ATR_{window}: {e}")
                        indicators[f'ATR_{window}'] = close.iloc[-1] * 0.01
                else:
                    logger.warning(f"Insufficient data for ATR_{window} (need {window+1}, got {len(data)})")
                    indicators[f'ATR_{window}'] = close.iloc[-1] * 0.01

            # Price Transform
            wclprice = (high + low + 2 * close) / 4
            indicators['WCLPRICE'] = safe_last_value(wclprice, 'WCLPRICE')

            # Statistical indicators (Variance)
            for window in [15, 60, 300]:
                if len(close) >= window:
                    indicators[f'VAR_{window}'] = safe_last_value(close.rolling(window=window).var(), f'VAR_{window}')
                else:
                    logger.warning(f"Insufficient data for VAR_{window} (need {window}, got {len(close)})")
                    indicators[f'VAR_{window}'] = close.var() if len(close) > 1 else 0.0

            # MFI indicators
            for window in [15, 60, 300]:
                if len(data) >= window:
                    try:
                        indicators[f'MFI_{window}'] = safe_last_value(
                            ta.volume.MFIIndicator(high, low, close, volume, window=window).money_flow_index(),
                            f'MFI_{window}'
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating MFI_{window}: {e}")
                        indicators[f'MFI_{window}'] = 50.0
                else:
                    logger.warning(f"Insufficient data for MFI_{window} (need {window}, got {len(data)})")
                    indicators[f'MFI_{window}'] = 50.0
            
            # MACD indicators (26, 12, 9 parameters as in feature_engineer.py)
            try:
                ema_fast = close.ewm(span=12, adjust=False).mean()
                ema_slow = close.ewm(span=26, adjust=False).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.rolling(window=9).mean()
                histogram = macd - signal_line
                indicators['MACD_default_macd'] = safe_last_value(macd, 'MACD_default_macd')
                indicators['MACD_default_signal'] = safe_last_value(signal_line, 'MACD_default_signal')
                indicators['MACD_default_histogram'] = safe_last_value(histogram, 'MACD_default_histogram')
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                indicators['MACD_default_macd'] = 0.0
                indicators['MACD_default_signal'] = 0.0
                indicators['MACD_default_histogram'] = 0.0
            
            # Stochastic oscillator (14, 3, 3 parameters as in feature_engineer.py)
            try:
                stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
                indicators['Stochastic_slowk'] = safe_last_value(stoch_indicator.stoch(), 'Stochastic_slowk')  # raw K
                indicators['Stochastic_slowd'] = safe_last_value(stoch_indicator.stoch_signal(), 'Stochastic_slowd')  # smoothed D
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {e}")
                indicators['Stochastic_slowk'] = 50.0
                indicators['Stochastic_slowd'] = 50.0

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def _init_live_environment(self, df):
        """Initialize the EnhancedTradingEnvironment with market data for consistent state calculation"""
        try:
            # Create a DataFrame with the required columns for the environment
            env_df = df.copy()
            
            # Ensure we have the required columns
            if 'close' not in env_df.columns and 'Close' in env_df.columns:
                env_df['close'] = env_df['Close']
            
            # Calculate technical indicators for the environment DataFrame
            # These are the indicators expected by EnhancedTradingEnvironment
            close = env_df['close']
            high = env_df['high']
            low = env_df['low']
            volume = env_df['volume']
            
            # RSI_15
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(window=15).mean()
            ma_down = down.rolling(window=15).mean()
            env_df['RSI_15'] = 100 - (100 / (1 + ma_up / ma_down))
            
            # Bollinger Bands 15
            sma_15 = close.rolling(window=15).mean()
            std_15 = close.rolling(window=15).std()
            env_df['BB_15_upper'] = sma_15 + 2 * std_15
            env_df['BB_15_lower'] = sma_15 - 2 * std_15
            
            # ATR_15
            env_df['ATR_15'] = ta.volatility.AverageTrueRange(high, low, close, window=15).average_true_range()
            
            # OBV
            env_df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            
            # AD (Accumulation/Distribution)
            env_df['AD'] = ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
            
            # MFI_15
            env_df['MFI_15'] = ta.volume.MFIIndicator(high, low, close, volume, window=15).money_flow_index()
            
            # MACD indicators (26, 12, 9 parameters as in feature_engineer.py)
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            env_df['MACD_default_macd'] = ema_fast - ema_slow
            env_df['MACD_default_signal'] = env_df['MACD_default_macd'].rolling(window=9).mean()
            env_df['MACD_default_histogram'] = env_df['MACD_default_macd'] - env_df['MACD_default_signal']
            
            # Stochastic oscillator (14, 3, 3 parameters as in feature_engineer.py)
            stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            env_df['Stochastic_slowk'] = stoch_indicator.stoch()  # raw K
            env_df['Stochastic_slowd'] = stoch_indicator.stoch_signal()  # smoothed D
            
            # Fill NaN values
            env_df = env_df.bfill().ffill()
            
            # Store the DataFrame
            self.env_df = env_df
            
            # Create the EnhancedTradingEnvironment
            # CRITICAL: Use training_initial_balance for correct state normalization
            self.live_env = EnhancedTradingEnvironment(
                df=env_df,
                initial_balance=self.training_initial_balance,  # Use training value for state normalization
                transaction_fee=0.0018,
                episode_length=len(env_df),
                start_step=len(env_df) - 1,  # Start at the latest data point
                debug=False,
                enable_strategy_balancing=True
            )
            
            # Sync the environment's internal state with our live trading state
            self._sync_env_state()
            
            logger.info(f"Initialized EnhancedTradingEnvironment with {len(env_df)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing live environment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _sync_env_state(self):
        """Synchronize the EnhancedTradingEnvironment's internal state with live trading state
        
        CRITICAL: State normalization uses training_initial_balance, not live balance.
        We need to scale the balance to match training normalization.
        """
        if self.live_env is None:
            return
        
        # CRITICAL: Scale balance to training_initial_balance for correct state normalization
        # The model expects states normalized with training_initial_balance (10000)
        # So we need to present the balance as if it were in the training scale
        balance_ratio = self.balance / self.initial_balance  # Current balance ratio
        scaled_balance = self.training_initial_balance * balance_ratio  # Scale to training units
        
        # Sync balance (scaled for correct normalization)
        self.live_env.balance = scaled_balance
        self.live_env.position = self.position
        
        # Sync margin trading state (also scaled)
        margin_ratio = self.margin_locked / self.initial_balance if self.initial_balance > 0 else 0
        self.live_env.margin_locked = self.training_initial_balance * margin_ratio
        self.live_env.short_position_value = self.short_position * (self.short_entry_price if self.short_entry_price > 0 else 0)
        
        # Sync position tracking
        self.live_env.entry_price = self.long_entry_price if self.long_position > 0 else self.short_entry_price
        self.live_env.highest_price_since_entry = self.highest_price_since_entry
        self.live_env.lowest_price_since_entry = self.lowest_price_since_entry
        self.live_env.trailing_stop_loss = self.trailing_stop_loss
        self.live_env.trailing_take_profit = self.trailing_take_profit
        
        # Sync portfolio tracking (scaled)
        self.live_env.total_fees = self.total_fees
        portfolio_value = self.balance + self.margin_locked + self.position * (self.env_df['close'].iloc[-1] if self.env_df is not None else 0)
        portfolio_ratio = portfolio_value / self.initial_balance if self.initial_balance > 0 else 1.0
        self.live_env.prev_portfolio_value = self.training_initial_balance * portfolio_ratio
        
        # Set current step to the last data point
        if self.env_df is not None:
            self.live_env.current_step = len(self.env_df) - 1

    def _update_env_dataframe(self, new_candle):
        """Update the environment DataFrame with new market data"""
        if self.env_df is None:
            return
        
        try:
            # Append new candle
            self.env_df = pd.concat([self.env_df, pd.DataFrame([new_candle])], ignore_index=True)
            
            # Recalculate indicators for the new row
            close = self.env_df['close']
            high = self.env_df['high']
            low = self.env_df['low']
            volume = self.env_df['volume']
            
            # Update RSI_15 for the last row
            delta = close.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(window=15).mean()
            ma_down = down.rolling(window=15).mean()
            self.env_df['RSI_15'] = 100 - (100 / (1 + ma_up / ma_down))
            
            # Update Bollinger Bands 15
            sma_15 = close.rolling(window=15).mean()
            std_15 = close.rolling(window=15).std()
            self.env_df['BB_15_upper'] = sma_15 + 2 * std_15
            self.env_df['BB_15_lower'] = sma_15 - 2 * std_15
            
            # Update ATR_15
            self.env_df['ATR_15'] = ta.volatility.AverageTrueRange(high, low, close, window=15).average_true_range()
            
            # Update OBV
            self.env_df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            
            # Update AD
            self.env_df['AD'] = ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
            
            # Update MFI_15
            self.env_df['MFI_15'] = ta.volume.MFIIndicator(high, low, close, volume, window=15).money_flow_index()
            
            # Update MACD indicators
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            self.env_df['MACD_default_macd'] = ema_fast - ema_slow
            self.env_df['MACD_default_signal'] = self.env_df['MACD_default_macd'].rolling(window=9).mean()
            self.env_df['MACD_default_histogram'] = self.env_df['MACD_default_macd'] - self.env_df['MACD_default_signal']
            
            # Update Stochastic oscillator
            stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            self.env_df['Stochastic_slowk'] = stoch_indicator.stoch()  # raw K
            self.env_df['Stochastic_slowd'] = stoch_indicator.stoch_signal()  # smoothed D
            
            # Fill NaN values
            self.env_df = self.env_df.bfill().ffill()
            
            # Update the environment's DataFrame reference
            self.live_env.df = self.env_df
            self.live_env.current_step = len(self.env_df) - 1
            
            # Update rolling normalization values
            window_size = min(100, len(self.env_df) // 10)
            self.live_env.price_rolling_mean = self.env_df['close'].rolling(window=window_size, min_periods=1).mean().values.copy()
            self.live_env.price_rolling_std = self.env_df['close'].rolling(window=window_size, min_periods=1).std().values.copy()
            self.live_env.price_rolling_std[self.live_env.price_rolling_std == 0] = 1
            
        except Exception as e:
            logger.error(f"Error updating environment DataFrame: {e}")

    def get_rl_state(self, indicators, current_price):
        """Create state vector for RL model using EnhancedTradingEnvironment's _get_state method
        
        This ensures the state calculation is EXACTLY the same as during training.
        """
        try:
            # If the environment is initialized, use its _get_state method for consistency
            if self.live_env is not None:
                # Sync the environment's internal state with our live trading state
                self._sync_env_state()
                
                # Get the state from the environment (same as training)
                state = self.live_env._get_state()
                
                logger.debug(f"State from EnhancedTradingEnvironment: {state[:3]}...")
                return state
            
            # Fallback: Manual state calculation (only if environment not initialized)
            logger.warning("EnhancedTradingEnvironment not initialized, using fallback state calculation")
            
            # Normalize values (same as in TradingEnvironment)
            balance_norm = self.balance / self.initial_balance - 1
            # Fix position normalization for short positions
            position_norm = self.position / (abs(self.balance) * 0.1) if self.balance != 0 else 0

            # Use historical price data for normalization if available
            if self.price_history:
                price_mean = np.mean(self.price_history[-100:])  # Use recent prices for normalization
                price_std = np.std(self.price_history[-100:])
            else:
                price_mean = current_price
                price_std = current_price * 0.1  # 10% std as fallback

            price_norm = (current_price - price_mean) / price_std if price_std > 0 else 0

            # Calculate price trends (short and medium term)
            short_trend = 0
            medium_trend = 0
            if len(self.price_history) >= 5:
                short_trend = (current_price - self.price_history[-5]) / self.price_history[-5] if self.price_history[-5] > 0 else 0
                short_trend = np.clip(short_trend, -0.1, 0.1) * 10
            if len(self.price_history) >= 20:
                medium_trend = (current_price - self.price_history[-20]) / self.price_history[-20] if self.price_history[-20] > 0 else 0
                medium_trend = np.clip(medium_trend, -0.2, 0.2) * 5

            # Technical indicators (same as in EnhancedTradingEnvironment)
            # Normalize MACD values
            macd = indicators.get('MACD_default_macd', 0)
            macd_signal = indicators.get('MACD_default_signal', 0)
            macd_hist = indicators.get('MACD_default_histogram', 0)
            macd_norm = macd / current_price * 100 if current_price > 0 else 0
            macd_signal_norm = macd_signal / current_price * 100 if current_price > 0 else 0
            macd_hist_norm = macd_hist / current_price * 100 if current_price > 0 else 0
            
            # Explicit position flags (same as in EnhancedTradingEnvironment)
            has_long = 1.0 if self.position > 0 else 0.0
            has_short = 1.0 if self.position < 0 else 0.0
            
            # Calculate unrealized PnL
            unrealized_pnl = 0
            if self.position > 0 and self.long_entry_price > 0:
                unrealized_pnl = (current_price - self.long_entry_price) / self.long_entry_price
            elif self.position < 0 and self.short_entry_price > 0:
                unrealized_pnl = (self.short_entry_price - current_price) / self.short_entry_price
            unrealized_pnl = np.clip(unrealized_pnl, -0.5, 0.5) * 2
            
            indicators_list = [
                indicators.get('RSI_15', 50) / 100 - 0.5,
                (indicators.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,
                (indicators.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,
                indicators.get('ATR_15', 100) / 1000,
                short_trend,  # 5-step price trend (replaced OBV)
                medium_trend,  # 20-step price trend (replaced AD)
                indicators.get('MFI_15', 50) / 100 - 0.5,
                # MACD indicators for trend confirmation
                np.clip(macd_norm, -1, 1),
                np.clip(macd_signal_norm, -1, 1),
                np.clip(macd_hist_norm, -1, 1),
                # Stochastic oscillator for overbought/oversold
                indicators.get('Stochastic_slowk', 50) / 100 - 0.5,
                indicators.get('Stochastic_slowd', 50) / 100 - 0.5,
            ]

            # Build state with explicit position flags (same format as EnhancedTradingEnvironment)
            state = np.array([
                balance_norm, 
                position_norm, 
                price_norm,
                has_long,  # Explicit flag: 1 if long position open
                has_short,  # Explicit flag: 1 if short position open
                unrealized_pnl  # Current unrealized PnL
            ] + indicators_list, dtype=np.float32)
            return state

        except Exception as e:
            logger.error(f"Error creating RL state: {e}")
            return np.zeros(18, dtype=np.float32)  # Fallback state (6 base + 12 indicators)

    def execute_virtual_trade(self, action, current_price):
        """Execute virtual trade with separate long/short position tracking"""
        try:
            # Check trading rate limit
            current_time = time.time()
            if current_time - self.last_trade_time < self.min_trade_interval:
                logger.debug(f"Trade rate limited. Last trade: {current_time - self.last_trade_time:.1f}s ago, min interval: {self.min_trade_interval}s")
                return False

            trade_executed = False

            if action == 1:  # Buy Long - открыть/добавить длинную позицию
                # Risk management: Limit long position size
                max_long_value = self.initial_balance * 0.7  # Maximum 70% of initial balance in long positions
                current_long_value = self.long_position * current_price
                available_for_long = max_long_value - current_long_value
                
                # Calculate position size with risk limits
                invest_amount = min(self.balance * 0.2, available_for_long, 200)  # Max 20% of balance, max $200, respect limits
                
                if invest_amount > 10 and available_for_long > 0:  # Minimum trade $10 and check limits
                    fee = self.calculate_fee(invest_amount)
                    # Check if we can afford the fee and investment
                    if self.balance >= invest_amount + fee:
                        btc_amount = (invest_amount - fee) / current_price

                        # Risk check: Don't open if already heavily long
                        if self.long_position > 0:
                            current_long_exposure = self.long_position * current_price
                            if current_long_exposure > self.initial_balance * 0.8:  # 80% exposure limit
                                logger.warning(f"⚠️  Long position exposure too high: ${current_long_exposure:.2f}, skipping new long")
                                return False

                        # Update long position with correct average price calculation
                        old_position_value = self.long_entry_price * self.long_position
                        new_position_value = current_price * btc_amount
                        total_value = old_position_value + new_position_value
                        total_btc = self.long_position + btc_amount
                        self.long_position = total_btc
                        if total_btc > 0:
                            self.long_entry_price = total_value / total_btc

                        # Update overall position
                        self.position = self.long_position - self.short_position
                        self.balance -= invest_amount + fee
                        self.total_fees += fee

                        # Reset trailing parameters for long position
                        self.highest_price_since_entry = current_price
                        self.trailing_stop_loss = current_price * (1 - self.trailing_stop_distance)
                        self.trailing_take_profit = current_price * (1 + self.take_profit_pct)

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'BUY_LONG',
                            'price': current_price,
                            'amount': btc_amount,
                            'value': invest_amount,
                            'fee': fee,
                            'balance_after': self.balance,
                            'position_after': self.position,
                            'long_position': self.long_position,
                            'short_position': self.short_position
                        }

                        self.trades.append(trade)
                        self.log_trade_to_csv(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL BUY LONG: {btc_amount:.6f} BTC at ${current_price:.2f}, Long Position: {self.long_position:.6f}, Long avg price: ${self.long_entry_price:.2f}")
                        self.last_trade_time = current_time

            elif action == 2:  # Sell Long - закрыть часть длинной позиции
                if self.long_position > 0:
                    # Always close the entire long position when signaled
                    sell_amount = self.long_position
                    revenue = sell_amount * current_price
                    fee = self.calculate_fee(revenue)
                    revenue_after_fee = revenue - fee

                    pnl = (current_price - self.long_entry_price) * sell_amount - fee

                    # Update long position
                    self.long_position = 0

                    # Update overall position
                    self.position = self.long_position - self.short_position
                    self.balance += revenue_after_fee
                    self.total_fees += fee

                    # Update cumulative P&L
                    self.cumulative_pnl += pnl

                    # Reset trailing parameters
                    self.highest_price_since_entry = 0
                    self.trailing_stop_loss = 0
                    self.trailing_take_profit = 0

                    trade = {
                        'timestamp': datetime.now(),
                        'type': 'SELL_LONG',
                        'price': current_price,
                        'amount': sell_amount,
                        'value': revenue_after_fee,
                        'fee': fee,
                        'pnl': pnl,
                        'cumulative_pnl': self.cumulative_pnl,
                        'balance_after': self.balance,
                        'position_after': self.position,
                        'long_position': self.long_position,
                        'short_position': self.short_position
                    }

                    self.trades.append(trade)
                    self.log_trade_to_csv(trade)
                    trade_executed = True

                    logger.info(f"VIRTUAL SELL LONG: {sell_amount:.6f} BTC at ${current_price:.2f}, Trade PnL: ${pnl:.2f}, Cumulative P&L: ${self.cumulative_pnl:.2f}")
                    self.last_trade_time = current_time

            elif action == 3:  # Sell Short - открыть короткую позицию
                # Risk management: Limit short position size
                max_short_value = self.initial_balance * 0.5  # Maximum 50% of initial balance in short positions
                current_short_value = self.short_position * current_price
                available_for_short = max_short_value - current_short_value
                
                # Calculate position size with risk limits
                short_value = min(self.balance * 0.1, available_for_short, 100)  # Max 10% of balance, max $100, respect limits
                
                if short_value > 10 and available_for_short > 0:  # Minimum trade $10 and check limits
                    fee = self.calculate_fee(short_value)
                    
                    # Calculate margin required for short position
                    margin_required = short_value * self.margin_requirement
                    available_balance = self.balance - self.margin_locked
                    
                    if available_balance >= margin_required:
                        btc_amount = short_value / current_price
                        
                        # Store opening fee to avoid double-counting on cover
                        self.short_opening_fees = fee

                        # Lock margin and receive proceeds (proceeds go to balance, but we owe btc_amount)
                        self.margin_locked += margin_required
                        self.balance -= margin_required
                        self.balance += short_value - fee  # Receive proceeds minus fee
                        self.short_position += btc_amount
                        self.total_fees += fee

                        # Risk check: Don't open if already heavily shorted
                        if self.short_position > 0:
                            current_short_exposure = self.short_position * current_price
                            if current_short_exposure > self.initial_balance * 0.8:  # 80% exposure limit
                                logger.warning(f"⚠️  Short position exposure too high: ${current_short_exposure:.2f}, skipping new short")
                                # Rollback the changes
                                self.margin_locked -= margin_required
                                self.balance += margin_required
                                self.balance -= short_value - fee
                                self.short_position -= btc_amount
                                self.total_fees -= fee
                                return False

                        # Update short position with correct average price calculation
                        old_position_value = self.short_entry_price * (self.short_position - btc_amount)
                        new_position_value = current_price * btc_amount
                        total_value = old_position_value + new_position_value
                        total_btc = self.short_position
                        if total_btc > 0:
                            self.short_entry_price = total_value / total_btc

                        # Update overall position (negative because it's short)
                        self.position = self.long_position - self.short_position

                        # Reset trailing parameters for short position
                        self.lowest_price_since_entry = current_price
                        self.trailing_stop_loss = current_price * (1 + self.trailing_stop_distance)
                        self.trailing_take_profit = current_price * (1 - self.take_profit_pct)

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'SELL_SHORT',
                            'price': current_price,
                            'amount': btc_amount,
                            'value': short_value,
                            'fee': fee,
                            'pnl': 0,  # No P&L on entry for short
                            'cumulative_pnl': self.cumulative_pnl,
                            'balance_after': self.balance,
                            'position_after': self.position,
                            'long_position': self.long_position,
                            'short_position': self.short_position
                        }

                        self.trades.append(trade)
                        self.log_trade_to_csv(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL SELL SHORT: {btc_amount:.6f} BTC at ${current_price:.2f}, Short Position: {self.short_position:.6f}, Short avg price: ${self.short_entry_price:.2f}, Margin Locked: ${self.margin_locked:.2f}")
                        self.last_trade_time = current_time
                    else:
                        logger.warning(f"Insufficient margin to open short position. Need: ${margin_required:.2f}, Available: ${available_balance:.2f}")
                        return False

            elif action == 4:  # Buy Short - закрыть короткую позицию
                if self.short_position > 0:  # Have short position
                    # Always close the entire short position when signaled
                    cover_amount = self.short_position
                    cost = cover_amount * current_price
                    fee = self.calculate_fee(cost)

                    # Calculate PnL for short position
                    # Price PnL: we sold at entry_price, buying back at current_price
                    price_pnl = (self.short_entry_price - current_price) * cover_amount
                    
                    # Use stored opening fee instead of recalculating
                    open_fee = self.short_opening_fees
                    close_fee = self.calculate_fee(cost)
                    
                    # Total PnL: price difference minus closing fee (opening fee already deducted)
                    pnl = price_pnl - close_fee

                    # Check sufficient balance to buy back the BTC
                    # We need: cost (to buy BTC) + close_fee
                    # We have: balance + margin_locked (which will be returned)
                    net_cash_needed = cost + close_fee - self.margin_locked
                    
                    if self.balance >= net_cash_needed:
                        # Update short position
                        self.short_position = 0

                        # Correct balance calculation:
                        # 1. Deduct cost of buying back BTC
                        # 2. Return margin_locked
                        # 3. Price PnL is already accounted for in the cost vs proceeds
                        self.balance = self.balance - cost - close_fee + self.margin_locked
                        self.margin_locked = 0
                        self.total_fees += close_fee  # Only count closing fee here
                        self.short_opening_fees = 0.0  # Reset

                        # Update overall position
                        self.position = self.long_position - self.short_position

                        # Update cumulative P&L
                        self.cumulative_pnl += pnl

                        # Reset trailing parameters
                        self.lowest_price_since_entry = float('inf')
                        self.trailing_stop_loss = 0
                        self.trailing_take_profit = 0

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'BUY_SHORT',
                            'price': current_price,
                            'amount': cover_amount,
                            'value': cost,
                            'fee': fee,
                            'pnl': pnl,
                            'cumulative_pnl': self.cumulative_pnl,
                            'balance_after': self.balance,
                            'position_after': self.position,
                            'long_position': self.long_position,
                            'short_position': self.short_position
                        }

                        self.trades.append(trade)
                        self.log_trade_to_csv(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL BUY SHORT: {cover_amount:.6f} BTC at ${current_price:.2f}, Trade PnL: ${pnl:.2f}, Cumulative P&L: ${self.cumulative_pnl:.2f}")
                        self.last_trade_time = current_time
                    else:
                        logger.warning(f"Insufficient funds to cover short position. Need: ${net_cash_needed:.2f}, Have: ${self.balance:.2f}")
                        return False

            return trade_executed

        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    def run_live_session_websocket(self, duration_minutes=60):
        """Run live trading session with WebSocket real-time price updates"""
        logger.info(f"🚀 Starting RL Live Trading Session (WebSocket) for {duration_minutes} minutes")
        print("=" * 70)
        print("🌐 WebSocket Real-Time Trading Mode")
        print("⚡ Rapid price change detection enabled")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Symbol: {self.symbol}")
        print(f"Test Mode: {self.test_mode}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print("=" * 70)

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # WebSocket price monitoring
        import websocket
        import json
        import threading

        last_price = None
        price_change_threshold = 0.001  # 0.1% price change threshold for decision
        last_decision_time = time.time() - 60  # Allow immediate first decision
        min_decision_interval = 5  # Minimum 5 seconds between decisions

        def on_message(ws, message):
            nonlocal last_price, last_decision_time

            try:
                data = json.loads(message)

                # Check if this is a trade event (Binance format)
                if 'e' in data and data['e'] == 'trade' and 'p' in data:
                    current_price = float(data['p'])
                # Alternative format with data array
                elif 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0 and 'price' in data['data'][0]:
                    current_price = float(data['data'][0]['price'])
                else:
                    return  # Skip non-trade messages

                # Log current price occasionally (every 10 seconds)
                current_time = time.time()
                if not hasattr(on_message, 'last_price_log') or current_time - on_message.last_price_log > 10:
                    logger.info(f"💰 Current BTC price: ${current_price:.2f}")
                    on_message.last_price_log = current_time

                # Initialize last_price on first message
                if last_price is None:
                    last_price = current_price
                    self.price_history.append(current_price)
                    logger.info(f"WebSocket connected. Initial price: ${current_price:.2f}")
                    return

                # Calculate price change
                price_change_pct = abs(current_price - last_price) / last_price

                # Add to price history
                self.price_history.append(current_price)

                # Decision trigger conditions
                time_since_last_decision = time.time() - last_decision_time
                significant_price_change = price_change_pct >= price_change_threshold
                minimum_time_passed = time_since_last_decision >= min_decision_interval

                if significant_price_change and minimum_time_passed:
                    # Price changed significantly - make trading decision
                    logger.info(f"⚡ Significant price change detected: {last_price:.2f} → {current_price:.2f} ({price_change_pct*100:.2f}%)")

                    # Process trading decision on significant price change
                    self.process_trading_decision(current_price)

                    last_decision_time = time.time()

                last_price = current_price

            except Exception as e:
                logger.error(f"WebSocket message error: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")

        def on_open(ws):
            # Subscribe to BTC/USDT trades stream (Binance format)
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [
                    f"{self.symbol.lower()}@trade"  # Real-time trade stream
                ],
                "id": 1
            }
            ws.send(json.dumps(subscribe_message))
            logger.info("Subscribed to real-time trade stream")

        # Initialize market data buffer
        print("Loading initial market data...")
        for attempt in range(5):
            df = self.get_market_data()
            if df is not None and len(df) >= 300:
                self.market_data_buffer = df
                last_price = df['close'].iloc[-1]
                print(f"✅ Initial data loaded with {len(df)} candles")
                break
            else:
                print(f"Attempt {attempt + 1}: Insufficient market data, retrying...")
                time.sleep(5)

        if len(self.market_data_buffer) < 300:
            logger.error("Failed to initialize market data buffer")
            return

        # Initialize the EnhancedTradingEnvironment with market data
        print("Initializing EnhancedTradingEnvironment for consistent state calculation...")
        if not self._init_live_environment(self.market_data_buffer):
            logger.error("Failed to initialize EnhancedTradingEnvironment")
            return
        print("✅ EnhancedTradingEnvironment initialized successfully")

        # Try Binance WebSocket first (public endpoint)
        try:
            import urllib.request
            urllib.request.urlopen('https://api.binance.com', timeout=5)
            use_binance = True
        except:
            use_binance = False

        if use_binance:
            # WebSocket URL for Binance (more reliable)
            ws_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
            logger.info("Using Binance WebSocket (fallback)")
        else:
            # WebSocket URL for BingX
            ws_url = "wss://ws.biex.com/ws"  # BingX official WebSocket endpoint

        # Start WebSocket connection
        ws = websocket.WebSocketApp(ws_url,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close,
                                  on_open=on_open)

        # Create and start WebSocket thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        try:
            while datetime.now() < end_time:
                time.sleep(1)  # Keep main thread alive

                # Progress logging every minute
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                if elapsed_minutes % 1 < 0.1 and int(elapsed_minutes) > 0 and int(elapsed_minutes) != getattr(on_message, 'last_progress_min', 0):
                    on_message.last_progress_min = int(elapsed_minutes)
                    # Calculate portfolio value that accounts for margin trading
                    # Portfolio = Available Balance + Margin Locked + Position_Value
                    position_value = self.position * (last_price or self.initial_balance) if self.position != 0 else 0
                    portfolio_value = self.balance + self.margin_locked + position_value
                    pnl = portfolio_value - self.initial_balance
                    logger.info(f"Session Progress: {elapsed_minutes:.0f}min | Portfolio: ${portfolio_value:.2f} | P&L: ${pnl:.2f} | Trades: {len(self.trades)}")

            ws.close()
            logger.info("RL Live trading session completed (WebSocket)")
            self.generate_report()

        except KeyboardInterrupt:
            ws.close()
            logger.info("Session interrupted by user")
            self.generate_report()

    def process_trading_decision(self, current_price):
        """Process trading decision when significant price change detected"""
        try:
            # Update market data buffer with latest price
            new_candle = {
                'timestamp': pd.Timestamp.now(),
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 1000.0  # Dummy volume for indicators
            }

            self.market_data_buffer = pd.concat([
                self.market_data_buffer,
                pd.DataFrame([new_candle])
            ]).tail(350)

            # Update the EnhancedTradingEnvironment DataFrame with new data
            if self.live_env is not None:
                self._update_env_dataframe(new_candle)

            # Calculate indicators with updated data
            indicators = self.calculate_indicators(self.market_data_buffer)
            if indicators is None:
                return

            # Create RL state using EnhancedTradingEnvironment
            state = self.get_rl_state(indicators, current_price)

            # Get action from RL model
            # Use stochastic sampling to allow exploration of all actions
            # Deterministic mode always picks the highest probability action (biased to SELL_SHORT)
            action, _ = self.model.predict(state, deterministic=False)
            if isinstance(action, (np.ndarray, torch.Tensor)):
                action = int(action.item())

            # Get action probabilities and log them
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            dist = self.model.policy.get_distribution(state_tensor)
            action_probs = dist.distribution.probs[0].detach().numpy()
            logger.info(f"Action weights: Hold={action_probs[0]:.4f}, Buy_Long={action_probs[1]:.4f}, Sell_Long={action_probs[2]:.4f}, Sell_Short={action_probs[3]:.4f}, Buy_Short={action_probs[4]:.4f}")

            # Log decision
            action_names = {0: 'HOLD', 1: 'BUY_LONG', 2: 'SELL_LONG', 3: 'SELL_SHORT', 4: 'BUY_SHORT'}
            logger.info(f"⚡ DECISION: Current price ${current_price:.2f}, Action: {action_names.get(action, f'UNKNOWN({action})')}")

            # Execute virtual trade
            trade_executed = self.execute_virtual_trade(action, current_price)

            # Record portfolio state - calculate correctly for margin trading
            # For margin trading, portfolio value = balance + margin_locked + position_value
            # Position value should account for short positions (negative for shorts)
            position_value = self.position * current_price
            portfolio_value = self.balance + self.margin_locked + position_value
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'price': current_price,
                'balance': self.balance,
                'position': self.position,
                'portfolio_value': portfolio_value,
                'action': action,
                'indicators': indicators
            })

            self.indicators_history.append(indicators)

        except Exception as e:
            logger.error(f"Error in trading decision: {e}")

    def run_live_session(self, duration_minutes=60, check_interval_seconds=10):
        """Run live trading session for specified duration with faster price checking"""
        logger.info(f"🚀 Starting RL Live Trading Session for {duration_minutes} minutes")
        logger.info(f"⚡ Price check interval: {check_interval_seconds} seconds")
        print("=" * 70)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Symbol: {self.symbol}")
        print(f"Test Mode: {self.test_mode}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print("=" * 70)

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Initialize market data buffer
        print("Initializing market data buffer...")
        for attempt in range(5):  # Try up to 5 times to get initial data
            df = self.get_market_data()
            if df is not None and len(df) >= 300:
                self.market_data_buffer = df
                print(f"✅ Market data buffer initialized with {len(df)} candles")
                break
            else:
                print(f"Attempt {attempt + 1}: Insufficient market data, retrying...")
                time.sleep(10)

        if len(self.market_data_buffer) < 300:
            logger.error("Failed to initialize market data buffer")
            return

        # Initialize the EnhancedTradingEnvironment with market data
        print("Initializing EnhancedTradingEnvironment for consistent state calculation...")
        if not self._init_live_environment(self.market_data_buffer):
            logger.error("Failed to initialize EnhancedTradingEnvironment")
            return
        print("✅ EnhancedTradingEnvironment initialized successfully")

        session_start = time.time()

        while datetime.now() < end_time:
            try:
                # Get fresh market data
                df = self.get_market_data()
                if df is None or df.empty:
                    logger.warning("No market data available, skipping iteration")
                    time.sleep(60)
                    continue

                # Update buffer with new data
                self.market_data_buffer = pd.concat([self.market_data_buffer, df]).drop_duplicates(subset=['timestamp']).tail(350)

                current_price = df['close'].iloc[-1]

                # Update the EnhancedTradingEnvironment DataFrame with new data
                if self.live_env is not None and len(df) > 0:
                    # Update environment with the latest candle
                    new_candle = {
                        'timestamp': df['timestamp'].iloc[-1],
                        'open': df['open'].iloc[-1],
                        'high': df['high'].iloc[-1],
                        'low': df['low'].iloc[-1],
                        'close': current_price,
                        'volume': df['volume'].iloc[-1]
                    }
                    self._update_env_dataframe(new_candle)

                # Calculate indicators
                indicators = self.calculate_indicators(self.market_data_buffer)
                if indicators is None:
                    logger.warning("Could not calculate indicators, skipping iteration")
                    time.sleep(60)
                    continue

                # Update trailing levels for long positions
                if self.long_position > 0:
                    # Update highest price since entry
                    if current_price > self.highest_price_since_entry:
                        self.highest_price_since_entry = current_price

                        # Update trailing stop-loss (3% below highest price)
                        self.trailing_stop_loss = self.highest_price_since_entry * (1 - self.trailing_stop_distance)

                        # Update trailing take-profit (10% above entry, but trails with price)
                        self.trailing_take_profit = max(self.trailing_take_profit,
                                                      self.long_entry_price * (1 + self.take_profit_pct),
                                                      self.highest_price_since_entry * 0.95)  # At least 5% profit

                # Update trailing levels for short positions
                if self.short_position > 0:
                    # Update lowest price since entry (good for shorts - price going down)
                    if current_price < self.lowest_price_since_entry:
                        self.lowest_price_since_entry = current_price

                        # Update trailing stop-loss (should be ABOVE current price for shorts - protect against price increases)
                        self.trailing_stop_loss = self.lowest_price_since_entry * (1 + self.trailing_stop_distance)

                        # Update trailing take-profit (should be BELOW entry price for shorts - profit from price decreases)
                        self.trailing_take_profit = min(self.trailing_take_profit,
                                                      self.short_entry_price * (1 - self.take_profit_pct),
                                                      self.lowest_price_since_entry * 0.95)  # Protection from too aggressive take-profit

                # Check for stop-loss or take-profit triggers
                stop_loss_triggered = False
                take_profit_triggered = False
                position_type = None

                if self.long_position > 0:
                    # Check stop-loss for long position
                    if current_price <= self.trailing_stop_loss:
                        stop_loss_triggered = True
                        position_type = "LONG"
                        logger.info(f"🛑 LONG STOP-LOSS triggered at ${current_price:.2f} (trailing: ${self.trailing_stop_loss:.2f})")

                    # Check take-profit for long position
                    elif current_price >= self.trailing_take_profit:
                        take_profit_triggered = True
                        position_type = "LONG"
                        logger.info(f"💰 LONG TAKE-PROFIT triggered at ${current_price:.2f} (trailing: ${self.trailing_take_profit:.2f})")

                elif self.short_position > 0:
                    # Check stop-loss for short position (price went up too much)
                    if current_price >= self.trailing_stop_loss:
                        stop_loss_triggered = True
                        position_type = "SHORT"
                        logger.info(f"🛑 SHORT STOP-LOSS triggered at ${current_price:.2f} (trailing: ${self.trailing_stop_loss:.2f})")

                    # Check take-profit for short position (price went down enough)
                    elif current_price <= self.trailing_take_profit:
                        take_profit_triggered = True
                        position_type = "SHORT"
                        logger.info(f"💰 SHORT TAKE-PROFIT triggered at ${current_price:.2f} (trailing: ${self.trailing_take_profit:.2f})")

                # Force close position if stop-loss or take-profit triggered
                if stop_loss_triggered or take_profit_triggered:
                    if position_type == "LONG" and self.long_position > 0:
                        revenue = self.long_position * current_price
                        fee = self.calculate_fee(revenue)
                        revenue_after_fee = revenue - fee

                        pnl = (current_price - self.long_entry_price) * self.long_position - fee

                        self.balance += revenue_after_fee
                        self.total_fees += fee
                        self.cumulative_pnl += pnl

                        trigger_type = "STOP-LOSS" if stop_loss_triggered else "TAKE-PROFIT"

                        trade = {
                            'timestamp': datetime.now(),
                            'type': f'FORCE_SELL_LONG_{trigger_type}',
                            'price': current_price,
                            'amount': self.long_position,
                            'value': revenue_after_fee,
                            'fee': fee,
                            'pnl': pnl,
                            'cumulative_pnl': self.cumulative_pnl,
                            'trigger_price': self.trailing_stop_loss if stop_loss_triggered else self.trailing_take_profit,
                            'balance_after': self.balance,
                            'position_after': self.long_position - self.short_position,
                            'long_position': 0,
                            'short_position': self.short_position
                        }

                        self.trades.append(trade)

                        # Reset long position
                        self.long_position = 0
                        self.long_entry_price = 0
                        self.highest_price_since_entry = 0
                        self.trailing_stop_loss = 0
                        self.trailing_take_profit = 0

                        # Update overall position
                        self.position = self.long_position - self.short_position

                        trade_executed = True
                        continue  # Skip normal RL action this step

                    elif position_type == "SHORT" and self.short_position > 0:
                        # For short positions, we need to BUY back the BTC we shorted
                        cost_to_cover = self.short_position * current_price  # Cost to buy back BTC
                        fee = self.calculate_fee(cost_to_cover)
                        total_cost = cost_to_cover + fee

                        # Calculate PnL for short position
                        # Price PnL: we sold at entry_price, buying back at current_price
                        price_pnl = (self.short_entry_price - current_price) * self.short_position
                        
                        # Use stored opening fee instead of recalculating
                        open_fee = self.short_opening_fees
                        close_fee = self.calculate_fee(cost_to_cover)
                        total_fee = close_fee  # Opening fee already counted
                        
                        pnl = price_pnl - close_fee  # Opening fee already deducted from balance

                        # Handle potential deficit (if price went up too much)
                        if self.balance >= total_cost:
                            # Return margin + PnL (opening fee was already deducted when opening)
                            self.balance = self.balance + self.margin_locked - total_cost
                            self.margin_locked = 0
                            self.total_fees += close_fee  # Only closing fee
                            self.short_opening_fees = 0.0  # Reset
                        else:
                            # Short position default - price went against us too much
                            deficit = total_cost - (self.balance + self.margin_locked)
                            self.balance = 0  # Balance cannot be negative
                            self.margin_locked = 0  # Release margin
                            self.short_opening_fees = 0.0  # Reset fees
                            self.cumulative_pnl -= deficit  # Additional loss from deficit
                            logger.error(f"🚨 SHORT POSITION DEFAULT! Deficit: ${deficit:.2f}")

                        self.cumulative_pnl += pnl

                        trigger_type = "STOP-LOSS" if stop_loss_triggered else "TAKE-PROFIT"

                        trade = {
                            'timestamp': datetime.now(),
                            'type': f'FORCE_BUY_SHORT_{trigger_type}',
                            'price': current_price,
                            'amount': self.short_position,
                            'value': total_cost,
                            'fee': fee,
                            'pnl': pnl,
                            'cumulative_pnl': self.cumulative_pnl,
                            'trigger_price': self.trailing_stop_loss if stop_loss_triggered else self.trailing_take_profit,
                            'balance_after': self.balance,
                            'position_after': self.long_position - self.short_position,
                            'long_position': self.long_position,
                            'short_position': 0
                        }

                        self.trades.append(trade)

                        # Reset short position
                        self.short_position = 0
                        self.short_entry_price = 0
                        self.lowest_price_since_entry = float('inf')
                        self.trailing_stop_loss = 0
                        self.trailing_take_profit = 0

                        # Update overall position
                        self.position = self.long_position - self.short_position

                        trade_executed = True
                        continue  # Skip normal RL action this step

                # Create RL state
                state = self.get_rl_state(indicators, current_price)

                # Get action from RL model
                # Use stochastic sampling to allow exploration of all actions
                # Deterministic mode always picks the highest probability action (biased to SELL_SHORT)
                action, _ = self.model.predict(state, deterministic=False)

                # Ensure action is a scalar int
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())

                # Get action probabilities (but don't log them)
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                dist = self.model.policy.get_distribution(state_tensor)
                action_probs = dist.distribution.probs[0].detach().numpy()

                # Execute virtual trade
                trade_executed = self.execute_virtual_trade(action, current_price)

                # Record portfolio state - calculate correctly for margin trading
                # For margin trading, portfolio value = balance + margin_locked + position_value
                # Position value should account for short positions (negative for shorts)
                position_value = self.position * current_price
                portfolio_value = self.balance + self.margin_locked + position_value
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'balance': self.balance,
                    'position': self.position,
                    'portfolio_value': portfolio_value,
                    'action': action,
                    'indicators': indicators
                })

                self.price_history.append(current_price)
                self.indicators_history.append(indicators)

                # Progress logging every minute (more frequent)
                elapsed_minutes = (time.time() - session_start) / 60
                if elapsed_minutes % 1 < 0.1:  # Log roughly every minute
                    pnl_pct = (self.cumulative_pnl / self.initial_balance) * 100
                    
                    # Calculate unrealized P&L for current positions
                    long_unrealized_pnl = 0
                    short_unrealized_pnl = 0
                    if self.long_position > 0 and self.long_entry_price > 0:
                        long_unrealized_pnl = (current_price - self.long_entry_price) * self.long_position
                    if self.short_position > 0 and self.short_entry_price > 0:
                        short_unrealized_pnl = (self.short_entry_price - current_price) * self.short_position
                    
                    total_unrealized_pnl = long_unrealized_pnl + short_unrealized_pnl
                    total_realized_pnl = self.cumulative_pnl
                    total_pnl = total_realized_pnl + total_unrealized_pnl
                    
                    # Calculate portfolio value that accounts for margin trading
                    # Portfolio = Available Balance + Margin Locked + Position_Value
                    # Position value should account for short positions (negative for shorts)
                    position_value = self.position * current_price
                    portfolio_value = self.balance + self.margin_locked + position_value
                    
                    # Create static dashboard display
                    self.display_dashboard(current_price, portfolio_value, elapsed_minutes)

                # Wait before next iteration (configurable interval)
                time.sleep(check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in live session: {e}")
                time.sleep(60)

        logger.info("RL Live trading session completed")
        self.generate_report()

    def display_dashboard(self, current_price, portfolio_value, elapsed_minutes):
        """Display static dashboard with colored blocks"""
        try:
            import sys
            
            # Calculate metrics
            pnl = portfolio_value - self.initial_balance
            pnl_pct = (pnl / self.initial_balance) * 100
            
            # Calculate unrealized P&L for current positions
            long_unrealized_pnl = 0
            short_unrealized_pnl = 0
            if self.long_position > 0 and self.long_entry_price > 0:
                long_unrealized_pnl = (current_price - self.long_entry_price) * self.long_position
            if self.short_position > 0 and self.short_entry_price > 0:
                short_unrealized_pnl = (self.short_entry_price - current_price) * self.short_position
            
            total_unrealized_pnl = long_unrealized_pnl + short_unrealized_pnl
            total_realized_pnl = self.cumulative_pnl
            total_pnl = total_realized_pnl + total_unrealized_pnl
            
            # Color codes
            GREEN = '\033[92m'
            RED = '\033[91m'
            YELLOW = '\033[93m'
            BLUE = '\033[94m'
            CYAN = '\033[96m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
            WHITE = '\033[97m'
            
            # Determine colors based on values
            pnl_color = GREEN if pnl >= 0 else RED
            pnl_symbol = "▲" if pnl >= 0 else "▼"
            
            long_color = GREEN if long_unrealized_pnl >= 0 else RED
            short_color = GREEN if short_unrealized_pnl >= 0 else RED
            
            # Build dashboard
            dashboard = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}
{BOLD}{CYAN}║                           🤖 RL TRADING DASHBOARD                           ║{RESET}
{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}TIME:{RESET} {elapsed_minutes:6.1f}min {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}PRICE:{RESET} ${current_price:10,.2f} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}TRADES:{RESET} {len(self.trades):3d} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}BALANCE:{RESET} ${self.balance:10,.2f} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}POSITION:{RESET} {self.position:+8.6f} BTC {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}LONG:{RESET} {self.long_position:+12.6f} BTC @${self.long_entry_price:8.2f} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}SHORT:{RESET} {self.short_position:+11.6f} BTC @${self.short_entry_price:8.2f} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}PORTFOLIO:{RESET} ${portfolio_value:10,.2f} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}P&L:{RESET} {pnl_color}{pnl_symbol}{pnl:+8.2f}{RESET} ({pnl_color}{pnl_pct:+6.2f}%{RESET}) {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}REALIZED:{RESET} {pnl_color}{total_realized_pnl:+8.2f}{RESET} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}UNREALIZED:{RESET} {pnl_color}{total_unrealized_pnl:+8.2f}{RESET} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}LONG P&L:{RESET} {long_color}{long_unrealized_pnl:+8.2f}{RESET} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}SHORT P&L:{RESET} {short_color}{short_unrealized_pnl:+8.2f}{RESET} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}TOTAL P&L:{RESET} {pnl_color}{total_pnl:+8.2f}{RESET} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}FEES:{RESET} ${self.total_fees:8.2f} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}
{BOLD}{CYAN}║{RESET} {BOLD}{WHITE}STATUS:{RESET} {GREEN if portfolio_value >= self.initial_balance else RED}{'🟢 PROFITABLE' if portfolio_value >= self.initial_balance else '🔴 IN LOSS'}{RESET} {BOLD}{WHITE}|{RESET} {BOLD}{WHITE}MARGIN:{RESET} {YELLOW if self.balance < 500 else GREEN}{self.balance:.2f} USDT{RESET} {BOLD}{CYAN}║{RESET}
{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
            
            # Clear screen and print dashboard
            sys.stdout.write('\033[2J\033[H')  # Clear screen and move cursor to top
            sys.stdout.write(dashboard)
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")

    def generate_report(self):
        """Generate trading report"""
        try:
            import pickle

            # Calculate final metrics
            if self.portfolio_history:
                final_portfolio_value = self.portfolio_history[-1]['portfolio_value']
            else:
                final_portfolio_value = self.initial_balance

            total_pnl = final_portfolio_value - self.initial_balance
            total_return = (total_pnl / self.initial_balance) * 100

            # Calculate Sharpe ratio (simplified)
            if len(self.portfolio_history) > 1:
                portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                if len(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0

            # Calculate max drawdown
            if self.portfolio_history:
                portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0

            # Generate report
            report = f"""
🤖 RL LIVE TRADING REPORT
{'=' * 70}

Session Summary:
- Symbol: {self.symbol}
- Test Mode: {self.test_mode}
- Duration: {len(self.portfolio_history)} minutes
- Initial Balance: ${self.initial_balance:.2f}
- Final Portfolio: ${final_portfolio_value:.2f}
- Total P&L: ${total_pnl:.2f}
- Total Return: {total_return:.2f}%

Risk Metrics:
- Sharpe Ratio: {sharpe_ratio:.4f}
- Max Drawdown: {max_drawdown:.4f}
- Total Fees: ${self.total_fees:.2f}

Trading Activity:
- Total Trades: {len(self.trades)}
- Buy Trades: {len([t for t in self.trades if t['type'] == 'BUY'])}
- Sell Trades: {len([t for t in self.trades if t['type'] == 'SELL'])}

Profit Analysis:
- Long Positions P&L: ${sum(t.get('pnl', 0) for t in self.trades if 'SELL_LONG' in t.get('type', '')):.2f}
- Short Positions P&L: ${sum(t.get('pnl', 0) for t in self.trades if 'BUY_SHORT' in t.get('type', '')):.2f}
- Forced Closures P&L: ${sum(t.get('pnl', 0) for t in self.trades if 'FORCE' in t.get('type', '')):.2f}

Final Position:
- Long BTC: {self.long_position:.6f}
- Short BTC: {self.short_position:.6f}
- Net BTC: {self.position:.6f}
- USDT Balance: ${self.balance:.2f}
"""

            print(report)
            logger.info("RL Live trading report generated")

            # Save detailed results
            results = {
                'summary': {
                    'initial_balance': self.initial_balance,
                    'final_portfolio': final_portfolio_value,
                    'total_pnl': total_pnl,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'total_fees': self.total_fees,
                    'total_trades': len(self.trades)
                },
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'price_history': self.price_history,
                'indicators_history': self.indicators_history
            }

            with open('rl_live_trading_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            logger.info("Detailed results saved to 'rl_live_trading_results.pkl'")

        except Exception as e:
            logger.error(f"Error generating report: {e}")

def main():
    """Main function to run RL live trading"""
    import argparse

    parser = argparse.ArgumentParser(description="RL Live Trading Bot")
    parser.add_argument("--model", default="ppo_trading_agent.zip",
                       help="Path to trained RL model")
    parser.add_argument("--symbol", default="BTC-USDT",
                       help="Trading symbol")
    parser.add_argument("--duration", type=int, default=2880,  # 2 days = 4
                       help="Trading duration in minutes")
    parser.add_argument("--balance", type=float, default=1000,
                       help="Initial virtual balance")
    parser.add_argument("--real-trades", action="store_true",
                       help="⚠️  WARNING: This would enable REAL trades (currently not implemented)")
    parser.add_argument("--websocket", action="store_true",
                       help="Enable WebSocket real-time trading mode (default: 1-minute intervals)")

    args = parser.parse_args()

    if args.real_trades:
        confirm = input("⚠️  WARNING: Real trading is not yet implemented for RL models! Use --real-trades only when ready. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborting.")
            return

    test_mode = not args.real_trades

    try:
        bot = RLLiveTradingBot(
            model_path=args.model,
            symbol=args.symbol,
            test_mode=test_mode,
            initial_balance=args.balance
        )

        # Choose trading method based on arguments
        if args.websocket:
            bot.run_live_session_websocket(duration_minutes=args.duration)
        else:
            bot.run_live_session(duration_minutes=args.duration)

    except Exception as e:
        logger.error(f"Error running RL live trading: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()