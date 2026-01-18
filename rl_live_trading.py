#!/usr/bin/env python3
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

# Import the unified trading environment
from trading_environment import TradingEnvironment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_live_trading.log'),
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

    def __init__(self, model_path, symbol='BTC-USDT', test_mode=True, initial_balance=1000):
        """
        Initialize the RL trading bot

        Args:
            model_path: Path to trained RL model (.zip)
            symbol: Trading symbol (default: BTC-USDT)
            test_mode: If True, only virtual trades (no real transactions)
            initial_balance: Virtual balance in USDT
        """
        self.symbol = symbol
        self.test_mode = test_mode
        self.initial_balance = initial_balance

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

        # Market data buffer (need enough data for indicators)
        self.market_data_buffer = []

        # Trading rate limiting (increased for real market conditions)
        self.last_trade_time = 0
        self.min_trade_interval = 300  # Minimum 5 minutes between trades for real market

        logger.info(f"Initialized RL Live Trading Bot for {symbol}")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Initial balance: {self.initial_balance} USDT")

    def calculate_fee(self, amount, fee_rate=0.0005):
        """Расчет комиссий для фьючерсов BingX (тейкер: 0.05%, мейкер: 0.02%)"""
        # Для бессрочных фьючерсов: 0.02% (мейкер) / 0.05% (тейкер)
        # Для стандартных фьючерсов: 0.045% (фиксированная, списывается при закрытии)
        # По умолчанию используем ставку тейкера (0.05%), так как бот использует рыночные ордера
        return amount * fee_rate

    def get_market_data(self):
        """Get current market data from BingX using CCXT"""
        try:
            # Get recent klines (1m interval, need at least 300 for indicators)
            symbol_ccxt = self.symbol.replace('-', '/')
            klines = self.exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=350)

            if not klines:
                logger.warning("No kline data received")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add missing columns for compatibility
            df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=1)
            df['quote_volume'] = df['volume'] * df['close']
            df['trades'] = 0
            df['taker_buy_volume'] = df['volume'] * 0.5
            df['taker_buy_quote_volume'] = df['taker_buy_volume'] * df['close']
            df['ignore'] = 0

            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)

            return df

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error accessing exchange: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Network error connecting to exchange: {e}")
            # Implement retry logic
            time.sleep(5)  # Wait before retry
            try:
                klines = self.exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=350)
                # Process data similarly
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
            except Exception as retry_e:
                logger.error(f"Retry failed: {retry_e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error getting market data: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators matching feature_engineer.py"""
        try:
            # Validate minimum data requirements
            min_candles = len(df)
            logger.info(f"Available candles: {min_candles}")

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

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def get_rl_state(self, indicators, current_price):
        """Create state vector for RL model"""
        try:
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

            # Technical indicators (same as in TradingEnvironment)
            indicators_list = [
                indicators.get('RSI_15', 50) / 100 - 0.5,
                (indicators.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,
                (indicators.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,
                indicators.get('ATR_15', 100) / 1000,
                indicators.get('OBV', 0) / 1e10,
                indicators.get('AD', 0) / 1e10,
                indicators.get('MFI_15', 50) / 100 - 0.5
            ]

            state = np.array([balance_norm, position_norm, price_norm] + indicators_list, dtype=np.float32)
            return state

        except Exception as e:
            logger.error(f"Error creating RL state: {e}")
            return np.zeros(10, dtype=np.float32)  # Fallback state

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
                # Limit position size to 50% of initial balance
                max_position_value = self.initial_balance * 0.5
                current_position_value = self.long_position * current_price
                available_for_long = max_position_value - current_position_value

                invest_amount = min(self.balance * 0.2, available_for_long)  # 20% от баланса
                if invest_amount > 10:  # Minimum trade
                    fee = self.calculate_fee(invest_amount)
                    if self.balance >= invest_amount:  # Check sufficient funds
                        btc_amount = (invest_amount - fee) / current_price

                        # Update long position with correct average price calculation
                        old_position_value = self.long_entry_price * self.long_position
                        new_position_value = current_price * btc_amount
                        total_value = old_position_value + new_position_value
                        self.long_position += btc_amount
                        self.long_entry_price = total_value / self.long_position if self.long_position > 0 else current_price

                        # Update overall position
                        self.position = self.long_position - self.short_position
                        self.balance -= invest_amount
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
                        trade_executed = True

                        logger.info(f"VIRTUAL BUY LONG: {btc_amount:.6f} BTC at ${current_price:.2f}, Long Position: {self.long_position:.6f}, Long avg price: ${self.long_entry_price:.2f}")
                        self.last_trade_time = current_time

            elif action == 2:  # Sell Long - закрыть часть длинной позиции
                if self.long_position > 0:
                    sell_amount = min(self.long_position * 0.5, self.long_position)  # Sell max 50% of long position
                    if sell_amount * current_price > 10:  # Minimum trade value
                        revenue = sell_amount * current_price
                        fee = self.calculate_fee(revenue)
                        revenue_after_fee = revenue - fee

                        pnl = (current_price - self.long_entry_price) * sell_amount - fee

                        # Update long position
                        self.long_position -= sell_amount

                        # Update overall position
                        self.position = self.long_position - self.short_position
                        self.balance += revenue_after_fee
                        self.total_fees += fee

                        # Update cumulative P&L
                        self.cumulative_pnl += pnl

                        # Reset trailing parameters if position closed
                        if self.long_position == 0:
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
                        trade_executed = True

                        logger.info(f"VIRTUAL SELL LONG: {sell_amount:.6f} BTC at ${current_price:.2f}, Trade PnL: ${pnl:.2f}, Cumulative P&L: ${self.cumulative_pnl:.2f}")
                        self.last_trade_time = current_time

            elif action == 3:  # Sell Short - открыть короткую позицию
                # Check if we can afford to open short
                short_value = min(self.balance * 0.1, self.balance)  # Max 10% of balance equivalent
                if short_value > 10:  # Minimum trade
                    fee = self.calculate_fee(short_value)
                    # Just check if we can afford the fee (more realistic margin check)
                    if self.balance >= fee:  # Can afford transaction fee
                        btc_amount = (short_value - fee) / current_price

                        # Update short position with correct average price calculation
                        old_position_value = self.short_entry_price * self.short_position
                        new_position_value = current_price * btc_amount
                        total_value = old_position_value + new_position_value
                        self.short_position += btc_amount
                        self.short_entry_price = total_value / self.short_position if self.short_position > 0 else current_price

                        # Update overall position
                        self.position = self.long_position - self.short_position
                        self.balance += short_value  # Receive money for shorting
                        self.total_fees += fee

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
                            'balance_after': self.balance,
                            'position_after': self.position,
                            'long_position': self.long_position,
                            'short_position': self.short_position
                        }

                        self.trades.append(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL SELL SHORT: {btc_amount:.6f} BTC at ${current_price:.2f}, Short Position: {self.short_position:.6f}")
                        self.last_trade_time = current_time

            elif action == 4:  # Buy Short - закрыть короткую позицию
                if self.short_position > 0:  # Have short position
                    cover_amount = min(self.short_position * 0.5, self.short_position)  # Cover max 50% short
                    if cover_amount * current_price > 10:  # Minimum trade value
                        cost = cover_amount * current_price
                        fee = self.calculate_fee(cost)
                        cost_with_fee = cost + fee

                        if self.balance >= cost_with_fee:  # Check sufficient funds to cover
                            # ПРАВИЛЬНЫЙ РАСЧЕТ P&L ДЛЯ ШОРТА
                            entry_price = self.short_entry_price if self.short_entry_price > 0 else current_price
                            pnl = (entry_price - current_price) * cover_amount - fee

                            # Update short position
                            self.short_position -= cover_amount

                            # Update overall position
                            self.position = self.long_position - self.short_position
                            self.balance -= cost_with_fee  # Pay to cover short
                            self.total_fees += fee

                            # Update cumulative P&L
                            self.cumulative_pnl += pnl

                            # Reset trailing parameters if position closed
                            if self.short_position == 0:
                                self.lowest_price_since_entry = float('inf')
                                self.trailing_stop_loss = 0
                                self.trailing_take_profit = 0

                            trade = {
                                'timestamp': datetime.now(),
                                'type': 'BUY_SHORT',
                                'price': current_price,
                                'amount': cover_amount,
                                'value': cost_with_fee,
                                'fee': fee,
                                'pnl': pnl,
                                'cumulative_pnl': self.cumulative_pnl,
                                'balance_after': self.balance,
                                'position_after': self.position,
                                'long_position': self.long_position,
                                'short_position': self.short_position
                            }

                            self.trades.append(trade)
                            trade_executed = True

                            logger.info(f"VIRTUAL BUY SHORT: {cover_amount:.6f} BTC at ${current_price:.2f}, Trade PnL: ${pnl:.2f}, Cumulative P&L: ${self.cumulative_pnl:.2f}")
                            self.last_trade_time = current_time
                        else:
                            logger.warning("Insufficient funds to cover short position")
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
                    portfolio_value = self.balance + (self.position * (last_price or self.initial_balance))
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

            # Calculate indicators with updated data
            indicators = self.calculate_indicators(self.market_data_buffer)
            if indicators is None:
                return

            # Create RL state
            state = self.get_rl_state(indicators, current_price)

            # Get action from RL model
            action, _ = self.model.predict(state, deterministic=True)
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

            # Record portfolio state
            portfolio_value = self.balance + (self.position * current_price)
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

                        # Profit from short: (entry_price - exit_price) * amount - fee
                        pnl = (self.short_entry_price - current_price) * self.short_position - fee

                        # Handle potential deficit (if price went up too much)
                        if self.balance >= total_cost:
                            self.balance -= total_cost
                        else:
                            # Short position default - price went against us too much
                            deficit = total_cost - self.balance
                            self.balance = 0  # Balance cannot be negative
                            self.cumulative_pnl -= deficit  # Additional loss from deficit
                            logger.error(f"🚨 SHORT POSITION DEFAULT! Deficit: ${deficit:.2f}")

                        self.total_fees += fee
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
                action, _ = self.model.predict(state, deterministic=True)

                # Ensure action is a scalar int
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())

                # Get action probabilities and log them
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                dist = self.model.policy.get_distribution(state_tensor)
                action_probs = dist.distribution.probs[0].detach().numpy()
                logger.info(f"Action weights: Hold={action_probs[0]:.4f}, Buy_Long={action_probs[1]:.4f}, Sell_Long={action_probs[2]:.4f}, Sell_Short={action_probs[3]:.4f}, Buy_Short={action_probs[4]:.4f}")

                # Determine action name for logging
                action_names = {0: 'HOLD', 1: 'BUY_LONG', 2: 'SELL_LONG', 3: 'SELL_SHORT', 4: 'BUY_SHORT'}
                logger.info(f"Current price: ${current_price:.2f}")
                logger.info(f"Chosen action: {action_names.get(action, f'UNKNOWN({action})')}")

                # Execute virtual trade
                trade_executed = self.execute_virtual_trade(action, current_price)

                # Record portfolio state
                portfolio_value = self.balance + (self.position * current_price)
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
                    logger.info(f"Session Progress: {elapsed_minutes:.1f}min | Portfolio: ${portfolio_value:.2f} | Balance: ${self.balance:.2f} | Long: {self.long_position:+.6f} BTC | Short: {self.short_position:+.6f} BTC | Net: {self.position:+.6f} BTC | Cumulative P&L: ${self.cumulative_pnl:+.2f} ({pnl_pct:+.2f}%) | Trades: {len(self.trades)}")

                # Wait before next iteration (configurable interval)
                time.sleep(check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in live session: {e}")
                time.sleep(60)

        logger.info("RL Live trading session completed")
        self.generate_report()

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
    parser.add_argument("--duration", type=int, default=60,
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
