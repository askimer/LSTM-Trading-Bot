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
import ccxt
import ta
from dotenv import load_dotenv
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

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

class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agent (same as in train_rl.py)
    """
    def __init__(self, df, initial_balance=10000, transaction_fee=0.0018):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.current_step = 0

        # Calculate dynamic price normalization based on data
        price_col = 'close' if 'close' in df.columns else 'Close'
        self.price_mean = df[price_col].mean()
        self.price_std = df[price_col].std()

        # Actions: 0=Hold, 1=Buy Long, 2=Sell Long, 3=Sell Short, 4=Buy Short
        self.action_space = spaces.Discrete(5)

        # State: [balance_norm, position_norm, price_norm, indicators...]
        n_indicators = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 + n_indicators,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_fees = 0
        self.portfolio_values = [self.initial_balance]

        return self._get_state(), {}

    def _get_state(self):
        """Get current state observation"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)

        row = self.df.iloc[self.current_step]

        # Normalize values
        balance_norm = self.balance / self.initial_balance - 1
        position_norm = self.position / (self.balance * 0.1) if self.balance > 0 else 0
        current_price = row.get('close', row.get('Close', self.price_mean))
        price_norm = (current_price - self.price_mean) / self.price_std

        # Technical indicators
        indicators = [
            row.get('RSI_15', 50) / 100 - 0.5,
            (row.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,
            (row.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,
            row.get('ATR_15', 100) / 1000,
            row.get('OBV', 0) / 1e10,
            row.get('AD', 0) / 1e10,
            row.get('MFI_15', 50) / 100 - 0.5
        ]

        state = np.array([balance_norm, position_norm, price_norm] + indicators, dtype=np.float32)
        return state

    def step(self, action):
        """Execute one step in environment"""
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            reward = 0
            return self._get_state(), reward, terminated, truncated, {}

        current_price = self.df.iloc[self.current_step].get('close', self.df.iloc[self.current_step].get('Close'))
        next_price = self.df.iloc[self.current_step + 1].get('close', self.df.iloc[self.current_step + 1].get('Close'))

        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_fee):
                invest_amount = min(self.balance * 0.1, self.balance - 100)
                fee = invest_amount * self.transaction_fee
                coins_bought = (invest_amount - fee) / current_price

                self.position += coins_bought
                self.balance -= invest_amount
                self.total_fees += fee
                reward -= 0.01

        elif action == 2:  # Sell
            if self.position > 0:
                sell_amount = self.position * 0.5
                revenue = sell_amount * current_price
                fee = revenue * self.transaction_fee

                self.position -= sell_amount
                self.balance += revenue - fee
                self.total_fees += fee
                reward -= 0.01

        # Calculate reward
        current_portfolio = self.balance + self.position * current_price
        next_portfolio = self.balance + self.position * next_price
        portfolio_change = (next_portfolio - current_portfolio) / current_portfolio if current_portfolio > 0 else 0

        if len(self.portfolio_values) >= 10:
            recent_portfolio_values = self.portfolio_values[-10:]
            returns = np.diff(recent_portfolio_values) / recent_portfolio_values[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            risk_penalty = volatility * 50
        else:
            risk_penalty = 0

        reward += (portfolio_change * 10000) - risk_penalty

        # Additional reward components
        price_change_pct = (next_price - current_price) / current_price if current_price > 0 else 0

        if action == 1 and price_change_pct > 0:
            reward += abs(price_change_pct) * 1000
        if action == 1 and price_change_pct < 0:
            reward -= abs(price_change_pct) * 500
        if action == 2 and price_change_pct < 0:
            reward += abs(price_change_pct) * 1000
        if action == 2 and price_change_pct > 0:
            reward -= abs(price_change_pct) * 500

        if action == 0 and self.position > 0:
            reward -= 0.01

        if current_portfolio < self.initial_balance * 0.5:
            reward -= 10

        self.portfolio_values.append(current_portfolio)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            terminated = True

        return self._get_state(), reward, terminated, truncated, {}

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

        logger.info(f"Initialized RL Live Trading Bot for {symbol}")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Initial balance: {self.initial_balance} USDT")

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

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
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
            position_norm = self.position / (self.balance * 0.1) if self.balance > 0 else 0

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
        """Execute virtual trade (no real transactions) - updated for short positions"""
        try:
            trade_executed = False

            if action == 1:  # Buy Long - открыть/добавить длинную позицию
                if self.balance > current_price * 1.002:  # Account for fee
                    invest_amount = min(self.balance * 0.1, self.balance * 0.5)  # Max 10% or 50% of balance
                    if invest_amount > 10:  # Minimum trade
                        fee = invest_amount * 0.0018
                        invest_after_fee = invest_amount - fee
                        btc_amount = invest_after_fee / current_price

                        self.position += btc_amount
                        self.balance -= invest_amount
                        self.entry_price = current_price
                        self.total_fees += fee

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'BUY_LONG',
                            'price': current_price,
                            'amount': btc_amount,
                            'value': invest_amount,
                            'fee': fee,
                            'balance_after': self.balance,
                            'position_after': self.position
                        }

                        self.trades.append(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL BUY LONG: {btc_amount:.6f} BTC at ${current_price:.2f}")

            elif action == 2:  # Sell Long - закрыть часть длинной позиции
                if self.position > 0:
                    sell_amount = min(self.position * 0.5, self.position)  # Sell max 50% position
                    if sell_amount * current_price > 10:  # Minimum trade value
                        revenue = sell_amount * current_price
                        fee = revenue * 0.0018
                        revenue_after_fee = revenue - fee

                        pnl = (current_price - self.entry_price) * sell_amount - fee

                        self.position -= sell_amount
                        self.balance += revenue_after_fee
                        self.total_fees += fee

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'SELL_LONG',
                            'price': current_price,
                            'amount': sell_amount,
                            'value': revenue_after_fee,
                            'fee': fee,
                            'pnl': pnl,
                            'balance_after': self.balance,
                            'position_after': self.position
                        }

                        self.trades.append(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL SELL LONG: {sell_amount:.6f} BTC at ${current_price:.2f}, PnL: ${pnl:.2f}")

            elif action == 3:  # Sell Short - открыть короткую позицию
                if self.balance > current_price * 1.002:  # Account for fee
                    short_value = min(self.balance * 0.1, self.balance)  # Max 10% of balance equivalent
                    if short_value > 10:  # Minimum trade
                        fee = short_value * 0.0018
                        short_after_fee = short_value - fee
                        btc_amount = short_after_fee / current_price

                        self.position -= btc_amount  # Negative position = short
                        self.balance += short_value  # Receive money for shorting
                        self.entry_price = current_price
                        self.total_fees += fee

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'SELL_SHORT',
                            'price': current_price,
                            'amount': btc_amount,
                            'value': short_value,
                            'fee': fee,
                            'balance_after': self.balance,
                            'position_after': self.position
                        }

                        self.trades.append(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL SELL SHORT: {btc_amount:.6f} BTC at ${current_price:.2f}")

            elif action == 4:  # Buy Short - закрыть короткую позицию
                if self.position < 0:  # Have short position
                    cover_amount = min(abs(self.position) * 0.5, abs(self.position))  # Cover max 50% short
                    if cover_amount * current_price > 10:  # Minimum trade value
                        cost = cover_amount * current_price
                        fee = cost * 0.0018
                        cost_with_fee = cost + fee

                        pnl = (self.entry_price - current_price) * cover_amount - fee  # Profit from short

                        self.position += cover_amount  # Reduce short position
                        self.balance -= cost_with_fee  # Pay to cover short
                        self.total_fees += fee

                        trade = {
                            'timestamp': datetime.now(),
                            'type': 'BUY_SHORT',
                            'price': current_price,
                            'amount': cover_amount,
                            'value': cost_with_fee,
                            'fee': fee,
                            'pnl': pnl,
                            'balance_after': self.balance,
                            'position_after': self.position
                        }

                        self.trades.append(trade)
                        trade_executed = True

                        logger.info(f"VIRTUAL BUY SHORT: {cover_amount:.6f} BTC at ${current_price:.2f}, PnL: ${pnl:.2f}")

            return trade_executed

        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    def run_live_session(self, duration_minutes=60):
        """Run live trading session for specified duration"""
        logger.info(f"🚀 Starting RL Live Trading Session for {duration_minutes} minutes")
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

                # Update trailing levels if we have a position
                if self.position > 0:
                    # Update highest price since entry
                    if current_price > self.highest_price_since_entry:
                        self.highest_price_since_entry = current_price

                        # Update trailing stop-loss (3% below highest price)
                        self.trailing_stop_loss = self.highest_price_since_entry * (1 - self.trailing_stop_distance)

                        # Update trailing take-profit (10% above entry, but trails with price)
                        self.trailing_take_profit = max(self.trailing_take_profit,
                                                      self.entry_price * (1 + self.take_profit_pct),
                                                      self.highest_price_since_entry * 0.95)  # At least 5% profit

                # Check for stop-loss or take-profit triggers
                stop_loss_triggered = False
                take_profit_triggered = False

                if self.position > 0:
                    # Check stop-loss
                    if current_price <= self.trailing_stop_loss:
                        stop_loss_triggered = True
                        logger.info(f"🛑 STOP-LOSS triggered at ${current_price:.2f} (trailing: ${self.trailing_stop_loss:.2f})")

                    # Check take-profit
                    elif current_price >= self.trailing_take_profit:
                        take_profit_triggered = True
                        logger.info(f"💰 TAKE-PROFIT triggered at ${current_price:.2f} (trailing: ${self.trailing_take_profit:.2f})")

                # Force sell if stop-loss or take-profit triggered
                if stop_loss_triggered or take_profit_triggered:
                    if self.position > 0:
                        revenue = self.position * current_price
                        fee = revenue * 0.0018
                        revenue_after_fee = revenue - fee

                        pnl = (current_price - self.entry_price) * self.position - fee

                        self.balance += revenue_after_fee
                        self.total_fees += fee

                        trigger_type = "STOP-LOSS" if stop_loss_triggered else "TAKE-PROFIT"

                        trade = {
                            'timestamp': datetime.now(),
                            'type': f'FORCE_SELL_{trigger_type}',
                            'price': current_price,
                            'amount': self.position,
                            'value': revenue_after_fee,
                            'fee': fee,
                            'pnl': pnl,
                            'trigger_price': self.trailing_stop_loss if stop_loss_triggered else self.trailing_take_profit,
                            'balance_after': self.balance,
                            'position_after': 0
                        }

                        self.trades.append(trade)

                        self.position = 0
                        self.entry_price = 0
                        self.trailing_stop_loss = 0
                        self.trailing_take_profit = 0
                        self.highest_price_since_entry = 0
                        self.lowest_price_since_entry = float('inf')

                        trade_executed = True
                        continue  # Skip normal RL action this step

                # Create RL state
                state = self.get_rl_state(indicators, current_price)

                # Get action from RL model
                action, _ = self.model.predict(state, deterministic=True)

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

                # Progress logging every 5 minutes
                elapsed_minutes = (time.time() - session_start) / 60
                if elapsed_minutes % 5 < 1:  # Log roughly every 5 minutes
                    pnl = portfolio_value - self.initial_balance
                    logger.info(f"Session Progress: {elapsed_minutes:.1f}min | Portfolio: ${portfolio_value:.2f} | P&L: ${pnl:.2f} | Trades: {len(self.trades)}")

                # Wait before next iteration (1 minute)
                time.sleep(60)

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

Final Position:
- BTC Held: {self.position:.6f}
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

        bot.run_live_session(duration_minutes=args.duration)

    except Exception as e:
        logger.error(f"Error running RL live trading: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
