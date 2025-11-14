#!/usr/bin/env python3
"""
Live Trading Module for BingX Futures
Performs virtual/paper trading on live market data without real transactions
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import pickle
import ccxt
import ta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingBot:
    """
    Live trading bot that uses trained LSTM model for virtual trading on BingX
    """

    def __init__(self, model_path, scaler_path, symbol='BTC-USDT', test_mode=True):
        """
        Initialize the trading bot

        Args:
            model_path: Path to trained LSTM model
            scaler_path: Path to data scaler
            symbol: Trading symbol (default: BTC-USDT)
            test_mode: If True, only virtual trades (no real transactions)
        """
        self.symbol = symbol
        self.test_mode = test_mode

        # Initialize BingX API
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
                'defaultType': 'future',  # Use futures
            }
        })

        # Load trained model and scalers
        self.load_model(model_path, scaler_path)

        # Trading parameters (from paper trading optimization)
        self.params = {
            'base_sell_threshold': 0.000318,
            'base_buy_threshold': 0.000541,
            'alpha_atr': 26.117467,
            'alpha_rsi': 29.549422,
            'sell_percentage': 0.314003,
            'buy_percentage': 0.671026,
            'window_size': 17,
            'min_profit_threshold': 0.768993
        }

        # Virtual portfolio
        self.initial_balance = 1000  # Virtual USDT balance
        self.balance = self.initial_balance
        self.position = 0  # BTC position (positive = long, negative = short)
        self.entry_price = 0
        self.total_fees = 0

        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.price_history = []

        # Technical indicators history
        self.indicators_history = []

        logger.info(f"Initialized Live Trading Bot for {symbol}")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Initial balance: {self.initial_balance} USDT")

    def load_model(self, model_path, scaler_path):
        """Load trained LSTM model and scalers"""
        try:
            # Load LSTM model
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if hasattr(checkpoint, 'module_'):
                self.model = checkpoint.module_
            else:
                self.model = checkpoint

            self.model.eval()
            logger.info(f"Loaded model from {model_path}")

            # Load scalers
            scaler_X_path = scaler_path.replace('scaler.pkl', 'scaler_X.pkl')
            scaler_y_path = scaler_path.replace('scaler.pkl', 'scaler_y.pkl')

            with open(scaler_X_path, 'rb') as f:
                self.scaler_X = pickle.load(f)
            logger.info(f"Loaded scaler_X from {scaler_X_path}")

            with open(scaler_y_path, 'rb') as f:
                self.scaler_y = pickle.load(f)
            logger.info(f"Loaded scaler_y from {scaler_y_path}")

        except Exception as e:
            logger.error(f"Error loading model/scalers: {e}")
            raise

    def get_market_data(self):
        """Get current market data from BingX using CCXT"""
        try:
            # Get recent klines (1m interval, need at least 300 for indicators)
            # CCXT uses 'BTC/USDT' format
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
            df['quote_volume'] = df['volume'] * df['close']  # Approximate
            df['trades'] = 0  # Not available in CCXT basic
            df['taker_buy_volume'] = df['volume'] * 0.5  # Approximate
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
            # feature_engineer.py uses window_size=50 for overlap, but we need full history
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
                        return 0.0  # Default fallback
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
                    indicators[f'BB_{window}_upper'] = close.iloc[-1]  # Fallback to current price
                    indicators[f'BB_{window}_lower'] = close.iloc[-1]

            # RSI indicators
            for window in [15, 60, 300]:
                if len(close) >= window + 1:  # Need extra candle for diff
                    delta = close.diff()
                    up = delta.clip(lower=0)
                    down = -1 * delta.clip(upper=0)
                    ma_up = up.rolling(window=window).mean()
                    ma_down = down.rolling(window=window).mean()
                    rsi = 100 - (100 / (1 + ma_up / ma_down))
                    indicators[f'RSI_{window}'] = safe_last_value(rsi, f'RSI_{window}')
                else:
                    logger.warning(f"Insufficient data for RSI_{window} (need {window+1}, got {len(close)})")
                    indicators[f'RSI_{window}'] = 50.0  # Neutral RSI

            # Ultimate Oscillator (requires at least 28 periods)
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
                        indicators[f'ATR_{window}'] = close.iloc[-1] * 0.01  # 1% of price as fallback
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
                        indicators[f'MFI_{window}'] = 50.0  # Neutral MFI
                else:
                    logger.warning(f"Insufficient data for MFI_{window} (need {window}, got {len(data)})")
                    indicators[f'MFI_{window}'] = 50.0

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def predict_price(self, indicators):
        """Use LSTM model to predict next price using all features from training"""
        try:
            # Prepare features in the exact same order as feature_engineer.py
            feature_names = [
                'EMA_15', 'EMA_60', 'EMA_300',
                'BB_15_upper', 'BB_15_lower', 'BB_60_upper', 'BB_60_lower', 'BB_300_upper', 'BB_300_lower',
                'RSI_15', 'RSI_60', 'RSI_300',
                'ULTOSC',
                'OBV', 'AD',
                'ATR_15', 'ATR_60',
                'WCLPRICE',
                'VAR_15', 'VAR_60', 'VAR_300',
                'MFI_15', 'MFI_60', 'MFI_300'
            ]

            features = np.array([indicators[name] for name in feature_names]).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler_X.transform(features)

            # Convert to tensor
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.model(input_tensor).item()

            # Convert back to original scale
            prediction = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]

            return prediction

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def calculate_trading_signal(self, current_price, prediction, indicators, past_predictions):
        """Calculate trading signal based on model prediction and technical indicators"""
        try:
            # Calculate dynamic thresholds
            atr_value = indicators['ATR_15']
            rsi_value = indicators['RSI_15']

            sell_threshold = (self.params['base_sell_threshold'] +
                            self.params['alpha_atr'] * atr_value +
                            self.params['alpha_rsi'] * max(0, rsi_value - 70))

            buy_threshold = (self.params['base_buy_threshold'] -
                           self.params['alpha_atr'] * atr_value -
                           self.params['alpha_rsi'] * max(0, 70 - rsi_value))

            # Calculate trend and confidence
            if past_predictions:
                past_average = np.mean(past_predictions[-self.params['window_size']:])
                trend_direction = 'up' if prediction > past_average else 'down'
                confidence = (prediction - past_average) / past_average if past_average != 0 else 0
            else:
                trend_direction = 'hold'
                confidence = 0

            # Calculate expected profit
            expected_price_increase = prediction - current_price
            expected_profit_percent = (expected_price_increase / current_price) * 100

            signal = {
                'trend': trend_direction,
                'confidence': confidence,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'expected_profit': expected_profit_percent,
                'should_buy': (trend_direction == 'up' and
                             confidence >= buy_threshold and
                             expected_profit_percent >= self.params['min_profit_threshold']),
                'should_sell': (trend_direction == 'down' and
                              confidence <= sell_threshold and
                              self.position > 0)
            }

            return signal

        except Exception as e:
            logger.error(f"Error calculating trading signal: {e}")
            return None

    def execute_virtual_trade(self, signal, current_price):
        """Execute virtual trade (no real transactions)"""
        try:
            trade_executed = False

            if signal['should_buy'] and self.balance > 10:  # Minimum balance check
                # Calculate investment amount
                investment = min(self.balance * abs(signal['confidence']) * self.params['buy_percentage'],
                               self.balance * 0.1)  # Max 10% of balance

                if investment > 10:  # Minimum trade size
                    # Apply trading fee (0.018%)
                    fee = investment * 0.00018
                    investment_after_fee = investment - fee

                    # Calculate BTC amount
                    btc_amount = investment_after_fee / current_price

                    # Update portfolio
                    self.position += btc_amount
                    self.balance -= investment
                    self.entry_price = current_price
                    self.total_fees += fee

                    trade = {
                        'timestamp': datetime.now(),
                        'type': 'BUY',
                        'price': current_price,
                        'amount': btc_amount,
                        'value': investment,
                        'fee': fee,
                        'confidence': signal['confidence']
                    }

                    self.trades.append(trade)
                    trade_executed = True

                    logger.info(f"VIRTUAL BUY: {btc_amount:.6f} BTC at ${current_price:.2f}")

            elif signal['should_sell'] and self.position > 0:
                # Calculate sell amount
                sell_amount = self.position * self.params['sell_percentage'] * abs(signal['confidence'])
                sell_amount = min(sell_amount, self.position)  # Don't sell more than owned

                if sell_amount * current_price > 10:  # Minimum trade value
                    # Calculate revenue
                    revenue = sell_amount * current_price
                    fee = revenue * 0.00018
                    revenue_after_fee = revenue - fee

                    # Update portfolio
                    self.position -= sell_amount
                    self.balance += revenue_after_fee
                    self.total_fees += fee

                    # Calculate P&L
                    pnl = (current_price - self.entry_price) * sell_amount - fee

                    trade = {
                        'timestamp': datetime.now(),
                        'type': 'SELL',
                        'price': current_price,
                        'amount': sell_amount,
                        'value': revenue_after_fee,
                        'fee': fee,
                        'pnl': pnl,
                        'confidence': signal['confidence']
                    }

                    self.trades.append(trade)
                    trade_executed = True

                    logger.info(f"VIRTUAL SELL: {sell_amount:.6f} BTC at ${current_price:.2f}, PnL: ${pnl:.2f}")

            return trade_executed

        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    def run_live_session(self, duration_minutes=60):
        """Run live trading session for specified duration"""
        logger.info(f"Starting live trading session for {duration_minutes} minutes")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        past_predictions = []

        while datetime.now() < end_time:
            try:
                # Get market data
                df = self.get_market_data()
                if df is None or df.empty:
                    logger.warning("No market data available, skipping iteration")
                    time.sleep(60)  # Wait 1 minute
                    continue

                current_price = df['close'].iloc[-1]

                # Calculate indicators
                indicators = self.calculate_indicators(df)
                if indicators is None:
                    logger.warning("Could not calculate indicators, skipping iteration")
                    time.sleep(60)
                    continue

                # Make price prediction
                prediction = self.predict_price(indicators)
                if prediction is None:
                    logger.warning("Could not make prediction, skipping iteration")
                    time.sleep(60)
                    continue

                past_predictions.append(prediction)
                if len(past_predictions) > 100:  # Keep only recent predictions
                    past_predictions = past_predictions[-100:]

                # Calculate trading signal
                signal = self.calculate_trading_signal(current_price, prediction, indicators, past_predictions)
                if signal is None:
                    logger.warning("Could not calculate trading signal, skipping iteration")
                    time.sleep(60)
                    continue

                # Execute virtual trade
                trade_executed = self.execute_virtual_trade(signal, current_price)

                # Record portfolio state
                portfolio_value = self.balance + (self.position * current_price)
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'balance': self.balance,
                    'position': self.position,
                    'portfolio_value': portfolio_value,
                    'prediction': prediction,
                    'signal': signal
                })

                self.price_history.append(current_price)
                self.indicators_history.append(indicators)

                # Log status every 5 minutes
                if len(self.portfolio_history) % 5 == 0:
                    pnl = portfolio_value - self.initial_balance
                    logger.info(f"Status - Portfolio: ${portfolio_value:.2f}, PnL: ${pnl:.2f}, Trades: {len(self.trades)}")

                # Wait before next iteration (1 minute)
                time.sleep(60)

            except Exception as e:
                logger.error(f"Error in live session: {e}")
                time.sleep(60)

        logger.info("Live trading session completed")
        self.generate_report()

    def generate_report(self):
        """Generate trading report"""
        try:
            # Calculate final metrics
            final_portfolio_value = self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else self.initial_balance
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
LIVE TRADING REPORT
===================

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
            logger.info("Trading report generated")

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

            # Save to file
            with open('live_trading_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            logger.info("Detailed results saved to live_trading_results.pkl")

        except Exception as e:
            logger.error(f"Error generating report: {e}")

def main():
    """Main function to run live trading"""
    import argparse

    parser = argparse.ArgumentParser(description="Live Trading Bot for BingX")
    parser.add_argument("--model", default="lstm_model_1472.pt", help="Path to trained model")
    parser.add_argument("--scaler", default="scaler.pkl", help="Path to data scaler")
    parser.add_argument("--symbol", default="BTC-USDT", help="Trading symbol")
    parser.add_argument("--duration", type=int, default=60, help="Trading duration in minutes")
    parser.add_argument("--real-trades", action="store_true", help="Enable REAL trades (WARNING: Use with caution!)")

    args = parser.parse_args()

    # Safety check
    if args.real_trades:
        confirm = input("⚠️  WARNING: This will execute REAL trades! Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborting real trading.")
            return

    test_mode = not args.real_trades

    try:
        bot = LiveTradingBot(
            model_path=args.model,
            scaler_path=args.scaler,
            symbol=args.symbol,
            test_mode=test_mode
        )

        bot.run_live_session(duration_minutes=args.duration)

    except Exception as e:
        logger.error(f"Error running live trading: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
