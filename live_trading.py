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

    def __init__(self, model_path, scaler_path='scaler_X.pkl', symbol='BTC-USDT', test_mode=True):
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

        # Load trained model and scaler
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
            'min_profit_threshold': 0.768993,
            'stop_loss_percent': 0.02  # 2% stop loss
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
        """Load trained LSTM model and scaler"""
        try:
            # Load LSTM model
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if hasattr(checkpoint, 'module'):
                self.model = checkpoint.module_
            else:
                self.model = checkpoint

            self.model.eval()
            logger.info(f"Loaded model from {model_path}")

            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_path}")

        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
            raise

    def get_market_data(self):
        """Get current market data from BingX using CCXT"""
        try:
            # Get recent klines (1m interval, last 100 candles)
            # CCXT uses 'BTC/USDT' format
            symbol_ccxt = self.symbol.replace('-', '/')
            klines = self.exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', limit=100)

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
        """Calculate technical indicators for the latest data point"""
        try:
            # Use last 50 candles for indicator calculation
            data = df.tail(50).copy()

            # Basic price data
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            # Calculate indicators using ta library
            indicators = {}

            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close, window=15)
            indicators['RSI_15'] = rsi_indicator.rsi().iloc[-1]

            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=15)
            indicators['ATR_15'] = atr_indicator.average_true_range().iloc[-1]

            # MACD
            macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            indicators['MACD_macd'] = macd_indicator.macd().iloc[-1]

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(close, window=15, window_dev=2)
            indicators['BB_15_upper'] = bb_indicator.bollinger_hband().iloc[-1]
            indicators['BB_15_lower'] = bb_indicator.bollinger_lband().iloc[-1]

            # OBV
            obv_indicator = ta.volume.OnBalanceVolumeIndicator(close, volume)
            indicators['OBV'] = obv_indicator.on_balance_volume().iloc[-1]

            # AD
            ad_indicator = ta.volume.AccDistIndexIndicator(high, low, close, volume)
            indicators['AD'] = ad_indicator.acc_dist_index().iloc[-1]

            # MFI
            mfi_indicator = ta.volume.MFIIndicator(high, low, close, volume, window=15)
            indicators['MFI_15'] = mfi_indicator.money_flow_index().iloc[-1]

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def predict_price(self, indicators):
        """Use LSTM model to predict next price"""
        try:
            # Prepare features (same order as in training)
            feature_names = [
                'RSI_15', 'MACD_macd', 'BB_15_upper', 'BB_15_lower',
                'ATR_15', 'OBV', 'AD', 'MFI_15'
            ]

            features = np.array([indicators[name] for name in feature_names]).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Convert to tensor
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor).item()

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
                if past_average != 0:
                    trend_direction = 'up' if prediction > past_average else 'down'
                    confidence = (prediction - past_average) / past_average
                else:
                    trend_direction = 'hold'
                    confidence = 0
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

            # Check stop-loss
            if self.position > 0 and current_price <= self.entry_price * (1 - self.params['stop_loss_percent']):
                # Sell all position due to stop-loss
                sell_amount = self.position
                revenue = sell_amount * current_price
                fee = revenue * 0.00018
                revenue_after_fee = revenue - fee

                self.position -= sell_amount
                self.balance += revenue_after_fee
                self.total_fees += fee

                pnl = (current_price - self.entry_price) * sell_amount - fee

                trade = {
                    'timestamp': datetime.now(),
                    'type': 'STOP_LOSS_SELL',
                    'price': current_price,
                    'amount': sell_amount,
                    'value': revenue_after_fee,
                    'fee': fee,
                    'pnl': pnl,
                    'confidence': 0
                }

                self.trades.append(trade)
                trade_executed = True

                logger.info(f"STOP LOSS SELL: {sell_amount:.6f} BTC at ${current_price:.2f}, PnL: ${pnl:.2f}")
                return trade_executed

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
