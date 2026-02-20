#!/usr/bin/env python3
"""
Unified RL Trading Script
Supports three trading modes:
  --paper  : Backtest on historical CSV data (no real money, no internet required)
  --live   : Virtual trading with live market data from exchange (no real orders)
  --real   : Real trading on exchange (USE WITH CAUTION)

Usage:
  python trade.py --paper --data btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv
  python trade.py --live --duration 60
  python trade.py --real --duration 60  # NOT IMPLEMENTED YET
"""

import os
import sys
import time
import logging
import argparse
import pickle
import json
import threading
import csv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

# ──────────────────────────────────────────────────────────────────────────────
# UTF-8 для Windows
# ──────────────────────────────────────────────────────────────────────────────
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Import trading environment (must match training exactly)
# ──────────────────────────────────────────────────────────────────────────────
from enhanced_trading_environment import EnhancedTradingEnvironment

# Action mapping
ACTION_NAMES = {0: 'HOLD', 1: 'BUY_LONG', 2: 'SELL_LONG', 3: 'SELL_SHORT', 4: 'COVER_SHORT'}


# ══════════════════════════════════════════════════════════════════════════════
# PAPER TRADING (Historical backtest)
# ══════════════════════════════════════════════════════════════════════════════

def run_paper_trading(model_path: str, data_path: str, initial_balance: float = 10000,
                      n_episodes: int = 1, verbose: bool = True):
    """
    Backtest RL agent on historical CSV data.
    Uses EnhancedTradingEnvironment directly — exact same setup as training.

    Args:
        model_path: Path to trained PPO model (.zip)
        data_path:  Path to CSV with OHLCV + indicators
        initial_balance: Starting balance in USD
        n_episodes: Number of independent episodes to run
        verbose: Print step-by-step info
    """
    print("\n" + "=" * 70)
    print("📊 PAPER TRADING MODE — Historical Backtest")
    print("=" * 70)

    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    # Load data
    print(f"Loading data: {data_path}")
    try:
        df = pd.read_csv(data_path).dropna()
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    # Normalize column names
    if 'close' not in df.columns and 'Close' in df.columns:
        df['close'] = df['Close']

    # Check required columns for EnhancedTradingEnvironment
    required = ['close', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'ATR_15', 'MFI_15',
                'MACD_default_macd', 'MACD_default_signal', 'MACD_default_histogram',
                'Stochastic_slowk', 'Stochastic_slowd']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return None

    all_results = []
    for episode in range(n_episodes):
        print(f"\n{'─' * 50}")
        print(f"Episode {episode + 1}/{n_episodes}")

        env = EnhancedTradingEnvironment(
            df=df,
            initial_balance=initial_balance,
            transaction_fee=0.0018,
            episode_length=len(df),
            start_step=0,
            debug=False,
            enable_strategy_balancing=False  # No balancing needed in evaluation
        )

        result = _run_episode(env, model, df, initial_balance, verbose=verbose)
        all_results.append(result)
        _print_episode_summary(result, episode + 1)

    # Aggregate results if multiple episodes
    if n_episodes > 1:
        _print_aggregated_summary(all_results)

    # Save results
    output_file = 'paper_trading_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results if n_episodes > 1 else all_results[0], f)
    print(f"\n💾 Results saved to {output_file}")

    return all_results if n_episodes > 1 else all_results[0]


def _run_episode(env: EnhancedTradingEnvironment, model: PPO, df: pd.DataFrame,
                 initial_balance: float, verbose: bool = True) -> dict:
    """Run a single paper trading episode and return results."""
    state, _ = env.reset()
    done = False
    step = 0
    action_counts = {i: 0 for i in range(5)}
    portfolio_history = []

    while not done:
        step += 1

        # Get current price safely
        price_idx = min(env.current_step, len(df) - 1)
        current_price = df.iloc[price_idx].get('close', df.iloc[price_idx].get('Close', 0))

        # Predict action (stochastic for exploration)
        action, _ = model.predict(state, deterministic=False)
        action = int(action)
        action_counts[action] = action_counts.get(action, 0) + 1

        if verbose and step <= 5:
            pos_type = "LONG" if env.position > 0 else ("SHORT" if env.position < 0 else "NONE")
            print(f"  Step {step:3d}: price=${current_price:,.0f} pos={pos_type:5s} "
                  f"action={ACTION_NAMES[action]:10s} balance=${env.balance:,.2f}")

        # Step environment
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_value = env.balance + env.margin_locked + env.position * current_price
        portfolio_history.append({
            'step': step,
            'price': current_price,
            'balance': env.balance,
            'position': env.position,
            'portfolio_value': portfolio_value,
            'action': action,
            'action_performed': info.get('action_performed', False),
            'pnl_pct': info.get('pnl_pct', 0)
        })

        if verbose and step % 500 == 0:
            pnl_pct = (portfolio_value / initial_balance - 1) * 100
            print(f"  Step {step}: portfolio=${portfolio_value:,.2f} ({pnl_pct:+.2f}%)")

    # Final metrics
    if portfolio_history:
        final_portfolio = portfolio_history[-1]['portfolio_value']
        pv = [p['portfolio_value'] for p in portfolio_history]
        returns = np.diff(pv) / np.array(pv[:-1])
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        running_max = np.maximum.accumulate(pv)
        max_dd = ((np.array(pv) - running_max) / running_max).min()
    else:
        final_portfolio = initial_balance
        sharpe = 0.0
        max_dd = 0.0

    return {
        'initial_balance': initial_balance,
        'final_portfolio': final_portfolio,
        'total_pnl': final_portfolio - initial_balance,
        'total_return': (final_portfolio / initial_balance - 1) * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'steps': step,
        'total_trades': env.total_trades,
        'win_count': env.win_count,
        'loss_count': env.loss_count,
        'long_trades': env.long_trades,
        'short_trades': env.short_trades,
        'action_counts': action_counts,
        'portfolio_history': portfolio_history,
    }


def _print_episode_summary(result: dict, episode_num: int):
    total_decisions = result['total_trades']
    win_rate = result['win_count'] / total_decisions * 100 if total_decisions > 0 else 0
    print(f"\n📊 Episode {episode_num} results:")
    print(f"  Return:       {result['total_return']:+.2f}%  (${result['total_pnl']:+,.2f})")
    print(f"  Final:        ${result['final_portfolio']:,.2f}")
    print(f"  Sharpe:       {result['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {result['max_drawdown'] * 100:.2f}%")
    print(f"  Trades:       {total_decisions}  (wins: {result['win_count']}, losses: {result['loss_count']}, win rate: {win_rate:.1f}%)")
    print(f"  Long/Short:   {result['long_trades']}/{result['short_trades']}")
    ac = result['action_counts']
    print(f"  Actions:      Hold={ac.get(0,0)} BuyL={ac.get(1,0)} SellL={ac.get(2,0)} SellS={ac.get(3,0)} CoverS={ac.get(4,0)}")


def _print_aggregated_summary(results: list):
    print("\n" + "=" * 70)
    print("🎯 AGGREGATED RESULTS")
    print("=" * 70)
    returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    print(f"  Episodes:     {len(results)}")
    print(f"  Avg Return:   {np.mean(returns):+.2f}% ± {np.std(returns):.2f}%")
    print(f"  Avg Sharpe:   {np.mean(sharpes):.4f}")
    print(f"  Avg Drawdown: {np.mean(drawdowns) * 100:.2f}%")
    best = max(range(len(results)), key=lambda i: results[i]['total_return'])
    worst = min(range(len(results)), key=lambda i: results[i]['total_return'])
    print(f"  Best:  Episode {best+1} ({returns[best]:+.2f}%)")
    print(f"  Worst: Episode {worst+1} ({returns[worst]:+.2f}%)")
    status = "✅ PROFITABLE" if np.mean(returns) > 0 else "❌ UNPROFITABLE"
    print(f"  Overall: {status}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING BOT (Virtual or Real)
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    """
    Trading bot that uses a trained RL model.
    Supports:
      mode='live'  — virtual paper trading with live market data
      mode='real'  — real order execution (NOT YET IMPLEMENTED)

    Position management follows the same rules as EnhancedTradingEnvironment:
      - BUY_LONG (1): opens long, BLOCKED if short is open
      - SELL_LONG (2): closes long, BLOCKED if no long
      - SELL_SHORT (3): opens short, BLOCKED if long is open
      - COVER_SHORT (4): closes short, BLOCKED if no short
    """

    TRANSACTION_FEE = 0.0018  # 0.18% (matches training)
    MARGIN_REQUIREMENT = 0.3   # 30% margin for shorts
    TRAILING_STOP_PCT = 0.05   # 5% trailing stop
    BASE_POSITION_SIZE = 0.08  # 8% of balance per trade

    def __init__(self, model_path: str, symbol: str = 'BTC-USDT',
                 initial_balance: float = 10000, mode: str = 'live'):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.mode = mode  # 'live' or 'real'

        # Load model
        logger.info(f"Loading model: {model_path}")
        self.model = PPO.load(model_path)
        logger.info("✅ Model loaded")

        # Exchange connection
        self._init_exchange()

        # ── Portfolio state ──────────────────────────────────────────────────
        self.balance = float(initial_balance)
        self.position = 0.0        # net position (+ long, - short)
        self.entry_price = 0.0
        self.entry_step = 0
        self.margin_locked = 0.0
        self.short_opening_fees = 0.0
        self.highest_price_since_entry = 0.0
        self.lowest_price_since_entry = float('inf')
        self.trailing_stop_loss = 0.0
        self.total_fees = 0.0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.long_trades_count = 0
        self.short_trades_count = 0

        # ── History ──────────────────────────────────────────────────────────
        self.trades: list = []
        self.portfolio_history: list = []
        self.price_history: list = []
        self.step_counter = 0

        # ── Environment for state calculation ────────────────────────────────
        # Will be initialized once we have market data
        self.live_env: EnhancedTradingEnvironment | None = None
        self.env_df: pd.DataFrame | None = None

        # ── Rate limiting ─────────────────────────────────────────────────────
        self.last_trade_time = 0.0
        self.min_trade_interval = 60  # seconds between trades

        # ── CSV log ───────────────────────────────────────────────────────────
        self.trade_log = 'trade_results.csv'
        self._init_csv()

        logger.info(f"Bot initialized | symbol={symbol} | mode={mode} | balance=${initial_balance:,.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_exchange(self):
        """Connect to exchange via ccxt."""
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('BINGX_API_KEY')
        secret = os.getenv('BINGX_SECRET_KEY')
        if not api_key or not secret:
            raise ValueError("BINGX_API_KEY and BINGX_SECRET_KEY must be in .env")
        import ccxt
        self.exchange = ccxt.bingx({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        logger.info("Exchange (BingX) connected")

    def _init_csv(self):
        if not os.path.exists(self.trade_log):
            with open(self.trade_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'action', 'price', 'btc_amount', 'usdt_value',
                    'fee', 'pnl', 'cumulative_pnl', 'balance_after', 'position_after'
                ])
                writer.writeheader()

    def _log_csv(self, row: dict):
        with open(self.trade_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'action', 'price', 'btc_amount', 'usdt_value',
                'fee', 'pnl', 'cumulative_pnl', 'balance_after', 'position_after'
            ])
            writer.writerow(row)

    # ─────────────────────────────────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────────────────────────────────

    def _klines_to_df(self, klines: list) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        return df

    def get_market_data(self, limit: int = 350) -> pd.DataFrame | None:
        """Fetch OHLCV from BingX (falls back to Binance public API)."""
        import ccxt
        sym = self.symbol.replace('-', '/')
        # Try BingX
        try:
            klines = self.exchange.fetch_ohlcv(sym, timeframe='1m', limit=limit)
            if klines:
                return self._klines_to_df(klines)
        except Exception as e:
            logger.warning(f"BingX fetch failed: {e}")
        # Fallback to Binance public
        try:
            binance = ccxt.binance({'enableRateLimit': True})
            klines = binance.fetch_ohlcv(sym, timeframe='1m', limit=limit)
            if klines:
                logger.info("Using Binance fallback")
                return self._klines_to_df(klines)
        except Exception as e:
            logger.warning(f"Binance fallback failed: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Technical indicators (matching enhanced_trading_environment.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators required by EnhancedTradingEnvironment."""
        import ta
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI_15
        delta = close.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        df['RSI_15'] = 100 - (100 / (1 + up.rolling(15).mean() / (down.rolling(15).mean() + 1e-9)))

        # Bollinger Bands 15
        sma = close.rolling(15).mean()
        std = close.rolling(15).std()
        df['BB_15_upper'] = sma + 2 * std
        df['BB_15_lower'] = sma - 2 * std

        # ATR_15
        df['ATR_15'] = ta.volatility.AverageTrueRange(high, low, close, window=15).average_true_range()

        # MFI_15
        df['MFI_15'] = ta.volume.MFIIndicator(high, low, close, volume, window=15).money_flow_index()

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD_default_macd'] = ema12 - ema26
        df['MACD_default_signal'] = df['MACD_default_macd'].ewm(span=9, adjust=False).mean()
        df['MACD_default_histogram'] = df['MACD_default_macd'] - df['MACD_default_signal']

        # Stochastic (14, 3)
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df['Stochastic_slowk'] = stoch.stoch()
        df['Stochastic_slowd'] = stoch.stoch_signal()

        return df.bfill().ffill()

    # ─────────────────────────────────────────────────────────────────────────
    # Environment initialization and state synchronisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_env(self, df: pd.DataFrame) -> bool:
        """Build EnhancedTradingEnvironment for consistent state calculation."""
        try:
            env_df = self._add_indicators(df.copy())
            self.env_df = env_df
            self.live_env = EnhancedTradingEnvironment(
                df=env_df,
                initial_balance=self.initial_balance,
                transaction_fee=self.TRANSACTION_FEE,
                episode_length=len(env_df),
                start_step=len(env_df) - 1,
                debug=False,
                enable_strategy_balancing=True,
            )
            self._sync_env()
            logger.info(f"Environment initialized with {len(env_df)} rows")
            return True
        except Exception as e:
            logger.error(f"Failed to init env: {e}")
            return False

    def _sync_env(self):
        """Sync internal bot state into the environment so _get_state() is correct."""
        if self.live_env is None:
            return
        self.live_env.balance = self.balance
        self.live_env.position = self.position
        self.live_env.margin_locked = self.margin_locked
        self.live_env.entry_price = self.entry_price
        self.live_env.entry_step = self.entry_step
        self.live_env.highest_price_since_entry = self.highest_price_since_entry
        self.live_env.lowest_price_since_entry = self.lowest_price_since_entry
        self.live_env.trailing_stop_loss = self.trailing_stop_loss
        self.live_env.total_fees = self.total_fees
        self.live_env.total_pnl = self.total_pnl
        self.live_env.win_count = self.win_count
        self.live_env.loss_count = self.loss_count
        self.live_env.total_trades = self.total_trades
        self.live_env.long_trades = self.long_trades_count
        self.live_env.short_trades = self.short_trades_count
        self.live_env.steps_since_last_trade = 0
        if self.env_df is not None:
            self.live_env.current_step = len(self.env_df) - 1

    def _update_env_df(self, new_row: dict):
        """Append a new candle to the environment DataFrame and recalculate indicators."""
        if self.env_df is None:
            return
        try:
            new_df = pd.concat([self.env_df, pd.DataFrame([new_row])], ignore_index=True)
            new_df = self._add_indicators(new_df)
            self.env_df = new_df
            self.live_env.df = new_df
            # Update rolling price normalization
            ws = min(100, len(new_df) // 10)
            self.live_env.price_rolling_mean = new_df['close'].rolling(ws, min_periods=1).mean().values.copy()
            self.live_env.price_rolling_std = new_df['close'].rolling(ws, min_periods=1).std().values.copy()
            self.live_env.price_rolling_std[self.live_env.price_rolling_std == 0] = 1
            self.live_env.current_step = len(new_df) - 1
        except Exception as e:
            logger.error(f"Error updating env df: {e}")

    def get_state(self) -> np.ndarray:
        """Return 18-dimensional state vector (consistent with training)."""
        if self.live_env is not None:
            self._sync_env()
            return self.live_env._get_state()
        return np.zeros(18, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Trade execution (Virtual mode — no real orders)
    # ─────────────────────────────────────────────────────────────────────────

    def _fee(self, amount_usdt: float) -> float:
        return amount_usdt * self.TRANSACTION_FEE

    def execute_trade(self, action: int, price: float) -> bool:
        """
        Execute action following the same rules as EnhancedTradingEnvironment:
          - BUY_LONG(1):    blocked if position < 0 (short open)
          - SELL_LONG(2):   blocked if position <= 0
          - SELL_SHORT(3):  blocked if position > 0 (long open)
          - COVER_SHORT(4): blocked if position >= 0

        Returns True if the trade was executed.
        """
        now = time.time()
        if now - self.last_trade_time < self.min_trade_interval and self.last_trade_time > 0:
            return False

        if self.mode == 'real':
            return self._execute_real_trade(action, price)

        # ── Virtual execution ─────────────────────────────────────────────────
        executed = False
        pnl = None

        if action == 1:  # BUY_LONG
            executed, pnl = self._vt_buy_long(price)

        elif action == 2:  # SELL_LONG
            executed, pnl = self._vt_sell_long(price)

        elif action == 3:  # SELL_SHORT
            executed, pnl = self._vt_sell_short(price)

        elif action == 4:  # COVER_SHORT
            executed, pnl = self._vt_cover_short(price)

        if executed:
            self.last_trade_time = now
            self.total_trades += 1
            if pnl is not None:
                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
            self._check_trailing_stop(price)

        return executed

    def _vt_buy_long(self, price: float) -> tuple[bool, None]:
        """Open long position. Blocked if short is open."""
        if self.position < 0:
            logger.debug("BUY_LONG blocked: short position is open. Use COVER_SHORT first.")
            return False, None
        if self.position > 0:
            logger.debug("BUY_LONG skipped: long already open")
            return False, None

        invest = self.balance * self.BASE_POSITION_SIZE
        if invest < 10 or self.balance < invest:
            return False, None

        fee = self._fee(invest)
        btc = (invest - fee) / price
        self.balance -= invest
        self.total_fees += fee
        self.position = btc
        self.entry_price = price
        self.entry_step = self.step_counter
        self.highest_price_since_entry = price
        self.trailing_stop_loss = price * (1 - self.TRAILING_STOP_PCT)
        self.long_trades_count += 1

        self._log_trade(ACTION_NAMES[1], price, btc, invest, fee, pnl=None)
        logger.info(f"BUY_LONG  {btc:.6f} BTC @ ${price:,.2f}  fee=${fee:.2f}  balance=${self.balance:,.2f}")
        return True, None

    def _vt_sell_long(self, price: float) -> tuple[bool, float | None]:
        """Close long position."""
        if self.position <= 0:
            logger.debug("SELL_LONG blocked: no long position")
            return False, None

        revenue = self.position * price
        fee = self._fee(revenue)
        pnl = revenue - (self.position * self.entry_price) - fee
        btc = self.position

        self.balance += revenue - fee
        self.total_fees += fee
        self._reset_position()

        self._log_trade(ACTION_NAMES[2], price, btc, revenue, fee, pnl)
        logger.info(f"SELL_LONG {btc:.6f} BTC @ ${price:,.2f}  pnl=${pnl:+.2f}  balance=${self.balance:,.2f}")
        return True, pnl

    def _vt_sell_short(self, price: float) -> tuple[bool, None]:
        """Open short position. Blocked if long is open."""
        if self.position > 0:
            logger.debug("SELL_SHORT blocked: long position is open. Use SELL_LONG first.")
            return False, None
        if self.position < 0:
            logger.debug("SELL_SHORT skipped: short already open")
            return False, None

        short_val = self.balance * self.BASE_POSITION_SIZE * 0.8
        margin = short_val * self.MARGIN_REQUIREMENT
        if short_val < 10 or self.balance < margin:
            return False, None

        fee = self._fee(short_val)
        btc = short_val / price
        self.short_opening_fees = fee
        self.margin_locked += short_val + margin
        self.balance -= margin + fee
        self.total_fees += fee
        self.position = -btc
        self.entry_price = price
        self.entry_step = self.step_counter
        self.lowest_price_since_entry = price
        self.trailing_stop_loss = price * (1 + self.TRAILING_STOP_PCT)
        self.short_trades_count += 1

        self._log_trade(ACTION_NAMES[3], price, btc, short_val, fee, pnl=None)
        logger.info(f"SELL_SHORT {btc:.6f} BTC @ ${price:,.2f}  margin=${margin:.2f}  balance=${self.balance:,.2f}")
        return True, None

    def _vt_cover_short(self, price: float) -> tuple[bool, float | None]:
        """Close short position."""
        if self.position >= 0:
            logger.debug("COVER_SHORT blocked: no short position")
            return False, None

        size = abs(self.position)
        sale_proceeds = size * self.entry_price
        buy_cost = size * price
        fee = self._fee(buy_cost)
        pnl = (sale_proceeds - buy_cost) - fee
        our_margin = sale_proceeds * self.MARGIN_REQUIREMENT

        self.balance += our_margin + pnl
        self.margin_locked = 0.0
        self.total_fees += fee
        self.short_opening_fees = 0.0
        self._reset_position()

        self._log_trade(ACTION_NAMES[4], price, size, buy_cost, fee, pnl)
        logger.info(f"COVER_SHORT {size:.6f} BTC @ ${price:,.2f}  pnl=${pnl:+.2f}  balance=${self.balance:,.2f}")
        return True, pnl

    def _reset_position(self):
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.highest_price_since_entry = 0.0
        self.lowest_price_since_entry = float('inf')
        self.trailing_stop_loss = 0.0
        self.margin_locked = 0.0

    def _check_trailing_stop(self, price: float):
        """Trigger trailing stop-loss if price moves against position."""
        if self.position == 0 or self.entry_price == 0:
            return

        if self.position > 0:  # Long
            if price > self.highest_price_since_entry:
                self.highest_price_since_entry = price
                self.trailing_stop_loss = self.highest_price_since_entry * (1 - self.TRAILING_STOP_PCT)
            if price <= self.trailing_stop_loss:
                logger.info(f"🛑 TRAILING STOP (long) triggered @ ${price:,.2f}")
                _, pnl = self._vt_sell_long(price)
                if pnl is not None:
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1

        elif self.position < 0:  # Short
            if price < self.lowest_price_since_entry:
                self.lowest_price_since_entry = price
                self.trailing_stop_loss = self.lowest_price_since_entry * (1 + self.TRAILING_STOP_PCT)
            if price >= self.trailing_stop_loss:
                logger.info(f"🛑 TRAILING STOP (short) triggered @ ${price:,.2f}")
                _, pnl = self._vt_cover_short(price)
                if pnl is not None:
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1

    def _log_trade(self, action_name: str, price: float, btc: float,
                   usdt_value: float, fee: float, pnl):
        self.trades.append({
            'timestamp': datetime.now().isoformat(),
            'action': action_name,
            'price': round(price, 2),
            'btc_amount': round(btc, 8),
            'usdt_value': round(usdt_value, 2),
            'fee': round(fee, 4),
            'pnl': round(pnl, 4) if pnl is not None else 0,
            'cumulative_pnl': round(self.total_pnl, 4),
            'balance_after': round(self.balance, 2),
            'position_after': round(self.position, 8),
        })
        self._log_csv(self.trades[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # Real trading (NOT IMPLEMENTED)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_real_trade(self, action: int, price: float) -> bool:
        """
        Placeholder for real order execution.
        Implement exchange-specific order placement here.
        """
        logger.warning("⚠️  REAL TRADING IS NOT YET IMPLEMENTED — trade skipped")
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main trading loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, duration_minutes: int = 60, check_interval: int = 60):
        """Main trading loop."""
        mode_label = "LIVE (virtual)" if self.mode == 'live' else "REAL"
        print("\n" + "=" * 70)
        print(f"🚀 TRADING BOT — {mode_label} MODE")
        print("=" * 70)
        print(f"Symbol:   {self.symbol}")
        print(f"Balance:  ${self.initial_balance:,.2f}")
        print(f"Duration: {duration_minutes} min  |  Interval: {check_interval}s")
        print("=" * 70)

        # Initial data load
        print("Fetching initial market data...")
        for attempt in range(5):
            df = self.get_market_data(limit=350)
            if df is not None and len(df) >= 50:
                print(f"✅ Got {len(df)} candles")
                break
            print(f"Attempt {attempt+1}/5 failed, retrying...")
            time.sleep(10)
        else:
            logger.error("Cannot fetch market data")
            return

        if not self._init_env(df):
            logger.error("Cannot initialize environment")
            return

        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        session_start = time.time()
        last_refresh = time.time()

        try:
            while datetime.now() < end_time:
                loop_start = time.time()

                # Refresh market data every minute
                if time.time() - last_refresh >= 60:
                    new_df = self.get_market_data(limit=350)
                    if new_df is not None and len(new_df) >= 50:
                        # Add latest candle to environment
                        last_row = new_df.iloc[-1].to_dict()
                        self._update_env_df(last_row)
                        last_refresh = time.time()

                # Get current price
                if self.env_df is not None and len(self.env_df) > 0:
                    current_price = float(self.env_df['close'].iloc[-1])
                else:
                    time.sleep(5)
                    continue

                self.step_counter += 1
                self.price_history.append(current_price)

                # Check trailing stop BEFORE model decision
                self._check_trailing_stop(current_price)

                # Get state and model prediction
                state = self.get_state()
                action, _ = self.model.predict(state, deterministic=False)
                action = int(action)

                # Log action probabilities
                state_t = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                dist = self.model.policy.get_distribution(state_t)
                probs = dist.distribution.probs[0].detach().numpy()
                prob_str = '  '.join(f"{ACTION_NAMES[i]}={probs[i]:.3f}" for i in range(5))
                logger.info(f"${current_price:,.2f}  {prob_str}  → {ACTION_NAMES[action]}")

                # Execute
                self.execute_trade(action, current_price)

                # Portfolio snapshot
                pv = self.balance + self.margin_locked + self.position * current_price
                self.portfolio_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'price': current_price,
                    'portfolio_value': round(pv, 2),
                    'balance': round(self.balance, 2),
                    'position': round(self.position, 8),
                    'action': ACTION_NAMES[action],
                })

                # Dashboard
                elapsed = (time.time() - session_start) / 60
                self._display_dashboard(current_price, pv, elapsed)

                # Sleep
                elapsed_loop = time.time() - loop_start
                sleep_time = max(0, check_interval - elapsed_loop)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        finally:
            self.generate_report()

    # ─────────────────────────────────────────────────────────────────────────
    # Display & Report
    # ─────────────────────────────────────────────────────────────────────────

    def _display_dashboard(self, price: float, portfolio: float, elapsed_min: float):
        G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'
        C = '\033[96m'; B = '\033[1m'; E = '\033[0m'
        pnl = portfolio - self.initial_balance
        pnl_c = G if pnl >= 0 else R
        pos_type = "LONG" if self.position > 0 else ("SHORT" if self.position < 0 else "NONE")
        wr = self.win_count / self.total_trades * 100 if self.total_trades > 0 else 0

        dashboard = (
            f"\n{B}{C}╔══════════════════════ RL TRADING BOT ══════════════════════╗{E}\n"
            f"{B}{C}║{E} Time: {elapsed_min:6.1f}min  Price: ${price:>10,.2f}  Trades: {self.total_trades:3d} {B}{C}║{E}\n"
            f"{B}{C}║{E} Balance: ${self.balance:>10,.2f}  Position: {pos_type:5s} {self.position:+.6f} BTC {B}{C}║{E}\n"
            f"{B}{C}║{E} Portfolio: ${portfolio:>9,.2f}  PnL: {pnl_c}{pnl:+8.2f}{E}  Win: {wr:.1f}%       {B}{C}║{E}\n"
            f"{B}{C}║{E} Long/Short trades: {self.long_trades_count}/{self.short_trades_count}  Fees: ${self.total_fees:.2f}            {B}{C}║{E}\n"
            f"{B}{C}╚════════════════════════════════════════════════════════════╝{E}"
        )
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.write(dashboard + '\n')
        sys.stdout.flush()

    def generate_report(self):
        """Print and save final trading report."""
        if self.portfolio_history:
            pv = [p['portfolio_value'] for p in self.portfolio_history]
            final = pv[-1]
            returns = np.diff(pv) / np.array(pv[:-1])
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            running_max = np.maximum.accumulate(pv)
            max_dd = ((np.array(pv) - running_max) / running_max).min()
        else:
            final = self.initial_balance
            sharpe = 0.0
            max_dd = 0.0

        total_return = (final / self.initial_balance - 1) * 100

        print("\n" + "=" * 70)
        print("📈 TRADING SESSION REPORT")
        print("=" * 70)
        print(f"  Mode:         {self.mode.upper()}")
        print(f"  Symbol:       {self.symbol}")
        print(f"  Initial:      ${self.initial_balance:,.2f}")
        print(f"  Final:        ${final:,.2f}")
        print(f"  Return:       {total_return:+.2f}%  (${final - self.initial_balance:+,.2f})")
        print(f"  Sharpe:       {sharpe:.4f}")
        print(f"  Max Drawdown: {max_dd * 100:.2f}%")
        print(f"  Trades:       {self.total_trades}  (W:{self.win_count} L:{self.loss_count})")
        print(f"  Long/Short:   {self.long_trades_count}/{self.short_trades_count}")
        print(f"  Fees:         ${self.total_fees:.2f}")

        # Save
        results = {
            'mode': self.mode,
            'symbol': self.symbol,
            'initial_balance': self.initial_balance,
            'final_portfolio': final,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'long_trades': self.long_trades_count,
            'short_trades': self.short_trades_count,
            'total_fees': self.total_fees,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
        }
        out = f"trading_results_{self.mode}.pkl"
        with open(out, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified RL Trading Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest on historical data (no internet needed):
  python trade.py --paper --data btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv

  # Multiple episodes backtest:
  python trade.py --paper --data btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv --episodes 5

  # Virtual trading with live market data (BingX API required):
  python trade.py --live --duration 120

  # Real trading (NOT IMPLEMENTED - will run in virtual mode):
  python trade.py --real --duration 60
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--paper', action='store_true',
                            help='Backtest on historical CSV data')
    mode_group.add_argument('--live', action='store_true',
                            help='Virtual paper trading with live market data')
    mode_group.add_argument('--real', action='store_true',
                            help='⚠️  Real trading (NOT YET IMPLEMENTED)')

    # Common args
    parser.add_argument('--model', default='ppo_trading_agent.zip',
                        help='Path to trained PPO model (default: ppo_trading_agent.zip)')
    parser.add_argument('--symbol', default='BTC-USDT',
                        help='Trading symbol (default: BTC-USDT)')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance in USDT (default: 10000)')

    # Paper trading args
    parser.add_argument('--data',
                        default='btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv',
                        help='Path to historical CSV data (for --paper mode)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run (for --paper mode, default: 1)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose step output (paper mode)')

    # Live/real args
    parser.add_argument('--duration', type=int, default=60,
                        help='Session duration in minutes (for --live/--real, default: 60)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Decision interval in seconds (for --live/--real, default: 60)')

    args = parser.parse_args()

    # ── Paper trading ─────────────────────────────────────────────────────────
    if args.paper:
        run_paper_trading(
            model_path=args.model,
            data_path=args.data,
            initial_balance=args.balance,
            n_episodes=args.episodes,
            verbose=args.verbose,
        )

    # ── Live or Real trading ──────────────────────────────────────────────────
    elif args.live or args.real:
        mode = 'live' if args.live else 'real'
        if mode == 'real':
            print("\n⚠️  WARNING: Real trading is not yet fully implemented.")
            print("   The bot will run in LIVE (virtual) mode instead.\n")
            confirm = input("Continue in virtual mode? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Aborted.")
                return
            mode = 'live'

        bot = TradingBot(
            model_path=args.model,
            symbol=args.symbol,
            initial_balance=args.balance,
            mode=mode,
        )
        bot.run(duration_minutes=args.duration, check_interval=args.interval)


if __name__ == '__main__':
    main()
