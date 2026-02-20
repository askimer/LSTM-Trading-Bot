#!/usr/bin/env python3
"""
State Utilities for RL Trading
Shared state calculation functions used by both training and live trading
to ensure consistency in state representation.
"""

import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.000006811):
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Array of returns (not portfolio values)
        risk_free_rate: Daily risk-free rate (default: ~2.5% annual)
    
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() > 0:
        return excess_returns.mean() / excess_returns.std()
    return 0.0


def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from portfolio values.
    
    Args:
        portfolio_values: Array of portfolio values over time
    
    Returns:
        float: Maximum drawdown as a fraction (negative value)
    """
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown.min()


def calculate_trading_state(
    balance: float,
    position: float,
    current_price: float,
    initial_balance: float,
    indicators: dict,
    price_mean: float = None,
    price_std: float = None,
    price_history: list = None
) -> np.ndarray:
    """
    Calculate state vector for RL model.
    
    This function MUST be used by both training environment and live trading
    to ensure identical state representation.
    
    Args:
        balance: Current balance in USDT
        position: Current position (positive=long, negative=short) in BTC
        current_price: Current price of the asset
        initial_balance: Initial balance for normalization
        indicators: Dictionary of technical indicators
        price_mean: Mean price for normalization (optional)
        price_std: Std price for normalization (optional)
        price_history: List of historical prices for rolling normalization (optional)
    
    Returns:
        np.ndarray: State vector of shape (10,) with dtype float32
    """
    # Normalize balance
    balance_norm = balance / initial_balance - 1
    
    # Position normalization - value relative to initial balance
    # Positive for long, negative for short
    position_value = abs(position) * current_price
    position_norm = position_value / initial_balance if initial_balance > 0 else 0.0
    if position < 0:
        position_norm = -position_norm
    position_norm = np.clip(position_norm, -2.0, 2.0)
    
    # Price normalization using rolling window
    if price_mean is not None and price_std is not None and price_std > 0:
        price_norm = (current_price - price_mean) / price_std
    elif price_history and len(price_history) >= 10:
        rolling_mean = np.mean(price_history[-100:]) if len(price_history) >= 100 else np.mean(price_history)
        rolling_std = np.std(price_history[-100:]) if len(price_history) >= 100 else np.std(price_history)
        if rolling_std > 0:
            price_norm = (current_price - rolling_mean) / rolling_std
        else:
            price_norm = 0.0
    else:
        # Fallback: use log return if we have previous price
        price_norm = 0.0
    
    # Clip price norm to prevent extreme values
    price_norm = np.clip(price_norm, -10, 10)
    
    # Technical indicators normalization
    indicators_list = [
        indicators.get('RSI_15', 50) / 100 - 0.5,  # RSI: [-0.5, 0.5]
        (indicators.get('BB_15_upper', current_price) / current_price - 1) if current_price > 0 else 0,  # BB upper
        (indicators.get('BB_15_lower', current_price) / current_price - 1) if current_price > 0 else 0,  # BB lower
        indicators.get('ATR_15', 100) / 1000,  # ATR normalized
        indicators.get('OBV', 0) / 1e10,  # OBV normalized
        indicators.get('AD', 0) / 1e10,  # AD normalized
        indicators.get('MFI_15', 50) / 100 - 0.5  # MFI: [-0.5, 0.5]
    ]
    
    # Build state vector
    state = np.array([balance_norm, position_norm, price_norm] + indicators_list, dtype=np.float32)
    
    # Protect against NaN and inf values
    state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
    
    # Clip to observation space bounds
    # bounds: balance[-2,2], position[-2.5,2.5], price[-15,15], indicators[-1,1]
    obs_low = np.array([-2.0, -2.5, -15.0] + [-1.0] * 7, dtype=np.float32)
    obs_high = np.array([2.0, 2.5, 15.0] + [1.0] * 7, dtype=np.float32)
    state = np.clip(state, obs_low, obs_high)
    
    return state


def get_action_name(action: int) -> str:
    """
    Get human-readable action name.
    
    Args:
        action: Action integer (0-4)
    
    Returns:
        str: Action name
    """
    action_names = {
        0: 'HOLD',
        1: 'BUY_LONG',
        2: 'SELL_LONG', 
        3: 'SELL_SHORT',
        4: 'BUY_SHORT (COVER)'
    }
    return action_names.get(action, f'UNKNOWN({action})')
