#!/usr/bin/env python3
"""
Advanced Risk Management for RL Trading Agent
Implements sophisticated risk controls and portfolio management strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class RiskMetrics:
    """Data class to store risk metrics"""
    value_at_risk: float
    conditional_var: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    treynor_ratio: Optional[float] = None


class RiskManager:
    """
    Advanced risk management system for trading
    Implements various risk controls and portfolio management strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.25,  # 25% of capital per position
        max_total_exposure: float = 0.50,  # 50% of capital total exposure
        stop_loss_pct: float = 0.08,  # 8% stop loss
        take_profit_pct: float = 0.15,  # 15% take profit
        max_drawdown_limit: float = 0.20,  # 20% max drawdown
        max_volatility_limit: float = 0.50,  # 50% annualized volatility limit
        correlation_limit: float = 0.7,  # Max correlation between positions
        rebalance_threshold: float = 0.05,  # 5% rebalance threshold
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.max_volatility_limit = max_volatility_limit
        self.correlation_limit = correlation_limit
        self.rebalance_threshold = rebalance_threshold
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}  # Asset -> position details
        self.portfolio_history = []
        self.risk_metrics_history = []
        
        # Market data tracking
        self.prices: Dict[str, List[float]] = {}
        self.returns: Dict[str, List[float]] = {}
        
        # Risk limits
        self.daily_loss_limit = initial_capital * 0.05  # 5% daily loss limit
        self.weekly_loss_limit = initial_capital * 0.10  # 10% weekly loss limit
        self.monthly_loss_limit = initial_capital * 0.15  # 15% monthly loss limit
        
        # Tracking for limits
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.last_reset_date = pd.Timestamp.now().date()
    
    def update_prices(self, asset: str, price: float):
        """Update price for an asset"""
        if asset not in self.prices:
            self.prices[asset] = []
            self.returns[asset] = []
        
        self.prices[asset].append(price)
        
        # Calculate returns
        if len(self.prices[asset]) > 1:
            ret = (price - self.prices[asset][-2]) / self.prices[asset][-2]
            self.returns[asset].append(ret)
    
    def calculate_position_size(self, asset: str, entry_price: float, side: PositionSide) -> float:
        """
        Calculate optimal position size based on risk management rules
        """
        # Calculate maximum position size based on capital and limits
        max_by_capital = self.capital * self.max_position_size
        max_by_exposure = self.capital * self.max_total_exposure
        
        # Calculate position size based on stop loss risk
        risk_per_trade = self.capital * 0.02  # Risk 2% of capital per trade
        stop_loss_amount = entry_price * self.stop_loss_pct
        position_size = risk_per_trade / stop_loss_amount if stop_loss_amount > 0 else 0
        
        # Apply position size limits
        position_size = min(position_size, max_by_capital, max_by_exposure)
        
        # Consider existing positions in same asset
        if asset in self.positions:
            existing_pos = self.positions[asset]
            if existing_pos['side'] == side:
                # Adding to existing position
                remaining_exposure = max_by_exposure - abs(existing_pos['value'])
                position_size = min(position_size, remaining_exposure / entry_price)
            else:
                # Opposite side position - need to reduce position
                opposite_value = existing_pos['value']
                position_size = min(position_size, (self.capital - abs(opposite_value)) * self.max_position_size / entry_price)
        
        return position_size
    
    def calculate_risk_metrics(self, portfolio_values: List[float], benchmark_returns: Optional[List[float]] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for portfolio
        """
        if len(portfolio_values) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0)
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate basic metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (assuming 2% annual risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * np.sqrt(252)  # Annualized VaR
        
        # Conditional Value at Risk (Expected shortfall)
        var_returns = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = np.mean(var_returns) * np.sqrt(252) if len(var_returns) > 0 else 0
        
        # Calculate advanced metrics if benchmark is provided
        beta = None
        alpha = None
        treynor_ratio = None
        
        if benchmark_returns and len(benchmark_returns) == len(returns):
            # Beta calculation
            cov_matrix = np.cov(returns, benchmark_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
            
            # Alpha calculation (Jensen's alpha)
            benchmark_return = np.mean(benchmark_returns) * 252
            expected_return = risk_free_rate * 252 + beta * (benchmark_return - risk_free_rate * 252)
            portfolio_return = np.mean(returns) * 252
            alpha = portfolio_return - expected_return
            
            # Treynor ratio
            if beta != 0:
                treynor_ratio = (portfolio_return - risk_free_rate * 252) / beta
        
        return RiskMetrics(
            value_at_risk=var_95,
            conditional_var=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            treynor_ratio=treynor_ratio
        )
    
    def check_risk_limits(self, asset: str, action: str, current_price: float) -> bool:
        """
        Check if proposed action violates any risk limits
        Returns True if action is allowed, False otherwise
        """
        # Check if we have enough data
        if asset not in self.prices or len(self.prices[asset]) < 2:
            return True  # Not enough data to make risk assessment
        
        # Check daily/weekly/monthly loss limits
        today = pd.Timestamp.now().date()
        if today != self.last_reset_date:
            # Reset daily/weekly/monthly counters
            if today.month != self.last_reset_date.month:
                self.monthly_pnl = 0.0
            if today.weekday() < self.last_reset_date.weekday():  # New week
                self.weekly_pnl = 0.0
            self.daily_pnl = 0.0
            self.last_reset_date = today
        
        # Check drawdown limit
        if hasattr(self, 'peak_capital'):
            current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            if current_drawdown > self.max_drawdown_limit:
                logger.warning(f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
                return False
        
        # Check if position size exceeds limits
        if asset in self.positions:
            position = self.positions[asset]
            position_value = abs(position['quantity'] * current_price)
            position_pct = position_value / self.capital
            
            if position_pct > self.max_position_size:
                logger.warning(f"Position size limit exceeded for {asset}: {position_pct:.2%} > {self.max_position_size:.2%}")
                return False
        
        # Check total exposure limit
        total_exposure = sum(abs(pos['quantity'] * self.prices[asset][0]) for pos in self.positions.values())
        total_exposure_pct = total_exposure / self.capital
        if total_exposure_pct > self.max_total_exposure:
            logger.warning(f"Total exposure limit exceeded: {total_exposure_pct:.2%} > {self.max_total_exposure:.2%}")
            return False
        
        # Check volatility limit if we have enough data
        if len(self.portfolio_history) > 30:  # At least 30 days of data
            recent_values = self.portfolio_history[-30:]
            returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            
            if volatility > self.max_volatility_limit:
                logger.warning(f"Volatility limit exceeded: {volatility:.2%} > {self.max_volatility_limit:.2%}")
                return False
        
        return True
    
    def update_portfolio(self, asset: str, quantity: float, price: float, side: PositionSide):
        """
        Update portfolio with new position
        """
        # Update or create position
        if asset in self.positions:
            # Existing position - adjust
            old_pos = self.positions[asset]
            old_qty = old_pos['quantity']
            old_avg_price = old_pos['avg_price']
            
            # Calculate new average price and quantity
            new_qty = old_qty + quantity if side == old_pos['side'] else old_qty - quantity
            if new_qty == 0:
                # Close position
                del self.positions[asset]
            else:
                if side == old_pos['side']:
                    # Adding to position
                    new_avg_price = (old_qty * old_avg_price + quantity * price) / abs(new_qty)
                else:
                    # Reducing position - use FIFO or average cost basis
                    if abs(new_qty) < abs(old_qty):
                        # Partial close - keep same average price
                        new_avg_price = old_avg_price
                    else:
                        # Opening opposite position
                        new_avg_price = price
                
                self.positions[asset] = {
                    'quantity': new_qty,
                    'avg_price': new_avg_price,
                    'side': PositionSide.LONG if new_qty > 0 else PositionSide.SHORT if new_qty < 0 else PositionSide.FLAT,
                    'entry_time': pd.Timestamp.now()
                }
        else:
            # New position
            self.positions[asset] = {
                'quantity': quantity,
                'avg_price': price,
                'side': side,
                'entry_time': pd.Timestamp.now()
            }
    
    def check_stop_loss_take_profit(self, asset: str, current_price: float) -> Tuple[bool, str, float]:
        """
        Check if stop loss or take profit conditions are met
        Returns (should_exit, reason, exit_price)
        """
        if asset not in self.positions:
            return False, "", 0.0
        
        position = self.positions[asset]
        avg_price = position['avg_price']
        
        if position['side'] == PositionSide.LONG:
            # Long position: stop loss when price falls, take profit when rises
            if current_price <= avg_price * (1 - self.stop_loss_pct):
                return True, "STOP_LOSS", current_price
            elif current_price >= avg_price * (1 + self.take_profit_pct):
                return True, "TAKE_PROFIT", current_price
        elif position['side'] == PositionSide.SHORT:
            # Short position: stop loss when price rises, take profit when falls
            if current_price >= avg_price * (1 + self.stop_loss_pct):
                return True, "STOP_LOSS", current_price
            elif current_price <= avg_price * (1 - self.take_profit_pct):
                return True, "TAKE_PROFIT", current_price
        
        return False, "", 0.0
    
    def rebalance_portfolio(self) -> List[Dict]:
        """
        Rebalance portfolio according to risk management rules
        Returns list of rebalancing actions to take
        """
        rebalance_actions = []
        
        for asset, position in self.positions.items():
            if asset not in self.prices or len(self.prices[asset]) == 0:
                continue
            
            current_price = self.prices[asset][-1]
            target_size = self.capital * self.max_position_size
            current_size = abs(position['quantity'] * current_price)
            
            # Check if rebalancing is needed
            if abs(current_size - target_size) / target_size > self.rebalance_threshold:
                # Calculate how much to buy/sell to reach target
                target_quantity = target_size / current_price * (1 if position['side'] == PositionSide.LONG else -1)
                quantity_to_adjust = target_quantity - position['quantity']
                
                if abs(quantity_to_adjust) > 0:
                    rebalance_actions.append({
                        'asset': asset,
                        'action': 'BUY' if quantity_to_adjust > 0 else 'SELL',
                        'quantity': abs(quantity_to_adjust),
                        'reason': 'REBALANCE'
                    })
        
        return rebalance_actions
    
    def update_capital(self, pnl: float):
        """
        Update account capital with realized P&L
        """
        self.capital += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        
        # Update peak capital for drawdown calculation
        if not hasattr(self, 'peak_capital') or self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Store portfolio value in history
        self.portfolio_history.append(self.capital)
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Get current portfolio allocation by asset
        """
        allocation = {}
        total_value = self.capital  # Include cash in total value
        
        for asset, position in self.positions.items():
            if asset in self.prices and len(self.prices[asset]) > 0:
                current_price = self.prices[asset][-1]
                position_value = position['quantity'] * current_price
                allocation[asset] = position_value / total_value
        
        return allocation


class AdvancedRiskManager(RiskManager):
    """
    Extended risk manager with additional sophisticated features
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Advanced risk metrics
        self.correlation_matrix = {}
        self.beta_tracking = {}
        self.skewness_kurtosis = {}
        
        # Stress testing scenarios
        self.stress_scenarios = [
            {'name': 'Market Crash', 'shock': -0.20},
            {'name': 'Interest Rate Hike', 'shock': -0.10},
            {'name': 'Volatility Spike', 'vol_mult': 2.0},
        ]
    
    def calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between all tracked assets
        """
        assets = list(self.returns.keys())
        correlations = {}
        
        for i, asset1 in enumerate(assets):
            correlations[asset1] = {}
            for j, asset2 in enumerate(assets):
                if i == j:
                    correlations[asset1][asset2] = 1.0
                else:
                    returns1 = self.returns[asset1]
                    returns2 = self.returns[asset2]
                    
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 1:
                        corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                        correlations[asset1][asset2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlations[asset1][asset2] = 0.0
        
        return correlations
    
    def stress_test_portfolio(self, scenario: Dict) -> float:
        """
        Perform stress test on portfolio under given scenario
        Returns estimated portfolio value after stress test
        """
        current_value = self.capital
        portfolio_value = current_value
        
        # Apply scenario to each position
        for asset, position in self.positions.items():
            if asset in self.prices and len(self.prices[asset]) > 0:
                current_price = self.prices[asset][-1]
                
                if 'shock' in scenario:
                    shocked_price = current_price * (1 + scenario['shock'])
                elif 'vol_mult' in scenario:
                    # Increase volatility around current price
                    returns = self.returns[asset] if asset in self.returns and len(self.returns[asset]) > 0 else [0.0]
                    avg_return = np.mean(returns) if len(returns) > 0 else 0.0
                    vol_mult = scenario['vol_mult']
                    shocked_price = current_price * (1 + avg_return + np.random.normal(0, np.std(returns) * vol_mult))
                else:
                    shocked_price = current_price
                
                position_value_change = position['quantity'] * (shocked_price - current_price)
                portfolio_value += position_value_change
        
        return portfolio_value
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """
        Calculate Expected Shortfall (CVaR) at given confidence level
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        var_returns = returns[returns <= np.percentile(returns, confidence_level * 100)]
        
        if len(var_returns) == 0:
            return 0.0
        
        # Calculate expected shortfall as average of worst returns
        expected_shortfall = np.mean(var_returns)
        return expected_shortfall
    
    def calculate_tail_risk_measures(self) -> Dict[str, float]:
        """
        Calculate tail risk measures
        """
        if len(self.portfolio_history) < 10:  # Need sufficient data
            return {'skewness': 0.0, 'kurtosis': 3.0, 'es_95': 0.0, 'es_99': 0.0}
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        # Skewness (measure of asymmetry)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret != 0:
            skewness = np.mean(((returns - mean_ret) / std_ret) ** 3)
        else:
            skewness = 0.0
        
        # Kurtosis (measure of tail thickness)
        if std_ret != 0:
            kurtosis = np.mean(((returns - mean_ret) / std_ret) ** 4)
        else:
            kurtosis = 3.0  # Normal distribution kurtosis
        
        # Expected Shortfall at 95% and 99%
        es_95 = self.calculate_expected_shortfall(0.05)
        es_99 = self.calculate_expected_shortfall(0.01)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'es_95': es_95,
            'es_99': es_99
        }


def apply_risk_management(
    action: int,
    current_price: float,
    asset: str = "BTC",
    risk_manager: RiskManager = None
) -> Tuple[int, Dict]:
    """
    Apply risk management to trading decisions
    
    Args:
        action: Original trading action (0=hold, 1=buy, 2=sell, 3=short, 4=cover)
        current_price: Current market price
        asset: Asset symbol
        risk_manager: RiskManager instance
    
    Returns:
        Tuple of (adjusted_action, risk_info_dict)
    """
    if risk_manager is None:
        # Create default risk manager if none provided
        risk_manager = RiskManager()
    
    # Update price tracking
    risk_manager.update_prices(asset, current_price)
    
    # Check risk limits for the action
    action_allowed = risk_manager.check_risk_limits(asset, "TRADE", current_price)
    
    # Check stop loss/take profit for existing positions
    exit_signal, exit_reason, exit_price = risk_manager.check_stop_loss_take_profit(asset, current_price)
    
    risk_info = {
        'action_allowed': action_allowed,
        'exit_signal': exit_signal,
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'risk_metrics': None,
        'positions': risk_manager.positions.copy()
    }
    
    # If exit signal is triggered, override action
    if exit_signal:
        if asset in risk_manager.positions:
            pos_side = risk_manager.positions[asset]['side']
            if pos_side == PositionSide.LONG:
                adjusted_action = 2  # Sell
            elif pos_side == PositionSide.SHORT:
                adjusted_action = 4  # Cover
            else:
                adjusted_action = 0  # Hold
        else:
            adjusted_action = 0  # Hold
    elif not action_allowed:
        # If action violates risk limits, hold
        adjusted_action = 0  # Hold
    else:
        # Action is allowed, return original action
        adjusted_action = action
    
    # Update risk metrics periodically
    if len(risk_manager.portfolio_history) > 1:
        recent_values = risk_manager.portfolio_history[-30:]  # Last 30 days
        if len(recent_values) >= 2:
            risk_metrics = risk_manager.calculate_risk_metrics(recent_values)
            risk_info['risk_metrics'] = risk_metrics
    
    return adjusted_action, risk_info


if __name__ == "__main__":
    # Example usage
    print("Testing Risk Manager...")
    
    # Create risk manager
    rm = RiskManager(initial_capital=10000)
    
    # Simulate some price data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))  # Geometric Brownian motion
    
    for i, price in enumerate(prices):
        rm.update_prices("BTC", price)
        
        # Simulate holding a position
        if i == 10:
            rm.update_portfolio("BTC", 1.0, price, PositionSide.LONG)
        
        # Check risk metrics every 10 steps
        if i % 10 == 0 and i > 0:
            metrics = rm.calculate_risk_metrics(rm.portfolio_history[-10:] if len(rm.portfolio_history) >= 10 else rm.portfolio_history)
            print(f"Step {i}: Price={price:.2f}, VaR={metrics.value_at_risk:.4f}, MaxDD={metrics.max_drawdown:.4f}, Sharpe={metrics.sharpe_ratio:.4f}")
    
    print("Risk management testing completed!")
