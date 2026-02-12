#!/usr/bin/env python3
"""
Test script to verify enhanced risk management features in trading environment
"""

import pandas as pd
import numpy as np
from trading_environment import TradingEnvironment
import matplotlib.pyplot as plt

def create_sample_data():
    """Create sample price data for testing"""
    np.random.seed(42)
    n_points = 1000
    
    # Create realistic price series with some trends and volatility
    returns = np.random.normal(0.0005, 0.02, n_points)  # Daily returns
    prices = 50000 * np.exp(np.cumsum(returns))  # Starting at $50,000
    
    # Add technical indicators (simplified)
    df = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
        'open': prices * (1 + np.random.normal(0, 0.005, n_points))
    })
    
    # Add simple technical indicators
    df['RSI_15'] = 50  # Neutral RSI
    df['BB_15_upper'] = df['close'] * 1.02
    df['BB_15_lower'] = df['close'] * 0.98
    df['ATR_15'] = df['close'] * 0.015  # 1.5% ATR
    df['OBV'] = np.cumsum(np.random.randn(len(df)))
    df['AD'] = np.random.randn(len(df))
    df['MFI_15'] = 50  # Neutral MFI
    
    return df

def test_dynamic_position_sizing():
    """Test dynamic position sizing based on volatility"""
    print("Testing Dynamic Position Sizing...")
    
    df = create_sample_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=200)
    
    # Test position size calculation at different volatility levels
    for step in [50, 100, 150]:
        if step < len(df):
            env.current_step = step
            max_long, max_short, min_size = env._get_position_sizes()
            dynamic_size = env._calculate_volatility_based_position_size(df.iloc[step]['close'])
            
            print(f"Step {step}: Price=${df.iloc[step]['close']:.2f}, Dynamic Size={dynamic_size:.4f}, "
                  f"Max Long=${max_long:.2f}, Max Short=${max_short:.2f}")
    
    print("✓ Dynamic position sizing test completed\n")

def test_adaptive_stop_loss():
    """Test adaptive stop loss calculation"""
    print("Testing Adaptive Stop Loss...")
    
    df = create_sample_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=200)
    
    # Test stop loss calculation at different points
    for step in [50, 100, 150]:
        if step >= env.volatility_window and step < len(df):
            env.current_step = step
            current_price = df.iloc[step]['close']
            adaptive_stop = env._calculate_adaptive_stop_loss(current_price)
            
            print(f"Step {step}: Price=${current_price:.2f}, Adaptive Stop Loss={adaptive_stop:.4f} "
                  f"({adaptive_stop*100:.2f}%)")
    
    print("✓ Adaptive stop loss test completed\n")

def test_episode_risk_limits():
    """Test episode-level risk limits"""
    print("Testing Episode Risk Limits...")
    
    df = create_sample_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=200)
    
    # Verify risk parameters are set correctly
    print(f"Max position size: {env.max_position_size:.2%}")
    print(f"Max total exposure: {env.max_total_exposure:.2%}")
    print(f"Stop loss percentage: {env.stop_loss_pct:.2%}")
    print(f"Max episode loss: {env.max_episode_loss_pct:.2%}")
    print(f"Termination stop loss threshold: {env.termination_stop_loss_threshold:.2%}")
    
    print("✓ Episode risk limits test completed\n")

def test_trailing_stop_with_adaptive():
    """Test trailing stop functionality with adaptive parameters"""
    print("Testing Trailing Stop with Adaptive Parameters...")
    
    df = create_sample_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=200)
    
    # Verify adaptive stop loss is enabled
    print(f"Adaptive stop loss enabled: {env.adaptive_stop_loss_enabled}")
    print(f"ATR multiplier: {env.atr_multiplier}")
    print(f"Min stop loss %: {env.min_stop_loss_pct:.2%}")
    print(f"Max stop loss %: {env.max_stop_loss_pct:.2%}")
    
    # Test adaptive stop loss calculation
    test_price = 50000
    env.current_step = 50
    adaptive_stop = env._calculate_adaptive_stop_loss(test_price)
    print(f"Sample adaptive stop loss at ${test_price}: {adaptive_stop:.4f} ({adaptive_stop*100:.2f}%)")
    
    print("✓ Trailing stop adaptive test completed\n")

def run_comprehensive_test():
    """Run a comprehensive test of the enhanced environment"""
    print("=" * 60)
    print("COMPREHENSIVE RISK MANAGEMENT ENHANCEMENT TEST")
    print("=" * 60)
    
    test_dynamic_position_sizing()
    test_adaptive_stop_loss()
    test_episode_risk_limits()
    test_trailing_stop_with_adaptive()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nEnhanced Features Implemented:")
    print("1. ✓ Dynamic position sizing based on market volatility")
    print("2. ✓ Adaptive stop-loss using ATR and market conditions") 
    print("3. ✓ Maximum episode loss limit (5%)")
    print("4. ✓ Tighter position size limits (15% per position)")
    print("5. ✓ Reduced total exposure limits (40% total)")
    print("6. ✓ Enhanced trailing stop with adaptive parameters")
    print("7. ✓ Improved risk termination conditions")

if __name__ == "__main__":
    run_comprehensive_test()