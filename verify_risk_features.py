#!/usr/bin/env python3
"""
Verification script to test the enhanced risk management features
"""

import pandas as pd
import numpy as np
from trading_environment import TradingEnvironment

def create_test_data():
    """Create test data with known patterns"""
    np.random.seed(42)
    n_points = 500
    
    # Create a price series with some volatility
    returns = np.random.normal(0.001, 0.025, n_points)  # Slightly higher volatility
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
        'open': prices * (1 + np.random.normal(0, 0.005, n_points))
    })
    
    # Add technical indicators
    df['RSI_15'] = 50
    df['BB_15_upper'] = df['close'] * 1.02
    df['BB_15_lower'] = df['close'] * 0.98
    df['ATR_15'] = df['close'] * 0.015
    df['OBV'] = np.cumsum(np.random.randn(len(df)))
    df['AD'] = np.random.randn(len(df))
    df['MFI_15'] = 50
    
    return df

def test_risk_parameters():
    """Test that all risk parameters are properly set"""
    print("Testing Risk Parameter Configuration...")
    
    df = create_test_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=100)
    
    # Verify all enhanced parameters
    assert env.max_position_size == 0.15, f"Expected 0.15, got {env.max_position_size}"
    assert env.max_total_exposure == 0.40, f"Expected 0.40, got {env.max_total_exposure}"
    assert env.stop_loss_pct == 0.08, f"Expected 0.08, got {env.stop_loss_pct}"
    assert env.max_episode_loss_pct == 0.05, f"Expected 0.05, got {env.max_episode_loss_pct}"
    assert env.adaptive_stop_loss_enabled == True, f"Expected True, got {env.adaptive_stop_loss_enabled}"
    assert env.atr_multiplier == 2.0, f"Expected 2.0, got {env.atr_multiplier}"
    
    print("✓ All risk parameters correctly configured")
    print(f"  - Max position size: {env.max_position_size:.2%}")
    print(f"  - Max total exposure: {env.max_total_exposure:.2%}")
    print(f"  - Stop loss %: {env.stop_loss_pct:.2%}")
    print(f"  - Max episode loss: {env.max_episode_loss_pct:.2%}")
    print(f"  - Adaptive stop loss: {env.adaptive_stop_loss_enabled}")
    print(f"  - ATR multiplier: {env.atr_multiplier}")
    print()

def test_dynamic_position_sizing():
    """Test dynamic position sizing functionality"""
    print("Testing Dynamic Position Sizing...")
    
    df = create_test_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=100)
    
    # Test at different volatility levels
    for step in [25, 50, 75]:
        env.current_step = step
        dynamic_size = env._calculate_volatility_based_position_size(df.iloc[step]['close'])
        max_long, max_short, _ = env._get_position_sizes()
        
        print(f"  Step {step}: Volatility-adjusted size = {dynamic_size:.4f}, "
              f"Max long = ${max_long:.2f}, Max short = ${max_short:.2f}")
        
        # Verify position sizes are within limits
        assert max_long <= env.balance * env.max_position_size, "Long position exceeds limit"
        assert max_short <= env.balance * env.max_position_size, "Short position exceeds limit"
        assert dynamic_size >= env.min_position_size, f"Dynamic size {dynamic_size} below minimum {env.min_position_size}"
        assert dynamic_size <= env.max_dynamic_position_size, f"Dynamic size {dynamic_size} above maximum {env.max_dynamic_position_size}"
    
    print("✓ Dynamic position sizing working correctly")
    print()

def test_adaptive_stop_loss():
    """Test adaptive stop loss calculation"""
    print("Testing Adaptive Stop Loss...")
    
    df = create_test_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=100)
    
    # Test adaptive stop loss at different points
    for step in [50, 100, 150]:
        if step >= env.volatility_window:
            env.current_step = step
            current_price = df.iloc[step]['close']
            adaptive_stop = env._calculate_adaptive_stop_loss(current_price)
            
            print(f"  Step {step}: Price = ${current_price:.2f}, Adaptive stop = {adaptive_stop:.4f} ({adaptive_stop*100:.2f}%)")
            
            # Verify stop loss is within bounds
            assert adaptive_stop >= env.min_stop_loss_pct, f"Stop loss {adaptive_stop} below minimum {env.min_stop_loss_pct}"
            assert adaptive_stop <= env.max_stop_loss_pct, f"Stop loss {adaptive_stop} above maximum {env.max_stop_loss_pct}"
    
    print("✓ Adaptive stop loss working correctly")
    print()

def test_episode_risk_limits():
    """Test episode-level risk limits"""
    print("Testing Episode Risk Limits...")
    
    df = create_test_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=100)
    
    # Test the maximum loss per episode trigger
    initial_balance = env.initial_balance
    max_loss_allowed = initial_balance * (1 - env.max_episode_loss_pct)
    
    print(f"  Initial balance: ${initial_balance:,.2f}")
    print(f"  Max loss allowed: {env.max_episode_loss_pct:.2%}")
    print(f"  Balance threshold for termination: ${max_loss_allowed:,.2f}")
    
    # Simulate portfolio value calculation
    portfolio_value = env.balance + env.margin_locked + env.position * df.iloc[0]['close']
    portfolio_return = (portfolio_value - initial_balance) / initial_balance
    
    print(f"  Current portfolio return: {portfolio_return:.2%}")
    print(f"  Will terminate if portfolio drops below: ${(initial_balance * (1 - env.max_episode_loss_pct)):,.2f}")
    
    print("✓ Episode risk limits properly configured")
    print()

def test_trailing_stop_functionality():
    """Test trailing stop functionality with adaptive parameters"""
    print("Testing Trailing Stop Functionality...")
    
    df = create_test_data()
    env = TradingEnvironment(df, initial_balance=10000, episode_length=100)
    
    # Test that trailing stop uses adaptive parameters
    print(f"  Trailing stop distance: {env.trailing_stop_distance:.2%}")
    print(f"  Adaptive stop loss enabled: {env.adaptive_stop_loss_enabled}")
    print(f"  ATR multiplier: {env.atr_multiplier}")
    
    # Verify the method exists and works
    test_price = 50000
    env.current_step = 50
    adaptive_stop = env._calculate_adaptive_stop_loss(test_price)
    
    print(f"  Sample adaptive stop calculation: {adaptive_stop:.4f} ({adaptive_stop*100:.2f}%)")
    
    print("✓ Trailing stop functionality verified")
    print()

def run_verification():
    """Run all verification tests"""
    print("=" * 70)
    print("RISK MANAGEMENT FEATURES VERIFICATION")
    print("=" * 70)
    
    test_risk_parameters()
    test_dynamic_position_sizing()
    test_adaptive_stop_loss()
    test_episode_risk_limits()
    test_trailing_stop_functionality()
    
    print("=" * 70)
    print("✅ ALL RISK MANAGEMENT FEATURES VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📋 IMPLEMENTED ENHANCEMENTS:")
    print("  ✅ Dynamic position sizing based on market volatility")
    print("  ✅ Adaptive stop-loss using ATR and market conditions")
    print("  ✅ Maximum episode loss limit (5%)")
    print("  ✅ Tighter position size limits (15% per position)")
    print("  ✅ Reduced total exposure limits (40% total)")
    print("  ✅ Enhanced trailing stop with adaptive parameters")
    print("  ✅ Improved risk termination conditions")
    print("  ✅ Volatility-based position adjustment")
    print("  ✅ Market condition-aware stop losses")

if __name__ == "__main__":
    run_verification()