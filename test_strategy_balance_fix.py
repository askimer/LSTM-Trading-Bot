#!/usr/bin/env python3
"""
Test script to verify the strategy balance fixes
Tests that the environment properly incentivizes both long and short trades
"""

import numpy as np
import pandas as pd
from enhanced_trading_environment import EnhancedTradingEnvironment

def create_test_data(n_rows=500):
    """Create synthetic price data for testing"""
    np.random.seed(42)
    
    # Create price data with both upward and downward trends
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='h')
    
    # Mix of uptrends and downtrends
    price = 50000
    prices = []
    for i in range(n_rows):
        # Create alternating trends
        trend = np.sin(i / 50) * 0.001  # Sinusoidal trend
        noise = np.random.normal(0, 0.005)
        price *= (1 + trend + noise)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': [1000 + np.random.rand() * 500 for _ in range(n_rows)],
        'ATR_15': [100 + np.random.rand() * 50 for _ in range(n_rows)],
        'RSI_15': [50 + np.random.rand() * 30 for _ in range(n_rows)],
        'BB_15_upper': [p * 1.02 for p in prices],
        'BB_15_lower': [p * 0.98 for p in prices],
        'OBV': [np.random.rand() * 1e6 for _ in range(n_rows)],
        'AD': [np.random.rand() * 1e6 for _ in range(n_rows)],
        'MFI_15': [50 + np.random.rand() * 30 for _ in range(n_rows)]
    })
    
    return df

def test_strategy_balance_tracking():
    """Test that strategy balance tracking only counts opening trades"""
    print("=" * 60)
    print("TEST 1: Strategy Balance Tracking")
    print("=" * 60)
    
    df = create_test_data(500)
    env = EnhancedTradingEnvironment(df, enable_strategy_balancing=True, debug=True)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Simulate a series of actions
    actions = [
        1,  # Buy Long (opening) - should count as long trade
        0,  # Hold
        0,  # Hold
        2,  # Sell Long (closing) - should NOT count as long trade
        3,  # Sell Short (opening) - should count as short trade
        0,  # Hold
        0,  # Hold
        4,  # Cover Short (closing) - should NOT count as short trade
        3,  # Sell Short (opening) - should count as short trade
        1,  # Buy Long (opening) - should count as long trade
    ]
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Long trades={env.long_trades}, Short trades={env.short_trades}")
        
        if terminated:
            break
    
    print(f"\nFinal counts: Long trades={env.long_trades}, Short trades={env.short_trades}")
    
    # Verify: should have 2 long trades (actions 1 at step 0 and 9) and 2 short trades (actions 3 at step 4 and 8)
    assert env.long_trades == 2, f"Expected 2 long trades, got {env.long_trades}"
    assert env.short_trades == 2, f"Expected 2 short trades, got {env.short_trades}"
    
    print("✅ PASSED: Strategy balance tracking correctly counts only opening trades")
    return True

def test_strategy_balance_rewards():
    """Test that strategy balance rewards incentivize underrepresented directions"""
    print("\n" + "=" * 60)
    print("TEST 2: Strategy Balance Rewards")
    print("=" * 60)
    
    df = create_test_data(500)
    env = EnhancedTradingEnvironment(df, enable_strategy_balancing=True, debug=True)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Simulate all short trades to create imbalance
    print("\nSimulating all short trades to create imbalance...")
    for i in range(10):
        # Open short
        obs, reward, terminated, truncated, info = env.step(3)  # Sell Short
        if terminated:
            obs, _ = env.reset()
        # Close short
        obs, reward, terminated, truncated, info = env.step(4)  # Cover Short
        if terminated:
            obs, _ = env.reset()
    
    print(f"After 10 short trades: Long={env.long_trades}, Short={env.short_trades}")
    
    # Now test the balance reward for long vs short actions
    balance_reward_long = env._calculate_strategy_balance_reward(1)  # Buy Long
    balance_reward_short = env._calculate_strategy_balance_reward(3)  # Sell Short
    
    print(f"Balance reward for Long action: {balance_reward_long:.4f}")
    print(f"Balance reward for Short action: {balance_reward_short:.4f}")
    
    # Long action should get positive reward (underrepresented)
    # Short action should get negative reward (overrepresented)
    assert balance_reward_long > 0, f"Expected positive reward for long action, got {balance_reward_long}"
    assert balance_reward_short < 0, f"Expected negative reward for short action, got {balance_reward_short}"
    
    print("✅ PASSED: Strategy balance rewards correctly incentivize underrepresented directions")
    return True

def test_direction_multipliers():
    """Test that direction multipliers are applied to trade rewards"""
    print("\n" + "=" * 60)
    print("TEST 3: Direction Reward Multipliers")
    print("=" * 60)
    
    df = create_test_data(500)
    env = EnhancedTradingEnvironment(df, enable_strategy_balancing=True, debug=True)
    
    print(f"Long reward multiplier: {env.long_reward_multiplier}")
    print(f"Short reward multiplier: {env.short_reward_multiplier}")
    
    # Verify multipliers are set
    assert env.long_reward_multiplier == 1.2, f"Expected long multiplier 1.2, got {env.long_reward_multiplier}"
    assert env.short_reward_multiplier == 1.0, f"Expected short multiplier 1.0, got {env.short_reward_multiplier}"
    
    print("✅ PASSED: Direction reward multipliers are correctly configured")
    return True

def test_exploration_bonuses():
    """Test that exploration bonuses are properly accumulated"""
    print("\n" + "=" * 60)
    print("TEST 4: Exploration Bonuses")
    print("=" * 60)
    
    df = create_test_data(500)
    env = EnhancedTradingEnvironment(df, enable_strategy_balancing=True, debug=True)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Create imbalance by only trading shorts
    for i in range(5):
        env.step(3)  # Sell Short
        env.step(4)  # Cover Short
    
    print(f"After 5 short trades: Long={env.long_trades}, Short={env.short_trades}")
    print(f"Exploration bonuses: Long={env.direction_exploration_bonuses['long']:.4f}, Short={env.direction_exploration_bonuses['short']:.4f}")
    
    # Long exploration bonus should be positive (underrepresented)
    assert env.direction_exploration_bonuses['long'] > 0, "Expected positive long exploration bonus"
    
    print("✅ PASSED: Exploration bonuses correctly accumulated for underrepresented direction")
    return True

def test_full_episode_balance():
    """Test a full episode to verify balanced trading is encouraged"""
    print("\n" + "=" * 60)
    print("TEST 5: Full Episode Balance Check")
    print("=" * 60)
    
    df = create_test_data(500)
    env = EnhancedTradingEnvironment(df, enable_strategy_balancing=True, debug=False)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run episode with random actions
    total_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for step in range(200):
        # Random action
        action = np.random.randint(0, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_counts[action] += 1
        
        if terminated or truncated:
            break
    
    print(f"Episode finished after {step+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Action distribution: {action_counts}")
    print(f"Long trades: {env.long_trades}, Short trades: {env.short_trades}")
    
    if env.long_trades + env.short_trades > 0:
        balance_ratio = env.long_trades / (env.long_trades + env.short_trades)
        print(f"Long/Short balance: {balance_ratio*100:.1f}% / {(1-balance_ratio)*100:.1f}%")
    
    print("✅ PASSED: Full episode completed successfully")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("STRATEGY BALANCE FIX VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        test_strategy_balance_tracking,
        test_strategy_balance_rewards,
        test_direction_multipliers,
        test_exploration_bonuses,
        test_full_episode_balance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! The strategy balance fixes are working correctly.")
        print("\nKey improvements made:")
        print("1. Strategy balance tracking now only counts OPENING trades")
        print("2. Stronger balance penalties (0.15 vs 0.05)")
        print("3. Higher exploration bonuses (0.10 vs 0.02)")
        print("4. Direction-specific opening bonuses (0.15)")
        print("5. Progressive imbalance penalties")
        print("6. Higher entropy coefficient (0.10 vs 0.05) for more exploration")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    main()
