#!/usr/bin/env python3
"""
Basic usage example for RL Trading Bot
Demonstrates how to use the main components of the system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import the main components
from trading_environment import TradingEnvironment
from train_rl import train_rl_agent, evaluate_agent_comprehensive
from rl_paper_trading import run_rl_paper_trading
from risk_management import RiskManager, apply_risk_management
from hyperparameter_optimization import run_optimization


def create_sample_data():
    """Create sample market data for demonstration"""
    print("📊 Creating sample market data...")
    
    # Generate synthetic price data using geometric brownian motion
    n_points = 1000
    dt = 1/252  # Daily time step
    mu = 0.05   # Drift
    sigma = 0.2 # Volatility
    
    # Generate random returns
    returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_points)
    
    # Calculate prices
    prices = 50000 * np.exp(np.cumsum(returns))  # Starting at $50,000
    
    # Create timestamps
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='D')
    
    # Create sample data with technical indicators placeholders
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.9, 1.01, n_points),
        'high': prices * np.random.uniform(1.0, 1.02, n_points),
        'low': prices * np.random.uniform(0.98, 1.0, n_points),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_points),
        'RSI_15': np.random.uniform(20, 80, n_points),  # Placeholder
        'BB_15_upper': prices * np.random.uniform(1.02, 1.05, n_points),  # Placeholder
        'BB_15_lower': prices * np.random.uniform(0.95, 0.98, n_points),  # Placeholder
        'ATR_15': np.random.uniform(100, 500, n_points),  # Placeholder
        'OBV': np.random.uniform(-1000, 1000000, n_points),  # Placeholder
        'AD': np.random.uniform(-500000, 500000, n_points),  # Placeholder
        'MFI_15': np.random.uniform(20, 80, n_points)  # Placeholder
    })
    
    print(f"✅ Created {len(data)} data points from {dates[0].date()} to {dates[-1].date()}")
    return data


def demonstrate_environment():
    """Demonstrate the trading environment"""
    print("\n" + "="*60)
    print("🏛️  DEMONSTRATING TRADING ENVIRONMENT")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Create trading environment
    env = TradingEnvironment(data, initial_balance=10000, transaction_fee=0.001)
    
    # Reset environment
    state, info = env.reset()
    print(f"✅ Environment created with initial balance: ${env.initial_balance:,.2f}")
    print(f"✅ Initial state shape: {state.shape}")
    print(f"✅ Action space: {env.action_space}")
    print(f"✅ Observation space: {env.observation_space}")
    
    # Take a few random steps to demonstrate
    print("\n🔄 Taking 5 random steps to demonstrate environment dynamics...")
    for step in range(5):
        action = env.action_space.sample()  # Random action
        new_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: Action={action}, Reward={reward:.4f}, "
              f"Balance=${info['balance']:.2f}, Position={info['position']:.6f}")
        
        if terminated or truncated:
            print("  🛑 Episode ended")
            break
    
    print("✅ Environment demonstration completed")


def demonstrate_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "="*60)
    print("🛡️  DEMONSTRATING RISK MANAGEMENT")
    print("="*60)
    
    # Create risk manager
    risk_manager = RiskManager(
        initial_capital=10000,
        max_position_size=0.25,
        max_total_exposure=0.50,
        stop_loss_pct=0.08,
        take_profit_pct=0.15
    )
    
    print(f"✅ Risk manager created with:")
    print(f"   - Initial capital: ${risk_manager.initial_capital:,.2f}")
    print(f"   - Max position size: {risk_manager.max_position_size:.1%}")
    print(f"   - Max total exposure: {risk_manager.max_total_exposure:.1%}")
    print(f"   - Stop loss: {risk_manager.stop_loss_pct:.1%}")
    print(f"   - Take profit: {risk_manager.take_profit_pct:.1%}")
    
    # Demonstrate position management
    print(f"\n💼 Updating portfolio with a long position...")
    risk_manager.update_portfolio("BTC", 0.1, 50000.0, risk_manager.__class__.__bases__[0].__dict__['PositionSide'].LONG)
    print(f"   Position: 0.1 BTC at $50,000 = ${0.1 * 50000:,.2f}")
    
    # Demonstrate risk limit checking
    print(f"\n⚖️  Checking risk limits...")
    allowed = risk_manager.check_risk_limits("BTC", "TRADE", 51000.0)
    print(f"   Trade allowed: {allowed}")
    
    # Demonstrate stop loss/take profit checking
    print(f"\n📊 Checking stop loss/take profit...")
    exit_signal, reason, exit_price = risk_manager.check_stop_loss_take_profit("BTC", 45000.0)  # Below stop loss
    print(f"   At $45,000: Exit signal = {exit_signal}, Reason = {reason}")
    
    exit_signal, reason, exit_price = risk_manager.check_stop_loss_take_profit("BTC", 58000.0)  # Above take profit
    print(f"   At $58,000: Exit signal = {exit_signal}, Reason = {reason}")
    
    print("✅ Risk management demonstration completed")


def demonstrate_model_training():
    """Demonstrate model training (conceptual - would need real data)"""
    print("\n" + "="*60)
    print("🎓 DEMONSTRATING MODEL TRAINING CONCEPT")
    print("="*60)
    
    print("ℹ️  Note: Actual training requires real market data")
    print("   This is a conceptual demonstration of the training process")
    
    print(f"\n🛠️  Training parameters would include:")
    print(f"   - Learning rate: 0.0003")
    print(f"   - Total timesteps: 100,000")
    print(f"   - Batch size: 128")
    print(f"   - Network architecture: [512, 256, 128]")
    print(f"   - Algorithm: PPO")
    
    print(f"\n📊 Training would produce metrics like:")
    print(f"   - Average return: ?%")
    print(f"   - Sharpe ratio: ?")
    print(f"   - Maximum drawdown: ?%")
    print(f"   - Win rate: ?%")
    
    print("✅ Training concept demonstrated")


def demonstrate_paper_trading():
    """Demonstrate paper trading (conceptual)"""
    print("\n" + "="*60)
    print("📈 DEMONSTRATING PAPER TRADING CONCEPT")
    print("="*60)
    
    print("ℹ️  Note: Actual paper trading requires a trained model")
    print("   This is a conceptual demonstration of the paper trading process")
    
    print(f"\n💳 Paper trading would simulate:")
    print(f"   - Initial balance: $10,000")
    print(f"   - Trading period: Historical data period")
    print(f"   - Trading strategy: RL model decisions")
    print(f"   - Transaction fees: Applied to each trade")
    
    print(f"\n📊 Paper trading would report metrics:")
    print(f"   - Total return: ?%")
    print(f"   - Final portfolio value: $?")
    print(f"   - Number of trades: ?")
    print(f"   - Win rate: ?%")
    print(f"   - Maximum drawdown: ?%")
    
    print("✅ Paper trading concept demonstrated")


def main():
    """Main function to run all demonstrations"""
    print("🤖 RL Trading Bot - Basic Usage Examples")
    print("This script demonstrates the main components of the RL trading system")
    
    try:
        # Demonstrate each component
        demonstrate_environment()
        demonstrate_risk_management()
        demonstrate_model_training()
        demonstrate_paper_trading()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📝 Next steps:")
        print("   1. Prepare real market data with technical indicators")
        print("   2. Train a model using: python -m train_rl")
        print("   3. Test with paper trading: python -m rl_paper_trading")
        print("   4. Evaluate risk management: python -m risk_management")
        print("   5. Optimize hyperparameters: python -m hyperparameter_optimization")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
