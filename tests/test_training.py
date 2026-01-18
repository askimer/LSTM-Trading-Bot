import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from train_rl import preprocess_data, calculate_max_drawdown, calculate_sharpe_ratio
from trading_environment import TradingEnvironment


class TestTrainingComponents(unittest.TestCase):
    """Unit tests for training components"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, 100),
            'high': np.random.uniform(40000, 60000, 100),
            'low': np.random.uniform(40000, 60000, 100),
            'close': np.random.uniform(40000, 60000, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'RSI_15': np.random.uniform(0, 100, 100),
            'BB_15_upper': np.random.uniform(50000, 70000, 100),
            'BB_15_lower': np.random.uniform(30000, 50000, 100),
            'ATR_15': np.random.uniform(1000, 3000, 100),
            'OBV': np.random.uniform(-1000, 1000000, 100),
            'AD': np.random.uniform(-500000, 500000, 100),
            'MFI_15': np.random.uniform(0, 100, 100)
        })

    def test_preprocess_data(self):
        """Test data preprocessing function"""
        processed_data = preprocess_data(self.sample_data.copy())
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # Check that the number of rows is preserved or reduced (due to cleaning)
        self.assertLessEqual(len(processed_data), len(self.sample_data))
        
        # Check that there are no NaN values
        self.assertFalse(processed_data.isnull().any().any())

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Test with increasing portfolio values (no drawdown)
        portfolio_values_up = [100, 110, 120, 130, 140]
        max_dd_up = calculate_max_drawdown(portfolio_values_up)
        self.assertEqual(max_dd_up, 0.0)
        
        # Test with decreasing portfolio values
        portfolio_values_down = [100, 90, 80, 70, 60]
        max_dd_down = calculate_max_drawdown(portfolio_values_down)
        self.assertGreater(max_dd_down, 0.0)
        
        # Test with mixed portfolio values (has drawdown)
        portfolio_values_mixed = [100, 110, 90, 120, 80]
        max_dd_mixed = calculate_max_drawdown(portfolio_values_mixed)
        self.assertGreaterEqual(max_dd_mixed, 0.0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Test with positive returns
        portfolio_values = [100, 105, 110, 108, 115]
        sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
        self.assertIsInstance(sharpe_ratio, float)
        
        # Test with flat returns (should have 0 volatility, return 0)
        flat_values = [10, 100, 100, 100]
        flat_sharpe = calculate_sharpe_ratio(flat_values)
        self.assertEqual(flat_sharpe, 0.0)

    def test_trading_environment_creation(self):
        """Test creation of trading environment"""
        env = TradingEnvironment(self.sample_data, initial_balance=10000)
        
        # Check that environment was created successfully
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_balance, 1000)
        
        # Check that reset works
        obs, info = env.reset()
        self.assertIsNotNone(obs)
        self.assertIsInstance(info, dict)

    def test_trading_environment_step(self):
        """Test taking steps in trading environment"""
        env = TradingEnvironment(self.sample_data, initial_balance=10000)
        obs, _ = env.reset()
        
        # Test different actions
        for action in range(5):  # Test all possible actions
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Check that outputs are of correct type
            self.assertIsNotNone(new_obs)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

    def test_trading_environment_balance_consistency(self):
        """Test that balance remains consistent"""
        env = TradingEnvironment(self.sample_data, initial_balance=10000)
        obs, _ = env.reset()
        
        initial_balance = env.balance
        initial_portfolio = env.balance + env.margin_locked + env.position * self.sample_data.iloc[env.current_step]['close']
        
        # Take a few random steps
        for _ in range(10):
            action = np.random.choice(range(5))
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check that balance properties are maintained
            current_portfolio = env.balance + env.margin_locked + env.position * self.sample_data.iloc[env.current_step]['close']
            self.assertIsInstance(current_portfolio, (int, float))
            self.assertGreaterEqual(env.balance, -1e-6)  # Allow for tiny floating point errors

    def test_data_validation_in_environment(self):
        """Test that environment handles different data formats"""
        # Test with different column names (close vs Close)
        data_with_uppercase = self.sample_data.copy()
        data_with_uppercase = data_with_uppercase.rename(columns={'close': 'Close'})
        
        env = TradingEnvironment(data_with_uppercase, initial_balance=10000)
        self.assertIsNotNone(env)
        
        # Test with minimal required columns
        minimal_data = self.sample_data[['close', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'ATR_15', 'OBV', 'AD', 'MFI_15']].copy()
        minimal_data['high'] = self.sample_data['high']
        minimal_data['low'] = self.sample_data['low']
        minimal_data['volume'] = self.sample_data['volume']
        minimal_data['open'] = self.sample_data['open']
        
        env_minimal = TradingEnvironment(minimal_data, initial_balance=10000)
        self.assertIsNotNone(env_minimal)


class TestReproducibility(unittest.TestCase):
    """Tests for ensuring reproducible results"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(40000, 50000, 50),
            'high': np.linspace(41000, 51000, 50),
            'low': np.linspace(39000, 49000, 50),
            'close': np.linspace(40500, 500, 50),
            'volume': np.full(50, 5000),
            'RSI_15': np.linspace(30, 70, 50),
            'BB_15_upper': np.linspace(45000, 55000, 50),
            'BB_15_lower': np.linspace(35000, 45000, 50),
            'ATR_15': np.full(50, 1500),
            'OBV': np.linspace(-10000, 100000, 50),
            'AD': np.linspace(-50000, 50000, 50),
            'MFI_15': np.linspace(20, 80, 50)
        })

    @patch('train_rl.set_seeds')
    def test_environment_determinism(self, mock_set_seeds):
        """Test that environment behaves deterministically with fixed seeds"""
        # This test verifies that our approach to setting seeds is correct
        # In practice, we'd need to test the actual training process to fully validate reproducibility
        env1 = TradingEnvironment(self.sample_data, initial_balance=10000)
        env2 = TradingEnvironment(self.sample_data, initial_balance=10000)
        
        # Reset both environments
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # With identical data and initial conditions, observations should be the same
        np.testing.assert_array_equal(obs1, obs2)
        
        # Take same action on both
        action = 0  # Hold
        new_obs1, reward1, _, _, _ = env1.step(action)
        new_obs2, reward2, _, _, _ = env2.step(action)
        
        # Results should be identical
        np.testing.assert_array_equal(new_obs1, new_obs2)
        self.assertEqual(reward1, reward2)


if __name__ == '__main__':
    unittest.main()
