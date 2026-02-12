#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Tests
Tests for evaluating RL trading model performance across multiple dimensions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from trading_environment import TradingEnvironment

class ModelEvaluationTester:
    """Comprehensive tester for RL trading model evaluation"""
    
    def __init__(self, model_path: str, test_data_path: str, n_episodes: int = 50):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data for evaluation
            n_episodes: Number of episodes for evaluation
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.n_episodes = n_episodes
        self.model = None
        self.test_data = None
        self.evaluation_results = {}
        
        # Load model and test data
        self._load_model_and_data()
        
    def _load_model_and_data(self):
        """Load the trained model and test data"""
        print("Loading model and test data...")
        
        # Load model
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
        
        # Load test data
        try:
            self.test_data = pd.read_csv(self.test_data_path)
            print(f"✅ Test data loaded with {len(self.test_data)} rows")
        except Exception as e:
            raise ValueError(f"Failed to load test data: {e}")
    
    def test_rewards_quality(self) -> Dict:
        """Test the quality and distribution of rewards"""
        print("\n🔍 Testing reward quality...")
        
        # Evaluate policy to get rewards
        env = TradingEnvironment(self.test_data, episode_length=200)
        env = Monitor(env)
        
        rewards = []
        episode_lengths = []
        
        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # Calculate reward statistics
        reward_stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'reward_range': np.max(rewards) - np.min(rewards),
            'episode_lengths_mean': np.mean(episode_lengths),
            'reward_variance': np.var(rewards),
            'reward_cv': np.std(rewards) / np.abs(np.mean(rewards)) if np.mean(rewards) != 0 else float('inf'),
            'degenerate_rewards': len([r for r in rewards if abs(r) < 1e-6]),
            'zero_rewards': len([r for r in rewards if r == 0]),
            'negative_rewards': len([r for r in rewards if r < 0]),
            'positive_rewards': len([r for r in rewards if r > 0])
        }
        
        # Quality checks
        reward_stats['quality_pass'] = (
            reward_stats['mean_reward'] != 0 and
            reward_stats['reward_range'] > 0 and
            reward_stats['degenerate_rewards'] < len(rewards) * 0.1  # Less than 10% degenerate
        )
        
        print(f"   Mean reward: {reward_stats['mean_reward']:.4f}")
        print(f"   Reward range: {reward_stats['reward_range']:.4f}")
        print(f"   Degenerate rewards: {reward_stats['degenerate_rewards']}/{len(rewards)}")
        print(f"   Quality pass: {reward_stats['quality_pass']}")
        
        self.evaluation_results['rewards'] = reward_stats
        return reward_stats
    
    def test_profitability_metrics(self) -> Dict:
        """Test profitability and trading performance metrics"""
        print("\n💰 Testing profitability metrics...")
        
        env = TradingEnvironment(self.test_data, episode_length=200)
        env = Monitor(env)
        
        portfolio_values = []
        returns = []
        trade_counts = []
        win_rates = []
        total_pnls = []
        trade_pnls = []
        
        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            done = False
            episode_pnls = []
            trade_count = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Collect trading info
                if info.get('action_performed', False):
                    trade_count += 1
                    pnl = info.get('pnl_pct', 0)
                    if pnl != 0:
                        episode_pnls.append(pnl)
                
                done = terminated or truncated
            
            # Calculate episode metrics
            if hasattr(env.unwrapped, 'portfolio_values') and len(env.unwrapped.portfolio_values) > 1:
                final_portfolio = env.unwrapped.portfolio_values[-1]
                initial_portfolio = env.unwrapped.portfolio_values[0]
                episode_return = (final_portfolio - initial_portfolio) / initial_portfolio
            else:
                final_portfolio = info.get('portfolio_value', 0)
                initial_portfolio = env.unwrapped.initial_balance
                episode_return = (final_portfolio - initial_portfolio) / initial_portfolio if initial_portfolio > 0 else 0
            
            portfolio_values.append(final_portfolio)
            returns.append(episode_return)
            trade_counts.append(trade_count)
            total_pnls.append(info.get('total_pnl', 0))
            trade_pnls.extend(episode_pnls)
            
            # Calculate win rate for this episode
            wins = len([pnl for pnl in episode_pnls if pnl > 0])
            win_rate = wins / len(episode_pnls) if len(episode_pnls) > 0 else 0
            win_rates.append(win_rate)
        
        # Calculate overall metrics
        profitability_stats = {
            'total_return_mean': np.mean(returns),
            'total_return_std': np.std(returns),
            'total_return_median': np.median(returns),
            'total_return_sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'win_rate_mean': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'avg_trades_per_episode': np.mean(trade_counts),
            'total_pnl_mean': np.mean(total_pnls),
            'total_pnl_std': np.std(total_pnls),
            'avg_trade_pnl_mean': np.mean(trade_pnls) if trade_pnls else 0,
            'avg_trade_pnl_std': np.std(trade_pnls) if trade_pnls else 0,
            'profitable_episodes': len([r for r in returns if r > 0]),
            'positive_pnl_episodes': len([p for p in total_pnls if p > 0]),
            'profit_factor': (
                sum(p for p in total_pnls if p > 0) / abs(sum(p for p in total_pnls if p < 0))
                if sum(p for p in total_pnls if p < 0) != 0 else float('inf')
            ) if total_pnls else 1.0
        }
        
        # Quality checks
        profitability_stats['profitability_pass'] = (
            profitability_stats['total_return_mean'] > 0.01 or  # At least 1% return
            profitability_stats['profitable_episodes'] > len(returns) * 0.5  # At least 50% profitable
        )
        
        print(f"   Mean total return: {profitability_stats['total_return_mean']:.4f} ({profitability_stats['total_return_mean']*100:.2f}%)")
        print(f"   Win rate: {profitability_stats['win_rate_mean']:.4f} ({profitability_stats['win_rate_mean']*100:.2f}%)")
        print(f"   Avg trades per episode: {profitability_stats['avg_trades_per_episode']:.2f}")
        print(f"   Profitable episodes: {profitability_stats['profitable_episodes']}/{len(returns)}")
        print(f"   Profit factor: {profitability_stats['profit_factor']:.3f}")
        print(f"   Profitability pass: {profitability_stats['profitability_pass']}")
        
        self.evaluation_results['profitability'] = profitability_stats
        return profitability_stats
    
    def test_risk_metrics(self) -> Dict:
        """Test risk and drawdown metrics"""
        print("\n🛡️  Testing risk metrics...")
        
        env = TradingEnvironment(self.test_data, episode_length=200)
        env = Monitor(env)
        
        portfolio_histories = []
        max_drawdowns = []
        
        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            done = False
            portfolio_history = [env.unwrapped.initial_balance]
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Calculate current portfolio value
                current_portfolio = info.get('portfolio_value', env.unwrapped.balance + env.unwrapped.position * info.get('current_price', 50000))
                portfolio_history.append(current_portfolio)
                
                done = terminated or truncated
            
            portfolio_histories.append(portfolio_history)
            
            # Calculate max drawdown for this episode
            if len(portfolio_history) > 1:
                running_max = np.maximum.accumulate(portfolio_history)
                drawdown = (portfolio_history - running_max) / running_max
                max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            else:
                max_dd = 0
            
            max_drawdowns.append(max_dd)
        
        # Calculate overall risk metrics
        risk_stats = {
            'max_drawdown_mean': np.mean(max_drawdowns),
            'max_drawdown_std': np.std(max_drawdowns),
            'max_drawdown_max': np.max(max_drawdowns),
            'max_drawdown_min': np.min(max_drawdowns),
            'drawdown_exceeds_10_percent': len([dd for dd in max_drawdowns if dd > 0.1]),
            'volatility_mean': np.mean([np.std(history) / np.mean(history) if np.mean(history) > 0 else 0 
                                       for history in portfolio_histories]),
            'volatility_std': np.std([np.std(history) / np.mean(history) if np.mean(history) > 0 else 0 
                                     for history in portfolio_histories]),
            'var_95': np.percentile(max_drawdowns, 95) if max_drawdowns else 0,
            'var_99': np.percentile(max_drawdowns, 99) if max_drawdowns else 0,
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_histories),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_histories, max_drawdowns)
        }
        
        # Risk quality checks
        risk_stats['risk_acceptable'] = (
            risk_stats['max_drawdown_mean'] < 0.15 and # Less than 15% average drawdown
            risk_stats['max_drawdown_max'] < 0.25 and   # Less than 25% maximum drawdown
            risk_stats['drawdown_exceeds_10_percent'] < len(max_drawdowns) * 0.3  # Less than 30% exceed 10%
        )
        
        print(f"   Mean max drawdown: {risk_stats['max_drawdown_mean']:.4f} ({risk_stats['max_drawdown_mean']*100:.2f}%)")
        print(f"   Max drawdown: {risk_stats['max_drawdown_max']:.4f} ({risk_stats['max_drawdown_max']*100:.2f}%)")
        print(f"   Drawdowns >10%: {risk_stats['drawdown_exceeds_10_percent']}/{len(max_drawdowns)}")
        print(f"   Risk acceptable: {risk_stats['risk_acceptable']}")
        
        self.evaluation_results['risk'] = risk_stats
        return risk_stats
    
    def _calculate_sortino_ratio(self, portfolio_histories: List[List[float]]) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if not portfolio_histories:
            return 0.0
        
        all_returns = []
        for history in portfolio_histories:
            if len(history) > 1:
                returns = np.diff(history) / history[:-1]
                all_returns.extend(returns)
        
        if len(all_returns) == 0:
            return 0.0
        
        # Only consider negative returns for downside deviation
        negative_returns = [r for r in all_returns if r < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(all_returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        excess_return = np.mean(all_returns)  # Assuming risk-free rate = 0
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, portfolio_histories: List[List[float]], max_drawdowns: List[float]) -> float:
        """Calculate Calmar ratio (return over max drawdown)"""
        if not portfolio_histories or not max_drawdowns:
            return 0.0
        
        all_returns = []
        for history in portfolio_histories:
            if len(history) > 1:
                total_return = (history[-1] - history[0]) / history[0]
                all_returns.append(total_return)
        
        if not all_returns:
            return 0.0
        
        avg_return = np.mean(all_returns)
        avg_drawdown = np.mean(max_drawdowns) if max_drawdowns else 1.0
        
        return avg_return / avg_drawdown if avg_drawdown > 0 else 0.0
    
    def test_convergence_stability(self) -> Dict:
        """Test model convergence and stability"""
        print("\n📈 Testing convergence and stability...")
        
        # This would typically analyze training logs
        # For now, we'll simulate by running multiple evaluations and checking consistency
        
        test_results = []
        
        for run in range(10):  # Multiple runs to test consistency
            env = TradingEnvironment(self.test_data, episode_length=100)
            env = Monitor(env)
            
            episode_rewards = []
            for episode in range(5):
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                
                episode_rewards.append(total_reward)
            
            test_results.append({
                'run': run,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'consistency_score': 1 - (np.std(episode_rewards) / np.abs(np.mean(episode_rewards)) if np.mean(episode_rewards) != 0 else float('inf'))
            })
        
        # Calculate stability metrics
        mean_rewards = [r['mean_reward'] for r in test_results]
        consistency_scores = [r['consistency_score'] for r in test_results]
        
        convergence_stats = {
            'reward_consistency_mean': np.mean(consistency_scores),
            'reward_consistency_std': np.std(consistency_scores),
            'reward_stability': 1 - (np.std(mean_rewards) / np.abs(np.mean(mean_rewards)) if np.mean(mean_rewards) != 0 else float('inf')),
            'runs_consistent': len([cs for cs in consistency_scores if cs > 0.8]),  # Consistency > 80%
            'convergence_indicator': np.mean(consistency_scores) > 0.7,  # Good convergence
            'performance_variance': np.var(mean_rewards),
            'coefficient_of_variation': np.std(mean_rewards) / np.abs(np.mean(mean_rewards)) if np.mean(mean_rewards) != 0 else float('inf')
        }
        
        print(f"   Reward consistency: {convergence_stats['reward_consistency_mean']:.4f} ± {convergence_stats['reward_consistency_std']:.4f}")
        print(f"   Reward stability: {convergence_stats['reward_stability']:.4f}")
        print(f"   Consistent runs: {convergence_stats['runs_consistent']}/10")
        print(f"   Convergence indicator: {convergence_stats['convergence_indicator']}")
        
        self.evaluation_results['convergence'] = convergence_stats
        return convergence_stats
    
    def test_overfitting_detection(self) -> Dict:
        """Test for signs of overfitting"""
        print("\n🔍 Testing for overfitting...")
        
        # Split test data into different time periods to test temporal generalization
        n_samples = len(self.test_data)
        split_point = n_samples // 2
        
        # Test on different data splits
        data_splits = {
            'early': self.test_data.iloc[:split_point].reset_index(drop=True),
            'late': self.test_data.iloc[split_point:].reset_index(drop=True)
        }
        
        split_performance = {}
        
        for split_name, split_data in data_splits.items():
            env = TradingEnvironment(split_data, episode_length=100)
            env = Monitor(env)
            
            rewards = []
            returns = []
            
            for episode in range(min(10, self.n_episodes // 5)):  # Fewer episodes for speed
                obs, _ = env.reset()
                done = False
                total_reward = 0
                initial_balance = env.unwrapped.initial_balance
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    done = terminated or truncated
                
                final_portfolio = info.get('portfolio_value', initial_balance)
                total_return = (final_portfolio - initial_balance) / initial_balance if initial_balance > 0 else 0
                
                rewards.append(total_reward)
                returns.append(total_return)
            
            split_performance[split_name] = {
                'mean_reward': np.mean(rewards) if rewards else 0,
                'mean_return': np.mean(returns) if returns else 0,
                'std_reward': np.std(rewards) if rewards else 0,
                'std_return': np.std(returns) if returns else 0,
                'n_episodes': len(rewards)
            }
        
        # Calculate overfitting metrics
        early_perf = split_performance.get('early', {'mean_return': 0})
        late_perf = split_performance.get('late', {'mean_return': 0})
        
        overfitting_stats = {
            'early_performance': early_perf['mean_return'],
            'late_performance': late_perf['mean_return'],
            'performance_gap': abs(early_perf['mean_return'] - late_perf['mean_return']),
            'overfitting_significant': abs(early_perf['mean_return'] - late_perf['mean_return']) > 0.05,  # 5% gap
            'temporal_stability': 1 - abs(early_perf['mean_return'] - late_perf['mean_return']),
            'early_vs_late_ratio': early_perf['mean_return'] / late_perf['mean_return'] if late_perf['mean_return'] != 0 else float('inf'),
            'consistent_across_periods': abs(early_perf['mean_return'] - late_perf['mean_return']) < 0.1  # 10% tolerance
        }
        
        print(f"   Early period return: {overfitting_stats['early_performance']:.4f}")
        print(f"   Late period return: {overfitting_stats['late_performance']:.4f}")
        print(f"   Performance gap: {overfitting_stats['performance_gap']:.4f}")
        print(f"   Overfitting detected: {overfitting_stats['overfitting_significant']}")
        print(f"   Temporal stability: {overfitting_stats['temporal_stability']:.4f}")
        
        self.evaluation_results['overfitting'] = overfitting_stats
        return overfitting_stats
    
    def test_trade_quality_metrics(self) -> Dict:
        """Test trade quality and frequency metrics"""
        print("\n🎯 Testing trade quality metrics...")
        
        env = TradingEnvironment(self.test_data, episode_length=200)
        env = Monitor(env)
        
        all_trade_pnls = []
        all_trade_returns = []
        all_trade_frequencies = []
        all_win_rates = []
        
        for episode in range(self.n_episodes):
            obs, _ = env.reset()
            done = False
            trade_pnls = []
            trade_returns = []
            trade_count = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track trades
                if info.get('action_performed', False) and info.get('pnl_pct', 0) != 0:
                    pnl_pct = info.get('pnl_pct', 0)
                    trade_pnls.append(pnl_pct)
                    trade_returns.append(info.get('pnl_pct', 0))
                    trade_count += 1
                
                done = terminated or truncated
            
            if trade_pnls:
                all_trade_pnls.extend(trade_pnls)
                all_trade_returns.extend(trade_returns)
                all_trade_frequencies.append(trade_count)
                
                wins = len([pnl for pnl in trade_pnls if pnl > 0])
                win_rate = wins / len(trade_pnls) if len(trade_pnls) > 0 else 0
                all_win_rates.append(win_rate)
        
        trade_quality_stats = {
            'avg_trade_pnl_mean': np.mean(all_trade_pnls) if all_trade_pnls else 0,
            'avg_trade_pnl_std': np.std(all_trade_pnls) if all_trade_pnls else 0,
            'avg_trade_pnl_median': np.median(all_trade_pnls) if all_trade_pnls else 0,
            'best_trade_pnl': np.max(all_trade_pnls) if all_trade_pnls else 0,
            'worst_trade_pnl': np.min(all_trade_pnls) if all_trade_pnls else 0,
            'avg_trades_per_episode': np.mean(all_trade_frequencies) if all_trade_frequencies else 0,
            'trade_frequency_std': np.std(all_trade_frequencies) if all_trade_frequencies else 0,
            'overall_win_rate': np.mean(all_win_rates) if all_win_rates else 0,
            'win_rate_std': np.std(all_win_rates) if all_win_rates else 0,
            'profitable_trades': len([pnl for pnl in all_trade_pnls if pnl > 0]) if all_trade_pnls else 0,
            'unprofitable_trades': len([pnl for pnl in all_trade_pnls if pnl < 0]) if all_trade_pnls else 0,
            'trade_efficiency': (
                np.mean(all_trade_pnls) / np.std(all_trade_pnls) if all_trade_pnls and np.std(all_trade_pnls) > 0 else 0
            ),
            'quality_pass': (
                len(all_trade_pnls) > 0 and
                np.mean(all_trade_pnls) > 0 if all_trade_pnls else False
            )
        }
        
        print(f"   Avg trade PnL: {trade_quality_stats['avg_trade_pnl_mean']:.6f} ({trade_quality_stats['avg_trade_pnl_mean']*100:.4f}%)")
        print(f"   Avg trades per episode: {trade_quality_stats['avg_trades_per_episode']:.2f}")
        print(f"   Overall win rate: {trade_quality_stats['overall_win_rate']:.4f} ({trade_quality_stats['overall_win_rate']*100:.2f}%)")
        print(f"   Profitable trades: {trade_quality_stats['profitable_trades']}")
        print(f"   Trade efficiency: {trade_quality_stats['trade_efficiency']:.4f}")
        print(f"   Quality pass: {trade_quality_stats['quality_pass']}")
        
        self.evaluation_results['trade_quality'] = trade_quality_stats
        return trade_quality_stats
    
    def run_all_tests(self) -> Dict:
        """Run all evaluation tests"""
        print("=" * 60)
        print("🤖 COMPREHENSIVE MODEL EVALUATION TESTS")
        print("=" * 60)
        
        # Run all individual tests
        self.test_rewards_quality()
        self.test_profitability_metrics()
        self.test_risk_metrics()
        self.test_convergence_stability()
        self.test_overfitting_detection()
        self.test_trade_quality_metrics()
        
        # Generate summary
        self._generate_summary()
        
        return self.evaluation_results
    
    def _generate_summary(self):
        """Generate summary of all test results"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        # Overall pass/fail based on key metrics
        overall_pass = True
        issues = []
        
        # Check key metrics
        if self.evaluation_results.get('rewards', {}).get('quality_pass', False):
            print("✅ Reward quality: PASS")
        else:
            print("❌ Reward quality: FAIL")
            overall_pass = False
            issues.append("Poor reward quality")
        
        if self.evaluation_results.get('profitability', {}).get('profitability_pass', False):
            print("✅ Profitability: PASS")
        else:
            print("❌ Profitability: FAIL")
            overall_pass = False
            issues.append("Low profitability")
        
        if self.evaluation_results.get('risk', {}).get('risk_acceptable', False):
            print("✅ Risk control: PASS")
        else:
            print("❌ Risk control: FAIL")
            overall_pass = False
            issues.append("High risk exposure")
        
        if self.evaluation_results.get('convergence', {}).get('convergence_indicator', False):
            print("✅ Convergence: PASS")
        else:
            print("❌ Convergence: FAIL")
            overall_pass = False
            issues.append("Poor convergence")
        
        if not self.evaluation_results.get('overfitting', {}).get('overfitting_significant', True):
            print("✅ Overfitting: PASS")
        else:
            print("❌ Overfitting: FAIL")
            overall_pass = False
            issues.append("Signs of overfitting")
        
        if self.evaluation_results.get('trade_quality', {}).get('quality_pass', False):
            print("✅ Trade quality: PASS")
        else:
            print("❌ Trade quality: FAIL")
            overall_pass = False
            issues.append("Poor trade quality")
        
        print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
        
        if issues:
            print(f"\n⚠️  Issues detected: {', '.join(issues)}")
        
        # Key metrics summary
        print(f"\n📈 KEY METRICS:")
        if 'profitability' in self.evaluation_results:
            profit_stats = self.evaluation_results['profitability']
            print(f"   Total Return: {profit_stats.get('total_return_mean', 0):.4f} ({profit_stats.get('total_return_mean', 0)*100:.2f}%)")
            print(f"   Win Rate: {profit_stats.get('win_rate_mean', 0):.4f} ({profit_stats.get('win_rate_mean', 0)*100:.2f}%)")
            print(f"   Profit Factor: {profit_stats.get('profit_factor', 0):.3f}")
        
        if 'risk' in self.evaluation_results:
            risk_stats = self.evaluation_results['risk']
            print(f"   Avg Max Drawdown: {risk_stats.get('max_drawdown_mean', 0):.4f} ({risk_stats.get('max_drawdown_mean', 0)*100:.2f}%)")
            print(f"   Max Drawdown: {risk_stats.get('max_drawdown_max', 0):.4f} ({risk_stats.get('max_drawdown_max', 0)*100:.2f}%)")
        
        if 'rewards' in self.evaluation_results:
            reward_stats = self.evaluation_results['rewards']
            print(f"   Avg Reward: {reward_stats.get('mean_reward', 0):.4f}")
            print(f"   Reward Range: {reward_stats.get('reward_range', 0):.4f}")
    
    def save_results(self, output_path: str = "model_evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_path}")
    
    def plot_results(self, output_dir: str = "evaluation_plots"):
        """Generate plots for evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot reward distribution
        if 'rewards' in self.evaluation_results:
            plt.figure(figsize=(10, 6))
            rewards = self.evaluation_results['rewards']
            plt.hist([0] * rewards.get('zero_rewards', 0) + 
                    [1] * rewards.get('positive_rewards', 0) + 
                    [-1] * rewards.get('negative_rewards', 0), bins=3)
            plt.title('Reward Distribution')
            plt.xlabel('Reward Type')
            plt.ylabel('Count')
            plt.savefig(f'{output_dir}/reward_distribution.png')
            plt.close()
        
        # Plot performance metrics
        if 'profitability' in self.evaluation_results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            profit_stats = self.evaluation_results['profitability']
            axes[0, 0].bar(['Return', 'Sharpe', 'Win Rate'], 
                          [profit_stats.get('total_return_mean', 0), 
                           profit_stats.get('total_return_sharpe', 0),
                           profit_stats.get('win_rate_mean', 0)])
            axes[0, 0].set_title('Profitability Metrics')
            axes[0, 0].set_ylabel('Value')
            
            risk_stats = self.evaluation_results['risk']
            axes[0, 1].bar(['Avg DD', 'Max DD', 'Volatility'], 
                          [risk_stats.get('max_drawdown_mean', 0), 
                           risk_stats.get('max_drawdown_max', 0),
                           risk_stats.get('volatility_mean', 0)])
            axes[0, 1].set_title('Risk Metrics')
            axes[0, 1].set_ylabel('Value')
            
            conv_stats = self.evaluation_results['convergence']
            axes[1, 0].bar(['Consistency', 'Stability'], 
                          [conv_stats.get('reward_consistency_mean', 0), 
                           conv_stats.get('reward_stability', 0)])
            axes[1, 0].set_title('Convergence Metrics')
            axes[1, 0].set_ylabel('Value')
            
            trade_stats = self.evaluation_results['trade_quality']
            axes[1, 1].bar(['Avg Trade PnL', 'Trade Efficiency'], 
                          [trade_stats.get('avg_trade_pnl_mean', 0), 
                           trade_stats.get('trade_efficiency', 0)])
            axes[1, 1].set_title('Trade Quality Metrics')
            axes[1, 1].set_ylabel('Value')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/performance_summary.png')
            plt.close()
        
        print(f"📊 Plots saved to {output_dir}/")


def main():
    """Main function to run model evaluation tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation Tests")
    parser.add_argument("--model", default="ppo_trading_agent.zip", help="Path to trained model")
    parser.add_argument("--test-data", default="btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv", 
                       help="Path to test data")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument("--output-dir", default="model_evaluation", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluationTester(
            model_path=args.model,
            test_data_path=args.test_data,
            n_episodes=args.episodes
        )
        
        # Run all tests
        results = evaluator.run_all_tests()
        
        # Save results
        evaluator.save_results(os.path.join(args.output_dir, "evaluation_results.json"))
        evaluator.plot_results(os.path.join(args.output_dir, "plots"))
        
        print(f"\n🎉 Model evaluation completed! Results saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()