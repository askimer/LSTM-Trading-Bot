#!/usr/bin/env python3
"""
Comprehensive Analysis of Strategy Balancing in RL Trading Agent
Tests for balanced long/short positioning and trading behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import json
import os
import pickle
from datetime import datetime
import torch
import warnings
from typing import Dict
warnings.filterwarnings('ignore')

from enhanced_trading_environment import EnhancedTradingEnvironment

# Rename the TradingEnvironment to TradingEnvironment for backward compatibility
TradingEnvironment = EnhancedTradingEnvironment

class StrategyBalancingAnalyzer:
    """Analyzer for evaluating strategy balancing in RL trading models"""
    
    def __init__(self, model_path: str, test_data_path: str, n_episodes: int = 20):
        """
        Initialize the strategy analyzer
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data for analysis
            n_episodes: Number of episodes for analysis
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.n_episodes = n_episodes
        self.model = None
        self.test_data = None
        self.analysis_results = {}
        
        # Load model and test data
        self._load_model_and_data()
    
    def _load_model_and_data(self):
        """Load the trained model and test data"""
        print("🔍 Loading model and test data...")
        
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
    
    def analyze_strategy_balance(self) -> Dict:
        """Analyze the balance between long and short strategies"""
        print("\n⚖️  Analyzing strategy balance (long vs short positions)...")
        
        # Run multiple episodes to collect statistics
        all_action_counts = []
        all_position_durations = []
        all_long_profits = []
        all_short_profits = []
        all_position_changes = []
        
        for episode in range(self.n_episodes):
            print(f"   Episode {episode + 1}/{self.n_episodes}")
            
            # Create environment with different data slices for each episode
            start_idx = (episode * len(self.test_data)) // self.n_episodes
            end_idx = ((episode + 1) * len(self.test_data)) // self.n_episodes
            episode_data = self.test_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            env = TradingEnvironment(episode_data)
            env = Monitor(env)
            
            obs, _ = env.reset()
            done = False
            
            # Track episode statistics
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # HOLD, BUY_LONG, SELL_SHORT, BUY_SHORT
            position_durations = {'long': 0, 'short': 0, 'neutral': 0}
            long_profits = []
            short_profits = []
            position_changes = 0
            current_position_type = 'neutral'
            position_start_step = 0
            
            step_count = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Ensure action is valid
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())
                else:
                    action = int(action)
                
                action = max(0, min(action, 4))  # Clamp to valid range
                
                # Count actions
                action_counts[action] += 1
                
                # Track position changes and durations
                old_position_type = current_position_type
                if action in [1, 2]:  # Long actions
                    current_position_type = 'long'
                elif action in [3, 4]:  # Short actions
                    current_position_type = 'short'
                else:  # Hold
                    current_position_type = 'neutral'
                
                if current_position_type != old_position_type:
                    # Calculate duration of previous position
                    if old_position_type == 'long':
                        position_durations['long'] += step_count - position_start_step
                    elif old_position_type == 'short':
                        position_durations['short'] += step_count - position_start_step
                    else:
                        position_durations['neutral'] += step_count - position_start_step
                    
                    position_start_step = step_count
                    if current_position_type != 'neutral':
                        position_changes += 1
                
                # Track profits by position type
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
            
            # Account for final position duration
            if current_position_type == 'long':
                position_durations['long'] += step_count - position_start_step
            elif current_position_type == 'short':
                position_durations['short'] += step_count - position_start_step
            else:
                position_durations['neutral'] += step_count - position_start_step
            
            all_action_counts.append(action_counts)
            all_position_durations.append(position_durations)
            all_long_profits.append(long_profits)
            all_short_profits.append(short_profits)
            all_position_changes.append(position_changes)
        
        # Calculate aggregate statistics
        strategy_stats = {
            'action_distribution': {
                'hold': np.mean([counts[0] for counts in all_action_counts]),
                'buy_long': np.mean([counts[1] for counts in all_action_counts]),
                'sell_long': np.mean([counts[2] for counts in all_action_counts]),
                'sell_short': np.mean([counts[3] for counts in all_action_counts]),
                'buy_short': np.mean([counts[4] for counts in all_action_counts])
            },
            'position_duration_stats': {
                'avg_long_duration': np.mean([durs['long'] for durs in all_position_durations]),
                'avg_short_duration': np.mean([durs['short'] for durs in all_position_durations]),
                'avg_neutral_duration': np.mean([durs['neutral'] for durs in all_position_durations])
            },
            'position_balance_ratio': {
                'long_vs_short_actions': (
                    np.mean([counts[1] + counts[2] for counts in all_action_counts]) /
                    max(1, np.mean([counts[3] + counts[4] for counts in all_action_counts]))
                ),
                'long_vs_short_duration': (
                    np.mean([durs['long'] for durs in all_position_durations]) /
                    max(1, np.mean([durs['short'] for durs in all_position_durations]))
                )
            },
            'trading_activity': {
                'avg_position_changes': np.mean(all_position_changes),
                'avg_total_actions': np.mean([sum(counts.values()) for counts in all_action_counts]),
                'trading_frequency': np.mean([
                    sum(counts[i] for i in [1, 2, 3, 4]) / max(1, sum(counts.values())) 
                    for counts in all_action_counts
                ])
            },
            'balance_metrics': {
                'action_balance_score': self._calculate_action_balance_score(all_action_counts),
                'duration_balance_score': self._calculate_duration_balance_score(all_position_durations),
                'overall_balance_score': 0.0
            }
        }
        
        # Calculate overall balance score
        strategy_stats['balance_metrics']['overall_balance_score'] = (
            strategy_stats['balance_metrics']['action_balance_score'] * 0.6 +
            strategy_stats['balance_metrics']['duration_balance_score'] * 0.4
        )
        
        # Quality assessment
        strategy_stats['balance_quality'] = self._assess_balance_quality(strategy_stats)
        
        print(f"   Long/Short action ratio: {strategy_stats['position_balance_ratio']['long_vs_short_actions']:.2f}")
        print(f"   Long/Short duration ratio: {strategy_stats['position_balance_ratio']['long_vs_short_duration']:.2f}")
        print(f"   Trading frequency: {strategy_stats['trading_activity']['trading_frequency']:.2f}")
        print(f"   Balance score: {strategy_stats['balance_metrics']['overall_balance_score']:.3f}")
        print(f"   Balance quality: {strategy_stats['balance_quality']}")
        
        self.analysis_results['strategy_balance'] = strategy_stats
        return strategy_stats
    
    def _calculate_action_balance_score(self, action_counts_list):
        """Calculate balance score based on action distribution (0-1 scale, 1 = perfect balance)"""
        if not action_counts_list:
            return 0.0
        
        # Calculate long and short action counts
        long_actions = [counts[1] + counts[2] for counts in action_counts_list]  # BUY_LONG + SELL_LONG
        short_actions = [counts[3] + counts[4] for counts in action_counts_list]  # SELL_SHORT + BUY_SHORT
        
        # Calculate balance ratios
        balance_ratios = []
        for long_count, short_count in zip(long_actions, short_actions):
            if long_count + short_count == 0:
                continue
            
            long_ratio = long_count / (long_count + short_count)
            short_ratio = short_count / (long_count + short_count)
            
            # Perfect balance is 0.5 for each, so score is based on distance from 0.5
            balance_score = 1 - abs(long_ratio - 0.5) * 2  # Scale to 0-1
            balance_ratios.append(max(0, balance_score))
        
        return np.mean(balance_ratios) if balance_ratios else 0.0
    
    def _calculate_duration_balance_score(self, position_durations_list):
        """Calculate balance score based on position duration (0-1 scale, 1 = perfect balance)"""
        if not position_durations_list:
            return 0.0
        
        # Calculate long and short durations
        long_durations = [durs['long'] for durs in position_durations_list]
        short_durations = [durs['short'] for durs in position_durations_list]
        
        # Calculate balance ratios
        balance_ratios = []
        for long_dur, short_dur in zip(long_durations, short_durations):
            total_duration = long_dur + short_dur
            if total_duration == 0:
                continue
            
            long_ratio = long_dur / total_duration
            short_ratio = short_dur / total_duration
            
            # Perfect balance is 0.5 for each
            balance_score = 1 - abs(long_ratio - 0.5) * 2  # Scale to 0-1
            balance_ratios.append(max(0, balance_score))
        
        return np.mean(balance_ratios) if balance_ratios else 0.0
    
    def _assess_balance_quality(self, strategy_stats):
        """Assess the quality of strategy balance"""
        balance_score = strategy_stats['balance_metrics']['overall_balance_score']
        
        if balance_score >= 0.7:
            return "EXCELLENT - Well balanced long/short strategy"
        elif balance_score >= 0.5:
            return "GOOD - Reasonably balanced strategy"
        elif balance_score >= 0.3:
            return "FAIR - Somewhat imbalanced but functional"
        else:
            return "POOR - Heavily skewed toward one direction"
    
    def analyze_risk_adjusted_performance(self) -> Dict:
        """Analyze risk-adjusted performance metrics"""
        print("\n🛡️  Analyzing risk-adjusted performance...")
        
        # Run episodes to collect performance data
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        episode_volatilities = []
        
        for episode in range(self.n_episodes):
            print(f"   Episode {episode + 1}/{self.n_episodes}")
            
            # Create episode-specific data slice
            start_idx = (episode * len(self.test_data)) // self.n_episodes
            end_idx = ((episode + 1) * len(self.test_data)) // self.n_episodes
            episode_data = self.test_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            env = TradingEnvironment(episode_data)
            env = Monitor(env)
            
            obs, _ = env.reset()
            done = False
            portfolio_values = [env.unwrapped.initial_balance]
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())
                else:
                    action = int(action)
                
                action = max(0, min(action, 4))
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate current portfolio value
                current_price = episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('close', 
                               episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('Close'))
                portfolio_value = (env.unwrapped.balance + 
                                 env.unwrapped.margin_locked + 
                                 env.unwrapped.position * current_price)
                portfolio_values.append(portfolio_value)
            
            # Calculate episode metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            if len(returns) > 1:
                # Total return
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                
                # Sharpe ratio (annualized)
                if np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
                # Maximum drawdown
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
                
                # Volatility (annualized)
                volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
                
                episode_returns.append(total_return)
                episode_sharpe_ratios.append(sharpe_ratio)
                episode_max_drawdowns.append(max_drawdown)
                episode_volatilities.append(volatility)
        
        # Calculate aggregate metrics
        risk_performance_stats = {
            'return_metrics': {
                'avg_total_return': np.mean(episode_returns) if episode_returns else 0,
                'std_total_return': np.std(episode_returns) if episode_returns else 0,
                'median_total_return': np.median(episode_returns) if episode_returns else 0,
                'best_return': np.max(episode_returns) if episode_returns else 0,
                'worst_return': np.min(episode_returns) if episode_returns else 0,
                'return_range': (np.max(episode_returns) - np.min(episode_returns)) if episode_returns else 0,
                'profitable_episodes': len([r for r in episode_returns if r > 0]) if episode_returns else 0
            },
            'risk_metrics': {
                'avg_sharpe_ratio': np.mean(episode_sharpe_ratios) if episode_sharpe_ratios else 0,
                'std_sharpe_ratio': np.std(episode_sharpe_ratios) if episode_sharpe_ratios else 0,
                'avg_max_drawdown': np.mean(episode_max_drawdowns) if episode_max_drawdowns else 0,
                'max_drawdown': np.max(episode_max_drawdowns) if episode_max_drawdowns else 0,
                'avg_volatility': np.mean(episode_volatilities) if episode_volatilities else 0,
                'var_95': np.percentile(episode_max_drawdowns, 95) if episode_max_drawdowns else 0,
                'var_99': np.percentile(episode_max_drawdowns, 99) if episode_max_drawdowns else 0
            },
            'performance_quality': self._assess_performance_quality(
                episode_returns, episode_sharpe_ratios, episode_max_drawdowns
            )
        }
        
        print(f"   Avg total return: {risk_performance_stats['return_metrics']['avg_total_return']:.4f} ({risk_performance_stats['return_metrics']['avg_total_return']*100:.2f}%)")
        print(f"   Avg Sharpe ratio: {risk_performance_stats['risk_metrics']['avg_sharpe_ratio']:.3f}")
        print(f"   Avg max drawdown: {risk_performance_stats['risk_metrics']['avg_max_drawdown']:.4f} ({risk_performance_stats['risk_metrics']['avg_max_drawdown']*100:.2f}%)")
        print(f"   Avg volatility: {risk_performance_stats['risk_metrics']['avg_volatility']:.4f}")
        print(f"   Profitable episodes: {risk_performance_stats['return_metrics']['profitable_episodes']}/{len(episode_returns)}")
        print(f"   Performance quality: {risk_performance_stats['performance_quality']}")
        
        self.analysis_results['risk_performance'] = risk_performance_stats
        return risk_performance_stats
    
    def _assess_performance_quality(self, returns, sharpe_ratios, drawdowns):
        """Assess the quality of risk-adjusted performance"""
        avg_return = np.mean(returns) if returns else 0
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        
        # Performance quality assessment
        if avg_return > 0.05 and avg_sharpe > 1.0 and avg_drawdown < 0.15:
            return "EXCELLENT - High return, good risk-adjusted performance"
        elif avg_return > 0.02 and avg_sharpe > 0.5:
            return "GOOD - Positive return with reasonable risk-adjusted performance"
        elif avg_return > 0:
            return "FAIR - Generating profit but with high risk or low risk-adjusted return"
        else:
            return "POOR - Negative return or very poor risk-adjusted performance"
    
    def analyze_trade_quality(self) -> Dict:
        """Analyze the quality of individual trades"""
        print("\n🎯 Analyzing trade quality metrics...")
        
        all_trade_data = []
        
        for episode in range(self.n_episodes):
            print(f"   Episode {episode + 1}/{self.n_episodes}")
            
            # Create episode-specific data slice
            start_idx = (episode * len(self.test_data)) // self.n_episodes
            end_idx = ((episode + 1) * len(self.test_data)) // self.n_episodes
            episode_data = self.test_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            env = TradingEnvironment(episode_data)
            env = Monitor(env)
            
            obs, _ = env.reset()
            done = False
            episode_trades = []
            position_entry_info = None
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())
                else:
                    action = int(action)
                
                action = max(0, min(action, 4))
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Track trade entries and exits
                current_price = episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('close', 
                               episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('Close'))
                
                # Simple trade tracking: record when position changes
                if action in [1, 3]:  # Entry actions (BUY_LONG or SELL_SHORT)
                    position_entry_info = {
                        'step': env.unwrapped.current_step,
                        'action': action,
                        'entry_price': current_price,
                        'position_size': abs(env.unwrapped.position),
                        'balance_before': env.unwrapped.balance
                    }
                elif action in [2, 4] and position_entry_info:  # Exit actions (SELL_LONG or BUY_SHORT)
                    exit_info = {
                        'step': env.unwrapped.current_step,
                        'exit_action': action,
                        'exit_price': current_price,
                        'entry_info': position_entry_info.copy()
                    }
                    
                    # Calculate PnL for this trade
                    if position_entry_info['action'] == 1:  # Long trade
                        pnl_pct = (current_price - position_entry_info['entry_price']) / position_entry_info['entry_price']
                    elif position_entry_info['action'] == 3:  # Short trade
                        pnl_pct = (position_entry_info['entry_price'] - current_price) / position_entry_info['entry_price']
                    else:
                        pnl_pct = 0
                    
                    exit_info['pnl_pct'] = pnl_pct
                    exit_info['direction'] = 'long' if position_entry_info['action'] == 1 else 'short'
                    
                    episode_trades.append(exit_info)
                    position_entry_info = None
            
            all_trade_data.extend(episode_trades)
        
        # Calculate trade quality metrics
        if all_trade_data:
            trade_metrics = {
                'total_trades': len(all_trade_data),
                'successful_trades': len([t for t in all_trade_data if t['pnl_pct'] > 0]),
                'unsuccessful_trades': len([t for t in all_trade_data if t['pnl_pct'] <= 0]),
                'avg_pnl_per_trade': np.mean([t['pnl_pct'] for t in all_trade_data]),
                'std_pnl_per_trade': np.std([t['pnl_pct'] for t in all_trade_data]),
                'best_trade': max([t['pnl_pct'] for t in all_trade_data]) if all_trade_data else 0,
                'worst_trade': min([t['pnl_pct'] for t in all_trade_data]) if all_trade_data else 0,
                'win_rate': len([t for t in all_trade_data if t['pnl_pct'] > 0]) / len(all_trade_data) if all_trade_data else 0,
                'profit_factor': self._calculate_profit_factor(all_trade_data),
                'trade_duration_stats': {
                    'avg_duration': np.mean([t['step'] - t['entry_info']['step'] for t in all_trade_data]) if all_trade_data else 0,
                    'min_duration': min([t['step'] - t['entry_info']['step'] for t in all_trade_data]) if all_trade_data else 0,
                    'max_duration': max([t['step'] - t['entry_info']['step'] for t in all_trade_data]) if all_trade_data else 0
                },
                'direction_analysis': {
                    'long_trades': len([t for t in all_trade_data if t['direction'] == 'long']),
                    'short_trades': len([t for t in all_trade_data if t['direction'] == 'short']),
                    'long_win_rate': len([t for t in all_trade_data if t['direction'] == 'long' and t['pnl_pct'] > 0]) / 
                                    max(1, len([t for t in all_trade_data if t['direction'] == 'long'])) if all_trade_data else 0,
                    'short_win_rate': len([t for t in all_trade_data if t['direction'] == 'short' and t['pnl_pct'] > 0]) / 
                                      max(1, len([t for t in all_trade_data if t['direction'] == 'short'])) if all_trade_data else 0,
                    'long_avg_pnl': np.mean([t['pnl_pct'] for t in all_trade_data if t['direction'] == 'long']) if [t for t in all_trade_data if t['direction'] == 'long'] else 0,
                    'short_avg_pnl': np.mean([t['pnl_pct'] for t in all_trade_data if t['direction'] == 'short']) if [t for t in all_trade_data if t['direction'] == 'short'] else 0
                }
            }
        else:
            trade_metrics = {
                'total_trades': 0,
                'successful_trades': 0,
                'unsuccessful_trades': 0,
                'avg_pnl_per_trade': 0,
                'std_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trade_duration_stats': {'avg_duration': 0, 'min_duration': 0, 'max_duration': 0},
                'direction_analysis': {
                    'long_trades': 0, 'short_trades': 0, 'long_win_rate': 0, 'short_win_rate': 0,
                    'long_avg_pnl': 0, 'short_avg_pnl': 0
                }
            }
        
        print(f"   Total trades: {trade_metrics['total_trades']}")
        print(f"   Win rate: {trade_metrics['win_rate']:.3f} ({trade_metrics['win_rate']*100:.1f}%)")
        print(f"   Avg PnL per trade: {trade_metrics['avg_pnl_per_trade']:.6f} ({trade_metrics['avg_pnl_per_trade']*100:.4f}%)")
        print(f"   Profit factor: {trade_metrics['profit_factor']:.3f}")
        print(f"   Long trades: {trade_metrics['direction_analysis']['long_trades']}, Short trades: {trade_metrics['direction_analysis']['short_trades']}")
        print(f"   Long win rate: {trade_metrics['direction_analysis']['long_win_rate']:.3f}, Short win rate: {trade_metrics['direction_analysis']['short_win_rate']:.3f}")
        
        self.analysis_results['trade_quality'] = trade_metrics
        return trade_metrics
    
    def _calculate_profit_factor(self, trade_data):
        """Calculate profit factor (gross profit / gross loss)"""
        if not trade_data:
            return 0
        
        gross_profit = sum(t['pnl_pct'] for t in trade_data if t['pnl_pct'] > 0)
        gross_loss = abs(sum(t['pnl_pct'] for t in trade_data if t['pnl_pct'] < 0))
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        else:
            return float('inf') if gross_profit > 0 else 1.0
    
    def analyze_convergence_stability(self) -> Dict:
        """Analyze model convergence and stability"""
        print("\n📈 Analyzing convergence and stability...")
        
        # Since we don't have training logs, we'll analyze consistency across episodes
        episode_rewards = []
        episode_returns = []
        
        for episode in range(self.n_episodes):
            print(f"   Episode {episode + 1}/{self.n_episodes}")
            
            # Create episode-specific data slice
            start_idx = (episode * len(self.test_data)) // self.n_episodes
            end_idx = ((episode + 1) * len(self.test_data)) // self.n_episodes
            episode_data = self.test_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            env = TradingEnvironment(episode_data)
            env = Monitor(env)
            
            obs, _ = env.reset()
            done = False
            total_reward = 0
            portfolio_values = [env.unwrapped.initial_balance]
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(action, (np.ndarray, torch.Tensor)):
                    action = int(action.item())
                else:
                    action = int(action)
                
                action = max(0, min(action, 4))
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                # Track portfolio for return calculation
                current_price = episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('close', 
                               episode_data.iloc[min(env.unwrapped.current_step, len(episode_data)-1)].get('Close'))
                portfolio_value = (env.unwrapped.balance + 
                                 env.unwrapped.margin_locked + 
                                 env.unwrapped.position * current_price)
                portfolio_values.append(portfolio_value)
            
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            episode_rewards.append(total_reward)
            episode_returns.append(total_return)
        
        # Calculate stability metrics
        stability_metrics = {
            'reward_stability': {
                'avg_total_reward': np.mean(episode_rewards) if episode_rewards else 0,
                'std_total_reward': np.std(episode_rewards) if episode_rewards else 0,
                'cv_reward': np.std(episode_rewards) / np.abs(np.mean(episode_rewards)) if episode_rewards and np.mean(episode_rewards) != 0 else float('inf'),
                'min_reward': np.min(episode_rewards) if episode_rewards else 0,
                'max_reward': np.max(episode_rewards) if episode_rewards else 0
            },
            'return_stability': {
                'avg_total_return': np.mean(episode_returns) if episode_returns else 0,
                'std_total_return': np.std(episode_returns) if episode_returns else 0,
                'cv_return': np.std(episode_returns) / np.abs(np.mean(episode_returns)) if episode_returns and np.mean(episode_returns) != 0 else float('inf'),
                'min_return': np.min(episode_returns) if episode_returns else 0,
                'max_return': np.max(episode_returns) if episode_returns else 0
            },
            'consistency_score': self._calculate_consistency_score(episode_returns),
            'stability_assessment': self._assess_stability(episode_rewards, episode_returns)
        }
        
        print(f"   Avg total reward: {stability_metrics['reward_stability']['avg_total_reward']:.4f}")
        print(f"   Reward CV: {stability_metrics['reward_stability']['cv_reward']:.3f}")
        print(f"   Avg total return: {stability_metrics['return_stability']['avg_total_return']:.4f}")
        print(f"   Return CV: {stability_metrics['return_stability']['cv_return']:.3f}")
        print(f"   Consistency score: {stability_metrics['consistency_score']:.3f}")
        print(f"   Stability assessment: {stability_metrics['stability_assessment']}")
        
        self.analysis_results['convergence_stability'] = stability_metrics
        return stability_metrics
    
    def _calculate_consistency_score(self, returns):
        """Calculate consistency score based on return variance (0-1 scale, 1 = perfect consistency)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        # Lower coefficient of variation means higher consistency
        cv = np.std(returns) / np.abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
        
        # Map CV to 0-1 scale (perfect consistency = 0 CV = 1 score)
        # Use sigmoid-like function to map CV to score
        consistency_score = 1 / (1 + cv) if cv < float('inf') else 0.0
        return min(1.0, consistency_score)  # Cap at 1.0
    
    def _assess_stability(self, rewards, returns):
        """Assess model stability"""
        avg_return = np.mean(returns) if returns else 0
        cv_return = np.std(returns) / np.abs(np.mean(returns)) if returns and np.mean(returns) != 0 else float('inf')
        
        if avg_return > 0.02 and cv_return < 0.5:
            return "STABLE - Good performance with low variance"
        elif avg_return > 0 and cv_return < 1.0:
            return "MODERATE - Positive return with acceptable variance"
        elif avg_return > 0:
            return "UNSTABLE - Positive return but high variance"
        else:
            return "UNSTABLE - Negative return or very high variance"
    
    def run_complete_analysis(self) -> Dict:
        """Run complete analysis of model performance"""
        print("=" * 70)
        print("🤖 COMPREHENSIVE MODEL STRATEGY BALANCING ANALYSIS")
        print("=" * 70)
        
        # Run all analysis modules
        self.analyze_strategy_balance()
        self.analyze_risk_adjusted_performance()
        self.analyze_trade_quality()
        self.analyze_convergence_stability()
        
        # Generate comprehensive summary
        self._generate_comprehensive_summary()
        
        return self.analysis_results
    
    def _generate_comprehensive_summary(self):
        """Generate comprehensive summary of all analysis results"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Strategy balance summary
        if 'strategy_balance' in self.analysis_results:
            sb = self.analysis_results['strategy_balance']
            print(f"\n⚖️  STRATEGY BALANCE:")
            print(f"   Long/Short action ratio: {sb['position_balance_ratio']['long_vs_short_actions']:.2f}")
            print(f"   Long/Short duration ratio: {sb['position_balance_ratio']['long_vs_short_duration']:.2f}")
            print(f"   Balance score: {sb['balance_metrics']['overall_balance_score']:.3f}")
            print(f"   Balance quality: {sb['balance_quality']}")
        
        # Risk-adjusted performance summary
        if 'risk_performance' in self.analysis_results:
            rp = self.analysis_results['risk_performance']
            print(f"\n🛡️  RISK-ADJUSTED PERFORMANCE:")
            print(f"   Avg return: {rp['return_metrics']['avg_total_return']:.4f} ({rp['return_metrics']['avg_total_return']*100:.2f}%)")
            print(f"   Avg Sharpe: {rp['risk_metrics']['avg_sharpe_ratio']:.3f}")
            print(f"   Avg drawdown: {rp['risk_metrics']['avg_max_drawdown']:.4f} ({rp['risk_metrics']['avg_max_drawdown']*100:.2f}%)")
            print(f"   Performance quality: {rp['performance_quality']}")
        
        # Trade quality summary
        if 'trade_quality' in self.analysis_results:
            tq = self.analysis_results['trade_quality']
            print(f"\n🎯 TRADE QUALITY:")
            print(f"   Total trades: {tq['total_trades']}")
            print(f"   Win rate: {tq['win_rate']:.3f} ({tq['win_rate']*100:.1f}%)")
            print(f"   Avg PnL per trade: {tq['avg_pnl_per_trade']:.6f} ({tq['avg_pnl_per_trade']*100:.4f}%)")
            print(f"   Profit factor: {tq['profit_factor']:.3f}")
            print(f"   Long trades: {tq['direction_analysis']['long_trades']}, Short trades: {tq['direction_analysis']['short_trades']}")
        
        # Stability summary
        if 'convergence_stability' in self.analysis_results:
            cs = self.analysis_results['convergence_stability']
            print(f"\n📈 STABILITY:")
            print(f"   Avg return: {cs['return_stability']['avg_total_return']:.4f}")
            print(f"   Return CV: {cs['return_stability']['cv_return']:.3f}")
            print(f"   Consistency: {cs['consistency_score']:.3f}")
            print(f"   Assessment: {cs['stability_assessment']}")
        
        # Overall assessment
        overall_assessment = self._generate_overall_assessment()
        print(f"\n🎯 OVERALL ASSESSMENT: {overall_assessment}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        if 'strategy_balance' in self.analysis_results:
            sb = self.analysis_results['strategy_balance']
            if sb['balance_metrics']['overall_balance_score'] < 0.3:
                print(f"   • ⚠️  Strategy heavily skewed toward one direction")
            elif sb['balance_metrics']['overall_balance_score'] < 0.5:
                print(f"   • ⚠️  Strategy somewhat imbalanced")
            else:
                print(f"   • ✅ Strategy shows good long/short balance")
        
        if 'risk_performance' in self.analysis_results:
            rp = self.analysis_results['risk_performance']
            if rp['return_metrics']['avg_total_return'] < 0:
                print(f"   • ❌ Model generates losses on average")
            elif rp['return_metrics']['avg_total_return'] < 0.01:
                print(f"   • ⚠️  Very low profitability")
            else:
                print(f"   • ✅ Model shows positive profitability")
        
        if 'trade_quality' in self.analysis_results:
            tq = self.analysis_results['trade_quality']
            if tq['win_rate'] < 0.4:
                print(f"   • ⚠️  Low win rate (<40%)")
            elif tq['profit_factor'] < 1.2:
                print(f"   • ⚠️  Low profit factor (<1.2)")
            else:
                print(f"   • ✅ Good trade quality metrics")
        
        print(f"\n📋 NEXT STEPS:")
        if 'strategy_balance' in self.analysis_results and sb['balance_metrics']['overall_balance_score'] < 0.5:
            print(f"   • Investigate strategy bias toward one direction")
            print(f"   • Consider rebalancing reward function")
            print(f"   • Review environment for directional bias")
        
        if 'risk_performance' in self.analysis_results and rp['return_metrics']['avg_total_return'] < 0.02:
            print(f"   • Optimize model hyperparameters")
            print(f"   • Review reward structure")
            print(f"   • Consider additional training")
        
        if 'trade_quality' in self.analysis_results and tq['win_rate'] < 0.5:
            print(f"   • Improve entry/exit conditions")
            print(f"   • Review risk management parameters")
            print(f"   • Consider position sizing optimization")
    
    def _generate_overall_assessment(self):
        """Generate overall assessment of model quality"""
        if 'strategy_balance' not in self.analysis_results:
            return "INSUFFICIENT DATA - Analysis incomplete"
        
        sb = self.analysis_results['strategy_balance']
        rp = self.analysis_results.get('risk_performance', {})
        tq = self.analysis_results.get('trade_quality', {})
        
        # Calculate composite score
        balance_score = sb['balance_metrics']['overall_balance_score']
        avg_return = rp.get('return_metrics', {}).get('avg_total_return', 0)
        win_rate = tq.get('win_rate', 0)
        profit_factor = tq.get('profit_factor', 0)
        
        score = 0
        if balance_score >= 0.5: score += 1
        if avg_return > 0.02: score += 1
        if win_rate > 0.5: score += 1
        if profit_factor > 1.5: score += 1
        
        if score >= 3:
            return "✅ MODEL READY - Good balance and performance"
        elif score >= 2:
            return "⚠️  NEEDS OPTIMIZATION - Decent but improvable"
        else:
            return "❌ NEEDS RETRAINING - Significant issues detected"
    
    def save_analysis_results(self, output_path="strategy_balancing_analysis.json"):
        """Save analysis results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"\n💾 Analysis results saved to {output_path}")
    
    def plot_analysis_results(self, output_dir="analysis_plots"):
        """Generate plots for analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot strategy balance
        if 'strategy_balance' in self.analysis_results:
            sb = self.analysis_results['strategy_balance']
            
            # Action distribution
            actions = ['HOLD', 'BUY_LONG', 'SELL_LONG', 'SELL_SHORT', 'BUY_SHORT']
            action_values = [sb['action_distribution'][key] for key in ['hold', 'buy_long', 'sell_long', 'sell_short', 'buy_short']]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(actions, action_values)
            plt.title('Action Distribution Analysis')
            plt.ylabel('Average Actions per Episode')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, action_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/action_distribution.png')
            plt.close()
        
        # Plot risk-adjusted performance
        if 'risk_performance' in self.analysis_results:
            rp = self.analysis_results['risk_performance']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Return distribution
            returns = rp.get('return_metrics', {})
            metrics = ['avg_total_return', 'best_return', 'worst_return']
            values = [returns.get(m, 0) for m in metrics]
            
            ax1.bar(metrics, values)
            ax1.set_title('Return Metrics')
            ax1.set_ylabel('Return')
            ax1.tick_params(axis='x', rotation=45)
            
            # Risk metrics
            risks = rp.get('risk_metrics', {})
            risk_metrics = ['avg_sharpe_ratio', 'avg_max_drawdown', 'avg_volatility']
            risk_values = [risks.get(m, 0) for m in risk_metrics]
            
            ax2.bar(risk_metrics, risk_values)
            ax2.set_title('Risk Metrics')
            ax2.set_ylabel('Value')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/risk_performance.png')
            plt.close()
        
        print(f"📊 Analysis plots saved to {output_dir}/")


def main():
    """Main function to run strategy balancing analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Strategy Balancing in RL Trading Model")
    parser.add_argument("--model", default="ppo_trading_agent.zip", 
                       help="Path to trained model")
    parser.add_argument("--test-data", default="btc_usdt_data/full_btc_usdt_data_feature_engineered.csv",
                       help="Path to test data")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes for analysis")
    parser.add_argument("--output-dir", default="strategy_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = StrategyBalancingAnalyzer(
            model_path=args.model,
            test_data_path=args.test_data,
            n_episodes=args.episodes
        )
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Save results
        analyzer.save_analysis_results(os.path.join(args.output_dir, "strategy_analysis_results.json"))
        analyzer.plot_analysis_results(os.path.join(args.output_dir, "plots"))
        
        print(f"\n🎉 Strategy balancing analysis completed!")
        print(f"Results saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()