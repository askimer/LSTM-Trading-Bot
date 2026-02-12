#!/usr/bin/env python3
"""
Quick test to verify direction balancing fixes
Compares old vs new environment behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

# Import both environments for comparison
from enhanced_trading_environment import EnhancedTradingEnvironment
from enhanced_trading_environment_v2 import EnhancedTradingEnvironmentV2


def test_environment(env_class, model_path, df, n_episodes=10, env_name="Environment"):
    """Test an environment with a trained model"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {env_name}")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    results = {
        'long_trades': [],
        'short_trades': [],
        'total_returns': [],
        'win_rates': [],
        'long_pnls': [],
        'short_pnls': []
    }
    
    for episode in range(n_episodes):
        # Create environment
        env = env_class(df, episode_length=200, debug=False)
        env = Monitor(env)
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Collect results
        results['long_trades'].append(info.get('long_trades', 0))
        results['short_trades'].append(info.get('short_trades', 0))
        results['total_returns'].append(info.get('total_return', 0))
        results['win_rates'].append(info.get('win_rate', 0))
        results['long_pnls'].append(info.get('long_pnl', 0) if 'long_pnl' in info else 0)
        results['short_pnls'].append(info.get('short_pnl', 0) if 'short_pnl' in info else 0)
        
        print(f"   Episode {episode+1}: Long={info.get('long_trades', 0)}, "
              f"Short={info.get('short_trades', 0)}, Return={info.get('total_return', 0):.2%}")
    
    # Calculate statistics
    stats = {
        'avg_long_trades': np.mean(results['long_trades']),
        'avg_short_trades': np.mean(results['short_trades']),
        'avg_total_return': np.mean(results['total_returns']),
        'avg_win_rate': np.mean(results['win_rates']),
        'avg_long_pnl': np.mean(results['long_pnls']),
        'avg_short_pnl': np.mean(results['short_pnls'])
    }
    
    # Calculate balance metrics
    total_directional = stats['avg_long_trades'] + stats['avg_short_trades']
    if total_directional > 0:
        stats['long_ratio'] = stats['avg_long_trades'] / total_directional
        stats['short_ratio'] = stats['avg_short_trades'] / total_directional
        stats['balance_score'] = 1 - abs(stats['long_ratio'] - 0.5) * 2
    else:
        stats['long_ratio'] = 0
        stats['short_ratio'] = 0
        stats['balance_score'] = 0
    
    print(f"\n📊 {env_name} Statistics:")
    print(f"   Average Long Trades: {stats['avg_long_trades']:.1f} ({stats['long_ratio']:.1%})")
    print(f"   Average Short Trades: {stats['avg_short_trades']:.1f} ({stats['short_ratio']:.1%})")
    print(f"   Balance Score: {stats['balance_score']:.2f}/1.0")
    print(f"   Average Return: {stats['avg_total_return']:.2%}")
    print(f"   Average Win Rate: {stats['avg_win_rate']:.1%}")
    
    return stats


def compare_environments(model_path, df_path, n_episodes=10):
    """Compare old vs new environment"""
    print("="*70)
    print("🔍 ENVIRONMENT COMPARISON TEST")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Data: {df_path}")
    print(f"Episodes: {n_episodes}")
    
    # Load data
    df = pd.read_csv(df_path)
    
    # Test old environment
    old_stats = test_environment(
        EnhancedTradingEnvironment, 
        model_path, 
        df, 
        n_episodes, 
        "OLD Environment (v1)"
    )
    
    # Test new environment
    new_stats = test_environment(
        EnhancedTradingEnvironmentV2, 
        model_path, 
        df, 
        n_episodes, 
        "NEW Environment (v2 - Aggressive Balancing)"
    )
    
    # Compare
    print("\n" + "="*70)
    print("📈 COMPARISON SUMMARY")
    print("="*70)
    
    if old_stats and new_stats:
        print(f"\n{'Metric':<30} {'Old (v1)':<15} {'New (v2)':<15} {'Change':<15}")
        print("-"*70)
        
        metrics = [
            ('Long Trades', 'avg_long_trades', '.1f'),
            ('Short Trades', 'avg_short_trades', '.1f'),
            ('Long Ratio', 'long_ratio', '.1%'),
            ('Short Ratio', 'short_ratio', '.1%'),
            ('Balance Score', 'balance_score', '.2f'),
            ('Total Return', 'avg_total_return', '.2%'),
            ('Win Rate', 'avg_win_rate', '.1%')
        ]
        
        for name, key, fmt in metrics:
            old_val = old_stats[key]
            new_val = new_stats[key]
            change = new_val - old_val
            
            # Format values based on type
            if fmt == '.1%':
                old_str = f"{old_val:.1%}"
                new_str = f"{new_val:.1%}"
                change_str = f"{change:+.1%}"
            elif fmt == '.2%':
                old_str = f"{old_val:.2%}"
                new_str = f"{new_val:.2%}"
                change_str = f"{change:+.2%}"
            elif fmt == '.2f':
                old_str = f"{old_val:.2f}"
                new_str = f"{new_val:.2f}"
                change_str = f"{change:+.2f}"
            else:
                old_str = f"{old_val:.1f}"
                new_str = f"{new_val:.1f}"
                change_str = f"{change:+.1f}"
            
            print(f"{name:<30} {old_str:<15} {new_str:<15} {change_str:<15}")
        
        # Assessment
        print("\n" + "="*70)
        print("🎯 ASSESSMENT")
        print("="*70)
        
        old_balance = old_stats['balance_score']
        new_balance = new_stats['balance_score']
        
        if new_balance > old_balance:
            improvement = new_balance - old_balance
            print(f"✅ Balance improved by {improvement:.2f} points!")
            
            if new_balance >= 0.7:
                print("🎯 EXCELLENT: New environment achieves good balance!")
            elif new_balance >= 0.5:
                print("⚠️  FAIR: New environment shows improvement but needs more work")
            else:
                print("❌ Model may need retraining with new environment")
        else:
            print("⚠️  No improvement in balance (model may need retraining)")
        
        # Save comparison results
        comparison = {
            'old_environment': old_stats,
            'new_environment': new_stats,
            'improvement': {
                'balance_score_change': new_balance - old_balance,
                'long_ratio_change': new_stats['long_ratio'] - old_stats['long_ratio'],
                'short_ratio_change': new_stats['short_ratio'] - old_stats['short_ratio']
            }
        }
        
        with open('balance_comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print("\n💾 Comparison results saved to balance_comparison_results.json")


def plot_comparison(old_stats, new_stats, output_path='balance_comparison.png'):
    """Plot comparison between old and new environment"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Direction balance comparison
    ax = axes[0, 0]
    categories = ['Long Ratio', 'Short Ratio']
    old_values = [old_stats['long_ratio'], old_stats['short_ratio']]
    new_values = [new_stats['long_ratio'], new_stats['short_ratio']]
    target = [0.5, 0.5]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax.bar(x - width, old_values, width, label='Old (v1)', alpha=0.8, color='lightcoral')
    ax.bar(x, new_values, width, label='New (v2)', alpha=0.8, color='lightgreen')
    ax.bar(x + width, target, width, label='Target', alpha=0.8, color='gold')
    
    ax.set_ylabel('Ratio')
    ax.set_title('Direction Balance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. Balance score
    ax = axes[0, 1]
    scores = [old_stats['balance_score'], new_stats['balance_score']]
    colors = ['lightcoral' if s < 0.5 else 'lightgreen' for s in scores]
    bars = ax.bar(['Old (v1)', 'New (v2)'], scores, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Fair threshold')
    ax.axhline(y=0.7, color='green', linestyle='--', label='Good threshold')
    ax.set_ylabel('Balance Score')
    ax.set_title('Balance Score Comparison')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Trade counts
    ax = axes[1, 0]
    trade_data = {
        'Old Long': old_stats['avg_long_trades'],
        'Old Short': old_stats['avg_short_trades'],
        'New Long': new_stats['avg_long_trades'],
        'New Short': new_stats['avg_short_trades']
    }
    colors = ['lightcoral', 'lightcoral', 'lightgreen', 'lightgreen']
    bars = ax.bar(trade_data.keys(), trade_data.values(), color=colors, alpha=0.8)
    ax.set_ylabel('Average Trades')
    ax.set_title('Trade Count Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Performance metrics
    ax = axes[1, 1]
    metrics = ['Return', 'Win Rate']
    old_perf = [old_stats['avg_total_return'], old_stats['avg_win_rate']]
    new_perf = [new_stats['avg_total_return'], new_stats['avg_win_rate']]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, old_perf, width, label='Old (v1)', alpha=0.8, color='lightcoral')
    ax.bar(x + width/2, new_perf, width, label='New (v2)', alpha=0.8, color='lightgreen')
    
    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test and compare environment balancing fixes'
    )
    parser.add_argument(
        '--model',
        default='ppo_trading_agent.zip',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        default='btc_usdt_data/full_btc_usdt_data_feature_engineered.csv',
        help='Path to test data'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    compare_environments(args.model, args.data, args.episodes)
    
    # Load stats and plot if requested
    if args.plot:
        try:
            with open('balance_comparison_results.json', 'r') as f:
                comparison = json.load(f)
            plot_comparison(comparison['old_environment'], comparison['new_environment'])
        except Exception as e:
            print(f"⚠️  Could not generate plot: {e}")


if __name__ == "__main__":
    main()
