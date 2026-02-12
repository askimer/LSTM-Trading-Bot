#!/usr/bin/env python3
"""
Retrain PPO model with EnhancedTradingEnvironmentV2 (Aggressive Balancing)
Fixes strategy imbalance issues by enforcing minimum trades in both directions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback, 
    StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import our enhanced environment
from enhanced_trading_environment_v2 import EnhancedTradingEnvironmentV2


def load_data(data_path):
    """Load and prepare training data"""
    print(f"📊 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_cols = ['close', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'ATR_15', 'OBV', 'AD', 'MFI_15']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Try alternative naming (capitalized)
        alt_cols = ['Close', 'RSI_15', 'BB_15_upper', 'BB_15_lower', 'ATR_15', 'OBV', 'AD', 'MFI_15']
        for col, alt in zip(required_cols, alt_cols):
            if col in missing_cols and alt in df.columns:
                df[col] = df[alt]
                missing_cols.remove(col)
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Loaded {len(df)} rows of data")
    return df


def make_env(df, rank=0, episode_length=200):
    """Create environment with proper monitoring"""
    def _init():
        # Use a different random seed for each environment
        env = EnhancedTradingEnvironmentV2(
            df=df,
            initial_balance=10000,
            transaction_fee=0.0018,
            episode_length=episode_length,
            debug=False,
            enable_strategy_balancing=True,
            min_long_ratio=0.30,  # Require at least 30% long trades
            min_short_ratio=0.30  # Require at least 30% short trades
        )
        env = Monitor(env)
        return env
    return _init


class DirectionBalanceCallback(BaseCallback):
    """Callback to monitor and log direction balance metrics"""
    def __init__(self, verbose=1):
        super(DirectionBalanceCallback, self).__init__(verbose)
        self.direction_ratios = []
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called during training"""
        # Access info from the environment
        info = self.locals.get('infos', [{}])[0]
        
        if info and 'long_trades' in info and 'short_trades' in info:
            long_trades = info['long_trades']
            short_trades = info['short_trades']
            total = long_trades + short_trades
            
            if total > 0:
                long_ratio = long_trades / total
                short_ratio = short_trades / total
                self.direction_ratios.append((long_ratio, short_ratio))
                
                if self.verbose > 0 and len(self.direction_ratios) % 100 == 0:
                    avg_long = np.mean([r[0] for r in self.direction_ratios[-100:]])
                    avg_short = np.mean([r[1] for r in self.direction_ratios[-100:]])
                    print(f"\n📊 Direction Balance (last 100 episodes):")
                    print(f"   Long: {avg_long:.1%}, Short: {avg_short:.1%}")
        
        return True


def create_model(env, tensorboard_log=None, hyperparams=None):
    """Create PPO model with optimized hyperparameters for balanced trading"""
    
    # Default hyperparameters tuned for balanced long/short trading
    # NOTE: use_sde=False because we have discrete action space (5 actions)
    default_params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.01,  # Entropy coefficient for exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,  # Disabled - gSDE only works with continuous actions
        'policy_kwargs': dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU,
        )
    }
    
    # Merge with provided hyperparameters
    if hyperparams:
        default_params.update(hyperparams)
    
    print("\n🔧 Model hyperparameters:")
    for key, value in default_params.items():
        if key != 'policy_kwargs':
            print(f"   {key}: {value}")
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **default_params
    )
    
    return model


def train_model(model, total_timesteps, callback=None, save_path='models'):
    """Train the model with callbacks"""
    
    # Setup checkpoint callback
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="ppo_balanced_model"
    )
    
    callbacks = [checkpoint_callback]
    if callback:
        callbacks.append(callback)
    
    callback_list = CallbackList(callbacks)
    
    print(f"\n🚀 Starting training for {total_timesteps:,} timesteps...")
    print(f"💾 Checkpoints will be saved to {save_path}/")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        print("\n✅ Training completed successfully!")
        return True
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate model and check direction balance"""
    print(f"\n📊 Evaluating model on {n_eval_episodes} episodes...")
    
    episode_returns = []
    episode_long_trades = []
    episode_short_trades = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_returns.append(total_reward)
        episode_long_trades.append(info.get('long_trades', 0))
        episode_short_trades.append(info.get('short_trades', 0))
        
        print(f"   Episode {episode+1}: Return={total_reward:.2f}, "
              f"Long={info.get('long_trades', 0)}, Short={info.get('short_trades', 0)}")
    
    # Calculate statistics
    avg_return = np.mean(episode_returns)
    avg_long = np.mean(episode_long_trades)
    avg_short = np.mean(episode_short_trades)
    total_direction_trades = avg_long + avg_short
    
    if total_direction_trades > 0:
        long_ratio = avg_long / total_direction_trades
        short_ratio = avg_short / total_direction_trades
    else:
        long_ratio = short_ratio = 0
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Average Return: {avg_return:.2f}")
    print(f"   Average Long Trades: {avg_long:.1f} ({long_ratio:.1%})")
    print(f"   Average Short Trades: {avg_short:.1f} ({short_ratio:.1%})")
    
    balance_score = 1 - abs(long_ratio - 0.5) * 2
    print(f"   Balance Score: {balance_score:.2f} (1.0 = perfect balance)")
    
    return {
        'avg_return': avg_return,
        'avg_long_trades': avg_long,
        'avg_short_trades': avg_short,
        'long_ratio': long_ratio,
        'short_ratio': short_ratio,
        'balance_score': balance_score
    }


def plot_training_results(log_dir, output_path='training_results.png'):
    """Plot training results from tensorboard logs"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Get available tags
        tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        if 'rollout/ep_rew_mean' in tags:
            rewards = ea.Scalars('rollout/ep_rew_mean')
            steps = [x.step for x in rewards]
            values = [x.value for x in rewards]
            axes[0, 0].plot(steps, values)
            axes[0, 0].set_title('Mean Episode Reward')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Reward')
        
        # Plot episode lengths
        if 'rollout/ep_len_mean' in tags:
            lengths = ea.Scalars('rollout/ep_len_mean')
            steps = [x.step for x in lengths]
            values = [x.value for x in lengths]
            axes[0, 1].plot(steps, values)
            axes[0, 1].set_title('Mean Episode Length')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Length')
        
        # Plot loss
        if 'train/loss' in tags:
            losses = ea.Scalars('train/loss')
            steps = [x.step for x in losses]
            values = [x.value for x in losses]
            axes[1, 0].plot(steps, values)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss')
        
        # Plot entropy
        if 'train/entropy_loss' in tags:
            entropy = ea.Scalars('train/entropy_loss')
            steps = [x.step for x in entropy]
            values = [x.value for x in entropy]
            axes[1, 1].plot(steps, values)
            axes[1, 1].set_title('Entropy Loss')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\n📊 Training plots saved to {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not plot training results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Retrain PPO model with aggressive direction balancing'
    )
    parser.add_argument(
        '--data', 
        default='btc_usdt_data/full_btc_usdt_data_feature_engineered.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=1_000_000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=200,
        help='Episode length'
    )
    parser.add_argument(
        '--envs', 
        type=int, 
        default=4,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--save-path', 
        default='rl_models_balanced',
        help='Directory to save models'
    )
    parser.add_argument(
        '--tensorboard', 
        default='rl_tensorboard_balanced',
        help='Tensorboard log directory'
    )
    parser.add_argument(
        '--eval-episodes', 
        type=int, 
        default=10,
        help='Number of evaluation episodes after training'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 PPO Model Retraining with AGGRESSIVE Direction Balancing")
    print("=" * 70)
    
    # Load data
    df = load_data(args.data)
    
    # Create vectorized environment
    print(f"\n🌐 Creating {args.envs} parallel environments...")
    if args.envs > 1:
        env = SubprocVecEnv([
            make_env(df, rank=i, episode_length=args.episodes) 
            for i in range(args.envs)
        ])
    else:
        env = DummyVecEnv([make_env(df, rank=0, episode_length=args.episodes)])
    
    # Create model
    model = create_model(env, tensorboard_log=args.tensorboard)
    
    # Setup balance monitoring callback
    balance_callback = DirectionBalanceCallback(verbose=1)
    
    # Train model
    success = train_model(
        model, 
        total_timesteps=args.timesteps,
        callback=balance_callback,
        save_path=args.save_path
    )
    
    if success:
        # Save final model
        final_model_path = os.path.join(args.save_path, 'ppo_balanced_final.zip')
        model.save(final_model_path)
        print(f"\n💾 Final model saved to {final_model_path}")
        
        # Evaluate model
        eval_env = DummyVecEnv([make_env(df, rank=0, episode_length=args.episodes)])
        eval_results = evaluate_model(model, eval_env.envs[0], n_eval_episodes=args.eval_episodes)
        
        # Save evaluation results
        eval_results_path = os.path.join(args.save_path, 'evaluation_results.json')
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n💾 Evaluation results saved to {eval_results_path}")
        
        # Plot training results
        plot_training_results(args.tensorboard, 
                            os.path.join(args.save_path, 'training_results.png'))
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 TRAINING SUMMARY")
        print("=" * 70)
        print(f"✅ Model trained for {args.timesteps:,} timesteps")
        print(f"✅ Final model: {final_model_path}")
        print(f"✅ Balance Score: {eval_results['balance_score']:.2f}")
        print(f"✅ Long/Short Ratio: {eval_results['long_ratio']:.1%} / {eval_results['short_ratio']:.1%}")
        
        if eval_results['balance_score'] >= 0.7:
            print("\n🎯 EXCELLENT: Model shows good directional balance!")
        elif eval_results['balance_score'] >= 0.5:
            print("\n⚠️  FAIR: Model shows moderate directional balance")
        else:
            print("\n❌ WARNING: Model is still imbalanced, consider more training")
    
    # Cleanup
    env.close()
    
    print("\n🎉 Retraining process completed!")


if __name__ == "__main__":
    main()
