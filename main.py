#!/usr/bin/env python3
"""
Main Entry Point for RL Algorithmic Trading Bot
Unified interface for training, paper trading, and live trading
"""

import argparse
import sys
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="🤖 RL Algorithmic Trading Bot - Advanced Reinforcement Learning Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --data data/btc_usdt_data.csv --timesteps 500000

  # Run paper trading simulation
  python main.py paper-trade --model models/ppo_trained.zip --balance 10000

  # Run live trading (paper/virtual mode)
  python main.py live-trade --model models/ppo_trained.zip --balance 5000

  # Evaluate model performance
  python main.py evaluate --model models/ppo_trained.zip --data data/test_data.csv
        """
    )
    
    parser.add_argument(
        "mode", 
        choices=['train', 'paper-trade', 'live-trade', 'evaluate', 'optimize'],
        help="Operation mode: train (train new model), paper-trade (simulate trading), "
             "live-trade (paper/live trading), evaluate (assess model), optimize (hyperparameter tuning)"
    )
    
    parser.add_argument(
        "--model", 
        default="models/ppo_trading_agent.zip",
        help="Path to trained model file (default: models/ppo_trading_agent.zip)"
    )
    
    parser.add_argument(
        "--data", 
        default="btc_usdt_data/full_btc_usdt_data_feature_engineered.csv",
        help="Path to training/evaluation data file (default: btc_usdt_data/full_btc_usdt_data_feature_engineered.csv)"
    )
    
    parser.add_argument(
        "--balance", 
        type=float, 
        default=10000.0,
        help="Initial trading balance in USDT (default: 10000.0)"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=200000,
        help="Number of training timesteps (default: 200000)"
    )
    
    parser.add_argument(
        "--symbol", 
        default="BTC-USDT",
        help="Trading symbol (default: BTC-USDT)"
    )
    
    parser.add_argument(
        "--test-mode", 
        action="store_true",
        help="Run in test mode (paper trading) even for live-trade command"
    )
    
    parser.add_argument(
        "--n-envs", 
        type=int, 
        default=4,
        help="Number of parallel environments for training (default: 4)"
    )
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=30,
        help="Number of optimization trials (for optimize mode, default: 30)"
    )
    
    args = parser.parse_args()
    
    print("🤖 RL Algorithmic Trading Bot")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        if args.mode == "train":
            logger.info("Starting RL model training...")
            from train_rl import train_rl_agent
            trained_model = train_rl_agent(
                data_path=args.data,
                total_timesteps=args.timesteps,
                n_envs=args.n_envs
            )
            logger.info(f"Training completed. Model saved to {args.model}")
            
        elif args.mode == "paper-trade":
            logger.info("Starting paper trading simulation...")
            from rl_paper_trading import run_rl_paper_trading
            results = run_rl_paper_trading(
                model_path=args.model,
                data_path=args.data,
                initial_balance=args.balance
            )
            if results:
                logger.info("Paper trading simulation completed successfully")
                print("\n📊 Paper Trading Results:")
                print(f"Final Portfolio Value: ${results.get('final_portfolio', 0):,.2f}")
                print(f"Total Return: {results.get('total_return', 0):.2f}%")
                print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
                print(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2%}")
            else:
                logger.error("Paper trading simulation failed")
                
        elif args.mode == "live-trade":
            logger.info("Starting live trading session...")
            from rl_live_trading import RLLiveTradingBot
            bot = RLLiveTradingBot(
                model_path=args.model,
                symbol=args.symbol,
                test_mode=args.test_mode,
                initial_balance=args.balance
            )
            bot.run_live_session(duration_minutes=60)  # Default 1 hour session
            logger.info("Live trading session completed")
            
        elif args.mode == "evaluate":
            logger.info("Evaluating trained model...")
            from train_rl import evaluate_agent_comprehensive
            from stable_baselines3 import PPO
            
            # Load the model
            if os.path.exists(args.model):
                model = PPO.load(args.model)
                logger.info(f"Model loaded from {args.model}")
            else:
                logger.error(f"Model file not found: {args.model}")
                sys.exit(1)
            
            # Run comprehensive evaluation
            evaluation_results = evaluate_agent_comprehensive(
                model=model,
                data_path=args.data,
                n_episodes=5
            )
            logger.info("Model evaluation completed")
            
        elif args.mode == "optimize":
            logger.info("Starting hyperparameter optimization...")
            from hyperparameter_optimization import run_optimization
            optimization_results = run_optimization(
                n_trials=args.trials,
                timeout=7200  # 2 hours timeout
            )
            logger.info("Hyperparameter optimization completed")
            
        else:
            # Show help if no valid command provided
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
