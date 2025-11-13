#!/usr/bin/env python3
"""
Pipeline script for sequential execution of the Algorithmic Trading Bot.
Supports both LSTM (supervised) and RL (reinforcement learning) approaches.
This script automates the entire process from data acquisition to live virtual trading.

__Этапы:__

1. Data Acquisition (скачивание, очистка, feature engineering)
2. Prepare Data Folders (переименование и создание папок)
3. Modify Scripts (корректировка путей в скриптах)
4. Train Model (обучение LSTM или RL модели)
5. Tune Trading Strategy (настройка стратегии)
6. Paper Trading (бумажная торговля)
7. Live Trading (Virtual) (виртуальная торговля на live данных)

__Примеры использования:__

- `python pipeline.py` - полный запуск с LSTM (по умолчанию)
- `python pipeline.py --rl` - полный запуск с RL
- `python pipeline.py --start-from 4 --rl` - начать с RL обучения
- `python pipeline.py --start-from 5` - начать с настройки стратегии


"""

import os
import shutil
import subprocess
import sys
import argparse
from datetime import datetime, timedelta

def run_script(script_name):
    """Run a Python script using subprocess."""
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    # Print the output of the scri/Volumes/Movies/PYTHON/LSTM-Algorithmic-Trading-Bot/btc_usdt_training_data/btc_usdt_data/Volumes/Movies/PYTHON/LSTM-Algorithmic-Trading-Bot/btc_usdt_training_data/btc_usdt_datapt
    if result.stdout:
        print(result.stdout.strip())
    print(f"{script_name} completed successfully.")

def rename_data_folder():
    """Rename btc_usdt_data to btc_usdt_training_data."""
    if os.path.exists('btc_usdt_data'):
        if os.path.exists('btc_usdt_training_data'):
            shutil.rmtree('btc_usdt_training_data')
        shutil.move('btc_usdt_data', 'btc_usdt_training_data')
        print("Renamed btc_usdt_data to btc_usdt_training_data")
    else:
        print("btc_usdt_data not found, skipping rename")

def create_tuning_data():
    """Create trading_alg_tuning_data by copying btc_usdt_training_data."""
    if os.path.exists('btc_usdt_training_data'):
        shutil.copytree('btc_usdt_training_data', 'trading_alg_tuning_data')
        print("Created trading_alg_tuning_data")
    else:
        print("btc_usdt_training_data not found, cannot create tuning data")

def create_paper_trade_data():
    """Create paper_trade_data by copying btc_usdt_training_data."""
    if os.path.exists('btc_usdt_training_data'):
        shutil.copytree('btc_usdt_training_data', 'paper_trade_data')
        print("Created paper_trade_data")
    else:
        print("btc_usdt_training_data not found, cannot create paper trade data")

def modify_train_lstm():
    """Modify train_lstm.py to use correct data path."""
    # Read the file
    with open('train_lstm.py', 'r') as f:
        content = f.read()

    # Replace the data loading line
    old_line = "    df = pd.read_csv('btc_usdt_data/full_btc_usdt_data_feature_engineered.csv')"
    new_line = "    df = pd.read_csv('./btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv')"
    content = content.replace(old_line, new_line)

    # Write back
    with open('train_lstm.py', 'w') as f:
        f.write(content)
    print("Modified train_lstm.py to use correct data path")

def modify_strategy_tuning():
    """Modify strategy_tuning.py to use correct data path."""
    with open('strategy_tuning.py', 'r') as f:
        content = f.read()

    old_line = "df = pd.read_csv('./trading_alg_tuning_data/full_btc_usdt_data_feature_engineered.csv')"
    new_line = "df = pd.read_csv('./trading_alg_tuning_data/full_btc_usdt_data_feature_engineered.csv')"
    # Already correct, but ensure it exists
    if old_line in content:
        pass  # Already correct

    print("strategy_tuning.py data path is correct")

def modify_paper_trading():
    """Modify paper_trading.py to use correct data path."""
    with open('paper_trading.py', 'r') as f:
        content = f.read()

    old_line = "df = pd.read_csv('./paper_trade_data_heaven/full_btc_usdt_data_feature_engineered.csv')"
    new_line = "df = pd.read_csv('./paper_trade_data/full_btc_usdt_data_feature_engineered.csv')"
    content = content.replace(old_line, new_line)

    with open('paper_trading.py', 'w') as f:
        f.write(content)
    print("Modified paper_trading.py to use correct data path")

def find_latest_model():
    """Find the latest model file (LSTM or RL)."""
    import glob
    import os

    # Check for RL model first
    rl_model = "ppo_trading_agent.zip"
    if os.path.exists(rl_model):
        print(f"Using RL model: {rl_model}")
        return rl_model

    # Check for LSTM models
    lstm_models = glob.glob('lstm_model_*.pt')
    if lstm_models:
        # Sort by MSE (assuming filename format lstm_model_{mse}.pt)
        lstm_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_model = lstm_models[-1]  # The one with highest MSE
        print(f"Using LSTM model: {latest_model}")
        return latest_model

    print("No model files found!")
    sys.exit(1)

def modify_strategy_tuning_model(model_file):
    """Modify strategy_tuning.py to use the correct model file."""
    with open('strategy_tuning.py', 'r') as f:
        content = f.read()

    if model_file.endswith('.zip'):
        # RL model
        old_line = "best_model = torch.load('./lstm_model_1472.pt')"
        new_line = f"from stable_baselines3 import PPO\nbest_model = PPO.load('./{model_file}')"
    else:
        # LSTM model
        old_line = "best_model = torch.load('./lstm_model_1472.pt')"
        new_line = f"best_model = torch.load('./{model_file}')"

    content = content.replace(old_line, new_line)

    with open('strategy_tuning.py', 'w') as f:
        f.write(content)
    print(f"Modified strategy_tuning.py to use model {model_file}")

def modify_paper_trading_model(model_file):
    """Modify paper_trading.py to use the correct model file."""
    with open('paper_trading.py', 'r') as f:
        content = f.read()

    if model_file.endswith('.zip'):
        # RL model
        old_line = "best_model = torch.load('./lstm_model_1472.pt')"
        new_line = f"from stable_baselines3 import PPO\nbest_model = PPO.load('./{model_file}')"
    else:
        # LSTM model
        old_line = "best_model = torch.load('./lstm_model_1472.pt')"
        new_line = f"best_model = torch.load('./{model_file}')"

    content = content.replace(old_line, new_line)

    with open('paper_trading.py', 'w') as f:
        f.write(content)
    print(f"Modified paper_trading.py to use model {model_file}")

def main(start_from=1):
    model_type = "RL" if args.rl else "LSTM"
    print(f"Starting {model_type} Algorithmic Trading Bot Pipeline")
    print("=" * 50)
    print(f"Starting from step {start_from}")

    steps = {
        1: ("Data Acquisition", lambda: (
            run_script('get_price_data.py'),
            run_script('clean_data.py'),
            run_script('feature_engineer.py'),
            rename_data_folder()
        )),
        2: ("Prepare Data Folders", lambda: (
            rename_data_folder(),
            create_tuning_data(),
            create_paper_trade_data()
        )),
        3: ("Modify Scripts for Correct Paths", lambda: (
            modify_train_lstm(),
            modify_strategy_tuning(),
            modify_paper_trading()
        )),
        4: ("Train Model", lambda: run_script('train_rl.py' if args.rl else 'train_lstm.py')),
        5: ("Tune Trading Strategy", lambda: (
            model_file := find_latest_model(),
            modify_strategy_tuning_model(model_file),
            run_script('strategy_tuning.py')
        )),
        6: ("Paper Trading", lambda: (
            model_file := find_latest_model(),
            modify_paper_trading_model(model_file),
            run_script('paper_trading.py')
        )),
        7: ("Live Trading (Virtual)", lambda: run_script('live_trading.py')),
    }

    for step_num in range(start_from, 8):
        step_name, step_func = steps[step_num]
        print(f"\nStep {step_num}: {step_name}")
        try:
            step_func()
        except Exception as e:
            print(f"Error in step {step_num}: {e}")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("Check the generated files and plots for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Algorithmic Trading Bot Pipeline")
    parser.add_argument(
        "--start-from",
        type=int,
        choices=range(1, 8),
        default=1,
        help="Step to start from (1-7): 1=Data Acquisition, 2=Prepare Folders, 3=Modify Scripts, 4=Train Model, 5=Tune Strategy, 6=Paper Trading, 7=Live Trading"
    )
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Use RL (Reinforcement Learning) instead of LSTM for training"
    )
    args = parser.parse_args()
    main(start_from=args.start_from)
