# Usage Guide for RL Algorithmic Trading Bot

This guide explains how to set up and use the RL Algorithmic Trading Bot.

## Prerequisites

- Python 3.11+
- UV package manager (recommended)
- Git installed

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RL-Algorithmic-Trading-Bot.git
cd RL-Algorithmic-Trading-Bot

# Install dependencies using UV
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RL-Algorithmic-Trading-Bot.git
cd RL-Algorithmic-Trading-Bot

# Install dependencies using pip
pip install -r requirements.txt
```

### Installing PyTorch and Stable-Baselines3

For Linux/macOS with CPU:
```bash
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]
```

For Linux/macOS with CUDA (select appropriate version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra]
```

For macOS with ARM64 (Apple Silicon):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3[extra]
```

## Environment Setup

Before running the live trading bot, set up your API keys in the `.env` file:

```bash
# Copy the template
cp .env.example .env

# Edit the file with your API keys
nano .env
```

⚠️ **Security Warning**: Never commit your actual API keys to version control. The `.env` file is already included in `.gitignore`.

## Data Preparation

1. Download historical market data (CSV format) for your preferred assets
2. Place data in `data/` directory
3. Run feature engineering:

```bash
python scripts/feature_engineering.py --data data/btc_usdt_historical.csv
```

## Training

Train PPO model with reproducible results:

```bash
python -m train_rl train ppo --data data/btc_usdt_feature_engineered.csv --timesteps 1000000
```

Optimize hyperparameters:

```bash
python -m hyperparameter_optimization --n_trials 100
```

## Paper Trading

Run paper trading simulation:

```bash
python -m rl_paper_trading --model models/dqn_trained.zip --data data/btc_usdt_test.csv --balance 10000
```

## Live Trading

Run live trading (virtual mode):

```bash
python -m rl_live_trading --model models/ppo_trained.zip --symbol BTC-USDT --balance 10000 --test-mode
```

## Configuration

Configuration parameters can be adjusted in `config/trading_config.yaml`:

```yaml
# Trading Parameters
initial_balance: 10000
transaction_fee: 0.001
max_position_size: 0.25
max_total_exposure: 0.50

# Risk Management
stop_loss_pct: 0.08
take_profit_pct: 0.15
max_drawdown_limit: 0.20
max_volatility_limit: 0.50

# RL Parameters
learning_rate: 0.0003
batch_size: 128
buffer_size: 100000
exploration_fraction: 0.1
```

## Monitoring

The system provides real-time monitoring through:
- Console logging with key metrics
- TensorBoard integration for training visualization
- Performance dashboards
- Risk alerts and notifications

## Testing

Run tests to verify system functionality:

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
