# RL Algorithmic Trading Bot

A sophisticated reinforcement learning-based algorithmic trading system that uses Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms to trade cryptocurrency markets.

## 🚀 Features

- **Advanced RL Algorithms**: Implements both DQN and PPO for optimal trading strategies
- **Multi-Asset Support**: Capable of trading various cryptocurrency pairs
- **Comprehensive Risk Management**: Built-in stop-loss, take-profit, position sizing, and portfolio controls
- **Real-time Trading**: Supports live trading with virtual/paper trading modes
- **Advanced Technical Analysis**: Incorporates 30+ technical indicators for decision making
- **Hyperparameter Optimization**: Automated optimization using Optuna
- **Performance Analytics**: Comprehensive backtesting and live performance metrics
- **Reproducible Results**: Fixed random seeds for consistent training outcomes

## 📊 Trading Strategies

The system implements multiple trading strategies:

1. **Trend Following**: Identifies and follows market trends
2. **Mean Reversion**: Capitalizes on price deviations from mean
3. **Breakout Trading**: Exploits price breakouts from consolidation patterns
4. **Momentum Trading**: Captures momentum-driven price movements

## 🏗️ Architecture

```
rl-trading-bot/
├── trading_environment.py      # Unified trading environment
├── train_rl.py                # RL training module with reproducible seeds
├── rl_paper_trading.py        # Paper trading simulation
├── rl_live_trading.py         # Live trading module with improved error handling
├── risk_management.py         # Advanced risk controls
├── hyperparameter_optimization.py  # Hyperparameter tuning with logging
├── data/
│   ├── btc_usdt_data/         # Historical market data
│   └── feature_engineered/    # Engineered features
├── models/                    # Trained RL models
├── notebooks/                 # Jupyter notebooks for analysis
└── utils/                     # Utility functions
```

## 📈 Technical Indicators

The system incorporates 30+ technical indicators:

- **Trend Indicators**: Moving Averages (EMA, WMA), MACD, ADX
- **Momentum Indicators**: RSI, Stochastic Oscillator, ROC, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels
- **Volume Indicators**: OBV, AD, MFI, VPT
- **Pattern Recognition**: Candlestick patterns, support/resistance levels

## 🎯 RL Implementation

### DQN (Deep Q-Network)
- Deep neural network with experience replay
- Double DQN with Dueling architecture
- Prioritized Experience Replay for efficient learning

### PPO (Proximal Policy Optimization)
- Actor-Critic architecture with shared feature extractor
- Clipped surrogate objective for stable training
- Adaptive learning rate scheduling

## 🛡️ Risk Management

Comprehensive risk controls:

- **Position Sizing**: Dynamic position sizing based on volatility and risk tolerance
- **Stop Loss**: Trailing stop-loss with adaptive distances
- **Take Profit**: Dynamic take-profit levels
- **Correlation Limits**: Maximum correlation between positions
- **Exposure Limits**: Individual and total portfolio exposure caps
- **Drawdown Control**: Automatic trading halt on excessive drawdown
- **Volatility Limits**: Position reduction during high volatility periods

## 📊 Performance Metrics

The system tracks multiple performance metrics:

- **Return Metrics**: Total return, annualized return, cumulative return
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk Metrics**: Value at Risk (VaR), Expected Shortfall (ES), Maximum drawdown
- **Efficiency Metrics**: Profit factor, Win rate, Average trade duration
- **Correlation Metrics**: Beta, Alpha, Information ratio

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- UV package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-trading-bot.git
cd rl-trading-bot

# Install dependencies using UV (recommended)
uv sync

# Or if UV is not available, use pip
pip install -r requirements.txt

# Install PyTorch and Stable-Baselines3 separately based on your platform:
# For Linux/macOS with CPU:
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]

# For Linux/macOS with CUDA (select appropriate version):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra]

# For macOS with ARM64 (Apple Silicon):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3[extra]
```

### Environment Setup

Before running the live trading bot, set up your API keys in the `.env` file:

```bash
# Copy the template
cp .env.example .env

# Edit the file with your API keys
nano .env
```

⚠️ **Security Warning**: Never commit your actual API keys to version control. The `.env` file is already included in `.gitignore`.

### Data Preparation

1. Download historical market data (CSV format) for your preferred assets
2. Place data in `data/` directory
3. Run feature engineering:

```bash
python scripts/feature_engineering.py --data data/btc_usdt_historical.csv
```

### Training

```bash
# Train PPO model with reproducible results
python -m train_rl train ppo --data data/btc_usdt_feature_engineered.csv --timesteps 1000000

# Optimize hyperparameters
python -m hyperparameter_optimization --n_trials 100
```

### Paper Trading

```bash
# Run paper trading simulation
python -m rl_paper_trading --model models/dqn_trained.zip --data data/btc_usdt_test.csv --balance 10000
```

### Live Trading

```bash
# Run live trading (virtual mode)
python -m rl_live_trading --model models/ppo_trained.zip --symbol BTC-USDT --balance 10000 --test-mode
```

## 📊 Model Performance

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Overall percentage return |
| Sharpe Ratio | Risk-adjusted return |
| Maximum Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Expectancy | Average profit per trade |

### Baseline Comparison

The RL model is compared against:

- Buy & Hold strategy
- Simple moving average crossover
- RSI-based strategy
- Random walk benchmark

## 🔧 Configuration

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

## 📈 Monitoring

The system provides real-time monitoring through:

- Console logging with key metrics
- TensorBoard integration for training visualization
- Performance dashboards
- Risk alerts and notifications

## 🧪 Testing

Run tests to verify system functionality:

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This is a research project for educational purposes. Trading cryptocurrencies involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Never invest more than you can afford to lose.**

The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Happy Trading!** 📈
