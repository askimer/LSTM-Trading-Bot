"""
Configuration file for paths and parameters
"""

# Data paths
TRAINING_DATA_PATH = './btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv'
TUNING_DATA_PATH = './trading_alg_tuning_data/full_btc_usdt_data_feature_engineered.csv'
PAPER_TRADE_DATA_PATH = './paper_trade_data/full_btc_usdt_data_feature_engineered.csv'

# Model paths
DEFAULT_LSTM_MODEL = './lstm_model_1472.pt'
DEFAULT_SCALER = './scaler_X.pkl'

# Other configs
LOG_LEVEL = 'INFO'
