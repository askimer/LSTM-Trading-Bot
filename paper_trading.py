import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# load the model
best_model = torch.load('./lstm_model_1472.pt')
best_model = best_model.module_
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)
best_model.eval()

# load completely new test data
df = pd.read_csv('./paper_trade_data/full_btc_usdt_data_feature_engineered.csv')

# Validate data
print(f"Loaded {len(df)} rows of data")
if df.empty:
    raise ValueError("No data found in paper_trade_data/full_btc_usdt_data_feature_engineered.csv")

# Check for required columns
required_columns = ['close', 'ATR_15', 'RSI_15']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Clean data
df = df.dropna()
print(f"After dropping NaN: {len(df)} rows")

if df.empty:
    raise ValueError("All data was removed after dropping NaN values")

# Check for constant columns
std_dev = df.std()
constant_columns = std_dev[std_dev == 0].index.tolist()
if constant_columns:
    print(f"Warning: Found constant columns: {constant_columns}")

non_constant_columns = std_dev[std_dev != 0].index.tolist()
df = df[non_constant_columns]

print(f"Final dataset: {len(df)} rows, {len(df.columns)} features")

# scale the data
X = df.drop('close', axis=1).values
y = df['close'].values.reshape(-1, 1)

# open the scalers (separate for X and y)
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

X = scaler_X.transform(X)

def calculate_past_average(predictions, window):
    if len(predictions) < window:
        return np.mean(predictions)
    else:
        return np.mean(predictions[-window:])
    
# env set up
initial_budget = 10000

def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown

# Function to run a single simulation with given parameters
def run_simulation(alpha_atr, alpha_rsi, base_sell_threshold, base_buy_threshold, sell_percentage, buy_percentage, window_size, min_profit_threshold):
    current_budget = initial_budget
    assets_held = 0
    past_predictions = []
    portfolio_values = []

    trading_fee_percentage = 0.0180 / 100  # Trading fee of 0.0180%

    # the simulated trading
    for i in range(X.shape[0]):
        # Reshape the data
        input_data = torch.tensor(X[i].reshape(1, 1, -1)).float().to(device)  # Reshape to (1, 1, num_features)

        atr_value = df['ATR_15'].iloc[i]
        rsi_value = df['RSI_15'].iloc[i]

        sell_threshold = base_sell_threshold + alpha_atr * atr_value + alpha_rsi * max(0, rsi_value - 70)
        buy_threshold = base_buy_threshold - alpha_atr * atr_value - alpha_rsi * max(0, 70 - rsi_value)

        # predictions (model outputs scaled values, need to inverse transform)
        prediction_scaled = best_model(input_data).item()
        prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

        past_predictions.append(prediction)
        past_average = calculate_past_average(past_predictions, window=window_size)
        
        trend_direction = 'up' if prediction > past_average else 'down'
        # positive -> probably going up
        # negative -> probably going down
        confidence = (prediction - past_average) / past_average if past_average != 0 else 0

        asset_price = y[i][0]

        # Calculate expected profit
        expected_price_increase = prediction - asset_price
        expected_profit_percent = (expected_price_increase / asset_price) * 100

        if trend_direction == 'up' and confidence >= buy_threshold and expected_profit_percent >= min_profit_threshold:
            investment = min(current_budget * abs(confidence) * buy_percentage, current_budget)
            fee = investment * trading_fee_percentage
            rounded_investment = round(investment - fee, 6)
            assets_bought = (rounded_investment / asset_price)
            assets_held += round(assets_bought, 8)
            current_budget -= rounded_investment
        elif trend_direction == 'down' and confidence <= sell_threshold and assets_held > 0:
            assets_to_sell = assets_held * sell_percentage * abs(confidence)
            sale_revenue = round(assets_to_sell, 8) * asset_price
            fee = sale_revenue * trading_fee_percentage
            assets_held -= assets_to_sell
            current_budget += round(sale_revenue - fee, 6)

        final_asset_value = assets_held * asset_price
        current_portfolio_value = current_budget + final_asset_value

        if i % 1000 == 0:
            print(f"atr_value: {atr_value}, rsi_value: {rsi_value}")
            print(f"Prediction: {trend_direction} Confidence: {confidence}")
            print(f"Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}")
            print(f"Second {i}, Budget: {current_budget}, Assets Held: {assets_held}, Asset Price: {asset_price}")
            print(f"Portfolio Value: {current_portfolio_value}")
            portfolio_values.append(current_portfolio_value)
    


    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate returns
    sharpe_ratio = calculate_sharpe_ratio(returns, 0.000006811)  # Calculate sharpe ratio, risk free rate per 1000 seconds is interpreted from a 3 month US treasury bill yield of 5.37%
    max_drawdown = calculate_max_drawdown(portfolio_values) # Calculate max drawdown


    print("LSTM strategy")
    print(f"Initial Budget: {initial_budget}, Final Portfolio Value: {current_portfolio_value}")
    print(f"Total Profit: {current_portfolio_value - initial_budget}")
    print("Sharpe Ratio:", sharpe_ratio)
    print("Maximum Drawdown:", max_drawdown)

    # plot the portfolio value over time in a plot, it will be shown at the end of the program
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Portfolio Value ($)')

    return current_portfolio_value - initial_budget, sharpe_ratio, max_drawdown  # Return profit


    
# [-inf, sell_threshold]  -> sell
# (sell_threshold, buy_threshold) -> hold
# [buy_threshold, inf] -> buy

# my chosen optimal parameters GOOD FOR STABLE MARKET
# profit of +191.34 or 1.934% profit (vs best of 203.80 (or 2.03%))
# sharpe_ratio of 0.031518248 (vs best of 0.032467788)
# max_drawdown of -0.013196566 (vs best of -0.012710932)
base_sell_threshold = 0.000387    # Confidence level to trigger a sell
base_buy_threshold = 0.000595     # Confidence level to trigger a buy
alpha_atr = 0.420739              # Weight of ATR's effect in thresholds
alpha_rsi = 0.284716              # Weight of RSI's effect in thresholds
sell_percentage = 0.178912        # Percentage of assets to sell
buy_percentage = 0.5         # Percentage of budget to buy with
window_size = 49                  # Number of past predictions to average over
min_profit_threshold = 0.731747      # Minimum profit threshold to trigger a buy

# optimal parameters number 2
# profit of +189.02 or 1.89% profit
# sharpe_ratio of 0.03428494
# max_drawdown of -0.012603482

base_sell_threshold = 0.000318
base_buy_threshold = 0.000541
alpha_atr = 26.117467
alpha_rsi = 29.549422
sell_percentage = 0.314003
buy_percentage = 0.671026
window_size = 17
min_profit_threshold = 0.768993

# Run the simulation
profit, sharpe, drawdown = run_simulation(alpha_atr, alpha_rsi, base_sell_threshold, base_buy_threshold, sell_percentage, buy_percentage, window_size, min_profit_threshold)

# compare with buy and hold
# search for the earliest price
earliest_time = df['Open time'].iloc[0]
earliest_price = df['close'].iloc[0]
latest_time = df['Open time'].iloc[-1]
latest_price = df['close'].iloc[-1]

print(f"Earliest time: {earliest_time}, Earliest price: {earliest_price}")
print(f"Latest time: {latest_time}, Latest price: {latest_price}")

initial_asset_value = initial_budget / earliest_price
final_asset_value = initial_asset_value * latest_price
total_value = final_asset_value
print("Buy and hold strategy")
print(f"Initial Budget: {initial_budget}, Final Total Value: {total_value}")
print(f"Total Profit: {total_value - initial_budget}")

# calculate the sharpe ratio using df['close']
# calculate the max drawdown using df['close']
portfolio_values = []
current_budget = initial_budget
assets_held = 0
trading_fee_percentage = 0.0180 / 100  # Trading fee of 0.0180%
for i in range(X.shape[0]):
    asset_price = y[i][0]
    if i == 0:
        assets_bought = (current_budget / asset_price)
        assets_held += assets_bought
        current_budget -= current_budget
    elif (i%1000 == 0):
        final_asset_value = assets_held * asset_price
        current_portfolio_value = current_budget + final_asset_value
        portfolio_values.append(current_portfolio_value)
        assets_held = assets_held

portfolio_values = np.array(portfolio_values)
returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate returns
sharpe_ratio = calculate_sharpe_ratio(returns, 0.000006811)  # Calculate sharpe ratio, risk free rate per 1000 seconds is interpreted from a 3 month US treasury bill yield of 5.37%
max_drawdown = calculate_max_drawdown(portfolio_values) # Calculate max drawdown

print("Buy and hold strategy")
print(f"Initial Budget: {initial_budget}, Final Portfolio Value: {current_portfolio_value}")
print(f"Total Profit: {current_portfolio_value - initial_budget}")
print("Sharpe Ratio:", sharpe_ratio)
print("Maximum Drawdown:", max_drawdown)

# plot the portfolio value over time in a subplot, it will be shown at the end of the program
plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Time (1000 seconds)')
plt.ylabel('Portfolio Value ($)')

# legend
plt.legend(['LSTM', 'Buy and Hold'])
plt.grid(True)

plt.savefig('lstm_vs_buy_and_hold.png')
