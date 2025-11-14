import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
from skorch.callbacks import Callback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skorch.callbacks import EarlyStopping, ProgressBar
import matplotlib.pyplot as plt

import pickle
import resource

# Define the LSTM model using PyTorch
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    
early_stopping = EarlyStopping(
    monitor='valid_loss',
    patience=5,
    threshold=0.0001,
    threshold_mode='rel',
    lower_is_better=True,
)

if __name__ == "__main__":
    # Try to set memory limit to encourage swap usage (optional)
    try:
        current_limits = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Current memory limits: soft={current_limits[0] // (1024**3)}GB, hard={current_limits[1] // (1024**3)}GB")
        # Only set if current soft limit is higher than 16GB
        if current_limits[0] > 16 * 1024 * 1024 * 1024:
            soft_limit = 12 * 1024 * 1024 * 1024  # 12GB in bytes
            hard_limit = current_limits[1]  # Keep current hard limit
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
            print(f"Memory limit set to {soft_limit // (1024**3)}GB soft (to encourage swap usage)")
        else:
            print("Memory limit not changed (already reasonable)")
    except ValueError as e:
        print(f"Could not set memory limit: {e}. System will manage memory automatically.")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")
    print("Loading data...")
    df = pd.read_csv('./btc_usdt_training_data/full_btc_usdt_data_feature_engineered.csv')
    df = df.dropna()

    print("Removing constant columns...")
    std_dev = df.std()
    non_constant_columns = std_dev[std_dev != 0].index.tolist()
    df = df[non_constant_columns]

    X = df.drop('close', axis=1).values
    y = df['close'].values.reshape(-1, 1)

    print("Scaling data...")
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    # pickle the scalers
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    n_features = X_train.shape[1]
    print("Converting data to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train.astype(np.float32).copy().reshape(-1, 1, n_features))
    y_train_tensor = torch.tensor(y_train.astype(np.float32).copy())
    X_test_tensor = torch.tensor(X_test.astype(np.float32).copy().reshape(-1, 1, n_features))
    y_test_tensor = torch.tensor(y_test.astype(np.float32).copy())

    # Create PyTorch datasets
    print("Creating PyTorch datasets...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    net = NeuralNetRegressor(
        module=LSTMRegressor,
        module__input_dim=n_features,
        module__hidden_dim=50,  # default value
        module__num_layers=1,   # default value
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.001,              # default learning rate
        batch_size=64,         # default batch size
        max_epochs=10,         # default number of epochs
        iterator_train__shuffle=True,
        device=device,
        callbacks=[early_stopping]
    )

    # Define the hyperparameters grid to search
    params = {
        'module__hidden_dim': [256, 384, 512, 768, 1024],
        'module__num_layers': [3, 4, 5, 6],
        'lr': [0.02, 0.015, 0.01],
        'batch_size': [128, 192, 256, 384, 512],
        'max_epochs': [10, 20, 30]
    }

    print("Searching for optimal hyperparameters...")
    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(net, params, n_iter=20, cv=3, scoring='neg_mean_squared_error', n_jobs=2, random_state=42, verbose=2)

    print("Training model...")
    # Fit the model
    random_search.fit(X_train_tensor, y_train_tensor)

    print("Get best params")
    best_params = random_search.best_params_
    print("Best hyperparameters:")
    for param_name, param_value in best_params.items():
        print(f"{param_name}: {param_value}")

    # Get the best model
    best_model = random_search.best_estimator_

    print("Evaluating model...")
    # Make predictions
    y_pred = best_model.predict(X_test_tensor)

    # Calculate MSE on original scale (not scaled)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scaler_y.inverse_transform(y_test_tensor.numpy())
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"Mean Squared Error (on original scale): {mse}")

    history = best_model.history

    plt.figure(figsize=(12, 6))
    plt.plot([epoch['epoch'] for epoch in history], [epoch['train_loss'] for epoch in history], label='Training Loss')
    plt.yscale('log')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save the model
    print("Saving model...")
    torch.save(best_model, f'lstm_model_{int(mse)}.pt')
