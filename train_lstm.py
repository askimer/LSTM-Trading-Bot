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
import time

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
    
class ProgressNeuralNetRegressor(NeuralNetRegressor):
    """NeuralNetRegressor with progress tracking"""

    def __init__(self, iteration=1, total_iterations=20, **kwargs):
        super().__init__(**kwargs)
        # Replace the default callback with our custom one
        self.callbacks_ = [
            cb for cb in self.callbacks
            if not isinstance(cb, TrainingProgressCallback)
        ]
        self.callbacks_.append(TrainingProgressCallback(
            total_iterations=total_iterations,
            current_iteration=iteration
        ))

class TrainingProgressCallback(Callback):
    """Custom callback to show detailed training progress"""

    def __init__(self, total_iterations=20, current_iteration=1):
        self.total_iterations = total_iterations
        self.current_iteration = current_iteration
        self.start_time = None

    def on_train_begin(self, net, X, y):
        self.start_time = time.time()
        print(f"\n🔄 Iteration {self.current_iteration}/{self.total_iterations}")
        print("=" * 50)

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        epoch = net.history[-1]['epoch']
        train_loss = net.history[-1]['train_loss']
        valid_loss = net.history[-1].get('valid_loss', 'N/A')

        # Calculate ETA
        elapsed = time.time() - self.start_time
        epochs_done = len(net.history)
        if epochs_done > 0:
            avg_epoch_time = elapsed / epochs_done
            remaining_epochs = net.max_epochs - epochs_done
            eta = avg_epoch_time * remaining_epochs
            eta_str = f"{eta/60:.1f}min" if eta > 60 else f"{eta:.1f}s"
        else:
            eta_str = "N/A"

        print(f"📈 Epoch {epoch:2d}/{net.max_epochs} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss} | ETA: {eta_str}")

        # Show progress bar
        progress = int(30 * epoch / net.max_epochs)
        bar = "█" * progress + "░" * (30 - progress)
        print(f"[{bar}] {epoch/net.max_epochs*100:.1f}%")

    def on_train_end(self, net, X, y):
        final_train_loss = net.history[-1]['train_loss']
        final_valid_loss = net.history[-1].get('valid_loss', 'N/A')
        total_time = time.time() - self.start_time

        print(f"\n✅ Iteration {self.current_iteration} completed in {total_time:.1f}s")
        print(f"   Final Train Loss: {final_train_loss:.6f}")
        print(f"   Final Valid Loss: {final_valid_loss}")
        print("-" * 50)

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

    # Create base network template
    def create_net(**kwargs):
        """Create NeuralNetRegressor with custom progress callback"""
        iteration = kwargs.pop('iteration', 1)
        return NeuralNetRegressor(
            module=LSTMRegressor,
            module__input_dim=n_features,
            criterion=nn.MSELoss,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            device=device,
            callbacks=[
                early_stopping,
                TrainingProgressCallback(total_iterations=20, current_iteration=iteration)
            ],
            **kwargs
        )

    net = create_net(
        module__hidden_dim=50,
        module__num_layers=1,
        lr=0.001,
        batch_size=64,
        max_epochs=10
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
    print("=" * 60)

    # Custom hyperparameter search with progress tracking
    best_score = float('-inf')
    best_params = None
    best_model = None

    # Sample parameter combinations
    np.random.seed(42)
    param_combinations = []
    for _ in range(20):
        combination = {}
        for param_name, param_values in params.items():
            combination[param_name] = np.random.choice(param_values)
        param_combinations.append(combination)

    for i, param_comb in enumerate(param_combinations, 1):
        print(f"\n🎯 Testing combination {i}/20:")
        for param_name, param_value in param_comb.items():
            print(f"   {param_name}: {param_value}")

        # Create model with progress tracking
        model = create_net(iteration=i, **param_comb)

        try:
            # Train model
            model.fit(X_train_tensor, y_train_tensor)

            # Evaluate on validation set (simple holdout for speed)
            y_pred = model.predict(X_test_tensor)
            score = -mean_squared_error(y_test_tensor, y_pred)  # Negative MSE

            print(f"   📊 Validation Score: {score:.6f}")

            if score > best_score:
                best_score = score
                best_params = param_comb.copy()
                best_model = model
                print(f"   🏆 New best score: {best_score:.6f}")

        except Exception as e:
            print(f"   ❌ Error training model: {e}")
            continue

    print(f"\n🏁 Hyperparameter search completed!")
    print(f"Best score: {best_score:.6f}")
    print("Best hyperparameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")

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
