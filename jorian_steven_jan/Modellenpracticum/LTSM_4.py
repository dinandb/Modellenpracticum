# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time
# No deque needed for forecasting anymore

# --- Configuration Parameters ---
# FILE PATH
csv_path = "C:\\Users\\steve\\Downloads\\CleanQP_data_36000.csv"

# DATA & MODEL PARAMS
# ---> Input Lookback Strategy Parameters <---
recent_lookback = 15
sparse_step_size = 3
sparse_max_offset = 50
# ---> Output Forecast Horizon <---
forecast_horizon = 150  # <<< Model will directly predict this many steps

hidden_size = 256 # Might need larger hidden size for direct multi-step
num_layers = 3
test_split_ratio = 0.75

# TRAINING PARAMS
learning_rate = 0.01
n_epochs = 20 # May need more epochs for this harder task
validation_frequency = 10
batch_size = 256 # Adjust based on memory, potentially smaller due to larger model output
num_workers = 4

# --- Calculate derived lookback parameters ---
sparse_offsets = list(range(sparse_step_size * 2, sparse_max_offset + 1, sparse_step_size))
recent_offsets = list(range(1, recent_lookback + 1))
required_offsets = sorted(sparse_offsets, reverse=True) + sorted(recent_offsets)
max_lookback_offset = sparse_max_offset # Max steps back needed for input
input_seq_len = len(required_offsets) # Length of the input sequence to LSTM

# print(f"Input Lookback Strategy: SeqLen={input_seq_len}, MaxOffset={max_lookback_offset}")
# print(f"Output Strategy: Direct Forecast Horizon={forecast_horizon}")
# print("-" * 30)

# --- Function and Class Definitions ---

def create_direct_forecast_dataset(dataset, required_offsets, max_lookback_offset, forecast_horizon):
    """
    Creates dataset for DIRECT multi-step forecasting.
    Input X uses custom lookback. Target y is the next forecast_horizon steps.

    Args:
        dataset (np.array): Time series (N, 1).
        required_offsets (list): Offsets back from time 't' for input X.
        max_lookback_offset (int): Max steps back needed for X.
        forecast_horizon (int): Number of future steps to predict directly (y).

    Returns:
        torch.Tensor (X): Input sequences (num_samples, input_seq_len, 1).
        torch.Tensor (y): Target sequences (num_samples, forecast_horizon, 1).
    """
    if dataset.ndim == 1: dataset = dataset.reshape(-1, 1)
    X_list, y_list = [], []
    input_seq_len = len(required_offsets)
    num_features = dataset.shape[1] # Should be 1 here

    # Loop control: Ensure we have enough past for input AND enough future for target
    # Start index 'i' represents the *first time step of the target sequence y*
    # So, input sequence X ends at time i-1
    # Need data back to i - max_lookback_offset for X
    # Need data forward to i + forecast_horizon - 1 for y
    for i in range(max_lookback_offset, len(dataset) - forecast_horizon + 1):
        # Input sequence indices (relative to end time i-1)
        input_indices = [(i - 1) - offset + 1 for offset in required_offsets] # +1 adjusts offset to be from i-1

        # Target sequence indices (relative to start time i)
        target_indices = list(range(i, i + forecast_horizon))

        feature_sequence = dataset[input_indices] # Shape (input_seq_len, 1)
        target_sequence = dataset[target_indices] # Shape (forecast_horizon, 1)

        X_list.append(feature_sequence)
        y_list.append(target_sequence)

    if not X_list:
        return torch.empty((0, input_seq_len, num_features), dtype=torch.float32), \
               torch.empty((0, forecast_horizon, num_features), dtype=torch.float32)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y


class DirectForecastModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, forecast_horizon=150):
        super().__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        # LSTM processes the input sequence based on custom lookback
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # ---> Output layer changed <---
        # Maps the last hidden state to the entire forecast horizon
        self.linear = nn.Linear(hidden_size, forecast_horizon * input_size)

    def forward(self, x):
        # x shape: (batch, input_seq_len, input_size=1)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, input_seq_len, hidden_size)

        # Use the output corresponding to the *last* element of the input sequence
        last_time_step_output = lstm_out[:, -1, :] # Shape: (batch, hidden_size)

        # ---> Linear layer predicts all future steps at once <---
        direct_forecast_flat = self.linear(last_time_step_output) # Shape: (batch, forecast_horizon * input_size)

        # ---> Reshape the output <---
        direct_forecast = direct_forecast_flat.view(x.size(0), self.forecast_horizon, self.input_size)
        # Output shape: (batch, forecast_horizon=150, input_size=1)
        return direct_forecast

# --- Main Execution Block ---
if __name__ == '__main__':
    # Optional: freeze_support()

    # --- Device Setup ---
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    pin_memory = cuda_available
    print(f"Device: {device}, Pin Mem: {pin_memory}, Workers: {num_workers}, Batch: {batch_size}")
    print("-" * 30)

    # --- Data Loading ---
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path, header=[0,1])
        timeseries = df["z_velocity"].values.astype('float32').reshape(-1, 1)
        print(f"Data loaded. Shape: {timeseries.shape}")
    except Exception as e: print(f"Data loading error: {e}"); exit()
    train_size = int(len(timeseries) * test_split_ratio)
    train_data, test_data = timeseries[:train_size], timeseries[train_size:]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print("-" * 30)

    # --- Create Datasets for Direct Forecasting ---
    print("Creating direct forecast datasets (CPU)...")
    t_start_dataset = time.time()
    X_train_cpu, y_train_cpu = create_direct_forecast_dataset(train_data, required_offsets, max_lookback_offset, forecast_horizon)
    X_test_cpu, y_test_cpu = create_direct_forecast_dataset(test_data, required_offsets, max_lookback_offset, forecast_horizon)
    print(f"Dataset creation took: {time.time() - t_start_dataset:.2f}s")

    if X_train_cpu.nelement() == 0: print("Error: Training dataset empty."); exit()
    if X_test_cpu.nelement() == 0: print("Warning: Test dataset empty.")
    # Note the new Y shape: (N, forecast_horizon, 1)
    print(f"Train X/Y shapes: {X_train_cpu.shape} / {y_train_cpu.shape}")
    print(f"Test X/Y shapes: {X_test_cpu.shape} / {y_test_cpu.shape}")
    print("-" * 30)

    # --- DataLoaders ---
    train_dataset = data.TensorDataset(X_train_cpu, y_train_cpu)
    test_dataset = data.TensorDataset(X_test_cpu, y_test_cpu)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=pin_memory)
    print("DataLoaders created.")
    print("-" * 30)

    # --- Model, Optimizer, Loss ---
    print("Initializing direct forecast model...")
    model = DirectForecastModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, forecast_horizon=forecast_horizon).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss() # Compares the full 150-step sequences
    print("-" * 30)

    # --- Training Loop ---
    print("--- Starting Training ---")
    total_training_time = 0
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader: # y_batch shape: (batch, 150, 1)
            X_batch, y_batch = X_batch.to(device, non_blocking=pin_memory), y_batch.to(device, non_blocking=pin_memory)

            # Forward pass - Model outputs the full 150 steps
            y_pred = model(X_batch) # y_pred shape: (batch, 150, 1)

            # ---> Loss Calculation Uses Full Sequences <---
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_epoch_loss:.6f} | Time: {epoch_duration:.2f}s", end="")

        # --- Validation ---
        if (epoch + 1) % validation_frequency == 0 or epoch == n_epochs - 1:
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for X_batch_test, y_batch_test in test_loader: # y_batch_test shape: (batch, 150, 1)
                    X_batch_test, y_batch_test = X_batch_test.to(device, non_blocking=pin_memory), y_batch_test.to(device, non_blocking=pin_memory)
                    y_pred_test = model(X_batch_test) # y_pred_test shape: (batch, 150, 1)
                    # ---> Loss Calculation Uses Full Sequences <---
                    test_loss = loss_fn(y_pred_test, y_batch_test)
                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0
            # RMSE is calculated over all 150*N prediction points in the test set
            test_rmse = np.sqrt(avg_test_loss) if avg_test_loss >= 0 else 0
            print(f" | Test RMSE (over {forecast_horizon} steps): {test_rmse:.6f}")
        else:
            print()

    print("-" * 30)
    print(f"--- Training Finished --- Total Time: {total_training_time:.2f}s ---")
    print("-" * 30)

    # --- Generate Forecast for Plotting ---
    print(f"Generating direct {forecast_horizon}-step forecast example...")
    model.eval()

    # Use the last available input sequence from the training data to start forecast
    # Or choose a specific point from the test set for visualization
    forecast_start_index = train_size # Start forecasting right after training data
    if forecast_start_index < max_lookback_offset:
        print("Warning: Not enough data before forecast start index for full lookback.")
        # Handle this case, e.g., choose a later start index or exit
        forecast_start_index = max_lookback_offset

    # Construct the single input sequence needed
    input_indices = [(forecast_start_index - 1) - offset + 1 for offset in required_offsets]
    input_sequence_np = timeseries[input_indices].reshape(1, input_seq_len, 1) # Batch size 1
    input_sequence_tensor = torch.tensor(input_sequence_np, dtype=torch.float32).to(device)

    # Get the direct forecast
    with torch.no_grad():
        direct_forecast_output = model(input_sequence_tensor) # Shape: (1, 150, 1)

    # Move to CPU and flatten for plotting
    direct_forecast_plot = direct_forecast_output.cpu().numpy().flatten()
    print(f"Generated direct forecast of length {len(direct_forecast_plot)}.")
    print("-" * 30)


    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(18, 8))

    # Original Data
    plt.plot(np.arange(len(timeseries)), timeseries.flatten(), label='Original Data', alpha=0.7, linewidth=1.0, color='blue')

    # Direct Forecast Plot
    # Time indices start right after the input sequence used
    future_time_indices = np.arange(forecast_start_index, forecast_start_index + forecast_horizon)
    plt.plot(future_time_indices, direct_forecast_plot,
             label=f'Direct Forecast ({forecast_horizon} steps)', linewidth=1.5, color='red')

    # Optionally plot actual test data for comparison over the forecast horizon
    actual_start_index = forecast_start_index
    actual_end_index = actual_start_index + forecast_horizon
    if actual_end_index <= len(timeseries): # Check if we have ground truth
        actual_data_plot = timeseries[actual_start_index:actual_end_index].flatten()
        plt.plot(future_time_indices, actual_data_plot,
                 label='Actual Data (Test Set)', linewidth=1.2, linestyle='--', color='green', alpha=0.8)


    # Plot Details
    plt.title(f"LSTM Direct Multi-Step Forecasting (InputSeq={input_seq_len}, Horizon={forecast_horizon})")
    plt.xlabel("Time Step")
    plt.ylabel("Z Velocity")
    plt.axvline(x=train_size, color='grey', linestyle=':', linewidth=1.5, label=f'Train/Test Split ({test_split_ratio*100:.0f}%)')
    # Mark forecast start
    plt.axvline(x=forecast_start_index, color='orange', linestyle=':', linewidth=1.5, label='Forecast Start')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    plot_start_index = max(0, forecast_start_index - max_lookback_offset - 50) # Show history
    plot_end_index = forecast_start_index + forecast_horizon + int(0.1 * forecast_horizon) # Show forecast + buffer
    plt.xlim(plot_start_index, plot_end_index)

    plt.tight_layout()
    plt.show()

    print("--- Script Finished ---")