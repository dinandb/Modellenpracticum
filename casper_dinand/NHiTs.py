import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS # LSTM, NBEATS were imported in original but not used for chosen model
from datetime import datetime
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 8,
    'pgf.rcfonts': False,
    'text.usetex': True,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'lines.linewidth' : 0.5,
     'lines.markersize'  : 5,
    'xtick.labelsize' : 8,
    'ytick.labelsize': 8})

# --- Configuration ---
# SET THIS FLAG: True to train and save, False to load and plot
TRAIN_MODEL = False  # Change to False to load a pre-trained model
MODEL_INDEX = 13
MODEL_SAVE_PATH =  f"./nhits_heave_model{MODEL_INDEX}"# Relative path for saving/loading the model

# --- (Original Data Loading and Preprocessing) ---
# Ensure this file path is correct for your system
file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\5415M_Hs=4m_Tp=10s_10h_clean.csv"
try:
    df_raw = pd.read_csv(file_path, header=[0,1])
except FileNotFoundError:
    print(f"Error: Data file not found at {file_path}. Please check the path.")
    exit()

heave_series = np.array(df_raw['z_wf'].values.flatten())
time_series = np.array(df_raw['t'].values.flatten()) # Used for length to create 'ds'

# Create DataFrame for NeuralForecast
# 'ds' should be sequential integers for freq=1
heave_df = pd.DataFrame({
    'unique_id': 'heave', # Single time series identifier
    'ds': np.arange(len(time_series)), # Time steps from 0 to N-1
    'y': heave_series # Target variable
})

# --- Model Configuration ---
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR) # Suppress verbose PyTorch Lightning logs

horizon = 150 # Forecast horizon: number of steps to predict
# NHITS model parameters
nhits_params = {
    'h': horizon,
    'input_size': 328, # Crucial: model looks back this many time steps for context
    'learning_rate': 0.001,
    'max_steps': 5000, # Number of training steps
    'dropout_prob_theta': 0,
    'batch_size': 64,
    'n_blocks': [2, 2, 2],
    'n_freq_downsample': [3, 2, 1],
    'accelerator': "gpu", # Change to "cpu" if no GPU is available or if issues arise
    'random_seed': 42 # For reproducibility of training
}

# --- Data Splitting ---
# Using the same split point as the original script for training
TRAIN_SAMPLES = 160000
if TRAIN_SAMPLES <= nhits_params['input_size']:
    raise ValueError(
        f"TRAIN_SAMPLES ({TRAIN_SAMPLES}) must be greater than model's input_size ({nhits_params['input_size']}) "
        "to ensure enough historical data for the first training step."
    )

train_df = heave_df.iloc[:TRAIN_SAMPLES]
# The rest of the data can be considered for validation/testing plots
validation_df_full = heave_df.iloc[TRAIN_SAMPLES:]


# --- Model Training or Loading ---
# Initialize NeuralForecast object. 
# If TRAIN_MODEL is True, models list with nhits_params will be used.
# If TRAIN_MODEL is False, nf will be overwritten by NeuralForecast.load().
current_models = [NHITS(**nhits_params)]
nf = NeuralForecast(models=current_models, freq=1) # freq=1 for integer 'ds' steps

if TRAIN_MODEL:
    print(f"Training model for {nhits_params['max_steps']} steps...")
    nf.fit(df=train_df) # val_size=0 can be added if you don't want fit to use internal validation
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    nf.save(MODEL_SAVE_PATH)
    print("Model trained and saved.")
else:
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    try:
        # NeuralForecast.load re-instantiates the model object with its saved configuration.
        nf = NeuralForecast.load(path=MODEL_SAVE_PATH)
        print("Model loaded successfully.")
        
        # Update horizon and input_size from the loaded model to ensure consistency for plotting
        # Assuming one model in the list, which is NHITS
        if nf.models and len(nf.models) > 0:
            loaded_model_params = nf.models[0]
            horizon = loaded_model_params.h # Use horizon from loaded model
            # nhits_params['input_size'] should ideally match loaded_model_params.input_size
            # For plotting, we will use loaded_model_params.input_size
            print(f"Loaded model parameters: horizon={horizon}, input_size={loaded_model_params.input_size}")
        else:
            print("Error: No models found in the loaded NeuralForecast object.")
            exit()

    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. "
              "Please train the model first by setting TRAIN_MODEL = True.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

# --- Plot 1: Forecast immediately after training data (Corrected) ---
# This plot shows how the model predicts the period immediately following its training data.
print("\n--- Generating Plot 1: Forecast immediately after training data ---")

# predict() will forecast `horizon` steps after the last data point nf has processed
# (either from fit() or from a `df` passed to a previous predict() call).
forecast_after_train = nf.predict()

# Get the actual values corresponding to this forecast period
actual_ds_start = forecast_after_train['ds'].min()
actual_ds_end = forecast_after_train['ds'].max()

actual_data_for_plot1 = heave_df[
    (heave_df['ds'] >= actual_ds_start) & (heave_df['ds'] <= actual_ds_end)
].copy()

if len(actual_data_for_plot1) == horizon:
    plt.figure(figsize=(14, 7))
    plt.plot(actual_data_for_plot1['ds'], actual_data_for_plot1['y'], label='Actual Heave', color='blue')
    plt.plot(forecast_after_train['ds'], forecast_after_train['NHITS'], label='NHITS Forecast', color='red', linestyle='--', marker='x', markersize=4)
    plt.legend()
    plt.xlabel('Time Step (ds)')
    plt.ylabel('Heave (z_wf)')
    plt.title(f'{horizon}-Step Forecast vs Actuals (Immediately After Training Data)\nTarget: ds {actual_ds_start} to {actual_ds_end}')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print(f"Warning for Plot 1: Expected {horizon} actual data points, but found {len(actual_data_for_plot1)}. Plotting may be misleading or skipped.")


# --- Plot 2: Multiple examples from the validation set ---
print("\n--- Generating Plot 2: Multiple forecast examples from the validation set ---")
num_validation_plots = 3
# Get input_size from the model (important if model was loaded)
current_input_size = nf.models[0].input_size 

# Check if validation_df_full is long enough to provide any forecast windows
if len(validation_df_full) < horizon:
     print(f"Validation set (length {len(validation_df_full)}) is shorter than horizon ({horizon}). Cannot generate validation plots.")
else:
    # Potential start indices for the *actuals* part of the plot, *within* validation_df_full
    # We need to ensure there are `horizon` points available from this start index.
    # (max index in validation_df_full is len(validation_df_full) - 1)
    # So, latest start for an actuals window of length `horizon` is `len(validation_df_full) - horizon`
    max_start_index_in_val = len(validation_df_full) - horizon 
    
    if max_start_index_in_val < 0: # Should be caught by the len(validation_df_full) < horizon check already
        print(f"Not enough data in validation_df_full to plot even one window of size {horizon}.")
    else:
        num_possible_plots = max_start_index_in_val + 1
        actual_num_plots_to_make = min(num_validation_plots, num_possible_plots)

        if actual_num_plots_to_make < num_validation_plots:
            print(f"Warning: Can only make {actual_num_plots_to_make} validation plots due to validation set size "
                  f"(requested {num_validation_plots}, possible {num_possible_plots}).")

        if actual_num_plots_to_make > 0:
            # Generate random start indices for plotting windows within the validation set
            # These indices are relative to the start of `validation_df_full`
            plot_start_indices_in_val_df = np.random.choice(
                np.arange(max_start_index_in_val + 1), 
                size=actual_num_plots_to_make, 
                replace=False # Ensure unique plot windows
            )
            plot_start_indices_in_val_df = np.sort(plot_start_indices_in_val_df) # Plot in chronological order

            for i, val_idx_start_actuals in enumerate(plot_start_indices_in_val_df):
                # Determine the actual 'ds' value where this validation forecast begins
                # This ds value is from the global heave_df['ds'] series
                first_ds_to_predict = validation_df_full['ds'].iloc[val_idx_start_actuals]
                # Extract the actual future values for this plot
                actual_values_this_plot = validation_df_full[
                    (validation_df_full['ds'] >= first_ds_to_predict) &
                    (validation_df_full['ds'] < first_ds_to_predict + horizon)
                ].copy()
                
                # Ensure we have exactly `horizon` points for actuals
                if len(actual_values_this_plot) != horizon:
                    print(f"Skipping validation plot {i+1} for ds={first_ds_to_predict}: "
                          f"Insufficient actual data points (expected {horizon}, got {len(actual_values_this_plot)}).")
                    continue
                
                # History for this prediction ends at 'ds' = first_ds_to_predict - 1
                # This history is taken from the *full* heave_df to provide context to the model
                history_for_this_prediction = heave_df[heave_df['ds'] < first_ds_to_predict].copy()
                
                if len(history_for_this_prediction) < current_input_size:
                    print(f"Skipping validation plot {i+1} for ds={first_ds_to_predict}: "
                          f"Insufficient history (need {current_input_size}, got {len(history_for_this_prediction)}).")
                    continue

                print(f"Generating validation plot {i+1}/{actual_num_plots_to_make} (forecasting from ds={first_ds_to_predict})...")
                # Predict using the historical data up to this point
                # nf.predict will use the tail of history_for_this_prediction (last input_size points)
                prediction_this_plot = nf.predict(df=history_for_this_prediction)
                fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.0))
                plt.plot(actual_values_this_plot['ds'], actual_values_this_plot['y'], label='Actual Heave', color='blue')
                plt.plot(prediction_this_plot['ds'], prediction_this_plot['NHITS'], label='NHITS Forecast', color='red', linestyle='--', marker='x', markersize=4)
                plt.legend()
                plt.xlabel('Time Step (ds)')
                plt.ylabel('Heave (m)')
                plt.title(f'Validation Forecast ds {first_ds_to_predict} to {first_ds_to_predict + horizon - 1})')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.25, left = 0.15, top=0.75)
                fig.set_size_inches(w=5.5, h=2.0)
                plt.savefig(r'C:\Users\caspe\OneDrive\Documents\GitHub\Modellenpracticum\casper_dinand\solution_' + f"{i}"  + '.pgf')
        else:
            print("No validation plots could be generated due to insufficient data after the training split point.")


print("\nScript finished.")