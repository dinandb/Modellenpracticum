# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split # Manual split
import os
import sys
import time
import datetime
import joblib
import multiprocessing
import matplotlib.pyplot as plt
import random

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

# --- Execution Mode ---
TRAINING = False # Set to True to train, False to load and predict/test

# --- File Paths ---
CSV_PATH = 'C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\CleanQP_data_36000.csv'
SAVED_MODELS_DIR = './saved_models_dual_multi_zeta'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ---> Timestamp for Loading (Only used if TRAINING = False) <---
LOAD_RUN_TIMESTAMP = "20250515_205224" # <<< EXAMPLE: From your error message <<<

# --- Data Parameters ---
TARGET_COLUMNS = ['z_wf', 'y_wf', 'x_wf', 'phi_wf', 'theta_wf', 'psi_wf']
INPUT_ONLY_COLUMNS = ['zeta']
EXTREMA_DETECTION_COLUMN = 'z_wf'
VALIDATION_SPLIT = 0.2

# --- Derived Data Parameters ---
ALL_INPUT_COLUMNS = TARGET_COLUMNS + INPUT_ONLY_COLUMNS
N_TARGET_FEATURES = len(TARGET_COLUMNS)
N_ALL_FEATURES = len(ALL_INPUT_COLUMNS)

# --- Model 1 Hyperparameters (Defaults, will be overridden if loading) ---
EXTREMA_SEQ_LEN = 3
M1_INPUT_FEATURES = 1 + N_TARGET_FEATURES
M1_OUTPUT_FEATURES = 1 + N_TARGET_FEATURES
EXTREMA_HIDDEN_SIZE = 16
EXTREMA_NUM_LAYERS = 2
EXTREMA_DROPOUT = 0.2
EXTREMA_LR = 0.005; EXTREMA_WEIGHT_DECAY = 0; EXTREMA_EPOCHS = 1000
EXTREMA_BATCH_SIZE = 256; EXTREMA_PATIENCE = 100

# --- Model 2 Hyperparameters (Defaults, will be overridden if loading) ---
SERIES_INPUT_LEN_M2 = 20; SERIES_PRED_HORIZON = 50
M2_INPUT_FEATURES = N_ALL_FEATURES + 1 + N_TARGET_FEATURES
M2_OUTPUT_FEATURES = N_TARGET_FEATURES
SERIES_HIDDEN_SIZE = 96
SERIES_NUM_LAYERS = 3
SERIES_DROPOUT = 0.15
SERIES_LR = 0.001; SERIES_WEIGHT_DECAY = 0; SERIES_EPOCHS = 250
SERIES_BATCH_SIZE = 256; SERIES_PATIENCE = 30

# --- Prediction Parameters (Used if TRAINING = False) ---
N_PREDICTION_EXTREMA = 7; N_RANDOM_SAMPLES = 4; PLOT_PAST_WINDOW_SIZE = 100

# --- General Training/System Parameters ---
NUM_WORKERS = 4; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = True if DEVICE == torch.device("cuda") else False

# ==============================================================================
#                             HELPER FUNCTIONS & CLASSES
# ==============================================================================
# (Dataset classes, Model classes, Data Processing functions, find_split_indices, train_model - KEEP THESE IDENTICAL TO THE PREVIOUS CORRECTED VERSION)
# --- Dataset Classes ---
class SeriesDatasetMulti(Dataset):
    def __init__(self, x_series, y_series):
        self.x_series=torch.tensor(x_series,dtype=torch.float32); self.y_series=torch.tensor(y_series,dtype=torch.float32)
    def __len__(self): return len(self.x_series)
    def __getitem__(self, idx): return self.x_series[idx], self.y_series[idx]

# --- Model Classes ---
class ExtremumPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=7, dropout_rate=0.2): # Default hidden_size used if not in params
        super().__init__(); self.input_size = input_size; self.hidden_size = hidden_size; self.num_layers = num_layers
        lstm_dropout = dropout_rate if num_layers > 1 and dropout_rate > 0 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout); self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, ext_seq):
        h0 = torch.zeros(self.num_layers, ext_seq.size(0), self.hidden_size).to(ext_seq.device); c0 = torch.zeros(self.num_layers, ext_seq.size(0), self.hidden_size).to(ext_seq.device)
        lstm_out, _ = self.lstm(ext_seq, (h0, c0)); last_lstm_hidden_state = lstm_out[:, -1, :]; prediction = self.fc(last_lstm_hidden_state); return prediction

class SeriesPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=3, output_horizon=50, output_features=6, dropout_rate=0.15): # Default hidden_size
        super().__init__(); self.hidden_size = hidden_size; self.num_layers = num_layers; self.output_horizon = output_horizon; self.output_features = output_features
        lstm_dropout = dropout_rate if num_layers > 1 and dropout_rate > 0 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout); self.fc = nn.Linear(hidden_size, output_horizon * output_features)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device); c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ =self.lstm(x, (h0, c0)); last_time_step_out=lstm_out[:, -1, :]
        out_flat=self.fc(last_time_step_out); out=out_flat.view(x.size(0), self.output_horizon, self.output_features); return out

# --- Data Processing Functions ---
def load_data_and_extrema(csv_path, all_input_columns, target_columns, extrema_detection_column):
    print("Loading multi-variate data...");
    try:
        df = pd.read_csv(csv_path, header=[0, 1]); timeseries_full_list = []; actual_columns_found = []
        target_indices_in_all = [] ; all_col_names_ordered = []
        for i, col_name in enumerate(all_input_columns):
             found = False
             for col_tuple in df.columns:
                 if col_name in str(col_tuple[0]).lower() or col_name in str(col_tuple[1]).lower():
                     timeseries_full_list.append(df[col_tuple].values.astype('float32').reshape(-1, 1))
                     actual_columns_found.append(col_tuple); all_col_names_ordered.append(col_name)
                     if col_name in target_columns: target_indices_in_all.append(i)
                     found = True; break
             if not found: raise ValueError(f"Col '{col_name}' not found.")
        timeseries_full = np.hstack(timeseries_full_list);
        if np.isnan(timeseries_full).any(): raise ValueError("NaNs in data!")
        print(f"Loaded cols: {actual_columns_found}"); print(f"Loaded shape: {timeseries_full.shape}"); time_indices_full = np.arange(len(timeseries_full)).astype('float32').reshape(-1, 1)
        try: extrema_col_idx_in_all = all_col_names_ordered.index(extrema_detection_column)
        except ValueError: raise ValueError(f"Extrema col '{extrema_detection_column}' not in loaded.")
        print(f"Extracting extrema based on: {extrema_detection_column} (idx {extrema_col_idx_in_all})..."); extrema_signal = timeseries_full[:, extrema_col_idx_in_all]
        peaks_indices, _ = find_peaks(extrema_signal, distance=5); troughs_indices, _ = find_peaks(-extrema_signal, distance=5)
        extrema_indices = np.sort(np.unique(np.concatenate([peaks_indices, troughs_indices])));
        if len(extrema_indices) > 1: extrema_indices = extrema_indices[np.insert(np.diff(extrema_indices) > 1, 0, True)]
        extrema_times = time_indices_full[extrema_indices]; extrema_values_all_features = timeseries_full[extrema_indices, :]
        extrema_data_full = np.hstack((extrema_times, extrema_values_all_features)); print(f"Found {len(extrema_data_full)} extrema.")
    except FileNotFoundError: print(f"Error: CSV not found: {csv_path}"); sys.exit(1)
    except Exception as e: print(f"Error loading/processing data: {e}"); sys.exit(1)
    return timeseries_full, time_indices_full, extrema_data_full, all_col_names_ordered, target_indices_in_all

def create_sequences_m1_multi(extrema_data, seq_length, target_indices_in_all):
    xs = []; ys = []; min_req = seq_length + 1; n_target_features = len(target_indices_in_all)
    if len(extrema_data) < min_req: return np.array([]), np.array([])
    value_indices_to_use = [idx + 1 for idx in target_indices_in_all]
    for i in range(seq_length, len(extrema_data)):
        x_seq_times = extrema_data[i - seq_length : i, 0:1]; x_seq_target_vals = extrema_data[i - seq_length : i, value_indices_to_use]
        x_seq_unprocessed = np.hstack((x_seq_times, x_seq_target_vals)); xs.append(x_seq_unprocessed)
        y_target_unprocessed = extrema_data[i]; last_time = x_seq_unprocessed[-1, 0]; target_time_diff = y_target_unprocessed[0] - last_time
        target_values = y_target_unprocessed[value_indices_to_use]; target = np.concatenate(([target_time_diff], target_values)); ys.append(target)
    if not xs: return np.array([]), np.array([])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def process_and_scale_m1_data_multi(sequences_unscaled, targets_unscaled, scaler_ext_time_diff, scaler_values):
    processed_ext_scaled = []; processed_tgt_scaled = []
    if sequences_unscaled.ndim != 3 or targets_unscaled.ndim != 2: return np.array([]), np.array([])
    seq_len = sequences_unscaled.shape[1]; n_target_features = sequences_unscaled.shape[2] - 1
    if n_target_features <= 0: print("Error: M1 scaling no target features."); return np.array([]), np.array([])
    for i in range(len(sequences_unscaled)):
        ext_seq = sequences_unscaled[i]; target = targets_unscaled[i]
        try:
            ext_proc = np.zeros((seq_len, 1 + n_target_features), dtype=np.float32)
            if seq_len > 1: ext_proc[1:, 0] = np.diff(ext_seq[:, 0]); ext_proc[0, 0] = 0; ext_proc[:, 1:] = ext_seq[:, 1:]
            ext_proc_scaled = ext_proc.copy()
            if seq_len > 1 and ext_proc[1:, 0].size > 0: ext_proc_scaled[1:, 0] = scaler_ext_time_diff.transform(ext_proc[1:, 0].reshape(-1, 1)).flatten()
            values_to_scale = ext_proc[:, 1:].reshape(-1, n_target_features); scaled_values = scaler_values.transform(values_to_scale); ext_proc_scaled[:, 1:] = scaled_values.reshape(seq_len, n_target_features)
            processed_ext_scaled.append(ext_proc_scaled)
            target_scaled = np.zeros_like(target)
            target_scaled[0] = scaler_ext_time_diff.transform(np.array([[target[0]]]))[0, 0]
            target_values_to_scale = target[1:].reshape(1, -1); scaled_target_values = scaler_values.transform(target_values_to_scale); target_scaled[1:] = scaled_target_values.flatten()
            processed_tgt_scaled.append(target_scaled)
        except Exception as e: print(f"Warn: Error scaling M1 multi sample idx {i}: {e}. Skip."); min_len=min(len(processed_ext_scaled), len(processed_tgt_scaled)); processed_ext_scaled=processed_ext_scaled[:min_len]; processed_tgt_scaled=processed_tgt_scaled[:min_len]; continue
    if not processed_ext_scaled or not processed_tgt_scaled: return np.array([]), np.array([])
    final_len = min(len(processed_ext_scaled), len(processed_tgt_scaled)); return np.array(processed_ext_scaled[:final_len]), np.array(processed_tgt_scaled[:final_len])

def create_series_sequences_multi(raw_data_scaled, extrema_info_unscaled, input_len, pred_horizon, last_known_extremum_raw_indices, scaler_ext_time_diff, scaler_values, target_indices_in_all):
    xs = []; ys = []; target_ext_info_scaled_out = []
    num_samples = len(extrema_info_unscaled); n_all_features = raw_data_scaled.shape[1]; n_target_features = len(target_indices_in_all)
    if last_known_extremum_raw_indices is None or len(last_known_extremum_raw_indices) != num_samples: print("Warn: Mismatch last known M2 multi"); return np.array([]), np.array([]), np.array([])
    for i in range(num_samples):
        target_extremum_props_m1 = extrema_info_unscaled[i]; last_known_extremum_index_in_raw = last_known_extremum_raw_indices[i]
        start_idx = last_known_extremum_index_in_raw - input_len + 1; end_idx = last_known_extremum_index_in_raw + 1
        if start_idx < 0 or end_idx > len(raw_data_scaled): continue
        x_series_raw_all = raw_data_scaled[start_idx : end_idx];
        if x_series_raw_all.shape[0] != input_len: continue
        pred_end_idx = end_idx + pred_horizon; y_series_targets = raw_data_scaled[end_idx : pred_end_idx, target_indices_in_all]
        actual_horizon = y_series_targets.shape[0];
        if actual_horizon == 0 and pred_horizon > 0: continue
        if actual_horizon < pred_horizon: padding_shape = (pred_horizon - actual_horizon, n_target_features); padding = np.full(padding_shape, -10.0, dtype=np.float32); y_series_targets = np.vstack((y_series_targets, padding))
        elif actual_horizon > pred_horizon: y_series_targets = y_series_targets[:pred_horizon]
        if y_series_targets.shape[0] != pred_horizon: continue
        target_info_scaled_sample = np.zeros(1 + n_target_features)
        try:
            target_info_scaled_sample[0] = scaler_ext_time_diff.transform(np.array([[target_extremum_props_m1[0]]]))[0, 0]
            target_values_to_scale = target_extremum_props_m1[1:].reshape(1, -1); scaled_target_values = scaler_values.transform(target_values_to_scale); target_info_scaled_sample[1:] = scaled_target_values.flatten()
        except Exception as e: print(f"Warn: Scaling M2 target info failed: {e}"); continue
        scaled_target_info_repeated = np.tile(target_info_scaled_sample, (input_len, 1))
        x_combined = np.hstack((x_series_raw_all, scaled_target_info_repeated))
        xs.append(x_combined); ys.append(y_series_targets); target_ext_info_scaled_out.append(target_info_scaled_sample)
    if not xs: return np.array([]), np.array([]), np.array([])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(target_ext_info_scaled_out, dtype=np.float32)

def find_split_indices(extrema_data, raw_data_len, validation_split_ratio, extrema_seq_len):
    n_ext_sequences = len(extrema_data) - extrema_seq_len;
    if n_ext_sequences <= 0: raise ValueError("Not enough extrema for sequences.");
    n_test_sequences = int(n_ext_sequences * validation_split_ratio); n_train_sequences = n_ext_sequences - n_test_sequences
    first_test_seq_last_input_idx = n_train_sequences + extrema_seq_len - 1
    if first_test_seq_last_input_idx >= len(extrema_data): print("Warn: Split index OOB. Fallback ratio."); split_index_raw = int(raw_data_len * (1 - validation_split_ratio)); split_index_extrema = np.searchsorted(extrema_data[:, 0], split_index_raw); return split_index_raw, split_index_extrema
    split_time_index_raw = int(extrema_data[first_test_seq_last_input_idx, 0]) + 1; split_index_extrema = first_test_seq_last_input_idx + 1; return split_time_index_raw, split_index_extrema

# --- Training Function ---
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
                epochs, patience, device, amp_enabled, best_model_path,
                model_params_to_save=None):
    print(f"\n--- Training {model_name} ---"); best_val_loss = float('inf'); best_epoch = 0; epochs_no_improve = 0; training_successful = False; last_train_loss = float('nan')
    if not train_loader or len(train_loader) == 0: print(f"Error: Train DataLoader empty."); return False, best_val_loss, best_epoch
    for epoch in range(epochs):
        model.train(); epoch_train_loss = 0.0; start_time = time.time(); processed_batches = 0; valid_loss_count_train_m2 = 0; epoch_train_loss_m2_accum = 0.0; num_valid_scalar = 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_success = False
            try:
                if model_name == "Model 1": batch_ext, batch_targets = batch_data; batch_ext=batch_ext.to(device,non_blocking=True)
                elif model_name == "Model 2": batch_features, batch_targets = batch_data; batch_features=batch_features.to(device,non_blocking=True)
                else: raise ValueError(f"Unknown model_name: {model_name}")
                batch_targets = batch_targets.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp_enabled):
                    if model_name == "Model 1": outputs = model(batch_ext)
                    else: outputs = model(batch_features)
                    if model_name == "Model 2":
                        mask = (batch_targets != -10.0).float(); loss_per_element = criterion(outputs, batch_targets)
                        if torch.isnan(loss_per_element).any(): raise ValueError("NaN loss_per_element")
                        masked_loss = loss_per_element * mask; total_loss = masked_loss.sum(); num_valid = mask.sum()
                        loss = total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=device, requires_grad=True); num_valid_scalar = num_valid.item()
                    else: loss = criterion(outputs, batch_targets); num_valid_scalar = batch_targets.numel()
                loss_item = loss.item()
                if not torch.isfinite(loss): raise ValueError(f"Invalid loss ({loss_item})")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); scaler.step(optimizer); scaler.update()
                batch_success = True
            except Exception as e: print(f"\n!!! Error processing {model_name} Train Batch {batch_idx} E{epoch+1}: {e}. Skip."); optimizer.zero_grad(set_to_none=True); continue
            if batch_success:
                if model_name == "Model 2":
                     if num_valid_scalar > 0: epoch_train_loss_m2_accum += loss_item * num_valid_scalar; valid_loss_count_train_m2 += num_valid_scalar
                else: epoch_train_loss += loss_item
                processed_batches += 1
        if processed_batches == 0: print(f"!!! Stop {model_name}: No batches processed E{epoch+1}. !!!"); break
        if model_name == "Model 2": avg_epoch_loss = epoch_train_loss_m2_accum / valid_loss_count_train_m2 if valid_loss_count_train_m2 > 0 else float('nan')
        else: avg_epoch_loss = epoch_train_loss / processed_batches if processed_batches > 0 else float('nan')
        last_train_loss = avg_epoch_loss
        if np.isnan(avg_epoch_loss): print(f"!!! Stop {model_name}: Avg train loss NaN E{epoch+1}. !!!"); break
        epoch_val_loss = float('nan')
        if val_loader:
            model.eval(); epoch_val_loss_accum = 0.0; val_batches = 0; valid_loss_count_val = 0
            with torch.no_grad():
                for batch_idx_val, batch_data_val in enumerate(val_loader):
                    try:
                        if model_name == "Model 1": batch_ext_val, batch_targets_val = batch_data_val; batch_ext_val=batch_ext_val.to(device,non_blocking=True)
                        else: batch_features_val, batch_targets_val = batch_data_val; batch_features_val=batch_features_val.to(device,non_blocking=True)
                        batch_targets_val = batch_targets_val.to(device, non_blocking=True)
                        with autocast(enabled=amp_enabled):
                            if model_name == "Model 1": outputs_val = model(batch_ext_val)
                            else: outputs_val = model(batch_features_val)
                            if model_name == "Model 2":
                                mask_val = (batch_targets_val != -10.0).float()
                                if torch.isnan(outputs_val).any() or torch.isnan(batch_targets_val).any(): raise ValueError("NaN input/target")
                                loss_per_element_val = criterion(outputs_val, batch_targets_val);
                                if torch.isnan(loss_per_element_val).any(): raise ValueError("NaN loss_per_element")
                                masked_loss_val = loss_per_element_val * mask_val; total_loss_val = masked_loss_val.sum(); num_valid_val = mask_val.sum()
                                if num_valid_val > 0: loss_val = total_loss_val / num_valid_val
                                else: loss_val = torch.tensor(float('nan'))
                            else:
                                if not torch.isnan(outputs_val).any() and not torch.isnan(batch_targets_val).any(): loss_val = criterion(outputs_val, batch_targets_val)
                                else: loss_val = torch.tensor(float('nan'))
                            if torch.isfinite(loss_val):
                                 if model_name == "Model 2": epoch_val_loss_accum += loss_val.item() * num_valid_val.item(); valid_loss_count_val += num_valid_val.item()
                                 else: epoch_val_loss_accum += loss_val.item(); val_batches += 1
                    except Exception as e: print(f"\n!!! Error processing {model_name} Val Batch {batch_idx_val} E{epoch+1}: {e}. Skip."); continue
            if model_name == "Model 2": epoch_val_loss = epoch_val_loss_accum / valid_loss_count_val if valid_loss_count_val > 0 else float('nan')
            else: epoch_val_loss = epoch_val_loss_accum / val_batches if val_batches > 0 else float('nan')
        end_time = time.time(); val_loss_str = f"{epoch_val_loss:.6f}" if not np.isnan(epoch_val_loss) else "N/A"
        print(f"Epoch [{epoch+1}/{epochs}] -> {model_name} Train Loss: {avg_epoch_loss:.6f} | Val Loss: {val_loss_str} | Time: {end_time - start_time:.2f}s", end="")
        if not np.isnan(epoch_val_loss):
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss; best_epoch = epoch + 1; training_successful = True; epochs_no_improve = 0
                print(" <<< Improvement found, saving best model...", end="")
                try:
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    save_dict = { 'model_state_dict': model.state_dict(), 'params': model_params_to_save if model_params_to_save is not None else {}, 'training_info': { 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_loss_at_best': last_train_loss, 'save_timestamp': datetime.datetime.now().isoformat() } }
                    tmp_save_path = best_model_path + ".tmp"; torch.save(save_dict, tmp_save_path); os.replace(tmp_save_path, best_model_path); print(" Saved!")
                except Exception as e: print(f"\n!!! Error saving best {model_name}: {e} !!!")
            else:
                epochs_no_improve += 1; print(f" (No imprv: {epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience: print(f"\n--- Early stopping {model_name} at Epoch {epoch+1} ---"); break
        else: print()
    if not training_successful: print(f"\n{model_name} did not improve or failed.")
    else: print(f"\n--- {model_name} Finished (Best E: {best_epoch}, Best Val Loss: {best_val_loss:.6f}) ---")
    return training_successful, best_val_loss, best_epoch

# --- Prediction Function (Corrected) ---
def run_predictions(model1, model2, scaler_ext_time_diff, scaler_values, scaler_raw_all,
                    timeseries_full, extrema_data_test, params1, params2, target_columns, device, amp_enabled,
                    n_samples, n_pred_extrema, all_input_cols_loaded_order, target_indices_in_all_loaded):
    print(f"\n--- Running Predictions for {n_samples} Samples ---"); predictions = []
    extrema_seq_len = params1.get('extrema_seq_len', 15)
    n_target_features = params1.get('n_target_features', len(target_columns))
    n_all_features = params1.get('n_all_features', len(all_input_cols_loaded_order))

    series_input_len_m2 = params2.get('series_input_len', 20)
    series_pred_horizon = params2.get('series_pred_horizon', 50)

    min_sample_idx_in_test = extrema_seq_len - 1; max_sample_idx_in_test = len(extrema_data_test) - 1
    if max_sample_idx_in_test < min_sample_idx_in_test: print("Error: Test set too small."); return []
    random_last_known_indices_in_test = random.sample(range(min_sample_idx_in_test, max_sample_idx_in_test + 1), n_samples)

    for i_sample, last_known_extremum_idx_in_test in enumerate(random_last_known_indices_in_test):
        print(f"\n--- Sample {i_sample+1}/{n_samples} (Test Extrema Idx {last_known_extremum_idx_in_test}) ---")
        history_start_idx_in_test = last_known_extremum_idx_in_test - extrema_seq_len + 1;
        initial_extrema_history = extrema_data_test[history_start_idx_in_test : last_known_extremum_idx_in_test + 1].copy()
        last_known_extremum_time_idx_abs = int(initial_extrema_history[-1, 0]); prediction_start_time_idx_abs = last_known_extremum_time_idx_abs + 1
        raw_hist_start_idx_abs = max(0, last_known_extremum_time_idx_abs - series_input_len_m2 + 1); raw_hist_end_idx_abs = last_known_extremum_time_idx_abs + 1
        if raw_hist_end_idx_abs > len(timeseries_full): print(f"Warn: Raw index OOB. Skip."); continue
        initial_raw_history = timeseries_full[raw_hist_start_idx_abs : raw_hist_end_idx_abs].copy()
        if len(initial_raw_history) < series_input_len_m2: print(f"Warn: Initial raw hist short. Skip."); continue

        predicted_extrema_list = []; predicted_series_segments = [];
        current_extrema_history = initial_extrema_history.copy();
        current_raw_series_input_m2 = initial_raw_history.copy();

        for i_pred_step in range(n_pred_extrema):
            m1_input_ext_proc = np.zeros((extrema_seq_len, 1 + n_target_features), dtype=np.float32)
            current_extrema_history_targets = current_extrema_history[:, [0] + [idx+1 for idx in target_indices_in_all_loaded]]
            if extrema_seq_len > 1: m1_input_ext_proc[1:, 0] = np.diff(current_extrema_history_targets[:, 0])
            m1_input_ext_proc[0, 0] = 0; m1_input_ext_proc[:, 1:] = current_extrema_history_targets[:, 1:]
            m1_input_ext_scaled = m1_input_ext_proc.copy()
            try:
                if extrema_seq_len > 1 and m1_input_ext_scaled[1:, 0].size > 0: m1_input_ext_scaled[1:, 0] = scaler_ext_time_diff.transform(m1_input_ext_scaled[1:, 0].reshape(-1, 1)).flatten()
                if n_target_features > 0 and m1_input_ext_scaled[:, 1:].size > 0: values_to_scale = m1_input_ext_scaled[:, 1:].reshape(-1, n_target_features); m1_input_ext_scaled[:, 1:] = scaler_values.transform(values_to_scale).reshape(extrema_seq_len, n_target_features)
            except Exception as e: print(f"Err scaling M1 ext step {i_pred_step}: {e}"); break
            m1_input_ext_tensor = torch.tensor(m1_input_ext_scaled,dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(enabled=amp_enabled): scaled_pred_m1 = model1(m1_input_ext_tensor)
            scaled_pred_m1_np = scaled_pred_m1.cpu().numpy()
            try:
                unscaled_time_diff = scaler_ext_time_diff.inverse_transform(scaled_pred_m1_np[:, 0:1])[0, 0]; unscaled_target_values = np.zeros(n_target_features)
                if n_target_features > 0 and scaled_pred_m1_np[:, 1:].size > 0: unscaled_target_values = scaler_values.inverse_transform(scaled_pred_m1_np[:, 1:]).flatten()
            except Exception as e: print(f"Err inv scaling M1 step {i_pred_step}: {e}"); break
            last_known_pred_time = current_extrema_history[-1, 0]; predicted_extremum_time_abs = last_known_pred_time + unscaled_time_diff;
            predicted_extrema_list.append(np.concatenate(([predicted_extremum_time_abs], unscaled_target_values)))
            m2_raw_input_scaled = scaler_raw_all.transform(current_raw_series_input_m2); m2_ext_info_scaled_repeated = scaled_pred_m1_np.repeat(series_input_len_m2, axis=0)
            m2_input_combined = np.hstack((m2_raw_input_scaled, m2_ext_info_scaled_repeated)); m2_input_tensor = torch.tensor(m2_input_combined, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(enabled=amp_enabled): scaled_pred_series_m2 = model2(m2_input_tensor)
            scaled_pred_series_m2_np = scaled_pred_series_m2.squeeze(0).cpu().numpy()
            try:
                 full_scaled_output = np.zeros((series_pred_horizon, n_all_features)); full_scaled_output[:, target_indices_in_all_loaded] = scaled_pred_series_m2_np
                 unscaled_pred_series_all = scaler_raw_all.inverse_transform(full_scaled_output)
                 unscaled_pred_series = unscaled_pred_series_all[:, target_indices_in_all_loaded]
            except Exception as e: print(f"Err inv scaling M2 step {i_pred_step}: {e}"); break
            delta_t_steps = max(1, int(np.round(unscaled_time_diff))); steps_to_use = min(delta_t_steps, series_pred_horizon); predicted_segment_target_values = unscaled_pred_series[:steps_to_use]
            start_time_index_segment_abs = last_known_pred_time + 1; end_time_index_segment_abs = start_time_index_segment_abs + steps_to_use; predicted_segment_times_abs = np.arange(start_time_index_segment_abs, end_time_index_segment_abs).reshape(-1,1)
            if len(predicted_segment_times_abs) > 0: predicted_series_segments.append((predicted_segment_times_abs, predicted_segment_target_values))

            new_extremum_all_features = np.zeros(n_all_features); new_extremum_all_features[target_indices_in_all_loaded] = unscaled_target_values
            input_only_indices = [idx for idx in range(n_all_features) if idx not in target_indices_in_all_loaded]

            # --- Construct segment_all_features_for_history ---
            segment_all_features_for_history = np.zeros((steps_to_use, n_all_features))
            if steps_to_use > 0: # Only if there are steps in the predicted segment
                segment_all_features_for_history[:, target_indices_in_all_loaded] = predicted_segment_target_values
                if len(input_only_indices) > 0:
                    last_raw_input_only_vals = current_raw_series_input_m2[-1, input_only_indices]
                    segment_all_features_for_history[:, input_only_indices] = np.tile(last_raw_input_only_vals, (steps_to_use, 1))
            # --- End Construction ---

            if len(predicted_segment_target_values) > 0 and steps_to_use >= delta_t_steps and delta_t_steps > 0:
                segment_time_offset = min(max(0, delta_t_steps - 1), steps_to_use - 1) # delta_t_steps is 1-based count
                new_extremum_all_features_from_segment = segment_all_features_for_history[segment_time_offset, :]
                if len(input_only_indices) > 0: new_extremum_all_features[input_only_indices] = new_extremum_all_features_from_segment[input_only_indices]
            elif len(input_only_indices) > 0:
                last_raw_input_only_vals = current_raw_series_input_m2[-1, input_only_indices]
                new_extremum_all_features[input_only_indices] = last_raw_input_only_vals

            new_extremum_hist = np.array([np.concatenate(([predicted_extremum_time_abs], new_extremum_all_features))]); current_extrema_history = np.vstack((current_extrema_history[1:], new_extremum_hist))
            if len(predicted_segment_target_values) > 0: # Use constructed segment_all_features_for_history
                 combined_raw = np.vstack((current_raw_series_input_m2, segment_all_features_for_history)); current_raw_series_input_m2 = combined_raw[-series_input_len_m2:]

            if predicted_extremum_time_abs > len(timeseries_full)*1.5 or predicted_extremum_time_abs <= last_known_pred_time: print(f"Stop sample {i_sample+1} step {i_pred_step+1}: Unrealistic time."); break
        predictions.append({'last_known_idx_in_test': last_known_extremum_idx_in_test, 'prediction_start_time_abs': prediction_start_time_idx_abs, 'predicted_extrema': predicted_extrema_list, 'predicted_series': predicted_series_segments})
        print(f"Finished sample {i_sample+1}. Predicted {len(predicted_extrema_list)} extrema.")
    return predictions

# --- Plotting Function ---
def plot_predictions(predictions, timeseries_full, time_indices_full, extrema_data_full, target_columns, all_input_cols_loaded_order, n_pred_extrema, past_window):
    print("\nGenerating multi-subplot plot..."); n_samples = len(predictions); n_target_features_to_plot = len(target_columns)
    if n_samples == 0: print("No predictions to plot."); return
    fig, axs = plt.subplots(n_samples, n_target_features_to_plot, figsize=(4 * n_target_features_to_plot, 3.5 * n_samples), sharex=False, sharey=False, squeeze=False)
    target_indices_in_all_loaded = [all_input_cols_loaded_order.index(col) for col in target_columns if col in all_input_cols_loaded_order]
    for i, p in enumerate(predictions):
        if not p['predicted_extrema']:
            for j_plot_idx in range(n_target_features_to_plot): axs[i, j_plot_idx].set_title(f"Sample {i+1}/{target_columns[j_plot_idx]}-Fail", fontsize=10); axs[i, j_plot_idx].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axs[i, j_plot_idx].transAxes); continue
        start_time = p['prediction_start_time_abs']; max_pred_time = p['predicted_extrema'][-1][0] if p['predicted_extrema'] else start_time
        min_plot = max(0, int(start_time - past_window)); max_plot = min(len(timeseries_full), int(max_pred_time + past_window * 0.2))
        if max_plot <= min_plot: max_plot = min_plot + past_window * 2
        pred_ext_array = np.array(p['predicted_extrema']) if p['predicted_extrema'] else np.empty((0, 1 + n_target_features_to_plot))
        for j_plot_idx in range(n_target_features_to_plot):
            ax = axs[i, j_plot_idx]; current_target_col_name = target_columns[j_plot_idx]; actual_col_idx_in_all = all_input_cols_loaded_order.index(current_target_col_name)
            time_mask_indices = np.where((time_indices_full.flatten() >= min_plot) & (time_indices_full.flatten() < max_plot))[0]
            if len(time_mask_indices)>0: ax.plot(time_indices_full[time_mask_indices].flatten(), timeseries_full[time_mask_indices, actual_col_idx_in_all].flatten(), label='Ground Truth', color='gray', lw=1)
            extrema_mask = (extrema_data_full[:, 0] >= min_plot) & (extrema_data_full[:, 0] < max_plot);
            ax.scatter(extrema_data_full[extrema_mask, 0], extrema_data_full[extrema_mask, actual_col_idx_in_all + 1], marker='x', color='k', label='Actual Extrema', s=30, zorder=5)
            color = plt.cm.viridis(i / max(1, n_samples - 1)); first_seg = True
            for t_s, v_s_target_features in p['predicted_series']:
                 seg_mask = (t_s >= min_plot) & (t_s < max_plot)
                 if np.any(seg_mask): ax.plot(t_s[seg_mask].flatten(), v_s_target_features[seg_mask.flatten(), j_plot_idx].flatten(), color=color, lw=1.5, ls='--', label='Pred. Series' if first_seg and j_plot_idx==0 else None); first_seg=False
            pred_ext_mask = (pred_ext_array[:, 0] >= min_plot) & (pred_ext_array[:, 0] < max_plot)
            if np.any(pred_ext_mask): ax.scatter(pred_ext_array[pred_ext_mask, 0], pred_ext_array[pred_ext_mask, j_plot_idx + 1], marker='o', facecolors='none', edgecolors=color, label='Pred. Extrema' if j_plot_idx==0 else None, s=40, zorder=6, lw=1.2)
            ax.set_title(f"Sample {i+1} - {current_target_col_name} (Start t={start_time:.0f})", fontsize=10); ax.set_xlim(min_plot, max_plot); ax.grid(True, ls=':', alpha=0.6)
            if j_plot_idx == 0: ax.set_ylabel(f'Sample {i+1}\nValue')
            if i == n_samples - 1: ax.set_xlabel('Time Step')
            if i == 0 and j_plot_idx == n_target_features_to_plot -1 : ax.legend(loc='best', fontsize='small')
    fig.suptitle(f'Multi-Variate Iterative Prediction ({n_pred_extrema} Extrema Steps)', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

# ==============================================================================
#                             MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    actual_num_workers = NUM_WORKERS
    if sys.platform == "win32" and NUM_WORKERS > 0: actual_num_workers = 0
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    print(f"--- Run ID: {run_timestamp} ---")
    timeseries_full, time_indices_full, extrema_data_full, all_columns_loaded, target_indices_in_all = load_data_and_extrema(CSV_PATH, ALL_INPUT_COLUMNS, TARGET_COLUMNS, EXTREMA_DETECTION_COLUMN)
    N_ALL_FEATURES = timeseries_full.shape[1]; N_TARGET_FEATURES = len(target_indices_in_all)
    M1_INPUT_FEATURES = 1 + N_TARGET_FEATURES; M1_OUTPUT_FEATURES = 1 + N_TARGET_FEATURES
    M2_INPUT_FEATURES = N_ALL_FEATURES + 1 + N_TARGET_FEATURES; M2_OUTPUT_FEATURES = N_TARGET_FEATURES
    if len(extrema_data_full) < EXTREMA_SEQ_LEN + 1: print("Error: Not enough overall extrema."); sys.exit(1)
    print(f"N Target Features: {N_TARGET_FEATURES}, N All Features: {N_ALL_FEATURES}"); print(f"Target Indices in All Loaded: {target_indices_in_all}")

    if TRAINING:
        print("\n=== TRAINING MODE ==="); current_best_model1_path = os.path.join(SAVED_MODELS_DIR, f'model1_extremum_multi_best_{run_timestamp}.pth'); current_best_model2_path = os.path.join(SAVED_MODELS_DIR, f'model2_series_multi_best_{run_timestamp}.pth'); scaler_ext_time_path = os.path.join(SAVED_MODELS_DIR, f'scaler_ext_time_diff_{run_timestamp}.pkl'); scaler_values_path = os.path.join(SAVED_MODELS_DIR, f'scaler_values_{run_timestamp}.pkl'); scaler_raw_all_path = os.path.join(SAVED_MODELS_DIR, f'scaler_raw_all_{run_timestamp}.pkl')
        print("\n--- Preparing Data for Model 1 ---"); X_ext_unscaled, y_ext_unscaled = create_sequences_m1_multi(extrema_data_full, EXTREMA_SEQ_LEN, target_indices_in_all)
        if X_ext_unscaled.size == 0: print("Error: No sequences M1."); sys.exit(1)
        tgt_m1_combined_unscaled = y_ext_unscaled; n_samples_m1 = X_ext_unscaled.shape[0]; split_idx_m1 = int(n_samples_m1 * (1.0 - VALIDATION_SPLIT))
        X_ext_train_unscaled=X_ext_unscaled[:split_idx_m1]; X_ext_val_unscaled=X_ext_unscaled[split_idx_m1:]; tgt_train_unscaled=tgt_m1_combined_unscaled[:split_idx_m1]; tgt_val_unscaled=tgt_m1_combined_unscaled[split_idx_m1:]
        print(f"M1 Samples: Train={len(X_ext_train_unscaled)}, Val={len(X_ext_val_unscaled)}")
        print("Fitting scalers on M1 training data..."); scaler_ext_time_diff = StandardScaler(); scaler_values = StandardScaler(); scaler_raw_all = StandardScaler()
        train_ext_rel_times=[]; train_ext_values_allfeat=[]
        for seq in X_ext_train_unscaled:
            if seq.shape[0] > 1: train_ext_rel_times.extend(np.diff(seq[:, 0]))
            train_ext_values_allfeat.append(seq[:, 1:])
        if not train_ext_rel_times: train_ext_rel_times=[0]
        if not train_ext_values_allfeat: print("Error: No M1 train values for scaler."); sys.exit(1)
        train_ext_values_allfeat_np = np.vstack(train_ext_values_allfeat)
        scaler_ext_time_diff.fit(np.vstack([np.array(train_ext_rel_times).reshape(-1,1), tgt_train_unscaled[:,0].reshape(-1,1)]))
        scaler_values.fit(np.vstack([train_ext_values_allfeat_np, tgt_train_unscaled[:,1:]]))
        print("Fitting raw features scaler...");
        try: split_index_raw_for_scaler, _ = find_split_indices(extrema_data_full, len(timeseries_full), VALIDATION_SPLIT, EXTREMA_SEQ_LEN)
        except Exception as e: print(f"Warn: Could not get split for raw scaler ({e}). Using all data."); split_index_raw_for_scaler = len(timeseries_full)
        if split_index_raw_for_scaler > 0: scaler_raw_all.fit(timeseries_full[:split_index_raw_for_scaler]); print(f"Raw scaler fitted up to index {split_index_raw_for_scaler}")
        else: print("Warning: Split index for raw scaler is 0. Fitting on all data."); scaler_raw_all.fit(timeseries_full)
        print("Scalers fitted.")
        X_ext_train_scaled, tgt_train_scaled = process_and_scale_m1_data_multi(X_ext_train_unscaled, tgt_train_unscaled, scaler_ext_time_diff, scaler_values)
        X_ext_val_scaled, tgt_val_scaled = process_and_scale_m1_data_multi(X_ext_val_unscaled, tgt_val_unscaled, scaler_ext_time_diff, scaler_values)
        if X_ext_train_scaled.size == 0: print("Error: M1 Training data scaling empty."); sys.exit(1)
        train_dataset_m1 = TensorDataset(torch.tensor(X_ext_train_scaled, dtype=torch.float32), torch.tensor(tgt_train_scaled, dtype=torch.float32))
        val_dataset_m1 = TensorDataset(torch.tensor(X_ext_val_scaled, dtype=torch.float32), torch.tensor(tgt_val_scaled, dtype=torch.float32)) if X_ext_val_scaled.size > 0 else None
        train_loader_m1 = DataLoader(train_dataset_m1, batch_size=EXTREMA_BATCH_SIZE, shuffle=True, num_workers=actual_num_workers, pin_memory=DEVICE.type=='cuda', persistent_workers=actual_num_workers>0)
        val_loader_m1 = DataLoader(val_dataset_m1, batch_size=EXTREMA_BATCH_SIZE, shuffle=False, num_workers=actual_num_workers, pin_memory=DEVICE.type=='cuda', persistent_workers=actual_num_workers>0) if val_dataset_m1 else None
        print("M1 DataLoaders created.")
        m1_init_params = {'input_size': M1_INPUT_FEATURES, 'hidden_size': EXTREMA_HIDDEN_SIZE, 'num_layers': EXTREMA_NUM_LAYERS, 'output_size': M1_OUTPUT_FEATURES, 'dropout_rate': EXTREMA_DROPOUT}
        m1_params_for_saving = m1_init_params.copy(); m1_params_for_saving['extrema_seq_len'] = EXTREMA_SEQ_LEN; m1_params_for_saving['target_columns'] = TARGET_COLUMNS; m1_params_for_saving['n_target_features'] = N_TARGET_FEATURES; m1_params_for_saving['all_input_columns'] = all_columns_loaded; m1_params_for_saving['n_all_features'] = N_ALL_FEATURES
        model1 = ExtremumPredictorLSTM(**m1_init_params).to(DEVICE)
        optimizer1 = optim.AdamW(model1.parameters(), lr=EXTREMA_LR, weight_decay=EXTREMA_WEIGHT_DECAY); scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.2, patience=10, verbose=True); criterion1 = nn.MSELoss(); scaler1 = GradScaler(enabled=AMP_ENABLED)
        m1_success, _, _ = train_model(model1, "Model 1", train_loader_m1, val_loader_m1, criterion1, optimizer1, scheduler1, scaler1, EXTREMA_EPOCHS, EXTREMA_PATIENCE, DEVICE, AMP_ENABLED, current_best_model1_path, m1_params_for_saving)
        print("\nSaving scalers...");
        try: joblib.dump(scaler_ext_time_diff, scaler_ext_time_path); joblib.dump(scaler_values, scaler_values_path); joblib.dump(scaler_raw_all, scaler_raw_all_path); print(f"Scalers saved for run {run_timestamp}")
        except Exception as e: print(f"!!! Error saving scalers: {e} !!!")
        if m1_success:
            print("\n--- Preparing Data for Model 2 ---")
            if 'scaler_raw_all' not in locals(): print("Error: Raw scaler missing."); sys.exit(1)
            timeseries_full_scaled = scaler_raw_all.transform(timeseries_full)
            last_known_indices_train = [int(seq[-1, 0]) for seq in X_ext_train_unscaled]; last_known_indices_val = [int(seq[-1, 0]) for seq in X_ext_val_unscaled]
            X_series_train, y_series_train, target_info_train_scaled = create_series_sequences_multi(timeseries_full_scaled, tgt_train_unscaled, SERIES_INPUT_LEN_M2, SERIES_PRED_HORIZON, last_known_indices_train, scaler_ext_time_diff, scaler_values, target_indices_in_all)
            X_series_val, y_series_val, target_info_val_scaled = create_series_sequences_multi(timeseries_full_scaled, tgt_val_unscaled, SERIES_INPUT_LEN_M2, SERIES_PRED_HORIZON, last_known_indices_val, scaler_ext_time_diff, scaler_values, target_indices_in_all)
            if X_series_train.size == 0: print("Error: Failed M2 train sequences."); sys.exit(1)
            if X_series_val.size == 0: print("Warning: Failed M2 val sequences.")
            print(f"M2 Train seq: {X_series_train.shape}, M2 Val seq: {X_series_val.shape}")
            train_dataset_series = SeriesDatasetMulti(X_series_train, y_series_train)
            val_dataset_series = SeriesDatasetMulti(X_series_val, y_series_val) if X_series_val.size > 0 else None
            train_loader_series = DataLoader(train_dataset_series, batch_size=SERIES_BATCH_SIZE, shuffle=True, num_workers=actual_num_workers, pin_memory=DEVICE.type=='cuda', persistent_workers=actual_num_workers>0)
            val_loader_series = DataLoader(val_dataset_series, batch_size=SERIES_BATCH_SIZE, shuffle=False, num_workers=actual_num_workers, pin_memory=DEVICE.type=='cuda', persistent_workers=actual_num_workers>0) if val_dataset_series else None
            print("M2 DataLoaders created.")
            m2_init_params = {'input_size': M2_INPUT_FEATURES, 'hidden_size': SERIES_HIDDEN_SIZE, 'num_layers': SERIES_NUM_LAYERS, 'output_horizon': SERIES_PRED_HORIZON, 'output_features': N_TARGET_FEATURES, 'dropout_rate': SERIES_DROPOUT}
            m2_params_for_saving = m2_init_params.copy(); m2_params_for_saving['series_input_len'] = SERIES_INPUT_LEN_M2; m2_params_for_saving['series_pred_horizon'] = SERIES_PRED_HORIZON; m2_params_for_saving['target_columns'] = TARGET_COLUMNS; m2_params_for_saving['all_input_columns'] = all_columns_loaded; m2_params_for_saving['n_target_features'] = N_TARGET_FEATURES; m2_params_for_saving['n_all_features'] = N_ALL_FEATURES
            model2 = SeriesPredictorLSTM(**m2_init_params).to(DEVICE)
            optimizer2 = optim.AdamW(model2.parameters(), lr=SERIES_LR, weight_decay=SERIES_WEIGHT_DECAY); scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.2, patience=10, verbose=True); criterion2 = nn.MSELoss(reduction='none'); scaler2 = GradScaler(enabled=AMP_ENABLED)
            m2_success, _, _ = train_model(model2, "Model 2", train_loader_series, val_loader_series, criterion2, optimizer2, scheduler2, scaler2, SERIES_EPOCHS, SERIES_PATIENCE, DEVICE, AMP_ENABLED, current_best_model2_path, m2_params_for_saving)
        else: print("\nSkipping Model 2 training.")

    else: # if not TRAINING:
        print("\n=== PREDICTION MODE ===")
        load_best_model1_path = os.path.join(SAVED_MODELS_DIR, f'model1_extremum_multi_best_{LOAD_RUN_TIMESTAMP}.pth')
        load_best_model2_path = os.path.join(SAVED_MODELS_DIR, f'model2_series_multi_best_{LOAD_RUN_TIMESTAMP}.pth')
        load_scaler_ext_time_path = os.path.join(SAVED_MODELS_DIR, f'scaler_ext_time_diff_{LOAD_RUN_TIMESTAMP}.pkl')
        load_scaler_values_path = os.path.join(SAVED_MODELS_DIR, f'scaler_values_{LOAD_RUN_TIMESTAMP}.pkl')
        load_scaler_raw_all_path = os.path.join(SAVED_MODELS_DIR, f'scaler_raw_all_{LOAD_RUN_TIMESTAMP}.pkl')
        model1 = None; model2 = None; params1_loaded = {}; params2_loaded = {} # Use different names for loaded params
        try:
            print("Loading scalers..."); scaler_ext_time_diff = joblib.load(load_scaler_ext_time_path); scaler_values = joblib.load(load_scaler_values_path); scaler_raw_all = joblib.load(load_scaler_raw_all_path); print("Scalers loaded.")
            print(f"Loading M1 checkpoint: {load_best_model1_path}"); checkpoint1 = torch.load(load_best_model1_path, map_location=DEVICE)
            if 'params' not in checkpoint1 or 'model_state_dict' not in checkpoint1: raise ValueError("M1 Checkpoint invalid."); params1_loaded = checkpoint1['params']; print("M1 Params:", params1_loaded)
            # Use target_columns and n_target_features from params1 if available
            loaded_target_columns_m1 = params1_loaded.get('target_columns', TARGET_COLUMNS)
            n_target_features_loaded_m1 = params1_loaded.get('n_target_features', len(loaded_target_columns_m1))
            m1_init_params = {k:v for k,v in params1_loaded.items() if k in ['input_size','hidden_size','num_layers','output_size','dropout_rate']}; m1_init_params.setdefault('input_size', 1 + n_target_features_loaded_m1); m1_init_params.setdefault('output_size', 1 + n_target_features_loaded_m1); m1_init_params.setdefault('hidden_size', EXTREMA_HIDDEN_SIZE); m1_init_params.setdefault('num_layers', EXTREMA_NUM_LAYERS); m1_init_params.setdefault('dropout_rate', EXTREMA_DROPOUT)
            model1 = ExtremumPredictorLSTM(**m1_init_params).to(DEVICE); model1.load_state_dict(checkpoint1['model_state_dict']); model1.eval(); print("Model 1 loaded.")

            print(f"Loading M2 checkpoint: {load_best_model2_path}"); checkpoint2 = torch.load(load_best_model2_path, map_location=DEVICE)
            if 'params' not in checkpoint2 or 'model_state_dict' not in checkpoint2: raise ValueError("M2 Checkpoint invalid."); params2_loaded = checkpoint2['params']; print("Model 2 Params:", params2_loaded)
            loaded_all_input_columns_m2 = params2_loaded.get('all_input_columns', ALL_INPUT_COLUMNS)
            loaded_target_columns_m2 = params2_loaded.get('target_columns', TARGET_COLUMNS)
            n_target_features_loaded_m2 = params2_loaded.get('n_target_features', len(loaded_target_columns_m2))
            n_all_features_loaded_m2 = params2_loaded.get('n_all_features', len(loaded_all_input_columns_m2))
            m2_init_params = {k:v for k,v in params2_loaded.items() if k not in ['series_input_len', 'series_pred_horizon', 'target_columns', 'all_input_columns', 'n_target_features', 'n_all_features']}
            m2_init_params.setdefault('input_size', n_all_features_loaded_m2 + 1 + n_target_features_loaded_m2); m2_init_params.setdefault('output_features', n_target_features_loaded_m2); m2_init_params.setdefault('output_horizon', SERIES_PRED_HORIZON); m2_init_params.setdefault('hidden_size', SERIES_HIDDEN_SIZE); m2_init_params.setdefault('num_layers', SERIES_NUM_LAYERS); m2_init_params.setdefault('dropout_rate', SERIES_DROPOUT)
            model2 = SeriesPredictorLSTM(**m2_init_params).to(DEVICE); model2.load_state_dict(checkpoint2['model_state_dict']); model2.eval(); print("Model 2 loaded.")
            final_target_columns = params1_loaded.get('target_columns', TARGET_COLUMNS) # Use from M1
            final_all_input_columns = params1_loaded.get('all_input_columns', ALL_INPUT_COLUMNS) # Use from M1
            final_target_indices_in_all = [final_all_input_columns.index(col) for col in final_target_columns if col in final_all_input_columns]
            print(f"Using Target Columns for Prediction: {final_target_columns}"); print(f"Using All Input Columns for Prediction: {final_all_input_columns}"); print("-" * 30)
        except FileNotFoundError as e: print(f"Error loading file: {e}. Check paths & LOAD_RUN_TIMESTAMP."); sys.exit(1)
        except Exception as e:
            print(f"Error loading/instantiation: {e}")
            if "size mismatch" in str(e) or "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e): print(">>> CRITICAL: Model architecture mismatch. Ensure loaded parameters match current model definitions in the script, or that the checkpoint parameters are correct for the model version it was saved with.")
            sys.exit(1)

        m1_extrema_seq_len_loaded = params1_loaded.get('extrema_seq_len', EXTREMA_SEQ_LEN)
        try: split_index_raw, _ = find_split_indices(extrema_data_full, len(timeseries_full), VALIDATION_SPLIT, m1_extrema_seq_len_loaded)
        except Exception as e: print(f"Error finding split indices: {e}"); sys.exit(1)
        extrema_data_test = extrema_data_full[extrema_data_full[:, 0] >= split_index_raw]
        print(f"Test set: Raw starts {split_index_raw}, Extrema Size={len(extrema_data_test)}")
        if len(extrema_data_test) < m1_extrema_seq_len_loaded + 1: print("Error: Test set too small."); sys.exit(1)

        predictions = run_predictions(model1, model2, scaler_ext_time_diff, scaler_values, scaler_raw_all, timeseries_full, extrema_data_test, params1_loaded, params2_loaded, final_target_columns, DEVICE, AMP_ENABLED, N_RANDOM_SAMPLES, N_PREDICTION_EXTREMA, final_all_input_columns, final_target_indices_in_all)
        if predictions: plot_predictions(predictions, timeseries_full, time_indices_full, extrema_data_full, final_target_columns, final_all_input_columns, N_PREDICTION_EXTREMA, PLOT_PAST_WINDOW_SIZE)
        else: print("Prediction failed, skipping plot.")

    print(f"\n--- Script Finished (Run ID: {run_timestamp}) ---")

#EOF