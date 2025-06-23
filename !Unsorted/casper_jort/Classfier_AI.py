import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import os
import sys
import time
import datetime
import joblib # For saving/loading scaler
import random

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

# --- Execution Mode ---
TRAINING = False  # Set to True to train, False to load and evaluate

# --- File Paths ---
# Make sure this CSV_PATH points to your RAW wave data,
# e.g., '5415M_Hs=4m_Tp=10s_10h_clean.csv' from your second script's example.
# NOT the 'Hs4_heavespeed_crit.csv' from your first script, unless that contains raw data.
CSV_PATH = r'C:\Users\caspe\OneDrive\Documents\Programming\Modellenpracticum\Data\5415M_Hs=4m_Tp=10s_10h_clean.csv'
SAVED_MODEL_DIR = './saved_extrema_classifier'
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, 'extrema_classifier_model.pth')
SCALER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, 'extrema_values_scaler.pkl')

# --- Data Parameters ---
# Column names from your raw data CSV
ALL_DATA_COLUMNS_TO_LOAD = ['z_wf', 'y_wf', 'x_wf', 'phi_wf', 'theta_wf', 'psi_wf', 'zeta'] # All columns you want to be available from CSV
EXTREMA_DETECTION_COLUMN = 'z_wf'      # Column used to find peaks/troughs (must be in ALL_DATA_COLUMNS_TO_LOAD)
CLASSIFICATION_TARGET_COLUMN = 'z_wf'  # Column whose EXTREMA VALUE is used for classification (must be in ALL_DATA_COLUMNS_TO_LOAD)
CLASSIFICATION_THRESHOLD = 1.0         # Threshold for the CLASSIFICATION_TARGET_COLUMN's extremum value
                                       # (e.g., if next z_wf extremum > 2.0, label is 1)

EXTREMA_SEQ_LEN = 10                   # Number of past extrema values to use as input sequence
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42                       # For reproducible splits

# --- Model Hyperparameters ---
INPUT_SIZE_CLASSIFIER = 1              # We are using one feature (the value of CLASSIFICATION_TARGET_COLUMN) per time step in the extrema sequence
HIDDEN_SIZE_CLASSIFIER = 48
NUM_LAYERS_CLASSIFIER = 2
DROPOUT_CLASSIFIER = 0.25
LR_CLASSIFIER = 0.001
EPOCHS_CLASSIFIER = 150
BATCH_SIZE_CLASSIFIER = 256
PATIENCE_CLASSIFIER = 20               # For early stopping
WEIGHT_DECAY_CLASSIFIER = 1e-5

# --- General Training/System Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = True if DEVICE.type == 'cuda' else False
NUM_WORKERS = 0 if sys.platform == "win32" else 4 # Safer default for Windows

# ==============================================================================
#                             HELPER FUNCTIONS & CLASSES
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# --- Data Loading and Extrema Detection (Adapted from your second script) ---
def load_data_and_extrema(csv_path, all_columns_to_load, extrema_detection_col_name):
    print("Loading multi-variate data...")
    try:
        # Assuming CSV might have a MultiIndex header like in your second script
        df = pd.read_csv(csv_path, header=[0, 1])
        print(df)
        timeseries_full_list = []
        actual_columns_found_tuples = [] # Store (original_tuple, simplified_name)
        all_col_names_ordered = [] # Store simplified names in order of loading

        for col_name_to_find in all_columns_to_load:
            found = False
            for col_tuple in df.columns: # col_tuple is like ('Z_WF', 'm')
                # Search in both parts of the tuple, case-insensitive
                if col_name_to_find.lower() in str(col_tuple[0]).lower() or \
                   col_name_to_find.lower() in str(col_tuple[1]).lower():
                    timeseries_full_list.append(df[col_tuple].values.astype('float32').reshape(-1, 1))
                    actual_columns_found_tuples.append(col_tuple)
                    all_col_names_ordered.append(col_name_to_find) # Use the desired name
                    found = True
                    break
            if not found:
                raise ValueError(f"Column '{col_name_to_find}' not found in CSV headers.")
        
        if not timeseries_full_list:
            raise ValueError("No data columns were loaded. Check ALL_DATA_COLUMNS_TO_LOAD and CSV content.")

        timeseries_full = np.hstack(timeseries_full_list)
        if np.isnan(timeseries_full).any():
            raise ValueError("NaNs found in loaded data after attempting to load all specified columns!")

        print(f"Successfully loaded columns (original names): {actual_columns_found_tuples}")
        print(f"Interpreted as (in order): {all_col_names_ordered}")
        print(f"Loaded full timeseries shape: {timeseries_full.shape}")
        
        time_indices_full = np.arange(len(timeseries_full)).astype('float32').reshape(-1, 1)

        try:
            extrema_col_idx_in_loaded_data = all_col_names_ordered.index(extrema_detection_col_name)
        except ValueError:
            raise ValueError(f"Extrema detection column '{extrema_detection_col_name}' not among successfully loaded columns: {all_col_names_ordered}")

        print(f"Extracting extrema based on: '{extrema_detection_col_name}' (column index {extrema_col_idx_in_loaded_data} in the hstacked data)...")
        extrema_signal = timeseries_full[:, extrema_col_idx_in_loaded_data]
        
        peaks_indices, _ = find_peaks(extrema_signal, distance=5) # distance can be tuned
        troughs_indices, _ = find_peaks(-extrema_signal, distance=5) # distance can be tuned
        
        extrema_indices = np.sort(np.unique(np.concatenate([peaks_indices, troughs_indices])))
        
        # Filter out consecutive indices if any (e.g. if distance is too small or signal is noisy)
        if len(extrema_indices) > 1:
             valid_extrema_mask = np.insert(np.diff(extrema_indices) > 1, 0, True) # Keep first, then only if diff > 1
             extrema_indices = extrema_indices[valid_extrema_mask]

        extrema_times = time_indices_full[extrema_indices] # Absolute time index of each extremum
        extrema_values_all_features = timeseries_full[extrema_indices, :] # Values of ALL loaded features at these extrema
        
        # extrema_data_full structure: [abs_time_idx, val_feat1, val_feat2, ...]
        extrema_data_full = np.hstack((extrema_times, extrema_values_all_features))
        print(f"Found {len(extrema_data_full)} extrema.")
        if len(extrema_data_full) == 0:
            raise ValueError("No extrema found. Check extrema_detection_column and data.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data loading or extrema detection: {e}")
        sys.exit(1)
    return timeseries_full, time_indices_full, extrema_data_full, all_col_names_ordered

# --- Sequence Creation for Classifier ---
def create_classification_sequences(extrema_data, all_col_names_ordered,
                                    classification_target_col_name, classification_threshold,
                                    seq_length):
    print(f"Creating classification sequences using target '{classification_target_col_name}' and threshold {classification_threshold}...")
    X_sequences = []
    y_labels = []

    try:
        # +1 because column 0 in extrema_data is the absolute time index
        target_col_idx_in_extrema_data = all_col_names_ordered.index(classification_target_col_name) + 1
    except ValueError:
        raise ValueError(f"Classification target column '{classification_target_col_name}' not in loaded columns: {all_col_names_ordered}")

    # We need seq_length past extrema + 1 future extremum for the label
    if len(extrema_data) < seq_length + 1:
        print(f"Warning: Not enough extrema ({len(extrema_data)}) to create any sequences of length {seq_length} + 1 label.")
        return np.array([]), np.array([])

    for i in range(seq_length, len(extrema_data)):
        # Input sequence: values of the target column from past 'seq_length' extrema
        # extrema_data[row_idx, col_idx_for_target_feature_value]
        past_extrema_values = extrema_data[i-seq_length : i, target_col_idx_in_extrema_data]
        X_sequences.append(past_extrema_values)
        
        # Label: based on the next extremum's value
        next_extremum_value = extrema_data[i, target_col_idx_in_extrema_data]
        label = 1.0 if next_extremum_value > classification_threshold else 0.0
        y_labels.append(label)
        
    if not X_sequences:
        print("No sequences were created.")
        return np.array([]), np.array([])

    X_out = np.array(X_sequences, dtype=np.float32)
    y_out = np.array(y_labels, dtype=np.float32)
    print(f"Created {len(X_out)} sequences. X_shape: {X_out.shape}, y_shape: {y_out.shape}")
    print(f"Class distribution in y: 0s={np.sum(y_out==0)}, 1s={np.sum(y_out==1)}")
    return X_out, y_out

# --- LSTM Classifier Model ---
class LSTMClassifierExtrema(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM dropout is applied between layers if num_layers > 1
        lstm_dropout = dropout_rate if num_layers > 1 and dropout_rate > 0 else 0.0
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=lstm_dropout)
        
        # Dropout layer after LSTM, before FC (optional, can be helpful)
        # self.dropout_fc = nn.Dropout(dropout_rate) # Uncomment if you want this
        
        self.fc = nn.Linear(hidden_size, 1)  # Output single logit for binary classification

        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data) # Orthogonal can be good for RNNs
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_seq):
        # x_seq shape: (batch_size, seq_length, input_size)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size, device=x_seq.device)
        c0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size, device=x_seq.device)
        
        lstm_out, _ = self.lstm(x_seq, (h0, c0))
        
        # We use the output of the last LSTM cell in the sequence
        last_hidden_state = lstm_out[:, -1, :]
        
        # If using dropout before FC:
        # last_hidden_state = self.dropout_fc(last_hidden_state)
        
        logit = self.fc(last_hidden_state)
        return logit.squeeze(-1) # Squeeze the last dim to get (batch_size) for BCEWithLogitsLoss

# ==============================================================================
#                             MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_timestamp}")

    if TRAINING:
        print("\n--- TRAINING MODE ---")
        # 1. Load data and detect extrema
        _, _, extrema_data_full, all_columns_loaded_ordered = \
            load_data_and_extrema(CSV_PATH, ALL_DATA_COLUMNS_TO_LOAD, EXTREMA_DETECTION_COLUMN)

        # 2. Create sequences for classification
        X_unscaled, y_labels = create_classification_sequences(
            extrema_data_full, all_columns_loaded_ordered,
            CLASSIFICATION_TARGET_COLUMN, CLASSIFICATION_THRESHOLD, EXTREMA_SEQ_LEN
        )

        if X_unscaled.size == 0:
            print("No data to train on. Exiting.")
            sys.exit(1)

        # 3. Scale input features (X_unscaled)
        # X_unscaled shape: (num_samples, seq_len_extrema), we need to scale based on all values
        scaler = StandardScaler()
        # Fit scaler on all sequence values reshaped to 2D (num_total_values, 1 feature)
        num_samples, seq_len = X_unscaled.shape
        X_reshaped_for_scaling = X_unscaled.reshape(-1, 1) # Treat each value as an independent sample for scaling
        scaler.fit(X_reshaped_for_scaling)
        X_scaled_reshaped = scaler.transform(X_reshaped_for_scaling)
        # Reshape back to (num_samples, seq_len) and add feature dimension for LSTM
        X_scaled = torch.tensor(X_scaled_reshaped.reshape(num_samples, seq_len, INPUT_SIZE_CLASSIFIER), dtype=torch.float32)
        
        y_tensor = torch.tensor(y_labels, dtype=torch.float32)
        
        print(f"X_scaled shape: {X_scaled.shape}, y_tensor shape: {y_tensor.shape}")

        # Save the scaler
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Scaler saved to {SCALER_SAVE_PATH}")

        # 4. Create Dataset and DataLoaders
        dataset = TensorDataset(X_scaled, y_tensor)
        train_size = int((1.0 - VALIDATION_SPLIT) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(RANDOM_SEED))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CLASSIFIER, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type=='cuda')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_CLASSIFIER, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type=='cuda')
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # 5. Initialize Model, Loss, Optimizer
        model = LSTMClassifierExtrema(
            input_size=INPUT_SIZE_CLASSIFIER,
            hidden_size=HIDDEN_SIZE_CLASSIFIER,
            num_layers=NUM_LAYERS_CLASSIFIER,
            dropout_rate=DROPOUT_CLASSIFIER
        ).to(DEVICE)
        print("\nModel Architecture:")
        print(model)

        # Calculate pos_weight for imbalanced classes
        count_neg = torch.sum(y_tensor == 0).item()
        count_pos = torch.sum(y_tensor == 1).item()
        if count_pos > 0:
            pos_weight_val = 0.05 * count_neg / count_pos
        else:
            pos_weight_val = 1.0 # Avoid division by zero
        pos_weight_tensor = torch.tensor([pos_weight_val], device=DEVICE)
        print(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_tensor.item():.4f} (Neg: {count_neg}, Pos: {count_pos})")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=LR_CLASSIFIER, weight_decay=WEIGHT_DECAY_CLASSIFIER)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=PATIENCE_CLASSIFIER // 2, verbose=True)
        grad_scaler = GradScaler(enabled=AMP_ENABLED)

        # 6. Training Loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        print("\n--- Starting Training ---")
        for epoch in range(EPOCHS_CLASSIFIER):
            model.train()
            total_train_loss = 0
            train_batches = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=AMP_ENABLED):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer) # For gradient clipping if needed
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                total_train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = total_train_loss / train_batches if train_batches > 0 else float('nan')

            # Validation
            model.eval()
            total_val_loss = 0
            val_batches = 0
            val_correct = 0
            val_total = 0
            val_TP, val_FN, val_FP, val_TN = 0, 0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    with autocast(enabled=AMP_ENABLED):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    val_batches +=1

                    preds = torch.round(torch.sigmoid(outputs)).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    val_TP += ((preds == 1) & (labels == 1)).sum().item()
                    val_FN += ((preds == 0) & (labels == 1)).sum().item()
                    val_FP += ((preds == 1) & (labels == 0)).sum().item()
                    val_TN += ((preds == 0) & (labels == 0)).sum().item()

            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('nan')
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_recall = val_TP / (val_TP + val_FN + 1e-8)
            val_precision = val_TP / (val_TP + val_FP + 1e-8)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)

            print(f"Epoch [{epoch+1}/{EPOCHS_CLASSIFIER}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2%}, Recall: {val_recall:.2f}, Prec: {val_precision:.2f}, F1: {val_f1:.2f}")
            print(f"  Val Counts -> TP: {val_TP}, FN: {val_FN}, FP: {val_FP}, TN: {val_TN}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'scaler_path': SCALER_SAVE_PATH, # Info about scaler used
                    'config': { # Save key config for reproducibility
                        'extrema_seq_len': EXTREMA_SEQ_LEN,
                        'classification_target_column': CLASSIFICATION_TARGET_COLUMN,
                        'classification_threshold': CLASSIFICATION_THRESHOLD,
                        'input_size_classifier': INPUT_SIZE_CLASSIFIER,
                        'hidden_size_classifier': HIDDEN_SIZE_CLASSIFIER,
                        'num_layers_classifier': NUM_LAYERS_CLASSIFIER,
                        'dropout_classifier': DROPOUT_CLASSIFIER,
                    }
                }, MODEL_SAVE_PATH)
                print(f"   -> New best validation loss. Model saved to {MODEL_SAVE_PATH}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE_CLASSIFIER:
                    print(f"   -> Early stopping triggered after {PATIENCE_CLASSIFIER} epochs with no improvement.")
                    break
        print("--- Training Finished ---")

    else: # if not TRAINING (i.e., evaluation mode)
        print("\n--- EVALUATION MODE (using pre-trained model) ---")
        if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(SCALER_SAVE_PATH):
            print(f"Error: Model ({MODEL_SAVE_PATH}) or Scaler ({SCALER_SAVE_PATH}) not found. Train first.")
            sys.exit(1)

        # Load scaler
        scaler = joblib.load(SCALER_SAVE_PATH)
        print(f"Scaler loaded from {SCALER_SAVE_PATH}")

        # Load model
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        cfg = checkpoint.get('config', {}) # Load saved config
        
        # Use saved config to instantiate model, falling back to current script config if keys missing
        model = LSTMClassifierExtrema(
            input_size=cfg.get('input_size_classifier', INPUT_SIZE_CLASSIFIER),
            hidden_size=cfg.get('hidden_size_classifier', HIDDEN_SIZE_CLASSIFIER),
            num_layers=cfg.get('num_layers_classifier', NUM_LAYERS_CLASSIFIER),
            dropout_rate=cfg.get('dropout_classifier', DROPOUT_CLASSIFIER)
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {MODEL_SAVE_PATH} (trained for {checkpoint.get('epoch','N/A')} epochs, best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f})")
        print("\nModel Architecture (from loaded):")
        print(model)

        # You would typically run this on a separate test set.
        # For demonstration, we'll re-use the validation set logic or create a small test portion.
        # For a proper test, you'd load test data, process it like training data:
        # 1. Load raw test data, detect extrema.
        # 2. Create classification sequences from test extrema.
        # 3. Scale X_test sequences using the *loaded* (fitted on train) scaler.
        # 4. Create a DataLoader for test data.
        # 5. Evaluate.
        
        print("\nTo evaluate on a test set, you would typically:")
        print("1. Prepare a separate test CSV or split your data to have a hold-out test set.")
        print("2. Run steps 1-4 from the TRAINING block on this test data (using the loaded scaler for step 3).")
        print("3. Then run the evaluation loop (like the validation loop in training).")
        print("This example does not implement a full test set evaluation pipeline for brevity.")
        # As a quick check, if you have val_loader from a previous training run (if script run with TRAINING=True before):
        # if 'val_loader' in locals() and val_loader is not None:
        #     print("\n--- Evaluating on existing val_loader (for quick check) ---")
        #     # ... (copy validation loop from training here) ...
        # else:
        #     print("\nNo val_loader available from previous run to perform a quick check.")

    print(f"\n--- Script Finished (Run ID: {run_timestamp}) ---")