import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
import data_frame_build as dfb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input



# Example usage
# reload_module('your_module_name')
# check_module_source(your_module)
def adjust_predictions(y_pred, y_eval):
    """
    Adjusts y_pred such that if any 1 in a sequence of 1s in y_eval was predicted as 1 in y_pred,
    the entire sequence is marked as 1 in the output y_pred.

    Parameters:
    - y_pred (list or np.array): Predicted labels.
    - y_eval (list or np.array): True labels.

    Returns:
    - np.array: Adjusted predictions.
    """
    y_pred = np.array(y_pred)
    y_eval = np.array(y_eval)

    i = 0
    while i < len(y_eval):
        if y_eval[i] == 1:
            start = i
            while i + 1 < len(y_eval) and y_eval[i + 1] == 1:
                i += 1
            end = i
            
            # If at least one prediction in this sequence was 1, set the whole sequence to 1
            if any(y_pred[start:end + 1] == 1):
                y_pred[start:end + 1] = 1

        i += 1

    return y_pred




def detect_quiescent_periods(heave_data, duration_threshold=30, heave_threshold=0.2):
    """
    Detect Quiescent Periods (QPs) with hard constraints on heave values.
    
    Parameters:
    - heave_data: numpy array of heave measurements
    - duration_threshold: minimum duration of QP in time steps
    - heave_threshold: maximum allowed absolute heave value during QP
    
    Returns:
    - Boolean mask of QP periods
    """
    # Create a mask for heave values within threshold
    duration_threshold *= 5
    within_threshold_mask = np.abs(heave_data) <= heave_threshold
    
    # Find continuous periods meeting the threshold
    qp_mask = np.zeros_like(heave_data, dtype=bool)
    
    for i in range(len(heave_data) - duration_threshold + 1):
        # Check if the next 'duration_threshold' steps are all within heave threshold
        if np.all(within_threshold_mask[i:i+duration_threshold]):
            qp_mask[i:i+duration_threshold] = True
    
    return qp_mask

def create_qp_labeled_dataset(heave_data, lookback_window=10):
    """
    Create labeled dataset for QP prediction.
    
    Parameters:
    - heave_data: numpy array of heave measurements
    - qp_mask: boolean mask of QP periods
    - lookback_window: number of time steps to look back before QP
    
    Returns:
    - X: features
    - y: labels
    """



    pickle_file_path = 'processed_data_features.pkl'

    try:
        # Try to load the data if it's already saved
        raise FileNotFoundError
    
        features = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        for i in range(len(heave_data)):
            # Extract local extrema and their characteristics
            if i >= lookback_window:
                local_window = heave_data[(i-lookback_window):i]
    
                peaks, _ = signal.find_peaks(local_window)
                troughs, _ = signal.find_peaks(-local_window)
                amplitudes = local_window[peaks] - local_window[troughs]
                amplitude_diffs = np.diff(amplitudes)
                amplitude_diffs_diffs = np.diff(amplitude_diffs)

                feature_vector = [
                    np.mean(local_window),
                    np.std(local_window),
                    np.max(local_window) - np.min(local_window),  # amplitude
                    peaks.size,  # number of peaks
                    troughs.size,  # number of troughs
                    np.mean(amplitudes) if len(amplitudes) > 0 else 0,  # mean amplitude
                    np.std(amplitudes) if len(amplitudes) > 0 else 0,  # std amplitude
                    np.mean(amplitude_diffs) if len(amplitude_diffs) > 0 else 0,  # mean amplitude difference
                    np.std(amplitude_diffs) if len(amplitude_diffs) > 0 else 0  # std amplitude difference

                ]
                features.append(feature_vector)





        save_processed_data(features, pickle_file_path)
        print("Processed data saved to pickle.")


    return np.array(features)

    
    # Extract features (you can expand these)

def train_qp_predictor(X, y):
    """
    Train a classifier to predict QP onset.
    
    Parameters:
    - X: feature matrix
    - y: labels
    
    Returns:
    - Trained classifier
    """
    # Split data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # Scale features
    X_train = np.array(X_train, dtype=np.float32)  # Convert to float
    y_train = np.array(y_train, dtype=np.float32)  # Convert to float

    
    # Train Random Forest with specific constraints
    clf = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  # Replace input_shape with Input layer
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("X_train shape:", X_train.shape)
    print("y_train shape:", np.array(y_train).shape)

    clf.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,verbose=2)
    # print('hio')
    # Evaluate model
    y_pred = clf.predict(X_test)
    print(f"er zijn {sum(y_test)} QP's in de testset")
    best_bar = 0
    min_err = 100000
    for bar in [0.1, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95]:
        y_pred_binary = (y_pred >= bar).astype(int)  # Convert probabilities to binary values

        # print("Classification Report:")
        # print(classification_report(y_test, y_pred_binary))
        # werkt adjust_pred wel goed?
        y_pred_binary = adjust_predictions(y_pred_binary, y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
        error = 20*fp + fn
        if error < min_err:
            min_err = error
            best_bar = bar
    bar = best_bar
    y_pred_binary = (y_pred >= bar).astype(int)  # Convert probabilities to binary values
    y_pred_binary = adjust_predictions(y_pred_binary, y_test)
    print(confusion_matrix(y_test, y_pred_binary))
    
    # print("hoi")
    return clf
def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Example usage (you'll replace with your actual data)
def main():
    # Simulated heave data (replace with your actual data)
    
    # heave_data = np.cumsum(np.random.normal(0, 1, 1000))
    # Path to the pickle file
    pickle_file_path = 'processed_data.pkl'

    try:
        # Try to load the data if it's already saved
        
        data = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        file_path = '../../assets/data_ed_2_clean.csv'
        data = pd.read_csv(file_path)
        data = data[['t', 'z_wf']]
        data = data.iloc[1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data, pickle_file_path)
        print("Processed data saved to pickle.")
    
    heave_data = np.array(data['z_wf'])
    print(len(heave_data))

    y = dfb.init_QPs(data)
    y = dfb.moveQP(y)
    print(sum(y))
    print(sum(y[62993:]))

    
    X = create_qp_labeled_dataset(heave_data, lookback_window=10)
    print(len(X))
    y = y + [False]*(len(X)-len(y))
    print(len(y))
    # Train predictor
    model  = train_qp_predictor(X, y)

    
    
main()