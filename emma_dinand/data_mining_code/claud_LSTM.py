import time
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
import pickle
import data_frame_build as dfb
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout




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
        
        features = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        for i in range(len(heave_data)):
            # Extract local extrema and their characteristics
            if i >= lookback_window:
                local_window = heave_data[i-lookback_window:i]
                
                feature_vector = [
                    np.mean(local_window),
                    np.std(local_window),
                    np.max(local_window) - np.min(local_window),  # amplitude
                    signal.find_peaks(local_window)[0].size,  # number of peaks
                    signal.find_peaks(-local_window)[0].size,  # number of troughs
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # Scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with specific constraints
    # clf = RandomForestClassifier(
    #     n_estimators=100,
    #     class_weight='balanced',  # Handle class imbalance
    #     min_samples_leaf=3,  # Reduce overfitting
    #     max_depth=5  # Prevent too complex models
    # )
    clf = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dense(1, activation='sigmoid')
    ])
    print("bliep")
    time.sleep(5)
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("going to fit with 1 epoch")
    time.sleep(5)
    clf.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluate model
    # y_pred = clf.predict(X_test)
    time.sleep(5)
    print("bloep")
    time.sleep(5)
    # y_pred = (clf.predict(X_test, verbose=0) > 0.5).astype(int)
    
    # Optional: print first few predictions
    # print("First few predictions:", y_pred[:10].flatten())
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    print("hoi")
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
    
    heave_data = data['z_wf']
    # Detect QPs
    


    y = dfb.init_QPs(data)
    y = dfb.moveQP(y)
    print(sum(y))
    
    

    
    X = create_qp_labeled_dataset(heave_data, lookback_window=10)
    print("create labeled dataset")
    print(len(X))
    print(len(y))
    y = y + [False]*(len(X)-len(y))
    # Create labeled dataset
    
    
    # Train predictor
    model = train_qp_predictor(X, y)

print("hallolstm")
main()