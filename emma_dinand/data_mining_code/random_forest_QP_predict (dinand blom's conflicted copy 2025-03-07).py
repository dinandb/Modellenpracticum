import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import data_frame_build as dfb
import extras

def create_feature_vector(heave_segment, extrema, diff_extrema, local_window):
    # Flatten the arrays to ensure consistent shape
    heave_segment_flat = np.ravel(heave_segment)  
    extrema_flat = np.ravel(extrema)        
    diff_extrema_flat = np.ravel(diff_extrema)
    
    # Check if lengths are consistent
    max_len = max(len(heave_segment_flat), len(extrema_flat), len(diff_extrema_flat))
    
    # Pad any shorter arrays to ensure they are the same length
    # heave_segment_flat = np.pad(heave_segment_flat, (0, max_len - len(heave_segment_flat)), mode='constant')
    # extrema_flat = np.pad(extrema_flat, (0, max_len - len(extrema_flat)), mode='constant')
    # diff_extrema_flat = np.pad(diff_extrema_flat, (0, max_len - len(diff_extrema_flat)), mode='constant')
    # print(heave_segment_flat)
    # print(extrema_flat)
    # print(diff_extrema_flat)
    # quit()
    feature_vector = np.concatenate([
        heave_segment_flat, 
        extrema_flat, 
        diff_extrema_flat,
        [np.mean(diff_extrema_flat)],  # Wrap scalar in a list to concatenate
        [np.mean(local_window)],       # Same for other scalars
        [np.std(local_window)],        
        [np.max(local_window) - np.min(local_window)],  # Peak-to-peak amplitude
    ])
    # print(feature_vector)
    
    return feature_vector

def create_qp_labeled_dataset(heave_data, lookback_window=100):
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
        # raise FileNotFoundError
        features = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        for i in range(len(heave_data)):
            # Extract local extrema and their characteristics
            if i >= lookback_window:
                if i % 200 == 0:
                    print(i)
                local_window = heave_data[i-lookback_window:i]
                # peaks = signal.find_peaks(local_window)[0],  # number of peaks
                # troughs = signal.find_peaks(-local_window)[0],  # number of troughs
                extrema = (np.array(extras.get_only_max_vals_vector(local_window)[-3:]))
                if len(extrema) !=3:
                    print("ahhhhhh")
                # print(extrema)
                # quit()
                diff_extrema = (np.array(np.diff(extrema)))
                heave_segment = heave_data[(i-3):i]
                if len(heave_segment) < 3:
                    heave_segment = np.pad(heave_segment, (3 - len(heave_segment), 0), mode='constant')
                if len(heave_segment) != 3:
                    print("a2hhhhhh")
                # quit()
                # feature_vector = [
                    
                    
                #     np.ravel(heave_segment),  # Flatten heave_segment
                #     np.ravel(extrema),        # Flatten extrema
                #     np.ravel(diff_extrema),   # Flatten diff_extrema
                #     np.mean(diff_extrema),    # Mean of diff_extrema
                #     np.mean(local_window),    # Mean of the local window
                #     np.std(local_window),     # Standard deviation of the local window
                #     np.max(local_window) - np.min(local_window),  # Peak-to-peak amplitude
                # ]
                feature_vector = create_feature_vector(heave_segment, extrema, diff_extrema, local_window) 
                features.append(feature_vector)





        save_processed_data(features, pickle_file_path)
        print("Processed data saved to pickle.")


    return np.array(features)

    
    # Extract features (you can expand these)

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

def create_qp_labeled_dataset(heave_data, lookback_window=100):
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
        # raise FileNotFoundError
        features = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        for i in range(len(heave_data)):
            # Extract local extrema and their characteristics
            if i >= lookback_window:
                if i % 200 == 0:
                    print(i)
                local_window = heave_data[i-lookback_window:i]
                # peaks = signal.find_peaks(local_window)[0],  # number of peaks
                # troughs = signal.find_peaks(-local_window)[0],  # number of troughs
                extrema = (np.array(extras.get_only_max_vals_vector(local_window)[-3:]))
                if len(extrema) !=3:
                    print("ahhhhhh")
                # print(extrema)
                # quit()
                diff_extrema = (np.array(np.diff(extrema)))
                heave_segment = heave_data[(i-3):i]
                if len(heave_segment) < 3:
                    heave_segment = np.pad(heave_segment, (3 - len(heave_segment), 0), mode='constant')
                if len(heave_segment) != 3:
                    print("a2hhhhhh")
                # quit()
                # feature_vector = [
                    
                    
                #     np.ravel(heave_segment),  # Flatten heave_segment
                #     np.ravel(extrema),        # Flatten extrema
                #     np.ravel(diff_extrema),   # Flatten diff_extrema
                #     np.mean(diff_extrema),    # Mean of diff_extrema
                #     np.mean(local_window),    # Mean of the local window
                #     np.std(local_window),     # Standard deviation of the local window
                #     np.max(local_window) - np.min(local_window),  # Peak-to-peak amplitude
                # ]
                feature_vector = create_feature_vector(heave_segment, extrema, diff_extrema, local_window) 
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with specific constraints
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # Handle class imbalance
        min_samples_leaf=3,  # Reduce overfitting
        max_depth=15  # Prevent too complex models
    )
    # adaboost_clf = AdaBoostClassifier(
    #     estimator=DecisionTreeClassifier(max_depth=5),  # Weak learner
    #     n_estimators=50,  # Number of weak learners
    #     learning_rate=0.1,  # Controls contribution of each learner
    #     random_state=42
    # )


    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = clf.predict_proba(X_test_scaled)
    y_pred = y_pred[:, 1] > 0.5  # Threshold at 0.5
    print(y_test)
    print(y_pred)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf, scaler

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
    

    data2_path = 'data2.pkl'

    try:
        # Try to load the data if it's already saved
        
        data2 = load_processed_data(data2_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        file_path = '../../assets/M5415_10kn_JONSWAP_3m_10s/output.csv'
        data2 = pd.read_csv(file_path)
        data2 = data2[['t', 'z_wf']]
        data2 = data2.iloc[1:]
        data2 = data2.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data2, data2_path)
        print("Processed data saved to pickle.")
    
        heave_data2 = data2['z_wf']

    # quit()
    y = dfb.init_QPs(data)
    y = dfb.moveQP(y)
    
    # quit()
    

    
    X = create_qp_labeled_dataset(heave_data[1:3000])
    print(len(X))
    # y = y + [False]*(len(X)-len(y))
    y  = y[:len(X)]

    print(len(y))
    # Train predictor

    pickle_model_path = 'pickel_model.pkl'

    try:
        # Try to load the data if it's already saved
        
        # raise FileNotFoundError
        model, scalar = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        model, scaler = train_qp_predictor(X, y)



    
    # nog testen op de andere data
    # we hebben data2$heave. hierop testen
    
    
main()