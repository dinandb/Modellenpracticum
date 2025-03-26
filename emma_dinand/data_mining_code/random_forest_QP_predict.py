import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import pickle

from sklearn.tree import DecisionTreeClassifier
import data_frame_build as dfb
import extras

def create_feature_vector(extrema, diff_extrema):
    # Flatten the arrays to ensure consistent shape
    
    extrema_flat = np.ravel(extrema)        
    diff_extrema_flat = np.ravel(diff_extrema)
    
    # Check if lengths are consistent
    # max_len = max(len(heave_segment_flat), len(extrema_flat), len(diff_extrema_flat))
    
    # Pad any shorter arrays to ensure they are the same length
    # heave_segment_flat = np.pad(heave_segment_flat, (0, max_len - len(heave_segment_flat)), mode='constant')
    # extrema_flat = np.pad(extrema_flat, (0, max_len - len(extrema_flat)), mode='constant')
    # diff_extrema_flat = np.pad(diff_extrema_flat, (0, max_len - len(diff_extrema_flat)), mode='constant')
    # print(heave_segment_flat)
    # print(extrema_flat)
    # print(diff_extrema_flat)
    # quit()
    feature_vector = np.concatenate([
        # local_window,
        # heave_segment_flat, 
        extrema_flat, 
        diff_extrema_flat,
        # [np.mean(diff_extrema_flat)],  # Wrap scalar in a list to concatenate
        # [np.mean(local_window)],       # Same for other scalars
        # [np.std(local_window)],        
        # [np.max(local_window) - np.min(local_window)],  # Peak-to-peak amplitude
    ])
    # print(feature_vector)
    
    return feature_vector


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

def create_qp_labeled_dataset(heave_data, i, lookback_window=100, new = False):
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



    pickle_file_path = f'processed_data_features{i}.pkl'

    try:

        # Try to load the data if it's already saved
        if new:
            raise FileNotFoundError
        
        
        features = load_processed_data(pickle_file_path)
        print("Loaded features from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        for i in range(len(heave_data)):
            # Extract local extrema and their characteristics
            if i >= lookback_window:
                if i % 500 == 0:
                    print(i)
                local_window = heave_data[i-lookback_window:i]
                # peaks = signal.find_peaks(local_window)[0],  # number of peaks
                # troughs = signal.find_peaks(-local_window)[0],  # number of troughs
                extrema = (np.array(extras.get_only_max_vals_vector(local_window)[-4:]))
                if len(extrema) !=4:
                    print("ahhhhhh")
                # print(extrema)
                # quit()
                diff_extrema = (np.array(np.diff(extrema)))
                # heave_segment = heave_data[(i-3):i]
                # if len(heave_segment) < 3:
                #     heave_segment = np.pad(heave_segment, (3 - len(heave_segment), 0), mode='constant')
                # if len(heave_segment) != 3:
                #     print("a2hhhhhh")
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
                feature_vector = create_feature_vector(extrema, diff_extrema) 
                # feature_vector = create_feature_vector(heave_segment, [], [], local_window)

                features.append(feature_vector)





        save_processed_data(features, pickle_file_path)
        print("Processed features saved to pickle.")


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with specific constraints
    # clf = RandomForestClassifier(
    #     n_estimators=100,
    #     class_weight='balanced',  # Handle class imbalance
    #     min_samples_leaf=3,  # Reduce overfitting
    #     max_depth=15  # Prevent too complex models
    # )
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=15),  # Weak learner
        n_estimators=100,  # Number of weak learners
        learning_rate=0.1,  # Controls contribution of each learner
        random_state=42
    )

    # clf = LogisticRegression(random_state=0, max_iter=1000)

    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = clf.predict_proba(X_test_scaled)
    print(sum(y_pred))
    print(len(y_pred))
    y_pred = y_pred[:, 1] > 0.5  # Threshold at 0.5
    # print(f"y_test {y_test[:100]}")
    # print(f"y_pred {y_pred[:100]}")
    # print(y_test)
    # print(y_pred)
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

def evaluate(model, scaler, X, y):
    # predict y based on X with model
    # evaluate performance with real y


    
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict_proba(X_scaled)
    y_pred = y_pred[:, 1] > 0.9  # Threshold at 0.5
    
    print("Classification Report:")
    print(classification_report(y, y_pred))


# Example usage (you'll replace with your actual data)
def main():
    # Simulated heave data (replace with your actual data)
    
    # heave_data = np.cumsum(np.random.normal(0, 1, 1000))
    # Path to the pickle file
    pickle_file_path = 'processed_data.pkl'

    try:
        # Try to load the data if it's already saved
        raise FileNotFoundError
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

    data2_path = 'data2.pkl'

    try:
        # Try to load the data if it's already saved
        
        data2 = load_processed_data(data2_path)
        print("Loaded data2 from pickle.")
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
    # y = dfb.moveQP(y)
    print(f"amount QPs in data {sum(y)}")
    # quit()
    

    print(len(heave_data))
<<<<<<< HEAD
    X = create_qp_labeled_dataset(heave_data[:2000], i=1, new = True)
=======
    X = create_qp_labeled_dataset(heave_data[6000:12000], i=1, new = True)
>>>>>>> 9342004f610d1e291faa8043d46017397a5404eb
    print(f"sum before trimming: {sum(y)    }")
    # y = y + [False]*(len(X)-len(y))
    y  = y[:len(X)]

    print(f"sum: {sum(y)    }")


    pickle_model_path = 'pickel_model.pkl'

    try:
        # Try to load the data if it's already saved
        
        raise FileNotFoundError
        model, scaler = load_processed_data(pickle_model_path)
        print("Loaded model from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        model, scaler = train_qp_predictor(X, y)
        save_processed_data((model, scaler), pickle_model_path)
        print("Processed data saved to pickle.")


    # quit()

    
    # nog testen op de andere data
    # we hebben data2$heave. hierop testen
    print(f"len heavedata2 {len(heave_data2)}")

    y2 = dfb.init_QPs(data2)
    y2 = dfb.moveQP(y2)
    print(f"amount QPs in data2 {sum(y2)}")
    y2 = y2[4000:10000]

    X2 = create_qp_labeled_dataset(heave_data2[4000:10000], i=2, new = True)


    
    evaluate(model, scaler, X2, y2)




    
    
main()