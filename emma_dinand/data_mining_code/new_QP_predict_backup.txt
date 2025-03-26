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
from odds_ratio_check import *

NO_EXTREMA_LOOKBACK = 3

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
    
    feature_vector = np.concatenate([
        # local_window,
        # heave_segment_flat, 
        extrema_flat, 
        # diff_extrema_flat,
        # [np.mean(diff_extrema_flat)],  # Wrap scalar in a list to concatenate
        # [np.mean(local_window)],       # Same for other scalars
        # [np.std(local_window)],        
        # [np.max(local_window) - np.min(local_window)],  # Peak-to-peak amplitude
    ])
    # print(feature_vector)
    
    return feature_vector


# def detect_quiescent_periods(data, duration_threshold=30, heave_threshold=0.2):
#     """
#     Detect Quiescent Periods (QPs) with hard constraints on heave values.
    
#     Parameters:
#     - heave_data: numpy array of heave measurements
#     - duration_threshold: minimum duration of QP in time steps
#     - heave_threshold: maximum allowed absolute heave value during QP
    
#     Returns:
#     - Boolean mask of QP periods
#     """
    
#     def can_be_part_of_QP():
#         pass
#     # Create a mask for heave values within threshold
#     duration_threshold *= 5
#     heave_data = data['z_wf']
#     within_threshold_mask = np.abs(heave_data) <= heave_threshold

    
#     # Find continuous periods meeting the threshold
#     qp_mask = np.zeros_like(heave_data, dtype=bool)
    
#     for i in range(len(heave_data) - duration_threshold + 1):
#         # Check if the next 'duration_threshold' steps are all within heave threshold
#         if np.all(within_threshold_mask[i:i+duration_threshold]):
#             qp_mask[i:i+duration_threshold] = True
    
#     return qp_mask

def create_qp_labeled_dataset_faster(data, dataset_id, new = False):
    """
    Create labeled dataset for QP prediction.
    
    Parameters:
    - heave_data: numpy array of heave measurements
    - qp_mask: boolean mask of QP periods
    - lookback_window: number of time steps to look back before QP
    
    Returns:
    - X: features
    
    """

    pickle_file_path = f'processed_data_features{dataset_id}.pkl'
    try:

        # Try to load the data if it's already saved
        if new:
            raise FileNotFoundError
        
        
        features, offset = load_processed_data(pickle_file_path)
        print("Loaded features from pickle.")

    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []
        """
        extract eerst max waarden van de dataset (nu alleen gebaseerd op heave) mbv extras.get_only_max_vals
        maak de eerste feature vector 
        dan voor elke
        """
        extrema, extrema_indices = extras.get_only_max_vals(data, colname="z_wf")
        
        # print(extrema_indices[:5])
        # print(data["z_wf"][:50])
        offset = extrema_indices[NO_EXTREMA_LOOKBACK-1] + 2
        #gebruik eerste 3 maxima als feature vector voor eerste en 
        # gebruik 4e index om te beplaen hoeveel en dan zo door en 
        # de offset gebruiken om te bepalen hoeveel y verschoven moet worden
        # bv extrema [12, 33, 56, 78, 103, ...]
        # dan vanaf 58 tot 80 krijgt als feature vector: [val(12), val(33), val(56)]
        # dan 80 tot 105 krijgt: [val(33), val(56), val(78)]

        for i in range(len(extrema_indices)-2):
            if i + 3 < len(extrema_indices):
                features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(extrema_indices[i+3]-extrema_indices[i+2]))
            else:
                features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(len(data)-extrema_indices[i+2]))
                
        # print(f"features[0] {features[0]}")
        # print(f"features[1] {features[1]}")
        # print(f"features[20] {features[20]}")
        # print(f"features[30] {features[30]}")
        # print(f"features[40] {features[40]}")
        print(f"len features {len(features)}")
        print(f"amount that did not get features = {len(data['z_wf']) - len(features)}")
        # ^ looking good
        print(f"offset (should be equal to above ? ) = {offset}")

        save_processed_data((features, offset), pickle_file_path)
    
    return features, offset



def create_qp_labeled_dataset(data, dataset_id, lookback_window=100, new = False):
    """
    Create labeled dataset for QP prediction.
    
    Parameters:
    - heave_data: numpy array of heave measurements
    - qp_mask: boolean mask of QP periods
    - lookback_window: number of time steps to look back before QP
    
    Returns:
    - X: features
    
    """
    heave_data = data['z_wf']


    pickle_file_path = f'processed_data_features{dataset_id}.pkl'

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
    #     n_estimators=200,
    #     class_weight='balanced',  # Handle class imbalance
    #     min_samples_leaf=3,  # Reduce overfitting
    #     max_depth=20  # Prevent too complex models
    # )

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=20),  # Weak learner
        n_estimators=300,  # Number of weak learners
        learning_rate=0.1,  # Controls contribution of each learner
        random_state=42
    )

    # clf = LogisticRegression(random_state=0, max_iter=1000)

    clf.fit(X_train_scaled, y_train)
    
    return clf, scaler, X_test, y_test

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
    max_score = 0
    best_bar = 0

    for bar in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        y_pred_bin = y_pred[:, 1] > bar  # Threshold at 0.5
        
        # print("Classification Report:")
        report = classification_report(y, y_pred_bin, output_dict=True)
        # macro_precision =  report['macro avg']['precision'] 
        # macro_recall = report['macro avg']['recall']    
        # macro_f1 = report['macro avg']['f1-score']
        # print(report['True']['precision'])
        # print(report['True']['recall'])
        score = report['True']['precision']*10 + report['True']['recall']
        print(score)
        if score > max_score:
            # print(f"new best bar = {bar}")
            max_score = score
            best_bar = bar
    # print(f"best bar = {best_bar}")
    # print(f"max score = {max_score}")
    y_pred_bin = y_pred[:, 1] > best_bar  # Threshold at 0.5
    # print(f"amount predicted QPs = {sum(y_pred_bin)}")
    # print(f"amount actual QPs = {sum(y)}")
    print("best Classification Report:")
    report = classification_report(y, y_pred_bin)

    print(report)

    """
    van de code hierboven een for loop maken die de beste threshold vindt
    """



def init(to_log=True,mark_first=True,mark_second=False):
    def import_data():
    # Path to the pickle file
        pickle_file_path = 'processed_data.pkl'

        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data = load_processed_data(pickle_file_path)
            print("Loaded data from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            file_path = '../../assets/M5415_10kn_JONSWAP_3m_10s/output.csv'
            data = pd.read_csv(file_path)
            data = data[['t', 'z_wf']]
            data = data.iloc[1:]
            data = data.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data, pickle_file_path)
            print("Processed data saved to pickle.")
        # heave_data = data['z_wf']

        data2_path = 'data2.pkl'

        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data2 = load_processed_data(data2_path)
            print("Loaded data2 from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            file_path = '../../assets/data_ed_2_clean.csv'
            data2 = pd.read_csv(file_path)
            data2 = data2[['t', 'z_wf']]
            data2 = data2.iloc[1:]
            data2 = data2.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data2, data2_path)
            print("Processed data saved to pickle.")
        
        
        return data, data2
    data, data2 = import_data() # en data3
    y = dfb.init_QPs(data, dataset_id=1, new = True)
    print(f"total length original y {len(y)}")
    y2 = dfb.init_QPs(data2, dataset_id=2, new = False)
    # y = dfb.moveQP(y)
    start_index1 = 0
    stop_index1 = len(data)-120
    start_index2 = 0
    stop_index2 = 2000

    


    
    if mark_first:
        X, offset1 = create_qp_labeled_dataset_faster(data[start_index1:stop_index1], dataset_id=1, new=True)
    else:
        X, offset1 = None, None
    if mark_second:
        X2, offset2 = create_qp_labeled_dataset_faster(data2[start_index2:stop_index2], dataset_id=2, new=False)
    else:
        X2, offset2 = None, None
    
    
    
    # print(f"features: {X[:10]}")
    # print(f"offset1: {offset1}")

    if to_log:
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
    
    if X is not None:
        y  = y[start_index1+offset1:stop_index1+2] # 100 is lookback window
    if X2 is not None:
        y2 = y2[start_index2+offset2:stop_index2+2]
    if to_log:
        print(len(X))
        print(f"after trimming y1 to fit to X sum = {sum(y)    }")

    return X, X2, y, y2

def model_train(X, y):
    pickle_model_path = 'pickel_model.pkl'

    try:
        # Try to load the data if it's already saved
        
        raise FileNotFoundError
        model, scaler, X_test, y_test = load_processed_data(pickle_model_path)
        print("Loaded model from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        model, scaler, X_test, y_test = train_qp_predictor(X, y)
        save_processed_data((model, scaler), pickle_model_path)
        print("Processed data saved to pickle.")

    return model, scaler, X_test, y_test


def main():


    X, X2, y, y2 = init()
    # print(X)
    print((y[3100:3400]))
    # eerste QP van data1 volgens de nieuwe def is in 3100-3150

    print(len(X))
    print(len(y))
    quit()
    
    model, scaler, X_test, y_test = model_train(X, y)

    # quit()

    evaluate(model, scaler, X_test, y_test)
    
    # evaluate(model, scaler, X2, y2)

    # odds_ratio_linear_check(X, y)

    # TODO het markeren efficienter maken



    
    
main()
# X, X2, y, y2 = init()
# odds_ratio_linear_check(X, y)



"""
first two different extrema vectors

 [ 0.1196906   0.08447302  0.08523453 -0.2532199 ]
 [ 0.08447302  0.08523453 -0.2532199   0.3849707 ]

"""