import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import pickle
from sklearn.decomposition import PCA



from sklearn.tree import DecisionTreeClassifier
import data_frame_build as dfb
from extras import load_processed_data, save_processed_data, get_only_max_vals, get_only_max_vals_vector
import odds_ratio_check 
import Detect_QP_CasperSteven

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

    pickle_file_path = f'slidingwindowclaudebackend/pickle_saves/vectors/processed_data_features{dataset_id}.pkl'
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
        extract eerst max waarden van de dataset (nu alleen gebaseerd op heave) mbv get_only_max_vals
        maak de eerste feature vector 
        dan voor elke
        """


        # heave, sway, surge, yaw, roll, pitch = ['z_wf'], [y_wf'],['x_wf'],['psi_wf'],['phi_wf'],['theta_wf']
        # extrema_heave, extrema_indices_heave = get_only_max_vals(data, colname="z_wf", name=str(dataset_id)+"_z_wf", new = True)
        # extrema_sway, extrema_indices_sway = get_only_max_vals(data, colname="y_wf", name=str(dataset_id)+"_y_wf", new = True)
        # extrema_surge, extrema_indices_surge = get_only_max_vals(data, colname="x_wf", name=str(dataset_id)+"_x_wf", new = True)
        # extrema_yaw, extrema_indices_yaw = get_only_max_vals(data, colname="psi_wf", name=str(dataset_id)+"_psi_wf", new = True)
        # extrema_roll, extrema_indices_roll = get_only_max_vals(data, colname="phi_wf", name=str(dataset_id)+"_phi_wf", new = True)
        # extrema_pitch, extrema_indices_pitch = get_only_max_vals(data, colname="theta_wf", name=str(dataset_id)+"_theta_wf", new = True)

        # offset_heave = extrema_indices_heave[NO_EXTREMA_LOOKBACK-1] + 2

        # features_heave = []
        # for i in range(len(extrema_indices)-2):
        #     if i + 3 < len(extrema_indices):
        #         features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(extrema_indices[i+3]-extrema_indices[i+2]))
        #     else:
        #         features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(len(data)-extrema_indices[i+2]))






        var_map = {
            "heave": "z_wf",
            "sway": "y_wf",
            "surge": "x_wf",
            "yaw": "psi_wf",
            "roll": "phi_wf",
            "pitch": "theta_wf"
        }
    
    
        extrema_dict = {}
        extrema_indices_dict = {}
        offset = 0
        offset_dict = {}
        features_dict = {}

        for var, col in var_map.items():
            extrema, extrema_indices = get_only_max_vals(data, colname=col, name=f"{dataset_id}_{col}", new=False)
            extrema_dict[var] = extrema
            extrema_indices_dict[var] = extrema_indices
            
            cur_offset = extrema_indices[NO_EXTREMA_LOOKBACK-1] + 2
            offset_dict[var] = cur_offset
            if cur_offset > offset:
                offset = extrema_indices_dict[var][NO_EXTREMA_LOOKBACK-1] + 2
            # print(f"cur offset {cur_offset}, var = {var}")

        # let op! rekening houden met verschillende offsets. als er niet goed rekening mee gehouden wordt 
        # dan kan de plaatsing van de features boven elkaar misschien niet kloppen. 

        # in schrift staat hoe te doen

        # dus eigenlijk moet je de offset van de extrema indices gebruiken om de features te maken.


        for var, col in var_map.items():
            extrema = extrema_dict[var]                  # dict holding extrema_{var} DataFrames
            extrema_indices = extrema_indices_dict[var]  # dict holding extrema_indices_{var} lists
            features_dict[var] = []

            for i in range(len(extrema_indices) - 2):
                if i + 3 < len(extrema_indices):
                    features_dict[var].extend([[extrema.iloc[i][col], extrema.iloc[i+1][col], extrema.iloc[i+2][col]]] * 
                                    (extrema_indices[i+3] - extrema_indices[i+2]))
                else:
                    features_dict[var].extend([[extrema.iloc[i][col], extrema.iloc[i+1][col], extrema.iloc[i+2][col]]] * 
                                    (len(data) - extrema_indices[i+2]))
                    
        # now, loop over all features_dict and remove the first offset - offset_dict[var] values from each list in features_dict[var]
        for var, col in var_map.items():
            offset_var = offset_dict[var]
            # print(f"removing first {offset - offset_var} values from {var}")
            features_dict[var] = features_dict[var][offset - offset_var:]
    
        # now, combine all features_dict into one list of features
        features = []
        for i in range(len(features_dict["heave"])):
            features.append([features_dict["heave"][i][0], features_dict["heave"][i][1], features_dict["heave"][i][2],
                             features_dict["sway"][i][0], features_dict["sway"][i][1], features_dict["sway"][i][2],
                             features_dict["surge"][i][0], features_dict["surge"][i][1], features_dict["surge"][i][2],
                             features_dict["yaw"][i][0], features_dict["yaw"][i][1], features_dict["yaw"][i][2],
                             features_dict["roll"][i][0], features_dict["roll"][i][1], features_dict["roll"][i][2],
                             features_dict["pitch"][i][0], features_dict["pitch"][i][1], features_dict["pitch"][i][2]])

        if False:
            print(f"len features {len(features)}")
            print(f"amount that did not get features = {len(data['z_wf']) - len(features)}")
            # ^ looking good
            print(f"offset (should be equal to above ? ) = {offset}")
        print(f"shape features before PCA = {np.array(features).shape}")
        pca = PCA(n_components=0.95)
        features = pca.fit_transform(features)
        print(f"shape features after PCA = {features.shape}")

        save_processed_data((features, offset), pickle_file_path)
        print(f"Processed features saved to pickle. id={dataset_id}")
    return features, offset





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
    # print(np.sum(y_train), np.sum(y_test))
    # print(len(y_train), len(y_test))
    
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


def evaluate(model, scaler, X, y):
    # predict y based on X with model
    # evaluate performance with real y


    
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict_proba(X_scaled)
    max_score = 0
    best_bar = 0
    # print(y_pred.shape)  # Should be (n_samples, 2) for binary classification
    # # quit()
    # for bar in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    #     y_pred_bin = y_pred[:, 1] > bar  # Threshold at 0.5
        
    #     # print("Classification Report:")
    #     report = classification_report(y, y_pred_bin, output_dict=True)
    #     # macro_precision =  report['macro avg']['precision'] 
    #     # macro_recall = report['macro avg']['recall']    
    #     # macro_f1 = report['macro avg']['f1-score']
    #     # print(report['True']['precision'])
    #     # print(report['True']['recall'])
    #     score = report[1]['precision']*10 + report[1]['recall']
    #     print(score)
    #     if score > max_score:
    #         # print(f"new best bar = {bar}")
    #         max_score = score
    #         best_bar = bar
    # # print(f"best bar = {best_bar}")
    # # print(f"max score = {max_score}")
    y_pred_bin = y_pred[:, 1] > 0.55  # Threshold at 0.5
    print(sum(y_pred_bin[:4000]))
    print(len(y_pred_bin))
    # print(f"amount predicted QPs = {sum(y_pred_bin)}")
    # print(f"amount actual QPs = {sum(y)}")
    print("best Classification Report:")
    report = classification_report(y, y_pred_bin)

    print(report)

    """
    van de code hierboven een for loop maken die de beste threshold vindt
    """
def format_data(data, data2, data3):
    path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data_clean1.pkl'
    try:
        data = load_processed_data(path)

    except:
    
        file_path = '../../assets/data1.csv'
        data = pd.read_csv(file_path)
        data = data[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        data = data.iloc[1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data, path)

    
    path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data_clean2.pkl'
    try:
        data2= load_processed_data(path)

    except:
    
        file_path = '../../assets/data2.csv'
        data2 = pd.read_csv(file_path)
        data2 = data2[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        data2 = data2.iloc[1:]
        data2 = data2.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data2, path)

    path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data_clean3.pkl'
    try:
        data3 = load_processed_data(path)

    except:
    
        file_path = '../../assets/data3.csv'
        data3 = pd.read_csv(file_path)
        data3 = data3[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        data3 = data3.iloc[1:]
        data3 = data3.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data3, path)

    return data, data2, data3


def init(to_log=True,mark_first=True,mark_second=True):
    def import_data():
    # Path to the pickle file
        pickle_file_path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data1.pkl'

        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data = load_processed_data(pickle_file_path)
            print("Loaded data from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            file_path = '../../assets/data1.csv'
            data = pd.read_csv(file_path, header=[0,1])
            data = data[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
            # data = data.iloc[1:]
            data = data.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data, pickle_file_path)
            print("Processed data saved to pickle.")
        # heave_data = data['z_wf']

        data2_path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data2.pkl'

        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data2 = load_processed_data(data2_path)
            print("Loaded data2 from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            file_path = '../../assets/data2.csv'
            data2 = pd.read_csv(file_path, header=[0,1])
            data2 = data2[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
            # data2 = data2.iloc[1:]
            data2 = data2.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data2, data2_path)
            print("Processed data saved to pickle.")
        
        data3_path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data3.pkl'

        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data3 = load_processed_data(data3_path)
            print("Loaded data3 from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            file_path = '../../assets/data3.csv'
            data3 = pd.read_csv(file_path, header=[0,1])
            data3 = data3[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
            # data2 = data2.iloc[1:]
            data3 = data3.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data3, data3_path)
            print("Processed data saved to pickle.")
        
        
        return data, data2, data3
    data, data2, data3 = import_data()
    
    y = Detect_QP_CasperSteven.mark_QP(data,name="QP1", new=False)

    # print(f"y {y}")
    # print(f"y shape {y.shape}")
    
    # print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
    # print(f"total length original y {len(y)}")
    y2 = Detect_QP_CasperSteven.mark_QP(data2,name="QP2", new=False)
    # print(f"y2 {y2[:40]}")
    # print(f"y2 shape {y2.shape}")
    y3 = Detect_QP_CasperSteven.mark_QP(data3, name="QP3", new=False)
    
    # print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
    # y = dfb.moveQP(y)
    start_index1 = 0
    stop_index1 = len(data)-120
    start_index2 = 0
    stop_index2 = len(data2)-120
    start_index3 = 0
    stop_index3 = len(data3)-120

    if to_log:
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
    
    y = dfb.moveQP(y)
    y2 = dfb.moveQP(y2)
    y3 = dfb.moveQP(y3)

    data, data2, data3 = format_data(data, data2, data3)
    
    
    if to_log:
        print("after moving QPs")
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
    
    if mark_first:
        X, offset1 = create_qp_labeled_dataset_faster(data[start_index1:stop_index1], dataset_id=1, new=False)
    else:
        X, offset1 = None, None
    if mark_second:
        X2, offset2 = create_qp_labeled_dataset_faster(data2[start_index2:stop_index2], dataset_id=2, new=False)
    else:
        X2, offset2 = None, None

    if mark_second:
        X3, offset3 = create_qp_labeled_dataset_faster(data3[start_index3:stop_index3], dataset_id=3, new=True)
    else:
        X3, offset3 = None, None
    

    
    if X is not None:
        y  = y[start_index1+offset1:stop_index1+2]
    if X2 is not None:
        y2 = y2[start_index2+offset2:stop_index2+2]
    if X3 is not None:
        y3 = y3[start_index3+offset3:stop_index3+2]

    return X, X2, X3, y, y2, y3

def model_train(X, y, id=1, new = False):
    pickle_model_path = f'slidingwindowclaudebackend/pickle_saves/modellen/model{str(id)}.pkl'


    
    
    try:
        # Try to load the data if it's already saved
        if new:
            raise FileNotFoundError
        model, scaler, X_test, y_test = load_processed_data(pickle_model_path)
        print("Loaded model from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        print("going to train model")
        model, scaler, X_test, y_test = train_qp_predictor(X, y)
        print("Trained model.")
        save_processed_data((model, scaler, X_test, y_test), pickle_model_path)
        print("Processed model saved to pickle.")

    return model, scaler, X_test, y_test


def main():


    X, X2, X3, y, y2, y3 = init()
    print(f"amount QPs in y{ sum(y)}")
    # print(X3[:100])
    # quit()
    


    model, scaler, X_test, y_test = model_train(X3, y3, id = 3, new = True)
    # quit()
    

    evaluate(model, scaler, X_test, y_test)
    
    

    # odds_ratio_linear_check(X, y)





    
    
main()
# X, X2, y, y2 = init()
# odds_ratio_linear_check(X, y)



"""
first two different extrema vectors

 [ 0.1196906   0.08447302  0.08523453 -0.2532199 ]
 [ 0.08447302  0.08523453 -0.2532199   0.3849707 ]

"""