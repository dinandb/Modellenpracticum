import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import pickle
from sklearn.decomposition import PCA
from emma_dinand import features
from emma_dinand.extras import load_processed_data, save_processed_data

pd.set_option('display.max_rows', None)
# Now write your code that displays DataFrames


from sklearn.tree import DecisionTreeClassifier

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

# def create_qp_labeled_dataset_faster(data, dataset_id, new = False):
#     """
#     Create labeled dataset for QP prediction.
    
#     Parameters:
#     - heave_data: numpy array of heave measurements
#     - qp_mask: boolean mask of QP periods
#     - lookback_window: number of time steps to look back before QP
    
#     Returns:
#     - X: features
    
#     """

#     pickle_file_path = f'slidingwindowclaudebackend/pickle_saves/vectors/processed_data_features{dataset_id}.pkl'
#     try:

#         # Try to load the data if it's already saved
#         if new:
#             raise FileNotFoundError
        
        
#         features, offset = load_processed_data(pickle_file_path)
#         print("Loaded features from pickle.")

#     except FileNotFoundError:
#         # If the pickle file doesn't exist, process the data and save it

#         features = []
#         """
#         extract eerst max waarden van de dataset (nu alleen gebaseerd op heave) mbv get_only_max_vals
#         maak de eerste feature vector 
#         dan voor elke
#         """


#         # heave, sway, surge, yaw, roll, pitch = ['z_wf'], [y_wf'],['x_wf'],['psi_wf'],['phi_wf'],['theta_wf']
#         # extrema_heave, extrema_indices_heave = get_only_max_vals(data, colname="z_wf", name=str(dataset_id)+"_z_wf", new = True)
#         # extrema_sway, extrema_indices_sway = get_only_max_vals(data, colname="y_wf", name=str(dataset_id)+"_y_wf", new = True)
#         # extrema_surge, extrema_indices_surge = get_only_max_vals(data, colname="x_wf", name=str(dataset_id)+"_x_wf", new = True)
#         # extrema_yaw, extrema_indices_yaw = get_only_max_vals(data, colname="psi_wf", name=str(dataset_id)+"_psi_wf", new = True)
#         # extrema_roll, extrema_indices_roll = get_only_max_vals(data, colname="phi_wf", name=str(dataset_id)+"_phi_wf", new = True)
#         # extrema_pitch, extrema_indices_pitch = get_only_max_vals(data, colname="theta_wf", name=str(dataset_id)+"_theta_wf", new = True)

#         # offset_heave = extrema_indices_heave[NO_EXTREMA_LOOKBACK-1] + 2

#         # features_heave = []
#         # for i in range(len(extrema_indices)-2):
#         #     if i + 3 < len(extrema_indices):
#         #         features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(extrema_indices[i+3]-extrema_indices[i+2]))
#         #     else:
#         #         features.extend([[extrema.iloc[i]['z_wf'], extrema.iloc[i+1]['z_wf'], extrema.iloc[i+2]['z_wf']]]*(len(data)-extrema_indices[i+2]))






#         var_map = {
#             "heave": "z_wf",
#             "sway": "y_wf",
#             "surge": "x_wf",
#             "yaw": "psi_wf",
#             "roll": "phi_wf",
#             "pitch": "theta_wf"
#         }
#         var_map_heave = {
#             "heave": "z_wf",
#         }
    
    
#         extrema_dict = {}
#         extrema_indices_dict = {}
#         offset = 0
#         offset_dict = {}
#         features_dict = {}

#         for var, col in var_map.items():
#             extrema, extrema_indices = get_only_max_vals(data, colname=col, name=f"{dataset_id}_{col}", new=False)
#             extrema_dict[var] = extrema
#             extrema_indices_dict[var] = extrema_indices
            
#             cur_offset = extrema_indices[NO_EXTREMA_LOOKBACK-1] + 2
#             offset_dict[var] = cur_offset
#             if cur_offset > offset:
#                 offset = extrema_indices_dict[var][NO_EXTREMA_LOOKBACK-1] + 2
#             print(f"cur offset {cur_offset}, var = {var}")

#         # let op! rekening houden met verschillende offsets. als er niet goed rekening mee gehouden wordt 
#         # dan kan de plaatsing van de features boven elkaar misschien niet kloppen. 

#         # in schrift staat hoe te doen

#         # dus eigenlijk moet je de offset van de extrema indices gebruiken om de features te maken.


#         for var, col in var_map.items():
#             extrema = extrema_dict[var]                  # dict holding extrema_{var} DataFrames
#             extrema_indices = extrema_indices_dict[var]  # dict holding extrema_indices_{var} lists
#             features_dict[var] = []

#             for i in range(len(extrema_indices) - 2):
#                 if i + 3 < len(extrema_indices):
#                     features_dict[var].extend([[extrema.iloc[i][col], extrema.iloc[i+1][col], extrema.iloc[i+2][col]]] * 
#                                     (extrema_indices[i+3] - extrema_indices[i+2]))
#                 else:
#                     features_dict[var].extend([[extrema.iloc[i][col], extrema.iloc[i+1][col], extrema.iloc[i+2][col]]] * 
#                                     (len(data) - extrema_indices[i+2]))
                    
#         # now, loop over all features_dict and remove the first offset - offset_dict[var] values from each list in features_dict[var]
#         for var, col in var_map.items():
#             offset_var = offset_dict[var]
#             # print(f"removing first {offset - offset_var} values from {var}")
#             features_dict[var] = features_dict[var][offset - offset_var:]
    
#         # now, combine all features_dict into one list of features

#         features = []
#         # for i in range(len(features_dict["heave"])):
#         #     features.append([features_dict["heave"][i][0].item(), features_dict["heave"][i][1].item(), features_dict["heave"][i][2].item(),
#         #                     features_dict["sway"][i][0].item(), features_dict["sway"][i][1].item(), features_dict["sway"][i][2].item(),
#         #                     features_dict["surge"][i][0].item(), features_dict["surge"][i][1].item(), features_dict["surge"][i][2].item(),
#         #                     features_dict["yaw"][i][0].item(), features_dict["yaw"][i][1].item(), features_dict["yaw"][i][2].item(),
#         #                     features_dict["roll"][i][0].item(), features_dict["roll"][i][1].item(), features_dict["roll"][i][2].item(),
#         #                     features_dict["pitch"][i][0].item(), features_dict["pitch"][i][1].item(), features_dict["pitch"][i][2].item()])
#         for i in range(len(features_dict["heave"])):
#             features.append([features_dict["heave"][i][0].item(), features_dict["heave"][i][1].item(), features_dict["heave"][i][2].item()])

#         if False:
#             print(f"len features {len(features)}")
#             print(f"amount that did not get features = {len(data['z_wf']) - len(features)}")
#             # ^ looking good
#             print(f"offset (should be equal to above ? ) = {offset}")
#         if dataset_id != 4:
#             print(f"shape features before PCA = {np.array(features).shape}")
#             pca = PCA(n_components=0.95)
#             features = pca.fit_transform(features)
#             print(f"shape features after PCA = {features.shape}")

#         save_processed_data((features, offset), pickle_file_path)
#         print(f"Processed features saved to pickle. id={dataset_id}")
#     return features, offset





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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    X_train = X[:int(len(X)/2)]
    X_test = X[int(len(X)/2):]
    y_train = y[:int(len(X)/2)]
    y_test = y[int(len(X)/2):]


    # print(np.sum(y_train), np.sum(y_test))
    # print(len(y_train), len(y_test))
    
    # Scale features
    scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = X_train
    # X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with specific constraints
    # clf = RandomForestClassifier(
    #     n_estimators=200,
    #     class_weight='balanced',  # Handle class imbalance
    #     min_samples_leaf=3,  # Reduce overfitting
    #     max_depth=20  # Prevent too complex models
    # )

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=20),  # Weak learner
        n_estimators=50,  # Number of weak learners
        learning_rate=0.1,  # Controls contribution of each learner
        random_state=42
    )

    # clf = LogisticRegression(random_state=0, max_iter=1000)

    clf.fit(X_train_scaled, y_train)
    
    return clf, scaler, X_test, y_test


def evaluate(model, scaler, X, y):
    # predict y based on X with model
    # evaluate performance with real y


    
    # X_scaled = scaler.transform(X)
    X_scaled = X
    
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
    y_pred_bin = y_pred[:, 1] > 0.5  # Threshold at 0.5
    print(f"in test set amount QPs = {sum(y)}")
    print(f"total predicted QP's {sum(y_pred_bin)}")
    print(len(y_pred_bin))
    # print(f"amount predicted QPs = {sum(y_pred_bin)}")
    # print(f"amount actual QPs = {sum(y)}")
    print("best Classification Report:")
    report = classification_report(y, y_pred_bin)

    print(report)

    """
    van de code hierboven een for loop maken die de beste threshold vindt
    """


def model_train(X, y, id=1, new = False):
    pickle_model_path = f'emma_dinand/pickle_saves/modellen/model{str(id)}_RF.pkl'


    
    
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

    Xs,ys = features.main()

    index = 2
    X = Xs[index]
    y = ys[index]

    model, scaler, X_test, y_test = model_train(X, y, id = index, new=True)
    
    
    evaluate(model, scaler, X_test, y_test)
    # evaluate(model, scaler, Xs[2], ys[2])
    
    

    # odds_ratio_linear_check(X, y)





    
    
main()
# X, X2, y, y2 = init()
# odds_ratio_linear_check(X, y)



"""
first two different extrema vectors

 [ 0.1196906   0.08447302  0.08523453 -0.2532199 ]
 [ 0.08447302  0.08523453 -0.2532199   0.3849707 ]

"""