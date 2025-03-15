import pandas as pd
import extras
import data_frame_build
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



    
def oversample_smote(X, y):
    n = len(X)
    # hier even in veranderen voor de oversample
    smote = SMOTE(sampling_strategy={1: int(n/15), 0: int(n*20)}, random_state=898)
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


import numpy as np

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


def grid_search_eval(X,y):
    oversample_smote(X, y)
    # print(X)
    # print(sum(y))
    # Create a pipeline with a scaler and a placeholder for classifier
    sample_weights = np.where(y==1, 1.0, 1.0)
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Add a scaler for preprocessing
        ('classifier', LogisticRegression())  # Placeholder for any classifier
    ])

    # Define the parameter grid
    param_grid = [
        {
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
        },
        {
            'classifier': [KNeighborsClassifier()],
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
        },
        # {
        #     'classifier': [RandomForestClassifier()],
        #     'classifier__n_estimators': [50, 100],
        #     'classifier__max_depth': [5, 10, None],
        # }
    ]

    # GridSearchCV will search across these classifiers and their corresponding hyperparameters
    grid_search = GridSearchCV(pipe, param_grid, cv=5)

    # Fit with cross-validation to find best params
    grid_search.fit(X, y)#,  classifier__sample_weight=sample_weights)

    # Get the best model and parameters
    best_clf = grid_search.best_estimator_
    print(best_clf)
    return best_clf

def eval_classifier(X, y, bar, skf, min_err, cur_best, param, clf, name):
    error = 0
    for _, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train, X_eval = X.iloc[train_index], X.iloc[test_index]
        y_train, y_eval = y.iloc[train_index], y.iloc[test_index]
        
        X_train, y_train = oversample_smote(X_train, y_train)

        clf.fit(X_train, y_train)

        # Predict and evaluate
        # y_pred = clf.predict(X_eval)
        y_proba = clf.predict_proba(X_eval)[:, 1]
        y_pred = (y_proba > bar).astype(int)
        # print(sum(y_pred))
        y_pred = adjust_predictions(y_pred, y_eval)
        TN, FP, FN, TP = calcConfusionMat(y_pred, y_eval)
        error += FN + FP*30
    if error < min_err:
        min_err = error
        cur_best = f"{name}, params={param}, bar = {bar}, err = {error}"
    return min_err, cur_best

def create_and_eval_classifier(X, y):



    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)

    classifiers = [
        {
            'name': 'LogisticRegression',
            'classifier': LogisticRegression(

            )
        },
        {
            'name': 'RandomForest',
            'classifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight=None
            )
        },
        {
            'name': 'KNeighborsClassifier',
            'classifier': KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean',
                weights='distance'
            )
        }
    ]
    cur_best = ""
    min_err = 999999999
    for clf_info in classifiers:
        print("next classifier")
        
        for bar in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            print("next bar")
            if clf_info['name'] == 'LogisticRegression':
                for i in range(1):
                    min_err, cur_best = eval_classifier(X, y, bar, skf, min_err, cur_best, i, LogisticRegression(
                        penalty='l2',
                        C=1.0,
                        solver='lbfgs',
                        max_iter=100,
                        fit_intercept=True,
                        class_weight=None,
                        random_state=42
                    ), clf_info['name'])
                        
            elif clf_info['name'] == 'KNeighborsClassifier':
                for i in range(10):
                    min_err, cur_best = eval_classifier(X, y, bar, skf, min_err, cur_best, i+1, KNeighborsClassifier(
                        n_neighbors=i+1,
                        metric='euclidean',
                        weights='distance'
                    ), clf_info['name'])
                        
            elif clf_info['name'] == 'RandomForst':
                for i in range(1):
                    min_err, cur_best = eval_classifier(X, y, bar, skf, min_err, cur_best, i, RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        random_state=42,
                        class_weight=None
                    ), clf_info['name'])

    return cur_best



# Function to save processed data as a pickle file
def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)



def init_data():
    # Path to the pickle file
    pickle_file_path = 'processed_data.pkl'

    try:
        # Try to load the data if it's already saved
        raise(FileNotFoundError)
        data = load_processed_data(pickle_file_path)
        print("Loaded data from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        file_path = '../../assets/data_ed_2_clean.csv'
        data = pd.read_csv(file_path)
        data = data[['t', 'z_wf']]
        data = data.iloc[1:]
        data = data.apply(pd.to_numeric, errors='coerce')
        

        
        
        # Save the processed data to a pickle file
        # save_processed_data(data, pickle_file_path)
        # print("Processed data saved to pickle.")
    
    # data = data_frame_build.generate_prev_heavs(3, 1, data)
    
    # Labels
    labels = data_frame_build.init_QPs(data)
    # print(sum(labels))
    labels = data_frame_build.moveQP(labels)
    # print(sum(labels))

    labels = labels + [False]*(len(data)-len(labels))



    y = pd.Series(labels, name="Label")
    X = data

    y = y.astype(int) if y.dtype == 'bool' else y

    return X, y



def main0():
    X, y = init_data()
    # print(f"len X = {len(X)}")
    print(X)
    # # print(f"len y = {len(y)}")
    # print(y)
    # Assuming 'data' is a pandas DataFrame and 'label' is a pandas Series


    
    

    # print(y.iloc[:1000].to_string())  # Ensures all values are printed in a readable format

    
    train_eval_X = X.iloc[:int(len(X) * 0.8)]
    train_eval_y = y.iloc[:int(len(y) * 0.8)]

    test_X = X.iloc[int((len(X)*.8)):]
    test_y = y.iloc[int((len(y)*.8)):]

    test_X, indices = extras.get_only_max_vals(test_X, test_y)
    test_y = pd.Series([test_y[test_y.index[i]] for i in indices], name="Label").iloc[1:]
    test_X = extras.get_diffs(test_X, power=1)

    train_eval_X, indices = extras.get_only_max_vals(train_eval_X, train_eval_y)
    
    train_eval_y = pd.Series([train_eval_y[train_eval_y.index[i]] for i in indices], name="Label").iloc[1:]

    train_eval_X = extras.get_diffs(train_eval_X, power=1)
    print(train_eval_X)
    print(train_eval_X['Derivative #1 ^1'].corr(train_eval_y))
    train_eval_X, train_eval_y = oversample_smote(train_eval_X, train_eval_y)


    # clf = create_and_eval_classifier(train_eval_X, train_eval_y)
    # print(clf)
    gen_err(LogisticRegression().fit(train_eval_X, train_eval_y), test_X, test_y)





def calcConfusionMat(y_pred, y_test):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    # print(y_test)
    # print(f"start {start}")
    assert len(y_pred) == len(y_test)
    # print(y_test)
    for i in range(len(y_pred)):
        if y_pred[i] and y_test[y_test.index[i]]:
            TP += 1
        elif not y_pred[i] and y_test[y_test.index[i]]:
            FN += 1
        elif not y_pred[i] and not y_test[y_test.index[i]]:
            TN += 1
        elif y_pred[i] and not y_test[y_test.index[i]]:
            FP += 1
    return TN, FP, FN, TP

    



def gen_err(clf, X_test, y_test):
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.99).astype(int)
    # print(sum(y_pred))
    y_pred = adjust_predictions(y_pred, y_test)
    pd.set_option('display.max_rows', 1300)  # Adjust this if needed
    # print(f"len test {len(y_test)}")
    # print(y_test[:(1000)])
    # print(y_pred[:(1000)])

    # weights = []
    # for x in list(y_test):
    #     weights.append(0.005 + .99*x) # for the first class (x=0), we get a weight 0.005, for the second class 0.995
    #     # weights.append(1)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    start = y_test.index[0]
    # print(y_test)
    # print(f"start {start}")
    for i in range(len(y_pred)):
        if y_pred[i] and y_test[y_test.index[i]]:
            TP += 1
        elif not y_pred[i] and y_test[y_test.index[i]]:
            FN += 1
        elif not y_pred[i] and not y_test[y_test.index[i]]:
            TN += 1
        elif y_pred[i] and not y_test[y_test.index[i]]:
            FP += 1
    FN, TP = int(FN/6), int(TP/6)
    print(f"TP {TP}")
    print(f"FP {FP}")
    print(f"TN {TN}")
    print(f"FN {FN}")
    confusion_matrix = np.array([[TN, FP],  # First row: TN, FP
                                 [FN, TP]])  # Second row: FN, TP

    print(confusion_matrix)
    
    # print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], sample_weight=weights, digits=4))
    # print(f"Accuracy score: {accuracy_score(y_test, y_pred, sample_weight=weights)}")
    # print("ROC-AUC Score (Evaluation Set):", roc_auc_score(y_test, y_proba, sample_weight=weights))  
main0()

