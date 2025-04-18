

import time
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_diffs(data, colname="z_wf", steps=1, power=1):
    """
    Calculate the derivatives of a given column in the data for the specified number of steps and power.

    Parameters:
    - data (pd.DataFrame): The data containing the column to differentiate.
    - colname (str): The column name for which derivatives are computed.
    - steps (int): The number of lag steps to create.
    - power (int): The number of times the difference operation is applied to the column.

    Returns:
    - pd.DataFrame: The original data with new columns for the derivatives.
    """
    n = len(data)
    relevant_sim_data_derivatives = data.copy()  # To ensure we don't modify the original data

    # Initial derivative (der0)
    der0 = data[colname].values.copy()
    der1 = np.zeros(n)

    # Loop to calculate derivatives (difference) with power
    for _ in range(power):
        for k in range(1, n):
            der1[k] = abs(der0[k] - der0[k - 1])
        der0 = der1.copy()

    # Set the first value as NA (we can use np.nan)
    der0[0] = np.nan

    # Add the lagged derivatives as new columns
    for i in range(steps, 0, -1):
        col_name = f"Derivative #{i} ^{power}"
        relevant_sim_data_derivatives[col_name] = pd.Series(np.roll(der0, i-1), index=data.index)

    return relevant_sim_data_derivatives[1:]

def get_only_max_vals(data, colname="z_wf", name="0", new = False):
    
    try:

        # Try to load the data if it's already saved
        if new:
            raise FileNotFoundError
        
        
        column = load_processed_data(f"emma_dinand/pickle_saves/vectors/{name}.pkl")
        print(f"Loaded column from pickle.{name}")

    except FileNotFoundError: 
        # print(f"could not find file {name}")
        column = data[colname].to_numpy(dtype=np.float32)


        save_processed_data(column, f"emma_dinand/pickle_saves/vectors/{name}.pkl")
        print(f"Saved column to pickle.{name}")

    
    return get_only_max_vals_vector(column, data)

        
         
def get_only_max_vals_vector(column, data):
    """
    Find local extrema (maxima and minima) in a time series.
    
    Parameters:
    data (DataFrame): Input data
    colname (str): Name of the column to analyze
    
    Returns:
    DataFrame: Rows from the original data where extrema were found
    """

    indices = []
    
    for i in range(2, len(column)-2):

        # Check for local maxima OR local minima
        try:
            if ((column[i-2]) < column[i-1] and column[i-1] < column[i] and column[i] > column[i+1] and column[i+1] > column[i+2]) or ((column[i-2]) > column[i-1] and column[i-1] > column[i] and column[i] < column[i+1] and column[i+1] < column[i+2]):
                indices.append(i)
        except Exception as e: 
            print(f"Error at index {i}: {column[i-1]}, {column[i]}, {column[i+1]}")
            print(e)
            quit()
    if data is None:
        return column[indices]
    else:
        # print(indices)
        # print(len(data))  
        return np.abs(data.iloc[indices]), indices


def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def display_information(X, y):
    
    
    # X is an array of arrays. every ith array belongs to the ith label in y.
    # every array has the same length, n.
    # do, for every 1 <= i <= n, the following:
    # display a histogram of all the ith values in the arrays in X, and colour them according to the label in y.
    # print(f"X shape {X.shape}")
    # print(f"y shape {y.shape}")
    import matplotlib.pyplot as plt

    # Transpose X to iterate over each "column" (i.e., ith values across all arrays)
    num_samples, n = X.shape

    # Convert y to a NumPy array if it's not already
    y = np.array(y)

    for i in range(n):
        values_class_0 = X[y == 0, i]
        values_class_1 = X[y == 1, i]

        plt.figure(figsize=(6, 4))
        plt.hist(values_class_0, bins=30, alpha=0.6, label='Class 0', color='blue')
        plt.hist(values_class_1, bins=30, alpha=0.6, label='Class 1', color='red')
        plt.title(f"Histogram of feature {i}")
        plt.xlabel(f"Value at index {i}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

# def train_model(model name, x train, y train) met cv



def evaluate(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(f"amount QPs in test set {sum(y_test)}, ")
    print(f"amount QPs in predicted test set {sum(y_pred)}")
    print('Confusion matrix')
    print(cnf_matrix)
    print('---------------')
    print('Precision:', metrics.precision_score(y_test, y_pred))
    print('Recall:', metrics.recall_score(y_test, y_pred))
    print('F1 Score:', metrics.f1_score(y_test, y_pred))

def evaluate_torch_ltsm(model, X_test_scaled, y_test, scaler):
    # Set the model to evaluation mode so dropout/batch norm work in inference mode.
    model.eval()
    
    # Prepare the test data (unsqueeze for the sequence dimension if needed)
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create Dataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16)
    
    all_preds = []
    all_targets = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)  # outputs shape: (batch, num_classes)
            preds = outputs.argmax(dim=1)  # Get index of highest logit (predicted class)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Overall accuracy calculation
    total_correct = (all_preds == all_targets).sum().item()
    accuracy = total_correct / len(all_targets)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classes for binary classification: 0 and 1
    classes = [0, 1]
    
    print(f"Class 0: {sum(all_targets == 0)}, Class 1: {sum(all_targets == 1)}")
    for cls in classes:
        # True Positives: predicted cls and true label is cls.
        TP = ((all_preds == cls) & (all_targets == cls)).sum().item()
        # False Positives: predicted cls but true label is not cls.
        FP = ((all_preds == cls) & (all_targets != cls)).sum().item()
        # False Negatives: didn't predict cls but true label is cls.
        FN = ((all_preds != cls) & (all_targets == cls)).sum().item()
        
        # Calculate precision and recall for this class.
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        print(f"\nClass {cls}:")
        print(f"  True Positives  (TP): {TP}")
        print(f"  False Positives (FP): {FP}")
        print(f"  False Negatives (FN): {FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

def split_data_scale(X, y, id, new = False):
    """
    Train a classifier to predict QP onset.
    
    Parameters:
    - X: feature matrix
    - y: labels
    
    Returns:
    - Trained classifier
    """
    # Split data
    try:
        if new:
            raise FileNotFoundError
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_processed_data(f'slidingwindowclaudebackend/pickle_saves/data/split{id}.pkl')

    except FileNotFoundError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        # print(np.sum(y_train), np.sum(y_test))
        # print(len(y_train), len(y_test))
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        save_processed_data((X_train_scaled, X_test_scaled, y_train, y_test, scaler), f'slidingwindowclaudebackend/pickle_saves/data/split{id}.pkl')
        
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


