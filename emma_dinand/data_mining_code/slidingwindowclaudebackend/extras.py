

import time
import pandas as pd
import numpy as np
import pickle



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
        
        
        column = load_processed_data(f"slidingwindowclaudebackend/pickle_saves/vectors/{name}")
        print(f"Loaded column from pickle.{name}")

    except FileNotFoundError: 
          
        column = np.array([data[colname].iloc[i] for i in range(len(data[colname]))])
        save_processed_data(column, f"slidingwindowclaudebackend/pickle_saves/vectors/{name}")
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
        return data.iloc[indices], indices


def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)



