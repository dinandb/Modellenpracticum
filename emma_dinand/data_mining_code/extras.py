"""
#getDerivatives move de vectors zoals in heave split en dan verschil

getDiffs <- function(data, colname="z_wf", steps=1, power=1)  {
  
  n = length(data$t)
  
  relevant_sim_data_derivatives <- data
  
  
  der0 <- c(relevant_sim_data_derivatives[1:n,colname])
  der1<-c(0)
  
  for(j in 1: power)
  {
    for (k in 2:n)
    {
      der1 [k] <- der0[k] - der0[k-1]
    }
    der0 <- der1
  }
  
  der0[1]<- NA

  for (i in steps:1) {
    col_name <- paste0("Derivative #", i, " ^", power)  # Create column name dynamically

    #relevant_sim_data_derivatives[[col_name]] <- der0 
    
    relevant_sim_data_derivatives[[col_name]] <- dplyr::lag(der0, n = i-1, default = NA)  
    
  }

  return(relevant_sim_data_derivatives)
}


"""





"""

#rename deze naar iets van data mining
getMaxVals <- function(data, colname="z_wf") {
  # colname moet een string zijn
  
  maxvals <- c()
  
  new_data <- c()
  indices <- c()
  
  for (i in 2:(length(data[[colname]])-1)) {
    # if (data[[colname]][i-1] < data[[colname]][i] & data[[colname]][i] > data[[colname]][i+1]) {
    # if (abs(data[[colname]][i-1]) < abs(data[[colname]][i]) & abs(data[[colname]][i]) > abs(data[[colname]][i+1])) {
    if ((data[[colname]][i-1]) < (data[[colname]][i]) & (data[[colname]][i]) > (data[[colname]][i+1])
        | (data[[colname]][i-1]) > (data[[colname]][i]) & (data[[colname]][i]) < (data[[colname]][i+1])) {
      maxvals <- c(maxvals, data[[colname]][i])
      # new_data <- c(new_data, data[i,])
      indices <- c(indices, i)
      
      
    }
    # else if (data$QP[i]) {
    #   # print("added bc QP = 1")
    #   maxvals <- c(maxvals, data[[colname]][i])
    #   # new_data <- c(new_data, data[i,])
    #   indices <- c(indices, i)
    # }
  }
  return(data[indices,])
  
}


"""

import pandas as pd
import numpy as np

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

def get_max_vals(data, labels, colname="z_wf"):
    """
    Find local extrema (maxima and minima) in a time series.
    
    Parameters:
    data (DataFrame): Input data
    colname (str): Name of the column to analyze
    
    Returns:
    DataFrame: Rows from the original data where extrema were found
    """
    indices = []
    
    for i in range(1, len(data[colname])-1):
        # Check for local maxima OR local minima
        if ((abs(data[colname].iloc[i-1]) < abs(data[colname].iloc[i]) and 
             abs(data[colname].iloc[i]) > abs(data[colname].iloc[i+1])) or 
            (abs(data[colname].iloc[i-1]) > abs(data[colname].iloc[i]) and 
             abs(data[colname].iloc[i]) < abs(data[colname].iloc[i+1]))):
            indices.append(i)
        # elif labels.iloc[i]:
        #     indices.append(i)
    
    return data.iloc[indices].copy(), indices
def get_only_max_vals(data, colname="z_wf"):
    """
    Find local extrema (maxima and minima) in a time series.
    
    Parameters:
    data (DataFrame): Input data
    colname (str): Name of the column to analyze
    
    Returns:
    DataFrame: Rows from the original data where extrema were found
    """
    indices = []
    
    for i in range(1, len(data[colname])-1):
        # Check for local maxima OR local minima
        if ((data[colname].iloc[i-1] < data[colname].iloc[i] and 
             data[colname].iloc[i] > data[colname].iloc[i+1]) or 
            (data[colname].iloc[i-1] > data[colname].iloc[i] and 
             data[colname].iloc[i] < data[colname].iloc[i+1])):
            indices.append(i)

    
    return data.iloc[indices].copy(), indices





def get_only_max_vals_vector(vector):
    """
    Find local extrema (maxima and minima) in a time series.
    
    Parameters:
    data (DataFrame): Input data
    colname (str): Name of the column to analyze
    
    Returns:
    DataFrame: Rows from the original data where extrema were found
    """
    indices = []
    
    for i in range(2, len(vector) - 2):
        if ((vector.iloc[i-2] > vector.iloc[i-1] > vector.iloc[i] < vector.iloc[i+1] < vector.iloc[i+2]) or
            (vector.iloc[i-2] < vector.iloc[i-1] < vector.iloc[i] > vector.iloc[i+1] > vector.iloc[i+2])):
            indices.append(i)

    
    return vector.iloc[indices].copy()