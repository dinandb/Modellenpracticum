import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_pca_lagged_features(X, y, input_columns=None, variance_threshold=0.95, lag_periods=5):
    """
    Apply PCA to input variables and create lagged features for time series analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the time series data
    target_column : str
        Name of the output/target variable column
    input_columns : list of str, optional
        Names of the input variable columns. If None, all columns except target_column are used
    variance_threshold : float, default=0.95
        Minimum variance to retain in PCA (between 0 and 1)
    lag_periods : int, default=5
        Number of past values to include for each principal component
        
    Returns:
    --------
    pandas.DataFrame
        A new dataframe with the target variable and lagged PCA components
    dict
        Dictionary containing PCA model, scaler, and relevant information
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = X.copy()
    
    # If input_columns is not specified, use all columns except the target
    if input_columns is None:
        input_columns = df_copy.columns.tolist()
    # Extract features and target
    X = df_copy.values
    
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Determine number of components needed to retain variance_threshold of variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Create a new PCA model with the desired number of components
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X_scaled)
    
    # Create dataframe with PCA components
    component_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=component_columns, index=df_copy.index)
    
    # Add target column
    df_pca['y'] = y
    
    # Create lagged features
    df_lagged = df_pca.copy()
    
    for component in component_columns:
        for lag in range(1, lag_periods + 1):
            df_lagged[f'{component}_lag{lag}'] = df_lagged[component].shift(lag)
    
    # Drop rows with NaN values (due to lagging)
    df_lagged = df_lagged.dropna()
    
    # Prepare the final dataframe with target and lagged features
    # Keep the current PCA components and their lags, drop the original PCA components
    features = [col for col in df_lagged.columns if '_lag' in col]
    df_final = df_lagged[['y'] + features]
    
    # Store information about the transformation
    info = {
        'pca_model': pca_final,
        'scaler': scaler,
        'n_components': n_components,
        'explained_variance_ratio': pca_final.explained_variance_ratio_,
        'cumulative_variance': np.sum(pca_final.explained_variance_ratio_),
        'input_columns': input_columns,
        'component_columns': component_columns
    }
    
    return df_final, info


# Example usage
if __name__ == "__main__":
    # Create a sample dataframe with 8 input variables and 1 target
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate random data for demonstration
    data = np.random.randn(100, 8)
    # Make some variables correlated
    data[:, 1] = data[:, 0] * 0.8 + np.random.randn(100) * 0.2
    data[:, 3] = data[:, 2] * 0.7 + np.random.randn(100) * 0.3
    data[:, 5] = data[:, 4] * 0.9 + np.random.randn(100) * 0.1
    
    # Create target variable with some relationship to inputs
    target = data[:, 0] * 0.5 + data[:, 2] * 0.3 + data[:, 4] * 0.2 + np.random.randn(100) * 0.1
    
    # Create dataframe
    df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(8)], index=dates)
    # df['target'] = target
    
    # Apply the function
    lag_periods = 6  # Number of lagged periods
    df_processed, info = create_pca_lagged_features(
        df,
        y=pd.Series(target),
        variance_threshold=0.95,
        lag_periods=lag_periods
    )
    
    # Print results
    print(f"Original dataframe shape: {df.shape}")
    print(f"Processed dataframe shape: {df_processed.shape}")
    print(f"Number of principal components retained: {info['n_components']}")
    print(f"Explained variance per component: {info['explained_variance_ratio']}")
    print(f"Total variance explained: {info['cumulative_variance']:.4f}")
    print("\nFirst few rows of processed dataframe:")
    print(df_processed.head())