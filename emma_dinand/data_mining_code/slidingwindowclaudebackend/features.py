

# from random_forest_QP_predict import create_qp_labeled_dataset_faster
import pickle
from sklearn.decomposition import PCA
import numpy as np

from extras import load_processed_data, save_processed_data, get_only_max_vals

NO_EXTREMA_LOOKBACK = 3

def build_features(data, dataset_id, new = False):

    pickle_file_path = f'slidingwindowclaudebackend/pickle_saves/vectors/processed_data_features{dataset_id}.pkl'
    # vervangen met eigen path
    try:

        # Try to load the data if it's already saved
        if new:
            raise FileNotFoundError
        
        
        features, offset = load_processed_data(pickle_file_path)
        print("Loaded features from pickle.")

    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it

        features = []

        var_map = {
            "heave": "z_wf",
            "sway": "y_wf",
            "surge": "x_wf",
            "yaw": "psi_wf",
            "roll": "phi_wf",
            "pitch": "theta_wf"
        }
        var_map_heave = {
            "heave": "z_wf",
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
            print(f"cur offset {cur_offset}, var = {var}")

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
        
        # for i in range(len(features_dict["heave"])):
        #     # hier voorloop afhnakeijk van no extrema lookback
        #     features.append([features_dict["heave"][i][0].item(), features_dict["heave"][i][1].item(), features_dict["heave"][i][2].item()])
        for i in range(len(features_dict["heave"])):
            temp = []
            for var, col in var_map.items():
                for j in range(NO_EXTREMA_LOOKBACK):
                    temp.append(features_dict[var][i][j].item())
                
            features.append(temp)
        
        

    
        print(f"shape features before PCA = {np.array(features).shape}")
        pca = PCA(n_components=0.99)
        features = pca.fit_transform(features)
        print(f"shape features after PCA = {features.shape}")

        save_processed_data((features, offset), pickle_file_path)
        print(f"Processed features saved to pickle. id={dataset_id}")
        
    return np.array(features), offset




