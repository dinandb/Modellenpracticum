from dataclasses import dataclass
from pathlib import Path
import pickle
import pandas as pd
from emma_dinand import data_frame_build as dfb
from emma_dinand import Detect_QP_CasperSteven

from sklearn.preprocessing import StandardScaler
# from slidingwindowclaudebackend import features
from sklearn.decomposition import PCA
import numpy as np
from constants import NO_EXTREMA_LOOKBACK, pos_helideck
# from jorian_steven_jan.Modellenpracticum import AIModel

from emma_dinand.extras import load_processed_data, save_processed_data, get_only_extrema_vals, get_only_extrema_vals_vector, get_only_max_vals_vector



# data_path = Path("../assets")

# data2 = data_path / "data2.csv"

@dataclass
class TimeSeriesData:
    data: pd.DataFrame
    file_path: Path
    saved_path:Path



def build_features(data, dataset_id, new = False):

    pickle_file_path = f'emma_dinand/pickle_saves/vectors/processed_data_features{dataset_id}.pkl'
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
            "pitch": "theta_wf",

        }
    
        heli_incl = np.array(Detect_QP_CasperSteven.heli_incl(np.array(data['z_wf']).flatten(),np.array(data['y_wf']).flatten(),np.array(data['x_wf']).flatten(),np.array(data['psi_wf']).flatten(),np.array(data['phi_wf']).flatten(),np.array(data['theta_wf']).flatten(), np.array(data['t']).flatten(),pos_helideck, name=f"heli_incl_QP{dataset_id}", new=False))
        heave_speed = np.array(Detect_QP_CasperSteven.heave_speed(np.array(data['z_wf']).flatten(), np.array(data['y_wf']).flatten(), np.array(data['x_wf']).flatten(), np.array(data['psi_wf']).flatten(), np.array(data['phi_wf']).flatten(), np.array(data['theta_wf']).flatten(), pos_helideck, len(data['t']), name=f"heave_speed_QP{dataset_id}", new=False))

        heli_incl_df = pd.DataFrame(heli_incl, columns=["heli_incl"])
        heave_speed_df = pd.DataFrame(heave_speed, columns=["heave_speed"])

        heli_incl_max, heli_incl_max_indices = get_only_max_vals_vector(heli_incl_df, "heli_incl", name=f"{dataset_id}_heli_incl", new=False)
        heave_speed_extrema, heave_speed_extrema_indices = get_only_extrema_vals(heave_speed_df, "heave_speed", name=f"{dataset_id}_heave_speed", new=False)
        
        

        extrema_dict = {}
        extrema_indices_dict = {}
        offset = 0
        offset_dict = {}
        features_dict = {}

        extrema_dict["heli_incl"] = heli_incl_max
        extrema_indices_dict["heli_incl"] = heli_incl_max_indices

        extrema_dict["heave_speed"] = heave_speed_extrema
        extrema_indices_dict["heave_speed"] = heave_speed_extrema_indices

        var = "heave_speed"
        col = "heave_speed"
        features_dict[var] = []

        for i in range(len(heave_speed_extrema_indices) - 2):
            if i + 3 < len(heave_speed_extrema_indices):
                features_dict[var].extend([[heave_speed_extrema.iloc[i][col], heave_speed_extrema.iloc[i+1][col], heave_speed_extrema.iloc[i+2][col]]] * 
                                (heave_speed_extrema_indices[i+3] - heave_speed_extrema_indices[i+2]))
            else:
                features_dict[var].extend([[heave_speed_extrema.iloc[i][col], heave_speed_extrema.iloc[i+1][col], heave_speed_extrema.iloc[i+2][col]]] * 
                                (len(data) - heave_speed_extrema_indices[i+2]))
        
        var = "heli_incl"
        col = "heli_incl"
        features_dict[var] = []

        for i in range(len(heli_incl_max_indices) - 2):
            if i + 3 < len(heli_incl_max_indices):
                features_dict[var].extend([[heli_incl_max.iloc[i][col], heli_incl_max.iloc[i+1][col], heli_incl_max.iloc[i+2][col]]] * 
                                (heli_incl_max_indices[i+3] - heli_incl_max_indices[i+2]))
            else:
                features_dict[var].extend([[heli_incl_max.iloc[i][col], heli_incl_max.iloc[i+1][col], heli_incl_max.iloc[i+2][col]]] * 
                                (len(data) - heli_incl_max_indices[i+2]))
        
        


        cur_offset = heave_speed_extrema_indices[NO_EXTREMA_LOOKBACK-1] + 2
        offset_dict["heli_incl"] = cur_offset
        if cur_offset > offset:
            offset = extrema_indices_dict['heli_incl'][NO_EXTREMA_LOOKBACK-1] + 2
        # print(f"cur offset {cur_offset}, var = heli_incl")

        cur_offset = heli_incl_max_indices[NO_EXTREMA_LOOKBACK-1] + 2
        offset_dict["heave_speed"] = cur_offset
        if cur_offset > offset:
            offset = extrema_indices_dict['heave_speed'][NO_EXTREMA_LOOKBACK-1] + 2
        # print(f"cur offset {cur_offset}, var = heave_speed")


        for var, col in var_map.items():
            extrema, extrema_indices = get_only_extrema_vals(data, colname=col, name=f"{dataset_id}_{col}", new=False)
            extrema_dict[var] = extrema
            extrema_indices_dict[var] = extrema_indices
            
            cur_offset = extrema_indices[NO_EXTREMA_LOOKBACK-1] + 2
            offset_dict[var] = cur_offset
            if cur_offset > offset:
                offset = extrema_indices_dict[var][NO_EXTREMA_LOOKBACK-1] + 2
            # print(f"cur offset {cur_offset}, var = {var}")


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
                    
        var_map["heave_speed"] = "heave_speed"
        var_map["heli_incl"] = "heli_incl"
        # print(f"len heave {len(features_dict['heave'])}" )
        # print(f"len heli incl {len(features_dict['heli_incl'])}" )
        # print(f"len sway {len(features_dict['sway'])}" )
        # print(f"len surge {len(features_dict['surge'])}" )
        # print(f"len yaw {len(features_dict['yaw'])}" )
        # print(f"len roll {len(features_dict['roll'])}" )
        # print(f"len pitch {len(features_dict['pitch'])}" )
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
        for i in range(min([len(features_dict["heli_incl"]), len(features_dict["heave_speed"]), len(features_dict["heave"]), len(features_dict["sway"]), len(features_dict["surge"]), len(features_dict["yaw"]), len(features_dict["roll"]), len(features_dict["pitch"])])):
            temp = []
            for var, col in var_map.items():
                if var == "heli_incl" or var == "heave_speed":

                    for j in range(NO_EXTREMA_LOOKBACK):
                        
                        temp.append(features_dict[var][i][j].item())

                if var == "heli_incl" or var == "heave_speed":
                    features.append(temp)
            
        

    
        # print(f"shape features before PCA = {np.array(features).shape}")
        # pca = PCA(n_components=7)
        # features = pca.fit_transform(features)
        # print(f"shape features after PCA = {features.shape}")

        # TODO: aantal extrema dynamisch maken
        # nog toevoegen aan de features: heli_incl laatste 8 maxima
        # laatste 8 extema heave speed
        # alles normaliseren
        


        save_processed_data((features, offset), pickle_file_path)
        print(f"Processed features saved to pickle. id={dataset_id}")
        # scaler = StandardScaler()
        # print(f"features[0] before scaling = {features[0]}")
        # features = scaler.fit_transform(features)
        # print(f"features[0] after scaling = {features[0]}")
        

    return np.array(features), offset



def load_data(saved_path: Path, file_path: Path):
    try:
        # Try to load the data if it's already saved
        # raise FileNotFoundError
        data = load_processed_data(saved_path)
        print(f"Loaded data from {saved_path} from pickle.")
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        
        data = pd.read_csv(file_path, header=[0,1])
        data = data[['t', 'z_wf', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        data = data.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data, saved_path)
        print(f"Read data from {file_path}. Processed data saved to pickle at {saved_path}.")

    ts_data = TimeSeriesData(data=data, file_path=file_path, saved_path=saved_path)
    return ts_data
    

def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def init(to_log=True):
    base_path_data = Path("assets")
    base_saved_data_path = Path("emma_dinand/pickle_saves/data")

    data1_data_path = base_path_data / "data1.csv"
    data1_saved_path = base_saved_data_path / "data1.pkl"
    
    # data2_data_path = base_path_data / "data2.csv"
    # data2_saved_path = base_saved_data_path / "data2.pkl"

    data3_data_path = base_path_data / "data3rustig.csv"
    data3_saved_path = base_saved_data_path / "data3.pkl"
    
    data4_data_path = base_path_data / "data4wilder.csv"
    data4_saved_path = base_saved_data_path / "data4.pkl"

    data5_data_path = base_path_data / "data5wildst.csv"
    data5_saved_path = base_saved_data_path / "data5.pkl"

    data1 = load_data(data1_saved_path, data1_data_path)
    # data2 = load_data(data2_saved_path, data2_data_path)
    data3 = load_data(data3_saved_path, data3_data_path)
    data4 = load_data(data4_saved_path, data4_data_path)
    data5 = load_data(data5_saved_path, data5_data_path)


    y = Detect_QP_CasperSteven.mark_QP(data1.data, name="QP1", new=False)


    # y2 = Detect_QP_CasperSteven.mark_QP(data2.data, name="QP2", new=False)
    
    y3 = Detect_QP_CasperSteven.mark_QP(data3.data, name="QP3", new=False)
    y4 = Detect_QP_CasperSteven.mark_QP(data4.data, name="QP4", new=False)
    y5 = Detect_QP_CasperSteven.mark_QP(data5.data, name="QP5", new=False)
    
    
    
    start_index1 = 0
    stop_index1 = len(data1.data)-120
    # start_index2 = 0
    # stop_index2 = len(data2.data)-120
    start_index3 = 0
    stop_index3 = len(data3.data)-120
    start_index4 = 0
    stop_index4 = len(data4.data)-120
    start_index5 = 0
    stop_index5 = len(data5.data)-120

    if to_log:
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        # print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
        print(f"amount QPs in data4 {sum(y4)}, total amount of data {len(y4)}")
        print(f"amount QPs in data5 {sum(y5)}, total amount of data {len(y5)}")
    
    y = dfb.moveQP(y)
    # y2 = dfb.moveQP(y2)
    y3 = dfb.moveQP(y3)
    y4 = dfb.moveQP(y4)
    y5 = dfb.moveQP(y5)

    

    if to_log:
        print("after moving QPs")
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        # print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
        print(f"amount QPs in data4 {sum(y4)}, total amount of data {len(y4)}")
        print(f"amount QPs in data5 {sum(y5)}, total amount of data {len(y5)}")
    
    
    X, offset1 = build_features(data1.data[start_index1:stop_index1], dataset_id=1, new=False)

    

    # X2, offset2 = build_features(data2.data[start_index2:stop_index2], dataset_id=2, new=False)

    


    X3, offset3 = build_features(data3.data[start_index3:stop_index3], dataset_id=3, new=False)

    
    


    X4, offset4 = build_features(data4.data[start_index4:stop_index4], dataset_id=4, new=False)
    X5, offset5 = build_features(data5.data[start_index5:stop_index5], dataset_id=5, new=False)

    


    
    
    y  = y[start_index1+offset1:len(X)+(start_index1+offset1)]

    # y2 = y2[start_index2+offset2:len(X2)+start_index2+offset2]

    y3 = y3[start_index3+offset3:len(X3)+start_index3+offset3]

    y4 = y4[start_index4+offset4:len(X4)+start_index4+offset4]

    y5 = y5[start_index5+offset5:len(X5)+start_index5+offset5]

    return [X,  None, X3, X4, X5], [y,None, y3, y4, y5]


def main():
    Xs, ys = init()
    
    return Xs, ys

main()