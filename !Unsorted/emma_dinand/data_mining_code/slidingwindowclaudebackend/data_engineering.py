

from dataclasses import dataclass
from features import build_features
from extras import load_processed_data, save_processed_data, get_only_max_vals, format_data
import pandas as pd
import data_frame_build as dfb
import Detect_QP_CasperSteven

from pathlib import Path

data_path = Path("../../assets")

data2 = data_path / "data2.csv"

@dataclass
class DataLocation:
    data: pd.DataFrame
    path: str | Path



def load_data(file_path: str | Path):
        try:
            # Try to load the data if it's already saved
            # raise FileNotFoundError
            data = load_processed_data(file_path)
            print("Loaded data from pickle.")
        except FileNotFoundError:
            # If the pickle file doesn't exist, process the data and save it
            data = pd.read_csv(file_path, header=[0,1])
            data = data[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
            # data = data.iloc[1:]
            data = data.apply(pd.to_numeric, errors='coerce')
            save_processed_data(data, file_path)
            print("Processed data saved to pickle.")

def init(to_log=True,mark_first=True,mark_second=True, mark_third=True, mark_fourth=True):
    
    pickle_file_path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data1.pkl'


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
    


    data4_path = 'slidingwindowclaudebackend/pickle_saves/data/processed_data4.pkl'

    try:
        # Try to load the data if it's already saved
        # raise FileNotFoundError
        data4 = load_processed_data(data4_path)
        print("Loaded data4 from pickle.")
        
    except FileNotFoundError:
        # If the pickle file doesn't exist, process the data and save it
        file_path = '../../assets/5415M_Hs=3m_Tp=10s_10h.csv'
        data4 = pd.read_csv(file_path, header=[0,1])
        data4 = data4[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        # data2 = data2.iloc[1:]
        data4 = data4.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data4, data4_path)
        print("Processed data saved to pickle.")


        
     
    # make pandasshow everything
    
    
    

    y = Detect_QP_CasperSteven.mark_QP(data,name="QP1", new=False)

    # print(f"y {y}")
    # print(f"y shape {y.shape}")
    
    # print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
    # print(f"total length original y {len(y)}")
    y2 = Detect_QP_CasperSteven.mark_QP(data2,name="QP2", new=False)
    # print(f"y2 {y2[:40]}")
    # print(f"y2 shape {y2.shape}")
    y3 = Detect_QP_CasperSteven.mark_QP(data3, name="QP3", new=False)
    y4 = Detect_QP_CasperSteven.mark_QP(data4[:30500], name="QP4", new=False)
    
    # print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
    # y = dfb.moveQP(y)
    start_index1 = 0
    stop_index1 = len(data)-120
    start_index2 = 0
    stop_index2 = len(data2)-120
    start_index3 = 0
    stop_index3 = len(data3)-120
    start_index4 = 0
    stop_index4 = 30000

    if to_log:
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
        print(f"amount QPs in data4 {sum(y4)}, total amount of data {len(y4)}")
    
    y = dfb.moveQP(y)
    y2 = dfb.moveQP(y2)
    y3 = dfb.moveQP(y3)
    y4 = dfb.moveQP(y4)

    data, data2, data3, data4 = format_data(data, data2, data3, data4)
    
    
    if to_log:
        print("after moving QPs")
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
        print(f"amount QPs in data4 {sum(y4)}, total amount of data {len(y4)}")
    
    if mark_first:
        X, offset1 = build_features(data[start_index1:stop_index1], dataset_id=1, new=False)
    else:
        X, offset1 = None, None
    if mark_second:
        X2, offset2 = build_features(data2[start_index2:stop_index2], dataset_id=2, new=False)
    else:
        X2, offset2 = None, None

    if mark_third:
        X3, offset3 = build_features(data3[start_index3:stop_index3], dataset_id=3, new=False)
    else:
        X3, offset3 = None, None

    if mark_fourth:
        X4, offset4 = build_features(data4[start_index4:stop_index4], dataset_id=4, new=False)
    else:
        X4, offset4 = None, None
    

    
    if X is not None:
        y  = y[start_index1+offset1:stop_index1+2]
    if X2 is not None:
        y2 = y2[start_index2+offset2:stop_index2+2]
    if X3 is not None:
        y3 = y3[start_index3+offset3:stop_index3+2]
    if X4 is not None:
        y4 = y4[start_index4+offset4:stop_index4+2]

    return [X, X2, X3, X4], [y, y2, y3, y4]