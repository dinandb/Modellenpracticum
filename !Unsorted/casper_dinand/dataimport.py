from dataclasses import dataclass
from pathlib import Path
import pickle
import pandas as pd
import Detect_QP_CasperSteven

def moveQP(QP):
    
    # amountToAdd = amountToMove - num_start_QP
    
    # # Remove first amountToMove elements, add amountToMove zeros to the end
    # QP = QP[amountToMove:] + [False] * amountToMove
    
    # toSkip = 0
    # for i in range(len(QP) - 1):
    #     if QP[i] and (not QP[i+1]) and toSkip <= 0:
    #         for j in range(1, amountToAdd + 1):
    #             if i+j < len(QP):
    #                 QP[i+j] = True
    #         toSkip = amountToAdd
    #     else:
    #         toSkip = toSkip - 1


    amount_to_remove_at_end = 150
    amount_to_add_at_start = 10

    for i in range(amount_to_add_at_start, len(QP)-amount_to_add_at_start):
        if QP[i] and not QP[i-1]: # we are at the start of a QP
            QP[i-amount_to_add_at_start:i] = [True] * amount_to_add_at_start
        elif QP[i] and not QP[i+1]: # we are at the end of a QP
            try:
                QP[i+1-amount_to_remove_at_end:i+1] = [False] * amount_to_remove_at_end
            except Exception as e:
                print(f"Error at index {i}: {QP[i-1]}, {QP[i]}, {QP[i+1]}")
                
                
                raise e
            

    return QP

def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from a pickle file
def load_processed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@dataclass
class TimeSeriesData:
    data: pd.DataFrame
    file_path: Path
    saved_path:Path

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
    
    y = moveQP(y)
    # y2 = moveQP(y2)
    y3 = moveQP(y3)
    y4 = moveQP(y4)
    y5 = moveQP(y5)

    

    if to_log:
        print("after moving QPs")
        print(f"amount QPs in data1 {sum(y)}, total amount of data {len(y)}")
        # print(f"amount QPs in data2 {sum(y2)}, total amount of data {len(y2)}")
        print(f"amount QPs in data3 {sum(y3)}, total amount of data {len(y3)}")
        print(f"amount QPs in data4 {sum(y4)}, total amount of data {len(y4)}")
        print(f"amount QPs in data5 {sum(y5)}, total amount of data {len(y5)}")


    return [data1.data, None, data3.data, data4.data,data5.data], [y, None, y3, y4, y5]