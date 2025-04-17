import numpy as np

num_start_QP = 3

def init_QPs(Data, dataset_id, new):
    # Data = relevant_sim_data_backup  # in logreg_for_predicting_QP's.R
    # Data = data.frame(lapply(Data, as.numeric))  # Convert all columns to numeric in R
    # For Python, assuming Data is a pandas DataFrame:
    # Data = Data.astype(float)  # Convert all columns to numeric

    time = Data['t']
    heave = Data['z_wf']
    # roll = Data['phi_wf']
    heaveThres = 0.2
    rollThres = 0.02
    timeThres = 30
    QP = [False]
    QPstart = [0]
    QPend = [0]
    
    i = Data.index[0]
    print(i)


    while i < len(time):
        # print(heave[i])
        if abs(heave[i]) < heaveThres:  # && roll[i] < rollThres in original
            j = 0
            # print(i+j)
            while (i+j) < len(time) and abs(heave[i + j]) < heaveThres:  # && roll[i + j] < rollThres && i + j < length(time) in original
                j = j + 1
            
            if i+j < len(time) and time[i + j] - time[i] >= timeThres:
                QP.extend([True] * (j-timeThres+1))
                QP.extend([False] * (timeThres-1))
                QP[(i-10):i] = [True]*10
                QPstart.append(time[i])
                QPend.append(time[i+j])
            elif i+j < len(time) and time[i + j] - time[i] < timeThres:
                QP.extend([False] * j)
            
            i = i + j
        else:
            QP.append(False)
            i = i + 1
    # QP.extend([False]*(len(time) - len(QP)))
    QP = QP[:(len(QP)-4)]
    return QP
def pretty_print(QP):
    for i in range(len(QP)):
        print(f"index {i}: {QP[i]}")
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



import pandas as pd

def generate_prev_heavs(steps, time_step, data):
    # Copy the original data to avoid modifying the original
    relevant_sim_data_splitted = data.copy()
    
    # Loop over the number of steps and generate the corresponding heave columns
    for i in range(steps, 0, -1):
        col_name = f"heave{i}"  # Create column name dynamically
        
        # Shift the 'z_wf' column by time_step * i to simulate the 'lag' behavior
        relevant_sim_data_splitted[col_name] = relevant_sim_data_splitted['z_wf'].shift(periods=time_step * i, fill_value=None)
    relevant_sim_data_splitted = relevant_sim_data_splitted.dropna()
    # print(relevant_sim_data_splitted)
    return relevant_sim_data_splitted



# def test_move_QP():
#     QP = 20 * [False] + [True] * 50 + [False] * 20 + [True] * 50 + [False] * 20
#     pretty_print(QP)
#     QP = moveQP(QP)
#     pretty_print(QP)

# test_move_QP()