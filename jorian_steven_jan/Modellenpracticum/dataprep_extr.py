import numpy as np
import pandas as pd



#hier laad je de dataset in de QP start vanuit Detect_QPv2(1).py
df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units2.csv", low_memory = True, header = [0,1])
df = df[1:10000]
#QP_start = pd.read_csv(r'C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_QPstarts.csv', dtype = np.float64)
#print(QP_start.head())
def find(n, array):
    counter = 0
    for i in range(len(array)):
        if array[i] == 0.0:
            counter += 1
    if counter >= n:
        return False
    if counter == 0:
        return True


def abs_extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    time = df['t']
    time = time.to_numpy()
    df = df[column]
    df = df.to_numpy()
    for i in range(1, len-2):
        if df[i-1] <= df[i] and df[i+1] <= df[i]:
            extremas_time += [time[i].item()]
            extremas += [abs(df[i].item())]
            extremas_ind += [i]
        elif df[i-1] >= df[i] and df[i+1] >= df[i]:
                extremas_time += [time[i].item()]
                extremas += [abs(df[i].item())] 
                extremas_ind += [i]
    return extremas_ind, extremas

def last_extremas(df, lookback, column):
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = []
    counter = lookback - 2
    for i in range(extremas_ind[lookback - 1], len(df.index)):
        if i > extremas_ind[-1]:
            break
        if i < extremas_ind[counter + 1]:
            array += [[i, extremas[counter - lookback + 1: counter + 1]]]
        if (i >= extremas_ind[counter + 1] and counter < len(extremas_ind) - 1):
            counter += 1
            array += [[i, extremas[counter - lookback + 1: counter + 1]]]
    print(len(array))
    return array
             
#print(last_extremas(df, 3, 'z_velocity'))       
      
def data_prep(df, column, lookback, time_increment):
    array = last_extremas(df, lookback, column)
    steps = time_increment*5
    if time_increment == 0:
        steps = 1
    array_2 = []
    QP = df['QP'].to_numpy()
    for i in range(array[0][0], array[-1][0] - 150, steps):
        if 0.0 in QP[i: i + 150]:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 0.0))
        else:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 1.0))   
    kolommen = [str(i) for i in range(lookback)]
    kolommen += ['label']
    dataframe = pd.DataFrame(array_2, columns=kolommen)
    return dataframe


df = data_prep(df, 'z_velocity', 5, 0)
df.to_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\heave_speed_for_model.csv")