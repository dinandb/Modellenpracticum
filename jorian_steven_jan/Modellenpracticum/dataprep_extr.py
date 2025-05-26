import numpy as np
import pandas as pd



#hier laad je de dataset in de QP start vanuit Detect_QPv2(1).py
df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units.csv", low_memory = True, header = [0,1])
df = df[1:100]
#QP_start = pd.read_csv(r'C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_QPstarts.csv', dtype = np.float64)
#print(QP_start.head())

def abs_extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    time = df['t']
    time = time.to_numpy()
    print(type(time[10]))
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
    print(extremas_ind[0:4])
    return extremas_ind, extremas

def last_extremas(df, time_increment, lookback, column):
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = []
    counter = lookback - 2
    for i in range(extremas_ind[lookback - 1] + 1, len(df.index)):
        print(i)
        print(extremas_ind[counter])
        if i > extremas_ind[-1]:
            break
        if i < extremas_ind[counter + 1]:
            array += [extremas[counter - lookback + 1: counter + 1], i]
        if (i >= extremas_ind[counter + 1] and counter < len(extremas_ind) - 1):
            counter += 1
            array += [extremas[counter - lookback + 1: counter + 1], i]
        print(extremas_ind[counter])
    return array
             
print(last_extremas(df, 1.0, 3, 'z_velocity'))       
        
               