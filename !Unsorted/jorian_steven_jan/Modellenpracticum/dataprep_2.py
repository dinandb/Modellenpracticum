import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_gooddata.csv", low_memory = True)

def dataprep(df, column, len_seq, threshold, len_threshold):
    counter = len_seq + 3.0
    array = []
    df = df[['t', column]]
    while counter < len(df.index) - len_seq - 15.0:
        data_1 = df[int((counter - len_seq)*5) : int(counter*5)]
        check_data = df[(df['t'] > counter) & (df['t'] <= counter + len_threshold)][column].to_numpy()
        for i in range(len(check_data)):
            if check_data[i] > threshold:
                array += [np.append(data_1[column].to_numpy(), 0.0)]
                break
            if i == (len(check_data) - 1):
                array += [np.append(data_1[column].to_numpy(), 1.0)]
            
        counter += 3.0
    cols = [str(i) for i in range(0, int(len_seq)*5)]
    cols += ['label']
    dataprep_2 = pd.DataFrame(array, columns=cols)
    print(dataprep_2.info)
    dataprep_2.to_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_heavespeed_crit.csv", index = False, header=True)
    return dataprep_2 

dataprep(df=df, column='z_velocity', len_seq=30.0, threshold=0.7, len_threshold=15.0)
