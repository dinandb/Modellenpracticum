import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs5_gooddata.csv", low_memory = True)
QP_start = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs5_QPstarts.csv")
QP_start = QP_start['QPstart_time'].to_numpy()
def spec_data_prep(df, QP_start , column):
    df = df[['t', column, 'QP']]

    array = []
    for times in QP_start:
        data = df[(df['t'] <= times) & (df['t'] > times - 30)]
        array += [np.append(data[column].to_numpy(), 1.0)]

    counter = 35.0
    array_1 = []
    while counter <  36000 - 80:
        counter += 10.2
        data_1 = df[(df['t'] <= counter) & (df['t'] > counter - 30.0)]
        if 0.0 in data_1['QP'].to_numpy():
            array_1 += [np.append(data_1[column].to_numpy(), 0.0)]


    rijtje = array + array_1
    Pxx = []
    for i in range(len(rijtje)):
        f, Pxx_den = signal.periodogram(rijtje[i][0:-2], fs=1.0)
        if rijtje[i][-1] == 1.0:
            plt.plot(f, Pxx_den, color='green', alpha=0.5, linewidth=0.45)
        if rijtje[i][-1] == 0.0:
            plt.plot(f, Pxx_den, color='red', alpha=0.2, linewidth=0.45)
        Pxx += [np.append(Pxx_den, rijtje[i][-1])]
    print(Pxx[0])
    plt.show()
    kolommen = [str(i) for i in range(150)]
    kolommen += ['label']
    dataprep_2 = pd.DataFrame(rijtje, columns=kolommen)
    #dataprep_2.to_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs5_heave.csv", index = False, header=True)
    return dataprep_2 

yolo = spec_data_prep(df, QP_start , 'z_wf')
print(yolo.info)
#f, Pxx_den = signal.periodogram(series)
# plt.plot(f, Pxx_den)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()

