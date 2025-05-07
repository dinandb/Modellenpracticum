import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added2.csv", low_memory = True)
QP_start = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_QPStarts.csv")
QP_start = QP_start['QPstart_time'].to_numpy()
df = df[['t', 'z_wf', 'QP']]

array = []
for times in QP_start:
    data = df[(df['t'] <= times) & (df['t'] > times - 30)]
    array += [np.append(data['z_wf'].to_numpy(), 1.0)]
# f, Pxx_den = signal.periodogram(series)
# plt.plot(f, Pxx_den)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
counter = 0
array_1 = []
for i in range(270):
    counter += 35
    data_1 = df[(df['t'] <= counter) & (df['t'] > counter - 30.0)]
    if 0.0 in data_1['QP'].to_numpy():
        array_1 += [np.append(data_1['z_wf'].to_numpy(), 0.0)]


rijtje = array + array_1
columns = [str(i) for i in range(len(rijtje[0]) - 1)]
columns += ['label']
dataprep = pd.DataFrame(rijtje, columns=columns)
#dataprep.to_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\spectrum_data.csv", index = False, header=True) 
Pxx = []
for i in range(len(rijtje)):
    f, Pxx_den = signal.periodogram(rijtje[i])
    Pxx += [np.append(Pxx_den, rijtje[i][-1])]


kolommen = [str(i) for i in range(len(Pxx[0]) - 1)]
kolommen += ['label']
dataprep_2 = pd.DataFrame(Pxx, columns=kolommen)

dataprep_2.to_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\spectrum_data2.csv", index = False, header=True) 
#f, Pxx_den = signal.periodogram(series)
# plt.plot(f, Pxx_den)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()

# f, Pxx_den = signal.periodogram(rijtje[2])
# plt.plot(f, Pxx_den)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()

# f, Pxx_den = signal.periodogram(rijtje[140])
# plt.plot(f, Pxx_den)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()


