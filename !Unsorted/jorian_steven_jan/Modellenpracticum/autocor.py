import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
#matplotlib.use('pgf')  # Set PGF backend before importing pyplot
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units2.csv", low_memory = True, header = [0,1])

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

time_series_extremas_ind, time_series_extr = abs_extr(df, 'z_velocity', len(df.index))


# fig, ax = plt.subplots(1, 1, figsize=(12, 5))
# plt.subplots_adjust(bottom=0.15, left = 0.15, top=0.85)
# fig.set_size_inches(w=5.5, h=3.5)

series = df['z_velocity'].to_numpy()

plot_acf(series, lags=30, alpha=1)
plt.ylim(0,1)
plt.xlabel('Lags')
plt.ylabel('Correlation')
plt.title('Autocorrelation of absolute value of heave rate extrema')
#plt.savefig(r'C:\Users\steve\OneDrive\Bureaublad\autocor_heaverate.pgf')
plt.show()