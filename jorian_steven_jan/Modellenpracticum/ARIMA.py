import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import statsmodels.api as sm
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units2.csv", low_memory = True, header = [0,1])
df = df[:2000]
#vind de extreme waarden
def extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    for i in range(1, len):
        if df[column][i-1] <= df[column][i] and df[column][i+1] <= df[column][i]:
            extremas_time += [df['t'][i]]
            extremas += [df[column][i]]
            extremas_ind += []
        elif df[column][i-1] >= df[column][i] and df[column][i+1] >= df[column][i]:
                extremas_time += [df['t'][i]]
                extremas += [df[column][i]]    
                extremas_ind += []
    return extremas_time, extremas, extremas_ind


HeaveTresh = 1.5



 

#test voor stationariteit
# def adf_test(series):
#     """Using an ADF test to determine if a series is stationary"""
#     test_results = adfuller(series)
#     print('ADF Statistic: ', test_results[0])
#     print('P-Value: ', test_results[1])
#     print('Critical Values:')
#     for thres, adf_stat in test_results[4].items():
#         print('\t%s: %.2f' % (thres, adf_stat))

# adf_test(np.asarray(extr(df, 'z_wf_1  [m]', 1000)[1]))

#autocorrelation en partialautocorrelation voor de parmaters voor het model
# plt.rc("figure", figsize=(8,4))
# plot_acf(np.asarray(extr(df, 'z_wf_1  [m]', 1200)[1]), lags=30)
# plt.ylim(0,1)
# plt.xlabel('Lags', fontsize=18)
# plt.ylabel('Correlation', fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.title('Autocorrelation Plot', fontsize=20)
# plt.tight_layout()
# plt.show()


plen = 100 #delen door vijf voor sec
mlen = 900 #hoeveel punten je meeneemt in de fit


# plots voor gewone series
plt.plot(df['t'][100:mlen].values, df['z_wf'][100:mlen].values, color='b', label='real')
plt.xlabel("time (s)")
plt.ylabel("Heave rate (m/s)")

plt.show()


