import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import statsmodels.api as sm
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv('Modellenpracticum/data_heave.csv')
#vind de extreme waarden
def extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    for i in range(1, len):
        if df[column][i-1] <= df[column][i] and df[column][i+1] <= df[column][i]:
            extremas_time += [df['t  [s]'][i]]
            extremas += [df[column][i]]
            extremas_ind += []
        elif df[column][i-1] >= df[column][i] and df[column][i+1] >= df[column][i]:
                extremas_time += [df['t  [s]'][i]]
                extremas += [df[column][i]]    
                extremas_ind += []
    return extremas_time, extremas, extremas_ind


HeaveTresh = 1.5



 

#test voor stationariteit
def adf_test(series):
    """Using an ADF test to determine if a series is stationary"""
    test_results = adfuller(series)
    print('ADF Statistic: ', test_results[0])
    print('P-Value: ', test_results[1])
    print('Critical Values:')
    for thres, adf_stat in test_results[4].items():
        print('\t%s: %.2f' % (thres, adf_stat))

adf_test(np.asarray(extr(df, 'z_wf_1  [m]', 1000)[1]))

#autocorrelation en partialautocorrelation voor de parmaters voor het model
plt.rc("figure", figsize=(8,4))
plot_acf(np.asarray(extr(df, 'z_wf_1  [m]', 1200)[1]), lags=30)
plt.ylim(0,1)
plt.xlabel('Lags', fontsize=18)
plt.ylabel('Correlation', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Autocorrelation Plot', fontsize=20)
plt.tight_layout()
plt.show()


plen = 100 #delen door vijf voor sec
mlen = 900 #hoeveel punten je meeneemt in de fit
model1=ARIMA(df['z_wf_1  [m]'][0:mlen], order=(2,0,2))
model1_fit=model1.fit()
forecast1 = model1_fit.predict(start=mlen,end=mlen+plen,dynamic=True)
model2=ARIMA(df['z_wf_1  [m]'][0:mlen],order=(23,0,8))
model2_fit=model2.fit()
forecast2 = model2_fit.predict(start=mlen,end=mlen+plen,dynamic=True)

# plots voor gewone series
plt.plot(df['t  [s]'][700:mlen].values, df['z_wf_1  [m]'][700:mlen].values, color='b', label='real')
plt.plot(df['t  [s]'][mlen - 1 :mlen + plen].values, forecast2.values, color='y', label='forecast 2 (13,0,7)')
plt.plot(df['t  [s]'][mlen - 1 :mlen + plen].values, df['z_wf_1  [m]'][mlen - 1 :mlen + plen].values, color='b')
plt.plot(df['t  [s]'][900], df['z_wf_1  [m]'][900], marker='o', color='k')
plt.legend(loc="upper left")
plt.show()


