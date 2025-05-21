import pandas as pd
import numpy as np

file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\5415M_Hs=4m_Tp=10s_10h_clean.csv"
df = pd.read_csv(file_path, header=[0,1])
print(df)
heave, sway, surge, yaw, roll, pitch = np.array(df['z_wf'].values.flatten()), np.array(df['y_wf'].values.flatten()), np.array(df['x_wf'].values.flatten()), np.array(df['psi_wf'].values.flatten()), np.array(df['phi_wf'].values.flatten()), np.array(df['theta_wf'].values.flatten())
dt, time = [0.2]*180001, np.array(df['t'].values.flatten())


import logging

import pandas as pd
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
horizon = 150

# Try different hyperparmeters to improve accuracy.
models = [
    LSTM(input_size=2 * horizon,
               h=horizon,                    # Forecast horizon
               max_steps=100,                # Number of steps to train
               scaler_type='standard',       # Type of scaler to normalize data
               encoder_hidden_size=64,       # Defines the size of the hidden state of the LSTM
               decoder_hidden_size=64,),     # Defines the number of hidden units of each layer of the MLP decoder
          NHITS(h=horizon,                   # Forecast horizon
                input_size=2 * horizon,      # Length of input sequence
                max_steps=250,               # Number of steps to train
                n_freq_downsample=[2, 1, 1]) # Downsampling factors for each stack output
          ]
nf = NeuralForecast(models=models, freq=1)
heave_df = pd.DataFrame({'unique_id': 'heave', 'ds': np.arange(len(time)), 'y': heave})

nf.fit(df=heave_df.head(120000))

heave_hat = nf.predict()

from utilsforecast.plotting import plot_series
# Create a DataFrame from the true values for plotting
actual_df = heave_df[['unique_id', 'ds', 'y']].iloc[-horizon:]
import matplotlib.pyplot as plt

print(heave_hat)
print(actual_df)

# Plot using the proper format
import matplotlib.pyplot as plt


# 2) grab the last 12 actuals
actual_tail = heave_df.iloc[-horizon:].copy()
# 3) overwrite the forecast 'ds' so it lines up with those same timestamps
heave_hat['ds'] = actual_tail['ds'].values

# 4) plot
plt.plot(actual_tail['ds'], actual_tail['y'], label='Actual (last 12)')
plt.plot(heave_hat['ds'], heave_hat['LSTM'],   marker='o', label='LSTM fcst')
plt.plot(heave_hat['ds'], heave_hat['NHITS'],  marker='x', label='NHITS fcst')
plt.legend()
plt.xlabel('ds')
plt.ylabel('heave')
plt.title('150-step Forecast vs Last 150 Observations')
plt.show()

