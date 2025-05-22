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
    NHITS(
        h=horizon,
        input_size=328,
        learning_rate=0.0010017435104832783,
        max_steps=10000,
        dropout_prob_theta=0.12491951168928624,
        batch_size=64,
        n_blocks=[3, 3, 3],
        n_freq_downsample=[7, 5, 1],
        accelerator="gpu",   # Enable GPU acceleration
    )]


nf = NeuralForecast(models=models, freq=1)
heave_df = pd.DataFrame({'unique_id': 'heave', 'ds': np.arange(len(time)), 'y': heave})

nf.fit(df=heave_df.head(120000))

heave_hat = nf.predict()


# Create a DataFrame from the true values for plotting
actual_df = heave_df[['unique_id', 'ds', 'y']].iloc[-horizon:]
import matplotlib.pyplot as plt

# 2) grab the last 12 actuals
actual_tail = heave_df.iloc[-horizon:].copy()
# 3) overwrite the forecast 'ds' so it lines up with those same timestamps
heave_hat['ds'] = actual_tail['ds'].values

# 4) plot
plt.plot(actual_tail['ds'], actual_tail['y'], label='Actual (last 12)')
plt.plot(heave_hat['ds'], heave_hat['NHITS'],  marker='x', label='NHITS fcst')
plt.legend()
plt.xlabel('ds')
plt.ylabel('heave')
plt.title('150-step Forecast vs Last 150 Observations')
plt.show()

