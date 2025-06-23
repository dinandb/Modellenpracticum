import numpy as np
import pandas as pd
import optuna
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from sklearn.metrics import mean_squared_error

# --- Load data
file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\5415M_Hs=4m_Tp=10s_10h_clean.csv"
df = pd.read_csv(file_path, header=[0, 1])
heave = np.array(df['z_wf'].values.flatten())
time = np.array(df['t'].values.flatten())

# --- Prepare forecasting dataframe
horizon = 150
heave_df = pd.DataFrame({
    'unique_id': 'heave',
    'ds': np.arange(len(time)),
    'y': heave
})



# Split data
train_df = heave_df.iloc[:120000]
val_df = heave_df.iloc[120000:120000 + horizon]
def objective(trial):
    input_size = trial.suggest_int('input_size', 300, 400)  # Broadened range around 340
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)  # Slightly broader
    max_steps = trial.suggest_int('max_steps', 100, 160)
    dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.05, 0.15)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Additional valid NHITS params to try tuning:
    n_blocks = trial.suggest_categorical('n_blocks', [[2, 2, 2], [3, 3, 3], [1, 1, 1]])
    n_freq_downsample = trial.suggest_categorical('n_freq_downsample', [[7, 5, 1], [5, 3, 1], [3, 1, 1]])

    model = NHITS(
        h=horizon,
        input_size=input_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        dropout_prob_theta=dropout_prob_theta,
        batch_size=batch_size,
        n_blocks=n_blocks,
        n_freq_downsample=n_freq_downsample,
        accelerator="gpu",   # Enable GPU acceleration
    )

    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(train_df)

    forecast = nf.predict()
    preds = forecast['NHITS'].values
    y_true = val_df['y'].values

    return mean_squared_error(y_true, preds)



# --- Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# --- Print best results
print("Best hyperparameters:", study.best_params)
print("Best MSE:", study.best_value)
