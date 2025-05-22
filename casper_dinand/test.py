import pandas as pd

df = pd.DataFrame({
    'unique_id': ['series_1'] * 100,
    'ds': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'y': [i + (i % 10) for i in range(100)],
})
from neuralforecast.auto import AutoNHITS
from neuralforecast import NeuralForecast



# build with Optuna instead of Ray
base_config = dict(max_steps=1, val_check_steps=1, input_size=8)
base_model = AutoNHITS(h=4, backend="optuna")


nf = NeuralForecast(models=[base_model], freq="D")
nf.fit(df)              # runs tuning via Optuna
forecast = nf.predict()
print(forecast.head())
