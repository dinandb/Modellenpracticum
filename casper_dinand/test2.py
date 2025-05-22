# from neuralforecast.utils import AirPassengersDF
import ray

print(ray.__version__)
# Y_df = AirPassengersDF
# print(Y_df.head())

from ray import tune
from neuralforecast.auto import AutoNHITS

nhits_config = AutoNHITS.get_default_config(h = 12, backend="ray")                      # Extract the default hyperparameter settings
nhits_config["random_seed"] = tune.randint(1, 10)                                       # Random seed
nhits_config["n_pool_kernel_size"] = tune.choice([[2, 2, 2], [16, 8, 1]])               # MaxPool's Kernelsize