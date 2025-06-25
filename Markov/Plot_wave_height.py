import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt

#setting up data
file_path = r"C:\Users\joria\OneDrive\Documents\Modellenpracticum\Cleandata.csv"
df = pd.read_csv(file_path)
df = df.rename(columns = {"t  [s]":"t", "Delta_t  [s]":"delta_t", "z_wf  [m]":"heave",
                          "phi_wf  [rad]":"roll", "theta_wf  [rad]":"pitch",
                          "zeta  [m]":"wave_height"})
df["wave_height"] = df["wave_height"].abs()
#axes for plot
time = df["t"].to_numpy()
wave_height = df["wave_height"].to_numpy()

#drawing the plot
plt.axhline(y=0, color = "black", linestyle = ":")
plt.axhline(y=0.35, color = "r", linestyle = ":")
plt.axhline(y=0.62, color = "r", linestyle = ":")
plt.axhline(y=0.9, color = "r", linestyle = ":")
plt.axhline(y=1.29, color = "r", linestyle = ":")

plt.plot(time, wave_height, label = "wave height")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("wave height (m)")
plt.show()
