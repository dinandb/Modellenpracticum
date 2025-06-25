import pandas as pd
import numpy as np
from math import *

#setting up data
file_path = r"C:\Users\joria\OneDrive\Documents\Modellenpracticum\Cleandata.csv"
df = pd.read_csv(file_path)
df = df.rename(columns = {"t  [s]":"t", "Delta_t  [s]":"delta_t", "z_wf  [m]":"heave",
                          "phi_wf  [rad]":"roll", "theta_wf  [rad]":"pitch",
                          "zeta  [m]":"wave_height"})

matrix = np.zeros((5,5))

minimum = df.min()["wave_height"]
maximum = df.max()["wave_height"]

bin_constant = (abs(minimum)+abs(maximum)+0.1)/5
print(bin_constant)

for index in range(17970):
    value_0 = df.loc()[index]["wave_height"]
    value_1 = df.loc()[index+30]["wave_height"]
    
    bin_0 = floor((value_0 + abs(minimum))/bin_constant)
    bin_1 = floor((value_1 + abs(minimum))/bin_constant)
    matrix[bin_0][bin_1] += 1

print(matrix)
     
