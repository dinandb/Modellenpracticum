import pandas as pd
import numpy as np
from math import *

#setting up data
file_path = r"CleanQP_data_36000.csv"
df = pd.read_csv(file_path)
df = df.rename(columns = {"t [s]":"t", "Delta_t [s]":"delta_t", "z_wf [m]":"heave",
                          "phi_wf [rad]":"roll", "theta_wf [rad]":"pitch",
                          "zeta [m]":"wave_height"})

matrix = np.zeros((25,25))
unmatrix2 = np.zeros((25,25))
matrix2 = np.zeros((25,25))
df["wave_height"] = df["wave_height"].abs()

#drop all entries which are not peaks
droplist = []

for index in range(1, 18000):
    previous = df.loc()[index-1]["wave_height"]
    following = df.loc()[index+1]["wave_height"]
    current = df.loc()[index]["wave_height"]
    
    if previous >= current or following >= current:
        droplist.append(index)

df = df.drop(droplist)
length = len(df.index)
df = df.reset_index()

#put all peaks in equal size bins (equal by amount of peaks per bin)
df = df.sort_values(by = ["wave_height"])
bin_constant = floor(length/5)

bins = [0 for i in range(bin_constant)]+[1 for i in range(bin_constant)]+[2 for
i in range(bin_constant)]+[3 for i in range(bin_constant)]+[4 for i
in range(length - 4*bin_constant)]

df["bin_value"] = bins
df = df.sort_index()
        
#create transition matrix
for index in range(length - 3):
    bin_0 = int(df.loc()[index]["bin_value"])
    bin_1 = int(df.loc()[index+1]["bin_value"])
    bin_2 = int(df.loc()[index+2]["bin_value"])
    bin_3 = int(df.loc()[index+3]["bin_value"])
    
    id_1 = 5*bin_0 + bin_1
    id_2 = 5*bin_1 + bin_2
    matrix[id_1][id_2] += 1


#normalize transition matrix
for i in range(25):
    matrix[i] = matrix[i]/np.sum(matrix[i])

print(matrix)

#comparison matrix
for index in range(length - 3):
    bin_0 = int(df.loc()[index]["bin_value"])
    bin_1 = int(df.loc()[index+1]["bin_value"])
    bin_2 = int(df.loc()[index+2]["bin_value"])
    bin_3 = int(df.loc()[index+3]["bin_value"])
    
    id_1 = 5*bin_0 + bin_1
    id_2 = 5*bin_2 + bin_3
    unmatrix2[id_1][id_2] += 1

#normalize
for i in range(25):
    matrix2[i] = unmatrix2[i]/np.sum(unmatrix2[i])

print(np.linalg.norm(unmatrix2*(matrix*matrix-matrix2)*(matrix*matrix-matrix2)))
