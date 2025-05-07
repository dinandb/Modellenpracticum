import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added.csv", low_memory = True, header = [0,1])
df = df.astype('float64').dtypes
QP_start = pd.read_csv(r'C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_QPstarts.csv', dtype = np.float64)
QP_start = QP_start[QP_start.columns[0]].to_numpy()

len_dens = 30
spec_dens = []

for times in QP_start:
    data = df[df['z_wf'] <= times]

print(data)