import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added2.csv", low_memory = True)
series = df['z_wf'].to_numpy()
f, Pxx_den = signal.periodogram(series)

plt.semilogy(f, Pxx_den)



plt.xlabel('frequency [Hz]')

plt.ylabel('PSD [V**2/Hz]')

plt.show()