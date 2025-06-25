import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 8,
    'pgf.rcfonts': False,
    'text.usetex': True,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'lines.linewidth' : 0.5,
     'lines.markersize'  : 5,
    'xtick.labelsize' : 8,
    'ytick.labelsize': 8})

file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\5415M_Hs=4m_Tp=10s_10h_clean.csv"
#file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\CleanQP_data_36000.csv"
df = pd.read_csv(file_path, header=[0,1])
N = len(df)
T = 0.2

constant_1= 2.8
constant_2 = 0.02
constant_3= 0.05
constant_4 = 1.4
constant_5= -0.7



def f(omega):
    return (constant_1 * np.exp(-(constant_2/omega)**4) * (constant_3**(constant_4*(omega-constant_5))))
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
plt.magnitude_spectrum(np.array(df['zeta']).reshape(-1,), Fs=5)
plt.xlim(0, 1)
#plt.plot([0.01*i for i in range(1, 100)], [f(0.01*i) for i in range(1, 100)], color='C2')
plt.title('Spectrogram of zeta')
fig.tight_layout()
plt.subplots_adjust(bottom=0.15, right=1, top=0.85)
fig.set_size_inches(w=5.5, h=3.5)
plt.savefig("C:\\Users\\caspe\\OneDrive\\Documents\\GitHub\\Modellenpracticum\\casper_jort\\Output\\" + Path(__file__).stem + ".pgf")
