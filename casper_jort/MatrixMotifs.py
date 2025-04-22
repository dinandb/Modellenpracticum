import pandas as pd
import stumpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt
import random as rand
from pathlib import Path


plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

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

file_path = "C:\\Users\\caspe\\OneDrive\\Documents\\Programming\\Modellenpracticum\\Data\\5415M_Hs=5m_Tp=10s_10h_clean.csv"

df = pd.read_csv(file_path, header=[0], skiprows=[1])
print(df)

#print(df['z_wf'][1 : 1000].reset_index(drop=True) - df['z_wf'][1001 : 2000].reset_index(drop=True))
m = 20000

mp = stumpy.gpu_stump(df['z_wf'], m)

motif_idx = np.argsort(mp[:, 0])[0]
print(f"The motif is located at index {motif_idx}")

nearest_neighbor_idx = mp[motif_idx, 1]
print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")


fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

axs[0].plot(df['z_wf'].values)
axs[0].set_ylabel('Heave', fontsize='20')
rect = Rectangle((motif_idx, -0.75), m, 1.5, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, -0.75), m, 1.5, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=motif_idx, linestyle="dashed")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
axs[1].plot(mp[:, 0])
#plt.show()



def Compare(index1, index2):
    first = df['z_wf'][index1 : index1 + 9000].reset_index(drop=True)
    second = df['z_wf'][index2 : index2 + 9000].reset_index(drop=True)

    plt.plot(first - second, alpha=0.1)
    plt.plot(df['z_wf'][index1 : index1 + 9000].reset_index(drop=True), alpha=0.5)
    plt.plot(df['z_wf'][index2 : index2 + 9000].reset_index(drop=True), alpha=0.5)
    print(first.corr(second))


    #plt.show()

plt.style.use('default')
# Compare(motif_idx, nearest_neighbor_idx)
# Compare(rand.randint(0, 6000), rand.randint(7000, 14000))
# Compare(rand.randint(0, 6000), rand.randint(7000, 14000))
# Compare(rand.randint(0, 6000), rand.randint(7000, 14000))


fig.tight_layout()
plt.subplots_adjust(bottom=0.15, right=1, top=0.85)
fig.set_size_inches(w=5.5, h=3.5)
plt.savefig("C:\\Users\\caspe\\OneDrive\\Documents\\GitHub\\Modellenpracticum\\casper_jort\\Output\\" + Path(__file__).stem + ".pgf")


