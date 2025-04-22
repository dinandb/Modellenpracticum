import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

plt.magnitude_spectrum(np.array(df['zeta']).reshape(-1,), Fs=5, color='C1')
#plt.plot([0.01*i for i in range(1, 100)], [f(0.01*i) for i in range(1, 100)], color='C2')
plt.title('Spectrogram of zeta')
plt.show()
plt.tight_layout()
plt.figure.set_size_inches(w=5.5, h=3.5)
plt.savefig("C:\\Users\\caspe\\OneDrive\\Documents\\GitHub\\Modellenpracticum\\casper_jort\\Output\\" + Path(__file__).stem + ".pgf")