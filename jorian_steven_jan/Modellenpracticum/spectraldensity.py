import numpy as np
import pandas as pd

from scipy import signal

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added2.csv", low_memory = True)
print(df.dtypes)