import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_heli_incl.csv")

df = df[['X5', 'X6', 'label']]
QP = df[df['label'] == 1.0]
x_QP = QP['X5'].to_numpy() 
y_QP = QP['X6'].to_numpy() 

P = df[df['label'] == 0.0]
x_P = P['X5'].to_numpy() 
y_P = P['X6'].to_numpy() 


plt.scatter(x_P, y_P,  color='r')
plt.scatter(x_QP, y_QP, s=100, color='g')
plt.show()

