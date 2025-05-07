import pandas as pd


df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added.csv", low_memory = True, header = 0)
df = df.drop([0])
columns = ['t', 'x_wf', 'y_wf', 'z_wf', 'phi_wf', 'theta_wf', 'psi_wf', 'zeta', 'z_velocity', 'heli_incl', 'QP']
for column in columns: 
    df[column] = df[column].astype(float)

df.to_csv (r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added2.csv", index = False, header=True) 
