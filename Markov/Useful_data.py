import pandas as pd
file_path = r"C:\Users\joria\OneDrive\Documents\Modellenpracticum\Data.xlsx"
out_path = r"C:\Users\joria\OneDrive\Documents\Modellenpracticum"

df = pd.read_excel(file_path, header=[0,1])
df = df[['t  [s]', 'Delta_t  [s]', 'z_wf  [m]', 'phi_wf  [rad]', 'theta_wf  [rad]', 'zeta  [m]']]
df.to_csv(out_path + 'out.csv', index=False)
print(df)
