import pandas as pd
import numpy as np

df = pd.read_excel('Modellenpracticum/Data-1.xlsx')
len(df['t  [s]'])-1
def extr(df, column):
    extremas_time = []
    extremas = []
    for i in range(1, 300):
        if df[column][i-1] <= df[column][i] and df[column][i+1] <= df[column][i]:
            extremas_time += [df['t  [s]'][i]]
            extremas += [df[column][i]]
        elif df[column][i-1] >= df[column][i] and df[column][i+1] >= df[column][i]:
                extremas_time += [df['t  [s]'][i]]
                extremas += [df[column][i]]    
    return extremas, extremas_time
          
print(extr(df, 'z_wf_1  [m]'))


