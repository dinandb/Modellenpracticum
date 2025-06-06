import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split





#hier laad je de dataset in de QP start vanuit Detect_QPv2(1).py
df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_added.csv", low_memory = True, header = [0,1])
length = len(df.index)
print(df.head())
QP_start = pd.read_csv(r'C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_QPstarts.csv', dtype = np.float64)
#print(QP_start.head())

#vindt de extreme waarden van één variabele
def abs_extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    time = df['t']
    time = time.to_numpy()
    print(type(time[10]))
    df = df[column]
    df = df.to_numpy()
    for i in range(1, len-2):
        if df[i-1] <= df[i] and df[i+1] <= df[i]:
            extremas_time += [time[i].item()]
            extremas += [abs(df[i].item())]
            extremas_ind += [i]
        elif df[i-1] >= df[i] and df[i+1] >= df[i]:
                extremas_time += [time[i].item()]
                extremas += [abs(df[i].item())] 
                extremas_ind += [i]
    #print(extremas_time)
    return pd.DataFrame({'time': extremas_time, 'abs_' + str(column): extremas}, dtype = pd.Float64Dtype())

#verdeeld de extrema in bins (genormaliseerd: bin k --> bin k/(num_bins))
def col_bins(df, num_bins, col):
    df = df[col].to_numpy()
    print(df)
    m = max(df)
    bins = []
    n = num_bins
    for i in range(len(df)):
        if df[i] > ((n-1)/n)*m:
            bins += [n/n]
        else:
            for j in range(1, num_bins):
                if df[i] <= (j/n)*m and df[i] > ((j-1)/n)*m:
                    bins += [j/n]                    
    return bins

#geeft dataframe met 9 kolommen: 8 voor de bins en dan label 1 is QP en 0 is geen QP
#inputs: hoeveel bins je wilt, dataframe van extreme waarden van één kolom, oorspronkelijke dataframe
def dataprep(df, QP_starts, column, num_extr):
    x = abs_extr(df, column, len(df.index))

    QP_starts = QP_starts['QPstart_time']
    print(QP_starts)


    for times in QP_starts:
        for j in range(len(time)):
            if QP[i] - time[j] < 1.5:
                seq += [np.append(b[j-num_extr:j], 1.0)]
                time = time[j:]
                break

    #biba = pd.DataFrame(seq, columns=['X1', 'X2','X3','X4','X5','X6','X7','X8', 'label'])
    QP_seq = df['QP'].to_numpy()
    steps = num_extr
    seq_2 = []
    time = x['time'].to_numpy()
    while steps < len(x.index) - (num_extr+2):
        for j in range(0 + int(5*time[steps]), 150 + int(5*time[steps])):
            if QP_seq[j] == 0.0:
                seq_2 += [np.append(b[steps-num_extr:steps], 0.0)]
                break
        steps += num_extr

    seq += seq_2
    boeba = pd.DataFrame(seq, columns=['X1', 'X2','X3','X4','X5','X6', 'label'])
    return boeba

number_bins = 50

pipoe = dataprep(number_bins, df, 'phi_wf', 6)
print(pipoe.info)
pipoe.to_csv (r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_phi.csv", index = False, header=True) 

#histogram of abs z velocity in bins
# plt.hist(x['bins'], bins = number_bins, density=False)
# plt.xlabel('bins')
# plt.ylabel('amount in bin')
# plt.title('Histogram of bins', fontweight='bold')
# plt.show()

# plt.plot(x['time'].to_numpy(), x['abs_z_velocity'].to_numpy(), 'r')
# plt.plot(QP_start['QPstart_time'].to_numpy(), np.ones(len(QP_start.index)), 'b')
# plt.show()