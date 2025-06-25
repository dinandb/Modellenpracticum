import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib

pgf_plot = True

if pgf_plot == True:
    matplotlib.use('pgf')  # Set PGF backend before importing pyplot
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
        'ytick.labelsize':8})

def f_beta(beta, TP, FP, TN, FN):
    return (1 + beta**2)*TP / ((1 + beta**2)*TP + (beta**2)*FN + FP)


#hier laad je de dataset in de QP start vanuit Detect_QPv2(1).py
df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units2.csv", low_memory = True, header = [0,1])
print(len(df.index))


def abs_extr(df, column, len):
    extremas_time = []
    extremas = []
    extremas_ind = []
    time = df['t']
    time = time.to_numpy()
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
    return extremas_ind, extremas

def last_extremas(df, lookback, column):
    if lookback < 2:
        print("lookback cant be smaller then 2")
        return
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = []
    counter = lookback - 2
    for i in range(extremas_ind[lookback - 1], len(df.index)):
        if i > extremas_ind[-1]:
            break
        if i < extremas_ind[counter + 1]:
            array += [[i, extremas[counter - lookback + 1: counter + 1]]]
        if (i >= extremas_ind[counter + 1] and counter < len(extremas_ind) - 1):
            counter += 1
            array += [[i, extremas[counter - lookback + 1: counter + 1]]]
    print(len(array))
    return array
             
      
def data_prep3(df, column, lookback, threshold, predicition_window, max_lookback):
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = last_extremas(df, lookback, column)
    # print(array)
    # print(extremas_ind)
    predicition_window = int(5*predicition_window)

    array_2 = []
    column = df[column].to_numpy()
    
    for i in extremas_ind[lookback - 1:len(extremas_ind) - lookback - 1:max_lookback]:
        x = True
        for j in column[i:i + predicition_window]:
            if abs(j) >= threshold:
                x = False
        if x == True:
            lijst = array[i - extremas_ind[lookback - 1]][1]
            array_2.append(np.append(lijst, 1.0))
        if x == False:
            lijst = array[i - extremas_ind[lookback - 1]][1]
            array_2.append(np.append(lijst, 0.0))
    kolommen = [str(i) for i in range(lookback)]
    kolommen += ['label']
    dataframe = pd.DataFrame(array_2, columns=kolommen)
    return dataframe



input_size = 1
lr = 0.001
num_epochs = 250
max_lookback = 13

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

f1_array = []
acc_array = []
RFPR_array = []
epoch_array = []
array = [i for i in range(3, max_lookback)]
for lookback in range(3, max_lookback):
    print(lookback)
    seq_length = lookback
    data = data_prep3(df, 'z_velocity', lookback, 1.0, 12.0, max_lookback - 1)    
    data = data.drop_duplicates()
    print(data['label'].value_counts())
    ratio = (len(data[data['label']==0.0].index))/(len(data[data['label']==1.0].index))
    print("ratio is: ", ratio)
    ratio = torch.tensor(ratio, device=device)

    wo = data[[str(i) for i in range(seq_length)]]

    X = torch.tensor(wo.to_numpy(), dtype=torch.float32).unsqueeze(-1)

    wl = data['label']

    y = torch.tensor(wl.to_numpy(), dtype=torch.float32)


    dataset = TensorDataset(X, y)
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ==== 2. Define the LSTM Model ====
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
            #self.fc = nn.Linear(hidden_size, 1)  # Output single logit
            self.fc = nn.Linear(hidden_size * 2, 1) #if bidirectional
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)


        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # Use the last time step
            out = self.fc(last_hidden)
            return out.squeeze()
        






    model = LSTMClassifier(input_size=input_size, hidden_size=(32), num_layers=3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=2,
                                                                    T_mult=2, 
                                                                    eta_min=1e-6)

    # ==== 4. Training Loop ====
    max_average = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=18.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs)).float()
                correct += (preds == labels).sum().item()
                TP += ((preds == 1) & (labels == 1)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                TN += ((preds == 0) & (labels == 0)).sum().item()
                total += labels.size(0)
        fb = f_beta(1, TP, FP, 0, FN)
        RFPR = 1 - (FP / (FP + TN + 1e-8))
        Acc = correct/total
        average = (1/3)*(fb + RFPR + Acc)
        if max_average <= average:
            max_average = average
            max_acc = Acc
            max_RFPR = RFPR
            m_epoch = epoch
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}], "
            f"Train Loss: {total_loss/len(train_loader):.2f}, "
            f"Val Acc: {correct/total:.2f}, "
            f"Val Recall: {TP / (TP + FN + 1e-8):.2f}, "
            f"Val FPR: {FP / (FP + TN + 1e-8):.2f}, "
            f"Val Prec: {TP / (TP + FP + 1e-8):.2f}")
            print("TP", TP, "FN", FN, "FP", FP, "TN", TN)
    print(max_average)
    print("best epoch", m_epoch)
    f1_array += [max_average]
    acc_array += [max_acc]
    RFPR_array += [max_RFPR]
    epoch_array += [m_epoch]

if pgf_plot == True:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.15, left = 0.15, top=0.85)
    fig.set_size_inches(w=5.5, h=3.5)
    plt.plot(array, f1_array, label='F1',)
    plt.plot(array, acc_array, label = "Acc")
    plt.plot(array, RFPR_array, label = "RFPR")
    plt.xlabel('Number of extrema')
    plt.ylabel('Acc/F1/RFPR')
    plt.title('Acc/F1/RFPR for different number of extrema')
    plt.legend()
    plt.savefig(r'C:\Users\steve\OneDrive\Bureaublad\Lookbackextremaheaverategood.pgf')

if pgf_plot == False:
    plt.plot(array, f1_array, label='F1')
    plt.plot(array, acc_array, label = "acc")
    plt.plot(array, RFPR_array, label = "RFPR")
    plt.xlabel('number of extrema')
    plt.ylabel('Acc/F1/RFPR')
    plt.title('Acc/F1/RFPR for different number of extrema')
    plt.legend()
    plt.show()

print(epoch_array)