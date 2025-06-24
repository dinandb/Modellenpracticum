import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import f1_score

def f_beta(beta, TP, FP, TN, FN):
    return (1 + beta**2)*TP / ((1 + beta**2)*TP + (beta**2)*FN + FP)


#hier laad je de dataset in de QP start vanuit Detect_QPv2(1).py
df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Hs4_data_without_units2.csv", low_memory = True, header = [0,1])



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
             
#print(last_extremas(df, 3, 'z_velocity'))       
      
def data_prep_QP1(df, column, lookback, time_increment):
    array = last_extremas(df, lookback, column)
    steps = time_increment*5
    if time_increment == 0:
        steps = 1
    array_2 = []
    QP = df['QP'].to_numpy()
    for i in range(array[0][0], array[-1][0] - 150, steps):
        if 0.0 in QP[i: i + 150]:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 0.0))
        else:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 1.0))   
    kolommen = [str(i) for i in range(lookback)]
    kolommen += ['label']
    dataframe = pd.DataFrame(array_2, columns=kolommen)
    return dataframe

def data_prep_QP2(df, column, lookback, time_increment):
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = last_extremas(df, lookback, column)

    array_2 = []
    QP = df['QP'].to_numpy()
    for i in extremas_ind[0:len(extremas_ind) - lookback - 5]:
        if 0.0 in QP[i: i + 150]:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 0.0))
        else:
            lijst = array[i][1]
            array_2.append(np.append(lijst, 1.0))   
    kolommen = [str(i) for i in range(lookback)]
    kolommen += ['label']
    dataframe = pd.DataFrame(array_2, columns=kolommen)
    return dataframe

def data_prep3(df, column, lookback, threshold, time_interval):
    extremas_ind, extremas = abs_extr(df, column, len=len(df.index))
    array = last_extremas(df, lookback, column)
    # print(array)
    # print(extremas_ind)
    time_interval = int(5*time_interval)

    array_2 = []
    column = df[column].to_numpy()
    
    for i in extremas_ind[lookback - 1:len(extremas_ind) - lookback - 1:lookback]:
        x = True
        for j in column[i:i + time_interval]:
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


def dataprep_wave(df, column, threshold, time_increment, lookback_time):
    steps = int(time_increment*5)
    if steps == 0:
        steps = 1
    array_2 = []
    column = df[column].to_numpy()
    lookback = int(lookback_time*5)

    for i in range(lookback, len(df.index) - 150 - int(time_increment + 1), steps):
        x = True
        for j in column[i:i + 50]:
            if abs(j) >= threshold:
                x = False
        if x == True:
            lijst = column[i - lookback: i]
            array_2.append(np.append(lijst, 1.0))
        if x == False:
            lijst = column[i - lookback: i]
            array_2.append(np.append(lijst, 0.0))
    kolommen = [str(i) for i in range(lookback)]
    kolommen += ['label']
    dataframe = pd.DataFrame(array_2, columns=kolommen)
    return dataframe

# df = data_prep3(df, 'z_velocity', 5, 1.0)
# print(df.info)
# print(df['label'].value_counts())




seq_length = 6
input_size = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

f1_array = []
array = [i for i in range(5, 51)]
for len_interval in range(5, 51):
    data = data_prep3(df, 'z_velocity', 6, 1.0, len_interval)    
    data = data.drop_duplicates()
    print(data['label'].value_counts())
    ratio = (len(data[data['label']==0.0].index))/(len(data[data['label']==1.0].index))
    ratio = torch.tensor(ratio, device=device)

    wo = data[[str(i) for i in range(seq_length)]]

    X = torch.tensor(wo.to_numpy(), dtype=torch.float32).unsqueeze(-1)

    wl = data['label']

    y = torch.tensor(wl.to_numpy(), dtype=torch.float32)


    dataset = TensorDataset(X, y)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

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







    model = LSTMClassifier(input_size=input_size, hidden_size=len_interval, num_layers=2).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # ==== 4. Training Loop ====
    num_epochs = 120
    max_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        TP = 0
        FN = 0
        FP = 0
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
                total += labels.size(0)
        f1 = 2*TP / (2*TP + FN + FP)
        if max_f1 < f1:
            max_f1 = f1
        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}], "
            f"Train Loss: {total_loss/len(train_loader):.2f}, "
            f"Val Acc: {correct/total:.2f}, "
            f"Val Recall: {TP / (TP + FN + 1e-8):.2f}, "
            f"Val Prec: {TP / (TP + FP + 1e-8):.2f}")
            print("TP", TP, "FN", FN, "FP", FP)
    print(max_f1)
    f1_array += [max_f1]


plt.plot(array, f1_array)
plt.show()
