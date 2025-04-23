import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

seq_length = 6
input_size = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_prepped3.csv")
ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([min(ratio, 15.0)], device=device)


wo = df[['X1', 'X2', 'X3','X4', 'X5','X6']]
X = torch.tensor(wo.to_numpy(), dtype=torch.float32).unsqueeze(-1)
wl = df['label']
y = torch.tensor(wl.to_numpy(), dtype=torch.float32)

print(X[0].shape)

dataset = TensorDataset(X, y)
train_size = int(0.75 * len(dataset))
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
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output single logit
        
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


# ==== 3. Training Setup ====


model = LSTMClassifier(input_size=input_size, hidden_size=32, num_layers=3).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 4. Training Loop ====
num_epochs = 800
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
    if epoch % 20 == 0:
        print(f"Epoch [{epoch+1}], "
          f"Train Loss: {total_loss/len(train_loader):.2f}, "
          f"Val Acc: {correct/total:.2f}, "
          f"Val Recall: {TP / (TP + FN + 1e-8):.2f}, "
          f"Val Prec: {TP / (TP + FP + 1e-8):.2f}")
        print("TP", TP, "FN", FN, "FP", FP)