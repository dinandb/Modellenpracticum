import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


num_samples = 500
X, y = make_blobs(n_samples=num_samples, 
                  centers=2, 
                  n_features=2, 
                  cluster_std=1.15,
                  random_state=0)


df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'label': y}, dtype=np.float32)
df_0 = df[df['label'] == 0.0]
df_1 = df[df['label'] == 1.0]

rijtje_0 = df_0[['X1','X2']].to_numpy()
rijtje_1 = df_1[['X1','X2']].to_numpy()
plt.scatter(rijtje_0[:, 0], rijtje_0[:, 1], color='red', edgecolors='k', label='0')
plt.scatter(rijtje_1[:, 0], rijtje_1[:, 1], color='blue', edgecolors='k', label='1')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Two clusters")
plt.show()
quit()
seq_length = 2
input_size = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([min(ratio, 15.0)], device=device)

wo = df[['X1','X2']]
X = torch.tensor(wo.to_numpy(), dtype=torch.float32).unsqueeze(-1)

wl = df['label']

y = torch.tensor(wl.to_numpy(), dtype=torch.float32)


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


model = LSTMClassifier(input_size=input_size, hidden_size=4, num_layers=1).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 4. Training Loop ====
num_epochs = 50
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

def plot_decision_boundary_lstm(model, val_dataset, device, resolution=100):
    model.eval()

    # Prepare validation inputs
    X_val = torch.stack([x for x, _ in val_dataset]).squeeze(-1).to(device)
    y_val = torch.tensor([y for _, y in val_dataset]).to(device)

    # Create meshgrid
    x_min, x_max = X_val[:,0].min().item() - 1, X_val[:,0].max().item() + 1
    y_min, y_max = X_val[:,1].min().item() - 1, X_val[:,1].max().item() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_seq = torch.tensor(grid, dtype=torch.float32).view(-1, 2, 1).to(device)

    # Predict probabilities over grid
    with torch.no_grad():
        preds_grid = torch.sigmoid(model(grid_seq)).cpu().numpy().reshape(xx.shape)

    # Plot heatmap of probabilities
    contour = plt.contourf(xx, yy, preds_grid, levels=25, cmap=plt.cm.RdBu, alpha=0.6, vmin=0.0, vmax=1.0)
    plt.colorbar(contour, label="Predicted Probability")

    # Scatter validation set
    plt.scatter(X_val[:,0].cpu(), X_val[:,1].cpu(), c=y_val.cpu(), cmap=plt.cm.RdBu)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("LSTM Decision Boundary (Validation Set Only)")
    plt.show()



plot_decision_boundary_lstm(model, val_dataset, device)