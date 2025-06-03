import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.utils import shuffle
import optuna
from sklearn.metrics import f1_score



seq_length = 250
input_size = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


df = pd.read_csv(r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\wave_prepped_threshold.csv")
df = df.drop_duplicates()
print(len(df.index))
ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([min(15, ratio)], device=device)

wo = df[[str(i) for i in range(seq_length)]]

X = torch.tensor(wo.to_numpy(), dtype=torch.float32).unsqueeze(-1)

wl = df['label']

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



def objective(trial):
    # === Hyperparameters to optimize ===
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # === Model setup ===
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # === Training Loop ===
    for epoch in range(30):  # fewer epochs for faster optimization
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    # === Validation Evaluation ===
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs)).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    f1 = f1_score(all_labels, all_preds)
    return f1




# model = LSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=ratio)
# optimizer = optim.Adam(model.parameters(), lr=0.0045)

# # ==== 4. Training Loop ====
# num_epochs = 251
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()
#         total_loss += loss.item()

#     # Validation
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0
#     TP = 0
#     FN = 0
#     FP = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             preds = torch.round(torch.sigmoid(outputs)).float()
#             correct += (preds == labels).sum().item()
#             TP += ((preds == 1) & (labels == 1)).sum().item()
#             FN += ((preds == 0) & (labels == 1)).sum().item()
#             FP += ((preds == 1) & (labels == 0)).sum().item()
#             total += labels.size(0)
#     if epoch % 10 == 0:
#         print(f"Epoch [{epoch+1}], "
#           f"Train Loss: {total_loss/len(train_loader):.2f}, "
#           f"Val Acc: {correct/total:.2f}, "
#           f"Val Recall: {TP / (TP + FN + 1e-8):.2f}, "
#           f"Val Prec: {TP / (TP + FP + 1e-8):.2f}")
#         print("TP", TP, "FN", FN, "FP", FP)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print("Best hyperparameters:", study.best_params)
print("Best F1-score:", study.best_value)


best_params = study.best_params
model = LSTMClassifier(input_size=input_size,
                       hidden_size=best_params['hidden_size'],
                       num_layers=best_params['num_layers']).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])


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
    plt.xlabel("3")
    plt.ylabel("4")
    plt.title("LSTM Decision Boundary (Validation Set Only)")
    plt.show()



