import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
import matplotlib
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

matplotlib.use('pgf')  # Set PGF backend before importing pyplot
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
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
plt.subplots_adjust(bottom=0.15, left = 0.15, top=0.85)
fig.set_size_inches(w=5.5, h=3.5)
plt.scatter(rijtje_0[:, 0], rijtje_0[:, 1], color='red', edgecolors='k', label='0')
plt.scatter(rijtje_1[:, 0], rijtje_1[:, 1], color='blue', edgecolors='k', label='1')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Two clusters")
plt.savefig(r'C:\Users\steve\OneDrive\Bureaublad\clusters.pgf')
quit()
len_seq = 2

ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([min(ratio, 15.0)], device=device)

wo = df[['X1', 'X2']]
X = wo.to_numpy()

wl = df['label']

y = wl.to_numpy()

# wo = val_set[[str(i) for i in range(76)]]
# X_2 = wo.to_numpy()

# wl = val_set['label']

# y_2 = wl.to_numpy()

# X_2 = torch.from_numpy(X).type(torch.float)
# no_input = len(X[0])
# y_2 = torch.from_numpy(y).type(torch.float)




X = torch.from_numpy(X).type(torch.float)
no_input = len(X[0])
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.75, # 20% test, 80% train
                                                    random_state=55) # make the random split reproducible



from torch import nn
class LinModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=no_input, out_features=8) # takes in 8 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=8, out_features=8)
        self.layer_3 = nn.Linear(in_features=8, out_features=8) # takes in 2 features (X), produces 5 features
        self.layer_4 = nn.Linear(in_features=8, out_features=1) # takes in 5 features, produces 1 feature (y)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Softmax()
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_4(self.layer_3(self.relu(self.layer_2(self.sigmoid(self.layer_1(x))))))

# 4. Create an instance of the model and send it to target device


model_0 = LinModel().to(device)


# pos_weight is voor dat de false positives zwaarder meetellen
loss_fn = nn.BCEWithLogitsLoss(pos_weight=ratio) # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.Adam(params=model_0.parameters(), 
                            lr=0.001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def precision_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    true_positives = ((y_pred == 1) & (y_true == 1)).sum().float()
    predicted_positives = (y_pred == 1).sum().float()
    precision = 100*(true_positives / (predicted_positives + 1e-8))  # avoid division by zero
    return precision

def recall_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    true_positives = ((y_pred == 1) & (y_true == 1)).sum().float()
    actual_positives = (y_true == 1).sum().float()
    recall = 100*(true_positives / (actual_positives + 1e-8))  # avoid division by zero
    return recall


# Set the number of epochs
epochs = 250

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs + 1):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 
    prec = precision_fn(y_true=y_train,
                        y_pred = y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        
        test_precision = precision_fn(y_true=y_test, y_pred=test_pred)
        test_recall = recall_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 2 == 0:
        print(f"Epoch: {epoch} | Test acc: {test_acc:.2f}% | Test prec: {test_precision:.2f}% | Test recall {test_recall:.2f}%"  )

# 1. Create mesh grid over the full input space
x_min, x_max = X_test[:, 0].min().item() - 1, X_test[:, 0].max().item() + 1
y_min, y_max = X_test[:, 1].min().item() - 1, X_test[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# 2. Prepare mesh input for model
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid).float().to(device)

# 3. Get model predictions (probabilities) on the mesh
model_0.eval()
with torch.inference_mode():
    logits = model_0(grid_tensor).squeeze()
    probs = torch.sigmoid(logits).cpu().numpy()

# 4. Reshape probabilities back to mesh shape
Z = probs.reshape(xx.shape)

# 5. Plot the probability background (graded)
plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, Z, 100, cmap='RdBu', alpha=0.7)
plt.colorbar(contour, label='Predicted Probability (class 1)')

# 6. Extract test set points by class
test_preds = y_test.cpu().numpy()
X_test_np = X_test.cpu().numpy()
test_0 = X_test_np[test_preds == 0]
test_1 = X_test_np[test_preds == 1]

# 7. Plot test set points only
plt.scatter(test_0[:, 0], test_0[:, 1], color='red', edgecolor='k', label='Test Class 0')
plt.scatter(test_1[:, 0], test_1[:, 1], color='blue', edgecolor='k', label='Test Class 1')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundary with Test Set Only")
plt.legend()
plt.show()