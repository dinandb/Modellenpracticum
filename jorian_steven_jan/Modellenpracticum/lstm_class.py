import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import dill as pickle
from dill import dump, load
from sklearn.model_selection import train_test_split
from torch import nn



# Define LSTM classifier model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Generate sample data
file_path = r"C:\Users\steve\OneDrive\Bureaublad\VS Code\git\Modellenpracticum\jorian_steven_jan\Modellenpracticum\Data4_prepped.csv"
df = pd.read_csv(file_path)
print(df.info)


subset = df[['X1', 'X2','X3','X4','X5','X6','X7','X8']]
lable = df['label']  
X = subset.to_numpy()
y = lable.to_numpy()

#determine pos weight
ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([ratio])
print(ratio)

# Convert data to PyTorch tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, # 20% test, 80% train
                                                    random_state=55) # make the random split reproducible



# Define model parameters
input_size = 8
hidden_size = 8
num_layers = 2
output_size = 1

# Instantiate the model
model_0 = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss(ratio) # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.Adam(params=model_0.parameters(), 
                            lr=0.0001)

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


# Train the model
num_epochs = 10
for epoch in range(num_epochs):
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
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Test acc: {acc}" )