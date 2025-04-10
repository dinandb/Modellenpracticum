import numpy as np
import torch
import pandas as pd
import dill as pickle
from dill import dump, load
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#data4 met helideck inclination


#pickle.dump(df, open('Data4.pkl', 'wb'))

file = open('Data4.pkl', 'rb')
df = pickle.load(file)

#Eerst drie rijen er uit voor te valideren
#val_set = df.iloc[[8,700,1850]]

#De gehele dataset zonder die drie
#tt_data = df.drop([8, 700, 1850])

#df = df.drop([i for i in range(100,1200)])
subset = df[['X1', 'X2','X3','X4','X5','X6','X7','X8']]
lable = df['label']  
X = subset.to_numpy()
y = lable.to_numpy()


ratio = (len(df[df['label']==0.0].index))/(len(df[df['label']==1.0].index))
ratio = torch.tensor([ratio])
ratio = ratio.to(device)


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, # 20% test, 80% train
                                                    random_state=55) # make the random split reproducible



from torch import nn
class LinModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=8, out_features=8) # takes in 8 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=8, out_features=8)
        self.layer_3 = nn.Linear(in_features=8, out_features=8) # takes in 2 features (X), produces 5 features
        self.layer_4 = nn.Linear(in_features=8, out_features=1) # takes in 5 features, produces 1 feature (y)
        self.relu = nn.ReLU()
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_4(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))

# 4. Create an instance of the model and send it to target device


model_0 = LinModel().to(device)


# pos_weight is voor dat de false positives zwaarder meetellen
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


# Set the number of epochs
epochs = 10001

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
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
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Test acc: {test_acc:.2f}% | Test prec: {test_precision:.2f}% | Test recall {test_recall:.2f}%"  )



# Saving model
# model_scripted = torch.jit.script(model_0) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save


