import numpy as np
import torch
import pandas as pd
import dill as pickle
from dill import dump, load
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt

#Checken of alles naar device gaat
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Loading model
model = torch.jit.load('model_scripted.pt')
model.eval()

file = open('Data4.pkl', 'rb')
df = pickle.load(file)
val_set = df.iloc[[8,700,1850]]

subset = val_set[['X1', 'X2','X3','X4','X5','X6','X7','X8']]
lable = val_set['label']  
X = subset.to_numpy()
y = lable.to_numpy()
X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)
X = X.to(device)
X = X.unsqueeze(0)
logit = model(X).squeeze(0)
y_pred = torch.round(torch.sigmoid(logit))
print(y_pred)
print(y)
