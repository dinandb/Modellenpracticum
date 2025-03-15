import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import csv


def oversample_smote(X, y):
    n = len(X)
    
    smote = SMOTE(sampling_strategy={1: int(n/2), 0: int((n/2) * (0.7/0.3))}, random_state=37)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(len(X_smote[y_smote == 1]))
    return X_smote, y_smote



data_train = pd.read_csv('95_train.csv', header=None)

labels = [0] * 2500 + [1] * 2500

X_train = data_train
y_train = pd.Series(labels, name="Label")

data_test = pd.read_csv('95_test.csv', header=None)

X_test = data_test

X_train, y_train = oversample_smote(X_train, y_train)

clf = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='euclidean')

clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)




with open('result1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(y_pred)

# print(sum([1 if p == 1 else 0 for p in y_pred])/len(y_pred))
# ^ to check ratio of second class (should be around 0.3)
