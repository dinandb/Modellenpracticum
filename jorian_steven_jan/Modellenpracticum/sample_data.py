import numpy as np
import pandas as pd

num_samples = 400
X = np.random.rand(num_samples, 2)
print(X[0])
Y = np.empty
for i in range(num_samples):
    if np.cos(X[i][0])*np.sqrt(X[i][0]) >= (X[i][1]):
        Y += [np.append(X[i], np.array([1]))]
    else:
        Y += [np.append(X[i], np.array([0]))]

print(Y[0])