import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

data = pd.read_csv('95_train.csv', header=None)	

labels = [0] * 2500 + [1] * 2500

X = data.values
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try PCA
apply_pca = input("Do you want to apply PCA? (yes/no): ").lower()
if apply_pca == "yes":
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"Reduced dimensions to {X_train.shape[1]} using PCA.")

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [7, 10, 12],           
    'metric': ['euclidean', 'manhattan'],  
    'weights': ['uniform', 'distance']    
}

# Calculate score if test set would have been 70/30
def balanced_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    
    weights = []
    for x in list(y):
        weights.append(0.3 + 0.4*x)
        
    return accuracy_score(y, y_pred, sample_weight=weights)

# Find best parameters
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=12,
    scoring=balanced_scorer,
)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_knn = grid_search.best_estimator_

y_pred = best_knn.predict(X_test)
y_proba = best_knn.predict_proba(X_test)[:, 1]

weights = []
for x in list(y_test):
    weights.append(0.3 + 0.4*x)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], sample_weight=weights, digits=4))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
