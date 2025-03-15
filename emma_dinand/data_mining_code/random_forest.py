import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('95_train.csv', header=None)

labels = [0] * 2500 + [1] * 2500

X = data
y = pd.Series(labels, name="Label")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)

y_eval_total = []
y_proba_total = []
y_pred_total = []
weights_total = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_eval = X.iloc[train_index], X.iloc[test_index]
    y_train, y_eval = y.iloc[train_index], y.iloc[test_index]

    clf = RandomForestClassifier(random_state=37, max_depth=10, min_samples_split=2, n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_eval)
    y_proba = clf.predict_proba(X_eval)[:, 1]

    weights = []
    for x in list(y_eval):
        weights.append(0.3 + 0.4*x)

    y_eval_total += list(y_eval)
    y_proba_total += list(y_proba)
    y_pred_total += list(y_pred)
    weights_total += weights

print(classification_report(y_eval_total, y_pred_total, target_names=["Class 0", "Class 1"], sample_weight=weights_total, digits=4))
print("ROC-AUC Score (Evaluation Set):", roc_auc_score(y_eval_total, y_proba_total))  
