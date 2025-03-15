import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv('95_train.csv', header=None)

labels = [0] * 2500 + [1] * 2500

X = data 
y = np.array(labels)

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(72,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') 
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_eval, y_eval),
    epochs=50,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluate
eval_results = model.evaluate(X_eval, y_eval)
print(f"Evaluation Results: {eval_results}")
