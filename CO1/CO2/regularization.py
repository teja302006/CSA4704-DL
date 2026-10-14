# Experiment 10: L2, Dropout and Early Stopping

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(10)
tf.random.set_seed(10)

X, y = make_classification(
    n_samples=500, n_features=8, n_informative=5,
    random_state=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(l2=0, dropout=0):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

results = []

configs = [
    ("Baseline", 0, 0, False),
    ("L2 0.001", 0.001, 0, False),
    ("L2 0.01", 0.01, 0, False),
    ("L2 0.1", 0.1, 0, False),
    ("Dropout 0.2", 0, 0.2, False),
    ("Dropout 0.3", 0, 0.3, False),
    ("Dropout 0.5", 0, 0.5, False),
    ("Early Stopping", 0, 0, True)
]

for name, l2, dropout, early in configs:
    model = build_model(l2, dropout)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )] if early else []

    history = model.fit(
        X_train, y_train, epochs=60, batch_size=16,
        validation_split=0.2, callbacks=callbacks, verbose=0
    )

    train_pred = (model.predict(X_train, verbose=0) >= 0.5).astype(int)
    test_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    results.append([name, train_acc, test_acc, train_acc - test_acc])

    if name == "Baseline":
        plt.plot(history.history["loss"], label="Training")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.title("Baseline Overfitting")
        plt.legend()
        plt.show()

print("Model | Train Accuracy | Test Accuracy | Gap")
for row in results:
    print(row)
