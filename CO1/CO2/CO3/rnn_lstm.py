# Experiment 16: RNN and LSTM Comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(10)

days = np.arange(730)
data = 500 + 100 * np.sin(2 * np.pi * days / 7) + 50 * np.sin(2 * np.pi * days / 365)
data += np.random.normal(0, 20, 730)

df = pd.DataFrame({"Orders": data})
df["Orders"] = df["Orders"].interpolate()

mean, std = df["Orders"].mean(), df["Orders"].std()
values = ((df["Orders"] - mean) / std).values

X, y = [], []
for i in range(30, len(values)):
    X.append(values[i-30:i])
    y.append(values[i])

X = np.array(X).reshape(-1, 30, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

results = []
forecasts = {}

for name, layer in [
    ("SimpleRNN", tf.keras.layers.SimpleRNN(32)),
    ("LSTM", tf.keras.layers.LSTM(32))
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 1)),
        layer,
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    start = time.time()
    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32,
        validation_split=0.2, verbose=0
    )
    duration = time.time() - start

    pred = model.predict(X_test, verbose=0).ravel()
    actual = y_test * std + mean
    predicted = pred * std + mean

    forecasts[name] = predicted
    results.append([
        name,
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
        np.mean(np.abs((actual - predicted) / actual)) * 100,
        duration
    ])

    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(name + " Loss")
    plt.legend()
    plt.show()

print(pd.DataFrame(results, columns=[
    "Model", "MAE", "RMSE", "MAPE", "Training Time"
]))

plt.plot(actual, label="Actual")
for name, pred in forecasts.items():
    plt.plot(pred, label=name)
plt.title("Order Forecast Comparison")
plt.legend()
plt.show()
