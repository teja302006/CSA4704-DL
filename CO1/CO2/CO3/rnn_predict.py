# Experiment 13: Sequential Dataset Preparation and RNN Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(10)

data = pd.DataFrame({
    "Demand": 100 + 20 * np.sin(np.arange(8760) * 2 * np.pi / 24)
    + np.random.normal(0, 5, 8760)
})

data = data.interpolate()
scaler = MinMaxScaler()
values = scaler.fit_transform(data).ravel()

X, y = [], []
for i in range(24, len(values)):
    X.append(values[i-24:i])
    y.append(values[i])

X = np.array(X).reshape(-1, 24, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 1)),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

pred = model.predict(X_test, verbose=0)
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
predicted = scaler.inverse_transform(pred).ravel()

print("MAE:", mean_absolute_error(actual, predicted))
print("RMSE:", np.sqrt(mean_squared_error(actual, predicted)))

plt.plot(actual[:200], label="Actual")
plt.plot(predicted[:200], label="Predicted")
plt.title("Electricity Demand Prediction")
plt.legend()
plt.show()
