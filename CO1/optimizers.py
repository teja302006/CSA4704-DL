# Experiment 6: Gradient Descent and Optimizers

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def loss(x, y):
    return x**2 + 5 * y**2

def gradient(x, y):
    return 2 * x, 10 * y

optimizers = ["GD", "SGD", "Momentum", "Adam"]

for opt in optimizers:
    x, y = 4.0, 4.0
    vx, vy = 0.0, 0.0
    mx, my = 0.0, 0.0
    sx, sy = 0.0, 0.0
    path_x, path_y = [], []

    for t in range(1, 101):
        gx, gy = gradient(x, y)
        lr = 0.05

        if opt == "SGD":
            gx += np.random.normal(0, 0.5)
            gy += np.random.normal(0, 0.5)

        if opt == "Momentum":
            vx = 0.9 * vx + gx
            vy = 0.9 * vy + gy
            gx, gy = vx, vy

        if opt == "Adam":
            mx = 0.9 * mx + 0.1 * gx
            my = 0.9 * my + 0.1 * gy
            sx = 0.999 * sx + 0.001 * gx**2
            sy = 0.999 * sy + 0.001 * gy**2
            gx = (mx / (1 - 0.9**t)) / (np.sqrt(sx / (1 - 0.999**t)) + 1e-8)
            gy = (my / (1 - 0.9**t)) / (np.sqrt(sy / (1 - 0.999**t)) + 1e-8)

        x -= lr * gx
        y -= lr * gy
        path_x.append(x)
        path_y.append(y)

    plt.plot(path_x, path_y, label=opt)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Optimizer Paths")
plt.legend()
plt.grid()
plt.show()

np.random.seed(10)
X = np.random.rand(500, 4).astype("float32")
y = (5 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + X[:, 3]).astype("float32")

results = {}

for name, optimizer in [
    ("SGD", tf.keras.optimizers.SGD(learning_rate=0.01)),
    ("Momentum", tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
    ("Adam", tf.keras.optimizers.Adam(learning_rate=0.01))
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss="mse")
    history = model.fit(X, y, epochs=50, verbose=0)
    results[name] = history.history["loss"]

for name, values in results.items():
    plt.plot(values, label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.grid()
plt.show()
