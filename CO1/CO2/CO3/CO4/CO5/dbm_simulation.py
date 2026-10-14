# Experiment 28: Deep Boltzmann Machine Simulation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(10)

X = (load_digits().data > 8).astype(float)[:500]
W1 = np.random.normal(0, 0.01, (64, 32))
W2 = np.random.normal(0, 0.01, (32, 16))
b1 = np.zeros(32)
b2 = np.zeros(16)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

errors = []

for epoch in range(20):
    error = 0

    for v in X:
        h1 = sigmoid(v @ W1 + b1)
        h2 = sigmoid(h1 @ W2 + b2)
        reconstructed = sigmoid(h1 @ W1.T)

        W1 += 0.001 * (np.outer(v, h1) - np.outer(reconstructed, h1))
        W2 += 0.001 * (np.outer(h1, h2) - np.outer(h1, h2))
        b1 += 0.001 * (h1 - sigmoid(reconstructed @ W1))
        b2 += 0.001 * (h2 - h2)
        error += np.mean((v - reconstructed) ** 2)

    errors.append(error / len(X))

plt.plot(errors)
plt.title("DBM Reconstruction Error")
plt.xlabel("Epoch")
plt.show()

plt.imshow(W1[:, :16].T.reshape(16, 8, 8)[0], cmap="gray")
plt.title("Learned Filter")
plt.show()

v = X[0]
h = sigmoid(v @ W1 + b1)
reconstructed = sigmoid(h @ W1.T)

plt.subplot(1, 2, 1)
plt.imshow(v.reshape(8, 8), cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed.reshape(8, 8), cmap="gray")
plt.title("Reconstruction")
plt.show()
