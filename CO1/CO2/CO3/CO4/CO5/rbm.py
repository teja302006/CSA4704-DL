# Experiment 25: Restricted Boltzmann Machine

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(10)

X = (load_digits().data > 8).astype(float)[:500]
nv, nh = X.shape[1], 64

W = np.random.normal(0, 0.01, (nv, nh))
a = np.zeros(nv)
b = np.zeros(nh)
lr = 0.01
errors, energies = [], []

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def energy(v, h):
    return -v @ a - h @ b - v @ W @ h

for epoch in range(50):
    total_error, total_energy = 0, 0

    for v0 in X:
        ph = sigmoid(v0 @ W + b)
        h0 = (np.random.rand(nh) < ph).astype(float)
        pv = sigmoid(h0 @ W.T + a)
        v1 = (np.random.rand(nv) < pv).astype(float)
        ph1 = sigmoid(v1 @ W + b)

        W += lr * (np.outer(v0, ph) - np.outer(v1, ph1))
        a += lr * (v0 - v1)
        b += lr * (ph - ph1)

        total_error += np.mean((v0 - pv) ** 2)
        total_energy += energy(v0, h0)

    errors.append(total_error / len(X))
    energies.append(total_energy / len(X))

plt.plot(errors, label="Reconstruction Error")
plt.plot(energies, label="Free Energy")
plt.legend()
plt.show()

v = X[0]
h = sigmoid(v @ W + b)
reconstructed = sigmoid(h @ W.T + a)

plt.subplot(1, 2, 1)
plt.imshow(v.reshape(8, 8), cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed.reshape(8, 8), cmap="gray")
plt.title("Reconstructed")
plt.show()
