# Experiment 26: Gibbs Sampling and Contrastive Divergence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import time

np.random.seed(10)

X = (load_digits().data > 8).astype(float)[:300]
nv, nh = X.shape[1], 32
W = np.random.normal(0, 0.01, (nv, nh))
a = np.zeros(nv)
b = np.zeros(nh)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def sample(p):
    return (np.random.rand(*p.shape) < p).astype(float)

def gibbs(v, steps):
    for _ in range(steps):
        h = sample(sigmoid(v @ W + b))
        v = sample(sigmoid(h @ W.T + a))
    return v

v = sample(np.random.rand(1, nv))

for steps in [1, 10, 100, 1000]:
    v = gibbs(v, steps)
    plt.imshow(v.reshape(8, 8), cmap="gray")
    plt.title(f"Gibbs Steps: {steps}")
    plt.show()

for k in [1, 5, 10]:
    start = time.time()
    errors = []

    for epoch in range(10):
        error = 0
        for v0 in X:
            ph = sigmoid(v0 @ W + b)
            h0 = sample(ph.reshape(1, -1))[0]
            vk = v0.reshape(1, -1)

            for _ in range(k):
                hk = sample(sigmoid(vk @ W + b))
                vk = sample(sigmoid(hk @ W.T + a))

            v1 = vk[0]
            ph1 = sigmoid(v1 @ W + b)

            W += 0.001 * (np.outer(v0, ph) - np.outer(v1, ph1))
            a += 0.001 * (v0 - v1)
            b += 0.001 * (ph - ph1)
            error += np.mean((v0 - v1) ** 2)

        errors.append(error / len(X))

    print("CD-", k, "Time:", time.time() - start)
    plt.plot(errors, label=f"CD-{k}")

plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.show()
