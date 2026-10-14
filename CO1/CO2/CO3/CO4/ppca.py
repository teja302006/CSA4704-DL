# Experiment 20: Probabilistic PCA and Reconstruction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(10)

Z = np.random.randn(500, 3)
W = np.random.randn(3, 5)
X = Z @ W + np.random.normal(0, 0.5, (500, 5))
X[::20, 0] = np.nan

mean = np.nanmean(X, axis=0)
X = np.where(np.isnan(X), mean, X)
X_centered = X - X.mean(axis=0)

errors = []

for k in range(1, 5):
    model = PCA(n_components=k)
    reduced = model.fit_transform(X_centered)
    reconstructed = model.inverse_transform(reduced)
    error = np.mean((X_centered - reconstructed) ** 2)
    errors.append(error)
    print("Components:", k, "Error:", error)

model = PCA(n_components=3)
model.fit(X_centered)

print("Estimated Noise Variance:", model.noise_variance_)

plt.plot(range(1, 5), errors, marker="o")
plt.xlabel("Components")
plt.ylabel("Reconstruction Error")
plt.title("PPCA Reconstruction Analysis")
plt.show()

samples = np.random.multivariate_normal(
    np.zeros(5), model.get_covariance(), 5
)
print("Generated Samples:\n", samples)
