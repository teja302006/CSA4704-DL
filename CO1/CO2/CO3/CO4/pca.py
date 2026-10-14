# Experiment 19: PCA and Explained Variance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X = load_digits().data
mean = X.mean(axis=0)
X_centered = X - mean

cov = np.cov(X_centered, rowvar=False)
values, vectors = np.linalg.eigh(cov)
idx = np.argsort(values)[::-1]
values, vectors = values[idx], vectors[:, idx]

variance = values / values.sum()
cumulative = np.cumsum(variance)
k = np.argmax(cumulative >= 0.95) + 1

print("Components for 95% variance:", k)

pca = PCA(n_components=0.95)
Z = pca.fit_transform(X)
print("Sklearn Components:", Z.shape[1])

plt.plot(variance, marker="o")
plt.title("Scree Plot")
plt.show()

plt.plot(cumulative)
plt.axhline(0.95, linestyle="--")
plt.title("Cumulative Explained Variance")
plt.show()

fig, ax = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    ax[i].imshow(vectors[:, i].reshape(8, 8), cmap="gray")
    ax[i].axis("off")
plt.suptitle("Top Eigenfaces / Eigenvectors")
plt.show()

for n in [10, 50, 100]:
    p = PCA(n_components=min(n, X.shape[1]))
    reconstructed = p.inverse_transform(p.fit_transform(X))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(X[0].reshape(8, 8), cmap="gray")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed[0].reshape(8, 8), cmap="gray")
    plt.title(f"{n} Components")
    plt.show()
