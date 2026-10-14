# Experiment 22: Independent Component Analysis

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, square
from sklearn.decomposition import FastICA, PCA

np.random.seed(10)

t = np.linspace(0, 8, 1000)
S = np.c_[np.sin(2*t), square(2*t), sawtooth(2*t)]
S = (S - S.mean(axis=0)) / S.std(axis=0)

A = np.random.rand(3, 3)
X = S @ A.T

ica = FastICA(n_components=3, random_state=10, max_iter=1000)
recovered = ica.fit_transform(X)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)

fig, ax = plt.subplots(4, 3, figsize=(12, 8))
for i in range(3):
    ax[0, i].plot(S[:, i])
    ax[0, i].set_title("Original")
    ax[1, i].plot(X[:, i])
    ax[1, i].set_title("Mixed")
    ax[2, i].plot(recovered[:, i])
    ax[2, i].set_title("ICA")
    ax[3, i].plot(pca_result[:, i])
    ax[3, i].set_title("PCA")

plt.tight_layout()
plt.show()

print("ICA Correlations:")
print(np.abs(np.corrcoef(S.T, recovered.T)[:3, 3:]))
print("PCA Correlations:")
print(np.abs(np.corrcoef(S.T, pca_result.T)[:3, 3:]))
