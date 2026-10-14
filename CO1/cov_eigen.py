# Experiment 2: Covariance and Eigenvalues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)
data = pd.DataFrame({
    "Haemoglobin": np.random.normal(13, 2, 200),
    "RBC": np.random.normal(5, 0.7, 200),
    "Glucose": np.random.normal(100, 15, 200),
    "Cholesterol": np.random.normal(190, 25, 200),
    "BMI": np.random.normal(25, 4, 200)
})

X = (data - data.mean()) / data.std()
cov = X.cov()
corr = X.corr()

print("Covariance:\n", cov)
print("Correlation:\n", corr)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cov, cmap="coolwarm")
plt.colorbar()
plt.title("Covariance")

plt.subplot(1, 2, 2)
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.title("Correlation")
plt.tight_layout()
plt.show()

values, vectors = np.linalg.eigh(cov)
idx = np.argsort(values)[::-1]
values = values[idx]
vectors = vectors[:, idx]

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)
print("Variance Percentage:", values / values.sum() * 100)

projected = X.to_numpy() @ vectors[:, :2]

plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
plt.arrow(0, 0, 3, 0, head_width=0.2)
plt.arrow(0, 0, 0, 3, head_width=0.2)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.show()
