# Experiment 21: Factor Analysis and Latent Factors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis, PCA

np.random.seed(10)

data = pd.DataFrame(
    np.random.randint(1, 6, (300, 10)),
    columns=[f"Q{i}" for i in range(1, 11)]
)

print(data.head())
print("Correlation:\n", data.corr())

for n in [2, 3, 4]:
    model = FactorAnalysis(n_components=n, random_state=10)
    model.fit(data)
    loadings = model.components_.T

    plt.figure()
    plt.imshow(loadings, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(range(n), [f"Factor {i+1}" for i in range(n)])
    plt.yticks(range(10), data.columns)
    plt.title(f"Factor Loadings: {n} Factors")
    plt.show()

    print(f"{n} Factors:\n", pd.DataFrame(loadings, index=data.columns))

pca = PCA(n_components=3).fit(data)
print("PCA Explained Variance:", pca.explained_variance_ratio_)
