# Experiment 4: Entropy and KL Divergence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

true = np.array([0.5, 0.3, 0.2])
model1 = np.array([0.6, 0.25, 0.15])
model2 = np.array([0.4, 0.4, 0.2])

H = entropy(true)
print("Entropy:", H)

for i, model in enumerate([model1, model2], 1):
    CE = -np.sum(true * np.log(model))
    KL = entropy(true, model)
    print("Model", i)
    print("Cross Entropy:", CE)
    print("KL Divergence:", KL)
    print("H + KL:", H + KL)

print("KL(P || Q):", entropy(true, model1))
print("KL(Q || P):", entropy(model1, true))

p = np.linspace(0.01, 0.99, 100)
y = np.ones(100)

loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

plt.plot(p, loss)
plt.xlabel("Predicted Probability")
plt.ylabel("Binary Cross Entropy")
plt.title("Binary Cross-Entropy Loss")
plt.grid()
plt.show()
