# Experiment 7: Perceptron and Forward Propagation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

X = np.random.rand(100, 2)
y = ((X[:, 0] + X[:, 1]) > 1).astype(int)

w = np.zeros(2)
b = 0
lr = 0.1

for epoch in range(20):
    errors = 0
    for i in range(len(X)):
        pred = int(np.dot(X[i], w) + b >= 0)
        error = y[i] - pred
        w += lr * error * X[i]
        b += lr * error
        errors += abs(error)
    print("Epoch", epoch + 1, "Errors:", errors, "Weights:", w)

plt.scatter(X[:, 0], X[:, 1], c=y)
x = np.linspace(0, 1, 100)
if w[1] != 0:
    plt.plot(x, -(w[0] * x + b) / w[1])
plt.title("Perceptron Decision Boundary")
plt.show()

XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
YOR = np.array([0, 1, 1, 0])
w = np.zeros(2)
b = 0

for epoch in range(20):
    errors = 0
    for i in range(4):
        pred = int(np.dot(XOR[i], w) + b >= 0)
        error = YOR[i] - pred
        w += lr * error * XOR[i]
        b += lr * error
        errors += abs(error)
    print("XOR Epoch:", epoch + 1, "Errors:", errors)

print("XOR does not converge with a single perceptron.")

z = np.array([-2, -1, 0, 1, 2])
print("Step:", (z >= 0).astype(int))
print("Sigmoid:", 1 / (1 + np.exp(-z)))
print("ReLU:", np.maximum(0, z))
