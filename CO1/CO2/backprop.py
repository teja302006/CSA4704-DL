# Experiment 8: Two-Layer Neural Network

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

lr = 0.5
losses = []

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

for epoch in range(10000):
    H = sigmoid(X @ W1 + b1)
    O = sigmoid(H @ W2 + b2)

    loss = np.mean((O - Y) ** 2)
    losses.append(loss)

    dO = 2 * (O - Y) * O * (1 - O) / len(X)
    dW2 = H.T @ dO
    db2 = np.sum(dO, axis=0, keepdims=True)

    dH = (dO @ W2.T) * H * (1 - H)
    dW1 = X.T @ dH
    db1 = np.sum(dH, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("Predictions:\n", np.round(O, 3))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Backpropagation Training Loss")
plt.show()

xx, yy = np.meshgrid(
    np.linspace(-0.2, 1.2, 100),
    np.linspace(-0.2, 1.2, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = sigmoid(sigmoid(grid @ W1 + b1) @ W2 + b2)
plt.contourf(xx, yy, Z.reshape(xx.shape), levels=20, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), edgecolors="black")
plt.title("XOR Decision Boundary")
plt.show()

original = W1[0, 0]
eps = 1e-5

def calculate_loss():
    output = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
    return np.mean((output - Y) ** 2)

W1[0, 0] = original + eps
loss1 = calculate_loss()
W1[0, 0] = original - eps
loss2 = calculate_loss()
W1[0, 0] = original

print("Numerical Gradient:", (loss1 - loss2) / (2 * eps))
