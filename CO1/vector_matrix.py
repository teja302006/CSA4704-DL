# Experiment 1: Vector and Matrix Operations

import numpy as np
import matplotlib.pyplot as plt

customer = np.array([
    [5, 4, 3, 2, 1],
    [4, 5, 2, 3, 1],
    [1, 2, 5, 4, 3],
    [2, 1, 4, 5, 3],
    [5, 3, 4, 2, 1],
    [1, 3, 2, 5, 4]
])

book = np.array([
    [1, 2, 3, 4],
    [2, 1, 4, 3],
    [3, 4, 1, 2],
    [4, 3, 2, 1],
    [2, 3, 4, 1]
])

score = np.dot(customer, book)

print("Dot Product:", np.dot(customer[0], book[:, 0]))
print("Matrix Product:\n", score)
print("Transpose:\n", customer.T)
print("Element-wise Addition:\n", customer + 1)

a = np.array([[2, 1], [1, 2]])
print("Inverse:\n", np.linalg.inv(a))

norm = np.linalg.norm(customer, axis=1)
similarity = customer @ customer.T / np.outer(norm, norm)
print("Cosine Similarity:\n", similarity)

i, j = np.unravel_index(
    np.argmax(similarity - np.eye(6)), (6, 6)
)
print("Most Similar Customers:", i + 1, "and", j + 1)

x = customer[0]
W = np.array([[1, 2, 1, 0, 1], [2, 1, 0, 1, 2]])
b = np.array([1, 2])
print("Dense Layer Output:", W @ x + b)

plt.imshow(score, cmap="viridis", aspect="auto")
plt.colorbar(label="Score")
plt.xlabel("Books")
plt.ylabel("Customers")
plt.title("Customer-Book Score Heatmap")
plt.show()
