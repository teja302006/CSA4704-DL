# Experiment 27: Deep Belief Network

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(10)

digits = load_digits()
X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

def train_rbm(data, hidden, epochs=10):
    W = np.random.normal(0, 0.01, (data.shape[1], hidden))
    b = np.zeros(hidden)
    a = np.zeros(data.shape[1])

    for _ in range(epochs):
        for v in data:
            ph = 1 / (1 + np.exp(-np.clip(v @ W + b, -50, 50)))
            h = (np.random.rand(hidden) < ph).astype(float)
            pv = 1 / (1 + np.exp(-np.clip(h @ W.T + a, -50, 50)))
            ph1 = 1 / (1 + np.exp(-np.clip(pv @ W + b, -50, 50)))
            W += 0.001 * (np.outer(v, ph) - np.outer(pv, ph1))
            a += 0.001 * (v - pv)
            b += 0.001 * (ph - ph1)

    return W, b

W1, b1 = train_rbm(X_train, 128)
H1 = 1 / (1 + np.exp(-X_train @ W1 - b1))
W2, b2 = train_rbm(H1, 64)

plt.imshow(W1[:, :25].T.reshape(25, 8, 8)[0], cmap="gray")
plt.title("First RBM Filter")
plt.show()

def build(pretrained=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(64, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    if pretrained:
        model.layers[0].set_weights([W1, b1])
        model.layers[1].set_weights([W2, b2])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

for pretrained in [True, False]:
    model = build(pretrained)
    model.fit(X_train[:1000], y_train[:1000], epochs=15, verbose=0)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("Pretrained:", pretrained,
          "Accuracy:", accuracy_score(y_test, pred))

features = PCA(n_components=2).fit_transform(X)
plt.scatter(features[:, 0], features[:, 1], c=y, cmap="tab10")
plt.title("Digit Representation")
plt.show()
