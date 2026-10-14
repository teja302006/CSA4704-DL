# Experiment 15: BPTT Gradient Analysis

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(10)

for activation in ["tanh", "relu"]:
    for length in [10, 50, 100]:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(length, 1)),
            tf.keras.layers.SimpleRNN(
                16, activation=activation, return_sequences=True
            ),
            tf.keras.layers.Dense(1)
        ])

        X = tf.random.normal((1, length, 1))
        Y = tf.random.normal((1, length, 1))

        with tf.GradientTape() as tape:
            tape.watch(X)
            output = model(X)
            loss = tf.reduce_mean((output - Y) ** 2)

        gradients = tape.gradient(loss, X)
        norms = tf.norm(gradients[0], axis=1).numpy()

        plt.plot(norms, label=f"Length {length}")

    plt.xlabel("Time Step")
    plt.ylabel("Gradient Norm")
    plt.title(f"BPTT Gradient Analysis: {activation}")
    plt.legend()
    plt.show()

print("Vanishing gradients approach zero.")
print("Exploding gradients become extremely large.")
