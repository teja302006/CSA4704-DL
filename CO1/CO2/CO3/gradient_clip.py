# Experiment 17: RNN Gradient Clipping

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(10)
text = ("deep learning models learn patterns from sequential data " * 1000)
chars = sorted(set(text))
mapping = {c: i for i, c in enumerate(chars)}
data = np.array([mapping[c] for c in text])

X, y = [], []
for i in range(0, len(data) - 20, 3):
    X.append(data[i:i+20])
    y.append(data[i+20])

X = np.array(X[:2000])
y = np.array(y[:2000])

results = {}

for name, clipnorm, clipvalue in [
    ("No Clipping", None, None),
    ("Clip Norm", 1.0, None),
    ("Clip Value", None, 0.5)
]:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(chars), 16),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(len(chars))
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    losses, norms = [], []

    for start in range(0, len(X), 32):
        xb = tf.convert_to_tensor(X[start:start+32])
        yb = tf.convert_to_tensor(y[start:start+32])

        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    yb, pred, from_logits=True
                )
            )

        grads = tape.gradient(loss, model.trainable_variables)
        norm = tf.linalg.global_norm(grads).numpy()
        norms.append(norm)

        if not np.isfinite(loss.numpy()) or not np.isfinite(norm):
            print(name, "unstable at step", len(losses))
            break

        if clipnorm is not None:
            grads, _ = tf.clip_by_global_norm(grads, clipnorm)

        if clipvalue is not None:
            grads = [tf.clip_by_value(g, -clipvalue, clipvalue) for g in grads]

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())

    results[name] = (losses, norms)

for name, (losses, norms) in results.items():
    plt.plot(norms, label=name)

plt.title("Gradient Norm Comparison")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.legend()
plt.show()

for name, (losses, norms) in results.items():
    plt.plot(losses, label=name)

plt.title("Training Loss Comparison")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("Recommended starting threshold: global clipnorm = 1.0")
