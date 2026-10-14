# Experiment 30: Generative Model Evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

(X_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=10
)

X_train = X_train[..., None]
X_val = X_val[..., None]

latent = 32
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)
mu = tf.keras.layers.Dense(latent)(x)
logvar = tf.keras.layers.Dense(latent)(x)

z = tf.keras.layers.Lambda(
    lambda a: a[0] + tf.exp(0.5 * a[1]) *
    tf.random.normal(tf.shape(a[0]))
)([mu, logvar])

encoder = tf.keras.Model(inputs, z)

decoder_input = tf.keras.Input(shape=(latent,))
x = tf.keras.layers.Dense(128, activation="relu")(decoder_input)
x = tf.keras.layers.Dense(784, activation="sigmoid")(x)
decoder = tf.keras.Model(
    decoder_input, tf.keras.layers.Reshape((28, 28, 1))(x)
)

vae_input = tf.keras.Input(shape=(28, 28, 1))
vae = tf.keras.Model(vae_input, decoder(encoder(vae_input)))
vae.compile(optimizer="adam", loss="binary_crossentropy")
vae.fit(X_train, X_train, epochs=3, batch_size=128, verbose=0)

vae_samples = decoder.predict(np.random.randn(1000, latent), verbose=0)

noise_dim = 100
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(noise_dim,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28, 1))
])

generator.compile(optimizer="adam", loss="mse")
generator.fit(
    np.random.randn(1000, noise_dim),
    X_train[:1000], epochs=3, verbose=0
)

gan_samples = generator.predict(np.random.randn(1000, noise_dim), verbose=0)

def diversity(samples):
    flat = samples.reshape(len(samples), -1)
    return np.mean(np.std(flat, axis=0))

results = pd.DataFrame({
    "Model": ["VAE", "GAN"],
    "Diversity": [diversity(vae_samples), diversity(gan_samples)],
    "Reconstruction/Generation Count": [len(vae_samples), len(gan_samples)]
})

print(results)

real = X_val[:500].reshape(500, -1)
vae_flat = vae_samples[:500].reshape(500, -1)
gan_flat = gan_samples[:500].reshape(500, -1)

combined = np.vstack([real, vae_flat, gan_flat])
labels = ["Real"] * 500 + ["VAE"] * 500 + ["GAN"] * 500

reduced = PCA(n_components=2).fit_transform(combined)

plt.scatter(reduced[:, 0], reduced[:, 1], c=[
    0 if x == "Real" else 1 if x == "VAE" else 2 for x in labels
], s=5)
plt.title("Real and Generated Feature Comparison")
plt.show()

print("Recommendation: Compare sample quality, diversity, and downstream classifier performance before deployment.")
