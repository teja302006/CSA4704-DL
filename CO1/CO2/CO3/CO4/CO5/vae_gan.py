# Experiment 29: VAE and GAN Sample Generation

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(X_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255
X_train = X_train[..., None]
latent = 2

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)
mu = tf.keras.layers.Dense(latent)(x)
logvar = tf.keras.layers.Dense(latent)(x)

def sample(args):
    mu, logvar = args
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps

z = tf.keras.layers.Lambda(sample)([mu, logvar])
encoder = tf.keras.Model(inputs, [mu, logvar, z])

decoder_input = tf.keras.Input(shape=(latent,))
x = tf.keras.layers.Dense(128, activation="relu")(decoder_input)
x = tf.keras.layers.Dense(784, activation="sigmoid")(x)
outputs = tf.keras.layers.Reshape((28, 28, 1))(x)
decoder = tf.keras.Model(decoder_input, outputs)

class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, logvar, z = self.encoder(data)
            reconstruction = self.decoder(z)
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=1)
            )
            loss = rec_loss + kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

vae = VAE()
vae.compile(optimizer="adam")
vae.fit(X_train, epochs=3, batch_size=128)

samples = decoder.predict(np.random.randn(10, latent), verbose=0)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(samples[i, :, :, 0], cmap="gray")
    plt.axis("off")
plt.show()

noise_dim = 100
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(noise_dim,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28, 1))
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(noise_dim,))
gan = tf.keras.Model(gan_input, discriminator(generator(gan_input)))
gan.compile(optimizer="adam", loss="binary_crossentropy")

g_losses, d_losses = [], []

for epoch in range(5):
    idx = np.random.randint(0, len(X_train), 128)
    real = X_train[idx]
    noise = np.random.randn(128, noise_dim)
    fake = generator.predict(noise, verbose=0)

    discriminator.trainable = True
    d_loss = discriminator.train_on_batch(
        np.concatenate([real, fake]),
        np.concatenate([np.ones((128, 1)), np.zeros((128, 1))])
    )

    discriminator.trainable = False
    g_loss = gan.train_on_batch(
        np.random.randn(128, noise_dim), np.ones((128, 1))
    )

    d_losses.append(float(np.mean(d_loss)))
    g_losses.append(float(np.mean(g_loss)))

plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.legend()
plt.show()
