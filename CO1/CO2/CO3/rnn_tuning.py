# Experiment 18: RNN, LSTM and GRU Hyperparameter Tuning

import time
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.random.seed(10)
tf.random.set_seed(10)

vocab = 10000
maxlen = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

X_val, y_val = X_train[-5000:], y_train[-5000:]
X_train, y_train = X_train[:-5000], y_train[:-5000]

def build_model(hp, cell_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(maxlen,)))

    embed = hp.Choice("embedding", [32, 64])
    model.add(tf.keras.layers.Embedding(vocab, embed))

    units = hp.Choice("units", [32, 64])
    layers = hp.Int("layers", 1, 2)
    dropout = hp.Choice("dropout", [0.2, 0.4])

    for i in range(layers):
        return_seq = i < layers - 1

        if cell_type == "RNN":
            layer = tf.keras.layers.SimpleRNN
        elif cell_type == "LSTM":
            layer = tf.keras.layers.LSTM
        else:
            layer = tf.keras.layers.GRU

        model.add(layer(units, return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    lr = hp.Choice("learning_rate", [0.001, 0.0005])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

results = []

for cell in ["RNN", "LSTM", "GRU"]:
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, cell),
        objective="val_accuracy",
        max_trials=3,
        directory="tuning",
        project_name=cell,
        overwrite=True
    )

    start = time.time()
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3, batch_size=64, verbose=1
    )
    duration = time.time() - start

    best = tuner.get_best_models(1)[0]
    loss, accuracy = best.evaluate(X_test, y_test, verbose=0)

    results.append([
        cell, accuracy, best.count_params(), duration,
        tuner.get_best_hyperparameters(1)[0].values
    ])

print("Model | Accuracy | Parameters | Training Time | Best Settings")
for row in results:
    print(row)

plt.bar([r[0] for r in results], [r[1] for r in results])
plt.xlabel("Recurrent Model")
plt.ylabel("Test Accuracy")
plt.title("RNN, LSTM and GRU Comparison")
plt.show()
