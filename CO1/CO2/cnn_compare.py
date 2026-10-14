# Experiment 11: CNN Image Classification

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
y_train = y_train.ravel()
y_test = y_test.ravel()

X_train, X_val = X_train[:-5000], X_train[-5000:]
y_train, y_val = y_train[:-5000], y_train[-5000:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    X_train, y_train, epochs=5, batch_size=64,
    validation_data=(X_val, y_val)
)

print("Test Results:", model.evaluate(X_test, y_test))

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.legend()
plt.show()

pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix")
plt.show()

wrong = np.where(pred != y_test)[0][:5]
for i in wrong:
    plt.imshow(X_test[i])
    plt.title(f"True: {y_test[i]}, Predicted: {pred[i]}")
    plt.axis("off")
    plt.show()

feature_model = tf.keras.Model(
    inputs=model.inputs, outputs=model.layers[0].output
)
features = feature_model.predict(X_test[:1], verbose=0)

for i in range(min(6, features.shape[-1])):
    plt.subplot(2, 3, i + 1)
    plt.imshow(features[0, :, :, i], cmap="viridis")
    plt.axis("off")

plt.suptitle("First Convolution Feature Maps")
plt.show()
