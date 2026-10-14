# Experiment 5: Bias Variance and Overfitting

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(10)

X = np.random.uniform(0, 10, 150).reshape(-1, 1)
y = 2 * X.ravel() + np.sin(X.ravel()) * 10 + np.random.normal(0, 5, 150)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

degrees = [1, 3, 9, 15]
train_errors = []
test_errors = []
grid = np.linspace(0, 10, 300).reshape(-1, 1)

plt.scatter(X, y, color="black", s=15)

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))
    plt.plot(grid, model.predict(grid), label=f"Degree {degree}")

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, test_errors, marker="o", label="Validation Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()
plt.show()

predictions = []

for seed in range(30):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), len(X), replace=True)
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(X[idx], y[idx])
    predictions.append(model.predict(grid).ravel())

predictions = np.array(predictions)
print("Estimated Variance:", np.mean(np.var(predictions, axis=0)))

best = degrees[np.argmin(test_errors)]
print("Best Degree:", best)
