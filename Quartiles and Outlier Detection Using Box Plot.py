
import matplotlib.pyplot as plt
import numpy as np

data = [12, 15, 18, 22, 28, 25, 30, 35, 40, 95]

data.sort()

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = [x for x in data if x < lower or x > upper]

print("Q1 =", q1)
print("Q3 =", q3)
print("IQR =", iqr)
print("Outliers =", outliers)

plt.boxplot(data)
plt.title("Box Plot")
plt.ylabel("Values")
plt.show()
