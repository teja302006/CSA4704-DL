
import random
import pandas as pd
import matplotlib.pyplot as plt

ages = [random.randint(18, 60) for i in range(100)]

bins = list(range(15, 66, 5))
freq = pd.cut(ages, bins=bins, right=False).value_counts().sort_index()

print("Frequency Table:")
print(freq)

plt.hist(ages, bins=bins, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Histogram")
plt.show()
