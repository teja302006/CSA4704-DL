# Experiment 24: Bayesian Inference and Sampling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling, RejectionSampling
from bayesian_network import model

infer = VariableElimination(model)
evidence_list = [
    {"Xray": 1, "Cough": 1},
    {"Xray": 1, "Cough": 1, "Smoking": 1},
    {"Xray": 0, "Cough": 0}
]

for evidence in evidence_list:
    result = infer.query(["LungDisease"], evidence=evidence)
    print(evidence, result)

sampler = BayesianModelSampling(model)
samples = sampler.forward_sample(size=10000, show_progress=False)
print(samples.head())

rejection = RejectionSampling(model)
sampled = rejection.query(
    variables=["LungDisease"],
    evidence=[("Xray", 1), ("Cough", 1)],
    show_progress=False
)
print("Rejection Sampling:\n", sampled)

sizes = [100, 500, 1000, 5000, 10000]
estimates = []

for n in sizes:
    subset = samples.iloc[:n]
    selected = subset[(subset["Xray"] == 1) & (subset["Cough"] == 1)]
    estimates.append(
        (selected["LungDisease"] == 1).mean()
        if len(selected) else np.nan
    )

exact = infer.query(
    ["LungDisease"], evidence={"Xray": 1, "Cough": 1}
).values[1]

print(pd.DataFrame({
    "Sample Size": sizes,
    "Estimated Probability": estimates,
    "Exact Probability": exact
}))

plt.plot(sizes, estimates, marker="o", label="Estimated")
plt.axhline(exact, linestyle="--", label="Exact")
plt.xlabel("Sample Size")
plt.ylabel("P(Lung Disease)")
plt.legend()
plt.show()
