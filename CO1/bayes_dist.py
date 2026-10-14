# Experiment 3: Probability Distributions and Bayes Theorem

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, uniform, norm

np.random.seed(10)

distributions = [
    ("Bernoulli", bernoulli(0.5), bernoulli.rvs(0.5, size=1000)),
    ("Binomial", binom(10, 0.5), binom.rvs(10, 0.5, size=1000)),
    ("Poisson", poisson(4), poisson.rvs(4, size=1000)),
    ("Uniform", uniform(0, 1), uniform.rvs(size=1000)),
    ("Gaussian", norm(0, 1), norm.rvs(size=1000))
]

for name, dist, samples in distributions:
    plt.figure()
    plt.hist(samples, bins=20, density=True, alpha=0.6)
    x = np.arange(0, 11) if name in ["Bernoulli", "Binomial", "Poisson"] else np.linspace(min(samples), max(samples), 100)
    if name in ["Bernoulli", "Binomial", "Poisson"]:
        plt.stem(x, dist.pmf(x))
    else:
        plt.plot(x, dist.pdf(x))
    plt.title(name)
    plt.show()

p = 0.01
sensitivity = 0.95
specificity = 0.90

posterior = sensitivity * p / (
    sensitivity * p + (1 - specificity) * (1 - p)
)
print("P(Disease | Positive):", posterior)

n = 100000
disease = np.random.random(n) < p
positive = np.where(
    disease,
    np.random.random(n) < sensitivity,
    np.random.random(n) < (1 - specificity)
)

print("Simulated Probability:", positive[disease].sum() / positive.sum())

prevalence = np.linspace(0.001, 0.20, 100)
posterior = sensitivity * prevalence / (
    sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
)

plt.plot(prevalence * 100, posterior * 100)
plt.xlabel("Disease Prevalence (%)")
plt.ylabel("Posterior Probability (%)")
plt.title("Bayes Theorem")
plt.grid()
plt.show()
