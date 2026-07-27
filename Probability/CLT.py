import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

import utils


#Gaussian Poplulation

mu = 10
sigma = 5

gaussian_population = np.random.normal(mu, sigma, 100_000)

sns.histplot(gaussian_population, stat = "density")
plt.show() 

#Sample means 

def sample_means(data, sample_size):
    means = []

    for _ in range(10_000):
        sample = np.random.choice(data, size= sample_size)

        means.append(np.mean(sample))

    return np.array(means)


gaussian_sample_means = sample_means(gaussian_population, sample_size=5)
sns.histplot(gaussian_sample_means, stat="density")
plt.show()

mu_sample_means = mu

#Compute the sigma 
sigma_sample_means = sigma /nq.sqrt(5)

#Define the x-range for the Gaussian curve
x_range = np.linspace(min(gaussian_sample_means), max(gaussian_sample_means),100)

#sns plot everything
sns.histplot(gaussian_sample_means, stat="density")
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
)
plt.show()

# Histogram of sample means (blue)
sns.histplot(gaussian_sample_means, stat="density", label="hist")

# Estimated PDF of sample means (red)
sns.kdeplot(
    data=gaussian_sample_means, 
    color="crimson",
    label="kde",
    linestyle="dashed",
    fill=True,
)

# Gaussian curve with estimated mu and sigma (black)
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
    label="gaussian",
)

plt.legend()
plt.show()

# Create the QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(gaussian_sample_means, plot=ax, fit=True)
plt.show()
utils.gaussian_clt()

n = 5
p = 0.8

binomial_population = np.random.binomial(n, p, 100_000)

binomial_pop_mean = np.mean(binomial_population)
binomial_pop_std = np.std(binomial_population)

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")

binomial_pop_mean = n * p
binomial_pop_std = np.sqrt(n * p * (1 - p))

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")

sample_size = 3
N = n * sample_size

condition_value = np.min([N * p, N * (1 - p)])
print(f"The condition value is: {condition_value:.1f}. CLT should hold?: {True if condition_value >= 5 else False}")

# Compute sample means
binomial_sample_means = sample_means(binomial_population, sample_size=sample_size)

# Compute estimated mu
mu_sample_means = n * p

# Compute estimated sigma
sigma_sample_means = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)