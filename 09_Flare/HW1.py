import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ==============================
#  CLT for Bernoulli Distribution
# ==============================
def clt_bernoulli():
    p = 0.3          # Bernoulli parameter
    n = 50           # sample size
    num_samples = 10000

    sample_means = []

    for _ in range(num_samples):
        sample = np.random.binomial(1, p, n)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Theoretical Normal distribution
    mu = p
    sigma = np.sqrt(p * (1 - p) / n)

    x = np.linspace(min(sample_means), max(sample_means), 200)
    y = norm.pdf(x, mu, sigma)

    plt.figure()
    plt.hist(sample_means, bins=40, density=True)
    plt.plot(x, y)
    plt.title("CLT for Bernoulli Distribution")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.show()


# ==============================
#  CLT for Exponential Distribution
# ==============================
def clt_exponential():
    lam = 1          # lambda
    n = 50
    num_samples = 10000

    sample_means = []

    for _ in range(num_samples):
        sample = np.random.exponential(1 / lam, n)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Theoretical Normal distribution
    mu = 1 / lam
    sigma = 1 / (lam * np.sqrt(n))

    x = np.linspace(min(sample_means), max(sample_means), 200)
    y = norm.pdf(x, mu, sigma)

    plt.figure()
    plt.hist(sample_means, bins=40, density=True)
    plt.plot(x, y)
    plt.title("CLT for Exponential Distribution")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.show()


# ==============================
#  Main
# ==============================
if __name__ == "__main__":
    print("Running CLT simulation for Bernoulli distribution...")
    clt_bernoulli()

    print("Running CLT simulation for Exponential distribution...")
    clt_exponential()