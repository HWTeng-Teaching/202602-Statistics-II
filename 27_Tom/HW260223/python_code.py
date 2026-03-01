import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Set up the figure
plt.figure(figsize=(15, 10))

# Part 1: Bernoulli Distribution
# ------------------------------
p = 0.3  # probability of success
sample_sizes = [1, 5, 10, 30, 50]

for i, n in enumerate(sample_sizes):
    # Generate 10000 samples, each with n Bernoulli trials
    samples = np.random.binomial(1, p, size=(10000, n))
    
    # Calculate the mean of each sample
    sample_means = samples.mean(axis=1)
    
    # Standardize the means
    if n > 1:  # Avoid division by zero for n=1
        standardized_means = (sample_means - p) / np.sqrt(p * (1-p) / n)
    else:
        standardized_means = sample_means
    
    # Plot
    plt.subplot(2, len(sample_sizes), i+1)
    sns.histplot(standardized_means, kde=True, stat="density")
    
    # Add normal distribution curve for comparison
    if n > 1:
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
    
    plt.title(f'Bernoulli CLT (n={n})')
    plt.xlabel('Standardized Sample Mean')
    plt.ylim(0, 1.0)

# Part 2: Exponential Distribution
# -------------------------------
lambda_rate = 2  # rate parameter
mean = 1/lambda_rate
variance = 1/(lambda_rate**2)

for i, n in enumerate(sample_sizes):
    # Generate 10000 samples, each with n exponential values
    samples = np.random.exponential(scale=1/lambda_rate, size=(10000, n))
    
    # Calculate the mean of each sample
    sample_means = samples.mean(axis=1)
    
    # Standardize the means
    if n > 1:
        standardized_means = (sample_means - mean) / np.sqrt(variance / n)
    else:
        standardized_means = sample_means
    
    # Plot
    plt.subplot(2, len(sample_sizes), i+1+len(sample_sizes))
    sns.histplot(standardized_means, kde=True, stat="density")
    
    # Add normal distribution curve for comparison
    if n > 1:
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
    
    plt.title(f'Exponential CLT (n={n})')
    plt.xlabel('Standardized Sample Mean')
    plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()

