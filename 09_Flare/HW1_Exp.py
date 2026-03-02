import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 參數設定
lam = 1          # lambda
n = 50
num_samples = 10000

sample_means = []

for _ in range(num_samples):
    sample = np.random.exponential(1/lam, n)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# 理論常態分布
mu = 1/lam
sigma = 1/(lam*np.sqrt(n))

x = np.linspace(min(sample_means), max(sample_means), 200)
y = norm.pdf(x, mu, sigma)

# 畫圖
plt.hist(sample_means, bins=40, density=True)
plt.plot(x, y)
plt.title("CLT for Exponential Distribution")
plt.show()