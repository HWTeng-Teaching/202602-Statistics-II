import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 參數設定
p = 0.3          # Bernoulli 參數
n = 50           # 每組樣本大小
num_samples = 10000   # 重複實驗次數

# 產生樣本平均
sample_means = []

for _ in range(num_samples):
    sample = np.random.binomial(1, p, n)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# 理論常態分布
mu = p
sigma = np.sqrt(p*(1-p)/n)

x = np.linspace(min(sample_means), max(sample_means), 200)
y = norm.pdf(x, mu, sigma)

# 畫圖
plt.hist(sample_means, bins=40, density=True)
plt.plot(x, y)
plt.title("CLT for Bernoulli Distribution")
plt.show()