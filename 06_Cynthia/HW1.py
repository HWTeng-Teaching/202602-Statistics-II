import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

def plot_clt(data_source, name, params, sample_size, iterations=10000):
            # 分布類型, 顯示名稱, 分布參數, 每次抽樣的樣本數 (n), 模擬次數
    
    means = []
    
    for _ in range(iterations):
        if data_source == 'bernoulli':
            # Bernoulli 試驗，參數為 p
            sample = np.random.binomial(1, params, sample_size)
        elif data_source == 'exponential':
            # 指數分布，參數為 scale (1/lambda)
            sample = np.random.exponential(1/params, sample_size)
        
        means.append(np.mean(sample))

    # 繪圖
    plt.hist(means, bins=50, density=True, alpha=0.6, color='pink', edgecolor='black')
    plt.title(f'CLT - {name} (n={sample_size})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')

# 參數設定
p_lambda = 0.5  # Bernoulli 的 p 與 Exponential 的 lambda
n_values = [2, 10, 50]  # 觀察不同樣本數 n 的變化

plt.figure(figsize=(15, 10))

for i, n in enumerate(n_values):
    # Bernoulli
    plt.subplot(2, 3, i + 1)
    plot_clt('bernoulli', 'Bernoulli', p_lambda, n)
    
    # Exponential
    plt.subplot(2, 3, i + 4)
    plot_clt('exponential', 'Exponential', p_lambda, n)

plt.tight_layout()
plt.show()