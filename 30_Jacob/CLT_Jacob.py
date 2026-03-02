import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp

def normal_pdf(x, mu=0.0, sigma=1.0):
    return (1.0/(sigma*sqrt(2*pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

np.random.seed(0)

M = 20000
n_list = [1, 2, 5, 10, 30, 50, 100]

# 
# 例1：Exponential(λ=1)
dist_name = "Exponential(λ=1)"
mu = 1.0
sigma = 1.0
def sample_X(size):
    return np.random.exponential(scale=1.0, size=size)

# # 例2：Bernoulli(p=0.2)
# dist_name = "Bernoulli(p=0.2)"
# p = 0.2
# mu = p
# sigma = sqrt(p*(1-p))
# def sample_X(size):
#     return np.random.binomial(n=1, p=p, size=size)

print("Distribution:", dist_name)
print("Theoretical mu =", mu, "sigma =", sigma)

fig, axes = plt.subplots(2, len(n_list), figsize=(4*len(n_list), 6))

for j, n in enumerate(n_list):
    # M 次抽樣，每次抽 n 個，算平均
    X = sample_X((M, n))
    Xbar = X.mean(axis=1)

    # (A) 樣本平均的分佈：應趨近 N(mu, sigma/sqrt(n))
    ax = axes[0, j]
    ax.hist(Xbar, bins=40, density=True)
    xs = np.linspace(Xbar.min(), Xbar.max(), 400)
    ax.plot(xs, normal_pdf(xs, mu, sigma/sqrt(n)), linewidth=2)
    ax.set_title(f"X̄, n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("density")

    # (B) 標準化後 Z：應趨近 N(0,1)
    Z = (Xbar - mu) / (sigma / sqrt(n))
    ax2 = axes[1, j]
    ax2.hist(Z, bins=40, density=True)
    zs = np.linspace(Z.min(), Z.max(), 400)
    ax2.plot(zs, normal_pdf(zs, 0, 1), linewidth=2)
    ax2.set_title(f"Z, n={n}")
    ax2.set_xlabel("z")
    ax2.set_ylabel("density")

plt.suptitle(f"CLT demonstration with {dist_name} (M={M})", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# 額外：數字檢查（平均、變異）
print("\nNumeric check (mean and std of X̄):")
for n in n_list:
    Xbar = sample_X((M, n)).mean(axis=1)
    print(f"n={n:3d} | mean(X̄)≈{Xbar.mean():.4f} (target {mu})"
          f" | std(X̄)≈{Xbar.std(ddof=0):.4f} (target {sigma/sqrt(n):.4f})")
