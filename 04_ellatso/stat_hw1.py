import numpy as np
import matplotlib.pyplot as plt

# ── 參數設定 ──────────────────────────────────────────────
N_SIMULATIONS = 10000
SAMPLE_SIZES  = [1, 5, 30, 100]

BERNOULLI_P   = 0.3
EXP_LAMBDA    = 2.0          # mean = 1/λ = 0.5

fig, axes = plt.subplots(2, len(SAMPLE_SIZES), figsize=(14, 6))
fig.suptitle("Central Limit Theorem — Bernoulli(p=0.3)  &  Exp(λ=2)", fontsize=14)


def plot_clt(ax, sample_means, dist_name, n):
    """畫樣本均值分佈直方圖，疊加理論常態曲線"""
    mu    = np.mean(sample_means)
    sigma = np.std(sample_means)

    ax.hist(sample_means, bins=60, density=True, color="steelblue",
            alpha=0.7, edgecolor="white", linewidth=0.4)

    # 理論常態 PDF
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, "r-", linewidth=1.8)

    ax.set_title(f"{dist_name}  n={n}", fontsize=10)
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density" if n == SAMPLE_SIZES[0] else "")
    ax.tick_params(labelsize=8)


# ── Bernoulli ─────────────────────────────────────────────
for col, n in enumerate(SAMPLE_SIZES):
    # 每次模擬：抽 n 個 Bernoulli 樣本，算均值
    data = np.random.binomial(1, BERNOULLI_P, size=(N_SIMULATIONS, n))
    sample_means = data.mean(axis=1)
    plot_clt(axes[0, col], sample_means, "Bernoulli", n)

# ── Exponential ───────────────────────────────────────────
for col, n in enumerate(SAMPLE_SIZES):
    data = np.random.exponential(1 / EXP_LAMBDA, size=(N_SIMULATIONS, n))
    sample_means = data.mean(axis=1)
    plot_clt(axes[1, col], sample_means, "Exp", n)


plt.tight_layout()
plt.savefig(r"C:\Users\ella.tso\Downloads\clt_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → clt_demo.png")