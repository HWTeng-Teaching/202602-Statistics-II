import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, kstest

def clt_demo(generator, mu, sigma, n_list=(5, 10, 30, 100), reps=20000, title=""):
    # generator: function(size)-> samples
    xgrid = np.linspace(-4, 4, 800)

    for n in n_list:
        # Simulate standardized sums
        X = generator((reps, n))
        Zn = (X.sum(axis=1) - n * mu) / (sigma * np.sqrt(n))

        # KS test vs N(0,1)
        ks_stat, ks_p = kstest(Zn, 'norm')

        # Plot histogram + normal pdf
        plt.figure()
        plt.hist(Zn, bins=60, density=True, alpha=0.75)
        plt.plot(xgrid, norm.pdf(xgrid, 0, 1), linewidth=2)
        plt.title(f"{title} | n={n} | KS stat={ks_stat:.4f}, p={ks_p:.4g}")
        plt.xlabel("Z_n")
        plt.ylabel("Density")
        plt.show()

        # Q-Q plot
        plt.figure()
        probplot(Zn, dist="norm", plot=plt)
        plt.title(f"{title} Q-Q Plot | n={n}")
        plt.show()


# ---- 1) Bernoulli(p) ----
p = 0.3
mu_b = p
sigma_b = np.sqrt(p * (1 - p))

def bernoulli_gen(shape):
    return np.random.binomial(1, p, size=shape)

clt_demo(
    generator=bernoulli_gen,
    mu=mu_b,
    sigma=sigma_b,
    n_list=(5, 10, 30, 100),
    reps=30000,
    title=f"Bernoulli(p={p}) CLT"
)

# ---- 2) Exponential(lambda) ----
lam = 1.0
mu_e = 1 / lam
sigma_e = 1 / lam  # std dev of Exp(lam) is 1/lam

def exp_gen(shape):
    # numpy uses scale = 1/lambda
    return np.random.exponential(scale=1/lam, size=shape)

clt_demo(
    generator=exp_gen,
    mu=mu_e,
    sigma=sigma_e,
    n_list=(5, 10, 30, 100),
    reps=30000,
    title=f"Exponential(lambda={lam}) CLT"
)