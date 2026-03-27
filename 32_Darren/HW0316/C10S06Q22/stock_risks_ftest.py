"""F-test for equality of variances + 99% CI for variance ratio: Stock Risks (Problem 22).

Given: variances directly (s^2), not standard deviations.

Part a: Two-tailed F-test
  H0: sigma1^2 = sigma2^2
  H1: sigma1^2 != sigma2^2

Part b: 99% confidence interval for sigma2^2 / sigma1^2
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import mpmath as mp

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-config"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def f_cdf(x: float, dfn: int, dfd: int) -> float:
    """CDF of the F distribution."""
    if x <= 0:
        return 0.0
    return float(
        mp.betainc(dfn / 2, dfd / 2, 0, (dfn * x) / (dfn * x + dfd), regularized=True)
    )


def f_pdf(x: float, dfn: int, dfd: int) -> float:
    """PDF of the F distribution."""
    if x <= 0:
        return 0.0
    num = math.sqrt((dfn * x) ** dfn * dfd ** dfd / (dfn * x + dfd) ** (dfn + dfd))
    den = x * float(mp.beta(dfn / 2, dfd / 2))
    return num / den


def f_inv(p: float, dfn: int, dfd: int, tol: float = 1e-10) -> float:
    """Inverse CDF of the F distribution via bisection."""
    lo, hi = 1e-9, 1.0
    while f_cdf(hi, dfn, dfd) < p:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if f_cdf(mid, dfn, dfd) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def plot_f_distribution(f_stat: float, dfn: int, dfd: int, p_value: float,
                        ci: tuple, outfile: Path) -> None:
    """Plot F-distribution with shaded tails and CI bounds marked."""
    x_max = 7.0
    n_points = 700
    xs = [i * x_max / n_points for i in range(1, n_points + 1)]
    ys = [f_pdf(x, dfn, dfd) for x in xs]

    f_low = 1.0 / f_stat   # symmetric lower tail bound

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(xs, ys, color="#2c6fad", lw=2.2, label=f"F({dfn}, {dfd}) distribution")

    # Shade right tail
    tail_r_xs = [x for x in xs if x >= f_stat]
    tail_r_ys = [f_pdf(x, dfn, dfd) for x in tail_r_xs]
    if tail_r_xs:
        ax.fill_between(tail_r_xs, tail_r_ys, color="#e05c3a", alpha=0.45,
                        label=f"Two-tailed P-value = {p_value:.4f}")

    # Shade left tail
    tail_l_xs = [x for x in xs if x <= f_low]
    tail_l_ys = [f_pdf(x, dfn, dfd) for x in tail_l_xs]
    if tail_l_xs:
        ax.fill_between(tail_l_xs, tail_l_ys, color="#e05c3a", alpha=0.45)

    # F-statistic line
    ax.axvline(f_stat, color="#e05c3a", ls="--", lw=1.8, label=f"F = {f_stat:.4f}")
    ax.axvline(f_low,  color="#e05c3a", ls=":",  lw=1.5, label=f"1/F = {f_low:.4f}")

    # 99% CI bounds
    ax.axvline(ci[0], color="#2ca02c", ls="--", lw=1.5, label=f"99% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    ax.axvline(ci[1], color="#2ca02c", ls="--", lw=1.5)

    # Annotation
    y_annot = f_pdf(f_stat, dfn, dfd) * 0.5
    ax.annotate(
        f"F = {f_stat:.4f}\np = {p_value:.4f}",
        xy=(f_stat, y_annot),
        xytext=(f_stat + 0.5, y_annot + 0.04),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=10, color="#333",
    )

    ax.set_xlabel("F", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "F-test for Equality of Variances (Two-tailed)\n"
        r"Stock 1 ($s_1^2=1.54$, $n_1=15$) vs Stock 2 ($s_2^2=2.96$, $n_2=15$)",
        fontsize=12,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"Plot saved to: {outfile}")


def main() -> None:
    # --- Data (variances given directly) ---
    var1, n1 = 1.54, 15   # Stock 1
    var2, n2 = 2.96, 15   # Stock 2

    # Put larger variance in numerator
    if var1 >= var2:
        f_stat = var1 / var2
        dfn, dfd = n1 - 1, n2 - 1
        label_num, label_den = "Stock 1", "Stock 2"
        ratio = var1 / var2
    else:
        f_stat = var2 / var1
        dfn, dfd = n2 - 1, n1 - 1
        label_num, label_den = "Stock 2", "Stock 1"
        ratio = var2 / var1   # s2^2 / s1^2

    # --- Part a: Two-tailed p-value ---
    p_one_tail = 1.0 - f_cdf(f_stat, dfn, dfd)
    p_value = min(1.0, 2 * p_one_tail)

    # --- Part b: 99% CI for sigma2^2 / sigma1^2 ---
    # CI = ( (s2^2/s1^2) / F_{alpha/2, dfn, dfd},
    #        (s2^2/s1^2) * F_{alpha/2, dfd, dfn} )
    alpha_ci = 0.01
    f_crit = f_inv(1 - alpha_ci / 2, dfn, dfd)   # upper critical value
    ci_low  = ratio / f_crit
    ci_high = ratio * f_crit

    # --- Output ---
    print("=" * 55)
    print("Stock Risks — F-test for Equality of Variances")
    print("=" * 55)
    print(f"  Stock 1:  s1^2 = {var1},  n1 = {n1}")
    print(f"  Stock 2:  s2^2 = {var2},  n2 = {n2}")
    print()
    print("  [Part a] Two-tailed F-test")
    print("  H0: sigma1^2 = sigma2^2")
    print("  H1: sigma1^2 != sigma2^2")
    print()
    print(f"  F = {label_num} / {label_den} = {f_stat:.4f}  (larger / smaller)")
    print(f"  df1 (numerator)   = {dfn}")
    print(f"  df2 (denominator) = {dfd}")
    print()
    print(f"  P(F({dfn},{dfd}) > {f_stat:.4f}) = {p_one_tail:.6f}  (one tail)")
    print(f"  Two-tailed P-value = {p_value:.6f}")
    print()

    alpha = 0.05
    if p_value < alpha:
        print(f"  => p = {p_value:.4f} < alpha = {alpha}")
        print("  => Reject H0. Significant difference in variabilities.")
    else:
        print(f"  => p = {p_value:.4f} >= alpha = {alpha}")
        print("  => Fail to reject H0. No significant difference in variabilities.")
    print()
    print("  [Part b] 99% Confidence Interval for sigma2^2 / sigma1^2")
    print(f"  F_{{0.005, {dfn}, {dfd}}} = {f_crit:.4f}")
    print(f"  CI = ({ratio:.4f} / {f_crit:.4f},  {ratio:.4f} x {f_crit:.4f})")
    print(f"  99% CI: ({ci_low:.4f}, {ci_high:.4f})")
    print()
    if 1.0 < ci_low or 1.0 > ci_high:
        print("  => 1 is outside the CI -> evidence of unequal variances.")
    else:
        print("  => 1 is inside the CI -> consistent with equal variances.")
    print("=" * 55)

    # --- Plot ---
    outfile = Path(__file__).with_name("stock_risks_fplot.png")
    plot_f_distribution(f_stat, dfn, dfd, p_value, (ci_low, ci_high), outfile)


if __name__ == "__main__":
    main()
