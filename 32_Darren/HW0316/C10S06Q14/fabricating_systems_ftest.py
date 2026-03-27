"""F-test for equality of variances: Fabricating Systems (Problem 14).

H0: sigma1^2 = sigma2^2
H1: sigma2^2 > sigma1^2  (old system more variable -> one-tailed upper)
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
    """CDF of the F distribution using the regularized incomplete beta function."""
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


def plot_f_distribution(f_stat: float, dfn: int, dfd: int, p_value: float, outfile: Path) -> None:
    """Plot F-distribution PDF with shaded right tail (p-value region)."""
    # x range: from 0 to slightly beyond x_max where pdf is negligible
    x_max = 6.0
    n_points = 600
    xs = [i * x_max / n_points for i in range(1, n_points + 1)]
    ys = [f_pdf(x, dfn, dfd) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Full PDF curve
    ax.plot(xs, ys, color="#2c6fad", lw=2.2, label=f"F({dfn}, {dfd}) distribution")

    # Shade the right tail (p-value area)
    tail_xs = [x for x in xs if x >= f_stat]
    tail_ys = [f_pdf(x, dfn, dfd) for x in tail_xs]
    if tail_xs:
        ax.fill_between(
            tail_xs, tail_ys,
            color="#e05c3a", alpha=0.45,
            label=f"P-value = {p_value:.4f}"
        )

    # Vertical line at F-statistic
    ax.axvline(f_stat, color="#e05c3a", ls="--", lw=1.8,
               label=f"F = {f_stat:.4f}")

    # Annotation
    y_annot = f_pdf(f_stat, dfn, dfd) * 0.5
    ax.annotate(
        f"F = {f_stat:.4f}\np = {p_value:.4f}",
        xy=(f_stat, y_annot),
        xytext=(f_stat + 0.5, y_annot + 0.05),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=10,
        color="#333",
    )

    ax.set_xlabel("F", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "F-test for Equality of Variances\n"
        f"New System ($s_1={15.6}$, $n_1=30$) vs Old System ($s_2={28.2}$, $n_2=30$)",
        fontsize=12,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"Plot saved to: {outfile}")


def main() -> None:
    # --- Data ---
    s1, n1 = 15.6, 30   # New system
    s2, n2 = 28.2, 30   # Old system

    var1 = s1 ** 2   # 243.36
    var2 = s2 ** 2   # 795.24

    # --- F-statistic (old / new, larger numerator for upper-tail test) ---
    f_stat = var2 / var1
    dfn = n2 - 1   # 29
    dfd = n1 - 1   # 29

    # --- One-tailed p-value: P(F > f_stat) ---
    p_value = 1.0 - f_cdf(f_stat, dfn, dfd)

    # --- Output ---
    print("=" * 50)
    print("Fabricating Systems — F-test for Variances")
    print("=" * 50)
    print(f"  New System:  s1 = {s1},  s1^2 = {var1:.2f},  n1 = {n1}")
    print(f"  Old System:  s2 = {s2},  s2^2 = {var2:.2f},  n2 = {n2}")
    print()
    print(f"  H0: sigma1^2 = sigma2^2")
    print(f"  H1: sigma2^2 > sigma1^2  (one-tailed, upper)")
    print()
    print(f"  F = s2^2 / s1^2 = {var2:.2f} / {var1:.2f} = {f_stat:.4f}")
    print(f"  df1 (numerator)   = {dfn}")
    print(f"  df2 (denominator) = {dfd}")
    print()
    print(f"  P-value = P(F({dfn},{dfd}) > {f_stat:.4f}) = {p_value:.6f}")
    print()

    alpha = 0.05
    if p_value < alpha:
        print(f"  => p = {p_value:.4f} < alpha = {alpha}")
        print("  => Reject H0. Sufficient evidence that the old system")
        print("     has greater variability -> increased maintenance warranted.")
    else:
        print(f"  => p = {p_value:.4f} >= alpha = {alpha}")
        print("  => Fail to reject H0.")
    print("=" * 50)

    # --- Plot ---
    outfile = Path(__file__).with_name("fabricating_systems_fplot.png")
    plot_f_distribution(f_stat, dfn, dfd, p_value, outfile)


if __name__ == "__main__":
    main()
