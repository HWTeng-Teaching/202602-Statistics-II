"""F-test for equality of variances: SAT Scores (Problem 16).

H0: sigma1^2 = sigma2^2
H1: sigma1^2 != sigma2^2  (two-tailed, testing pooled-t assumption)
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
    """Plot F-distribution PDF with both tails shaded (two-tailed test)."""
    x_max = 6.0
    n_points = 600
    xs = [i * x_max / n_points for i in range(1, n_points + 1)]
    ys = [f_pdf(x, dfn, dfd) for x in xs]

    # Lower critical value: F_low = 1 / F_stat (by symmetry of two-tailed F-test)
    f_low = 1.0 / f_stat

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Full PDF curve
    ax.plot(xs, ys, color="#2c6fad", lw=2.2, label=f"F({dfn}, {dfd}) distribution")

    # Shade right tail (F >= f_stat)
    tail_r_xs = [x for x in xs if x >= f_stat]
    tail_r_ys = [f_pdf(x, dfn, dfd) for x in tail_r_xs]
    if tail_r_xs:
        ax.fill_between(tail_r_xs, tail_r_ys, color="#e05c3a", alpha=0.45,
                        label=f"Two-tailed P-value = {p_value:.4f}")

    # Shade left tail (F <= f_low)
    tail_l_xs = [x for x in xs if x <= f_low]
    tail_l_ys = [f_pdf(x, dfn, dfd) for x in tail_l_xs]
    if tail_l_xs:
        ax.fill_between(tail_l_xs, tail_l_ys, color="#e05c3a", alpha=0.45)

    # Vertical lines
    ax.axvline(f_stat, color="#e05c3a", ls="--", lw=1.8, label=f"F = {f_stat:.4f}")
    ax.axvline(f_low,  color="#e05c3a", ls=":",  lw=1.5, label=f"1/F = {f_low:.4f}")

    # Annotation
    y_annot = f_pdf(f_stat, dfn, dfd) * 0.6
    ax.annotate(
        f"F = {f_stat:.4f}\np = {p_value:.4f}",
        xy=(f_stat, y_annot),
        xytext=(f_stat + 0.4, y_annot + 0.06),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=10, color="#333",
    )

    ax.set_xlabel("F", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "F-test for Equality of Variances (Two-tailed)\n"
        f"Chemistry ($s_1=114$, $n_1=15$) vs Physics ($s_2=103$, $n_2=15$)",
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
    s1, n1 = 114.0, 15   # Chemistry
    s2, n2 = 103.0, 15   # Physics

    var1 = s1 ** 2   # 12996
    var2 = s2 ** 2   # 10609

    # Put larger variance in numerator
    if var1 >= var2:
        f_stat = var1 / var2
        dfn, dfd = n1 - 1, n2 - 1
        label_num, label_den = "Chemistry", "Physics"
    else:
        f_stat = var2 / var1
        dfn, dfd = n2 - 1, n1 - 1
        label_num, label_den = "Physics", "Chemistry"

    # Two-tailed p-value: 2 * P(F > f_stat)
    p_one_tail = 1.0 - f_cdf(f_stat, dfn, dfd)
    p_value = min(1.0, 2 * p_one_tail)

    # --- Output ---
    print("=" * 52)
    print("SAT Scores — F-test for Equality of Variances")
    print("=" * 52)
    print(f"  Chemistry:  s1 = {s1},  s1^2 = {var1:.0f},  n1 = {n1}")
    print(f"  Physics:    s2 = {s2},  s2^2 = {var2:.0f},  n2 = {n2}")
    print()
    print("  H0: sigma1^2 = sigma2^2")
    print("  H1: sigma1^2 != sigma2^2  (two-tailed)")
    print()
    print(f"  F = {label_num} / {label_den} = {f_stat:.4f}  (larger / smaller)")
    print(f"  df1 (numerator)   = {dfn}")
    print(f"  df2 (denominator) = {dfd}")
    print()
    print(f"  P(F({dfn},{dfd}) > {f_stat:.4f}) = {p_one_tail:.6f}  (one tail)")
    print(f"  Two-tailed P-value = 2 x {p_one_tail:.6f} = {p_value:.6f}")
    print()

    alpha = 0.05
    if p_value < alpha:
        print(f"  => p = {p_value:.4f} < alpha = {alpha}")
        print("  => Reject H0. Variances are significantly different.")
        print("     Pooled t-test assumption is NOT satisfied.")
    else:
        print(f"  => p = {p_value:.4f} >= alpha = {alpha}")
        print("  => Fail to reject H0. No significant difference in variances.")
        print("     Pooled t-test assumption is satisfied.")
    print("=" * 52)

    # --- Plot ---
    outfile = Path(__file__).with_name("sat_scores_fplot.png")
    plot_f_distribution(f_stat, dfn, dfd, p_value, outfile)


if __name__ == "__main__":
    main()
