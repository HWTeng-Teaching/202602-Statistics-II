"""Paired t-test: Reaction Times (Problem 11, Chapter 10 Section 7).

Same 16 people (8 pairs shown) measured under two stimuli.
H0: mu_d = 0  (no difference in mean reaction times)
H1: mu_d != 0  (two-tailed, alpha = 0.05)
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


def t_cdf(x: float, df: float) -> float:
    """CDF of Student t distribution."""
    if df <= 0:
        raise ValueError("df must be positive.")
    if x == 0:
        return 0.5
    base = float(mp.betainc(df / 2, 0.5, 0, df / (df + x * x), regularized=True))
    return 1 - 0.5 * base if x > 0 else 0.5 * base


def t_pdf(x: float, df: float) -> float:
    """PDF of Student t distribution."""
    coeff = float(mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2)))
    return coeff * (1 + x * x / df) ** (-(df + 1) / 2)


def t_inv(p: float, df: float, tol: float = 1e-10) -> float:
    """Inverse CDF of t distribution via bisection."""
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def plot_t_distribution(t_stat: float, df: float, p_value: float,
                        t_crit: float, outfile: Path) -> None:
    """Plot t-distribution with two-tailed rejection regions and test statistic."""
    x_max = 4.5
    n_points = 600
    xs = [-x_max + i * 2 * x_max / n_points for i in range(n_points + 1)]
    ys = [t_pdf(x, df) for x in xs]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(xs, ys, color="#2c6fad", lw=2.2, label=f"t({int(df)}) distribution")

    # Rejection regions (|t| > t_crit)
    for sign in [1, -1]:
        bound = sign * t_crit
        rej_xs = [x for x in xs if (sign == 1 and x >= bound) or (sign == -1 and x <= bound)]
        rej_ys = [t_pdf(x, df) for x in rej_xs]
        if rej_xs:
            label = f"Rejection region (α=0.05)\n|t| > {t_crit:.4f}" if sign == 1 else None
            ax.fill_between(rej_xs, rej_ys, color="#e05c3a", alpha=0.35, label=label)

    # P-value tails
    for sign in [1, -1]:
        bound = sign * abs(t_stat)
        pv_xs = [x for x in xs if (sign == 1 and x >= bound) or (sign == -1 and x <= bound)]
        pv_ys = [t_pdf(x, df) for x in pv_xs]
        if pv_xs:
            label = f"P-value = {p_value:.4f}" if sign == 1 else None
            ax.fill_between(pv_xs, pv_ys, color="#ff7f0e", alpha=0.55, label=label)

    # Test statistic line
    ax.axvline(t_stat, color="#d62728", ls="--", lw=1.8,
               label=f"t = {t_stat:.4f}")
    ax.axvline(-t_crit, color="#555", ls=":", lw=1.3)
    ax.axvline( t_crit, color="#555", ls=":", lw=1.3)

    # Annotation
    ax.annotate(
        f"t = {t_stat:.4f}\np = {p_value:.4f}",
        xy=(t_stat, t_pdf(t_stat, df)),
        xytext=(t_stat - 1.2, t_pdf(t_stat, df) + 0.06),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=10, color="#333",
    )

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Paired t-test: Reaction Times\n"
        r"$H_0: \mu_d = 0$ vs $H_1: \mu_d \neq 0$  ($\alpha = 0.05$)",
        fontsize=12,
    )
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"Plot saved to: {outfile}")


def main() -> None:
    # --- Data ---
    stim1 = [1, 3, 2, 1, 2, 1, 3, 2]
    stim2 = [4, 2, 3, 3, 1, 2, 3, 3]
    n = len(stim1)

    # Paired differences d = stim1 - stim2
    diffs = [s1 - s2 for s1, s2 in zip(stim1, stim2)]

    d_bar = sum(diffs) / n
    s_d_sq = sum((d - d_bar) ** 2 for d in diffs) / (n - 1)
    s_d = math.sqrt(s_d_sq)
    se = s_d / math.sqrt(n)

    t_stat = d_bar / se
    df = n - 1

    # Two-tailed p-value
    p_value = 2 * min(t_cdf(t_stat, df), 1 - t_cdf(t_stat, df))

    # Critical value at alpha = 0.05 (two-tailed)
    alpha = 0.05
    t_crit = t_inv(1 - alpha / 2, df)

    # --- Output ---
    print("=" * 55)
    print("Reaction Times — Paired t-test")
    print("=" * 55)
    print(f"  Stimulus 1: {stim1}")
    print(f"  Stimulus 2: {stim2}")
    print(f"  Differences (d = S1 - S2): {diffs}")
    print()
    print(f"  n  = {n}")
    print(f"  d̄  = {d_bar:.4f}")
    print(f"  s_d = {s_d:.4f}")
    print(f"  SE  = s_d / √n = {se:.4f}")
    print()
    print(f"  H0: mu_d = 0")
    print(f"  H1: mu_d != 0  (two-tailed, alpha = {alpha})")
    print()
    print(f"  t = d̄ / SE = {d_bar:.4f} / {se:.4f} = {t_stat:.4f}")
    print(f"  df = {df}")
    print()
    print(f"  Two-tailed P-value = {p_value:.6f}")
    print(f"  Critical value t_{{0.025, {df}}} = ±{t_crit:.4f}")
    print()

    if p_value < alpha:
        print(f"  => p = {p_value:.4f} < alpha = {alpha}")
        print("  => Reject H0. Sufficient evidence of a difference")
        print("     in mean reaction times between the two stimuli.")
    else:
        print(f"  => p = {p_value:.4f} >= alpha = {alpha}")
        print("  => Fail to reject H0. No significant difference")
        print("     in mean reaction times.")
    print("=" * 55)

    # --- Plot ---
    outfile = Path(__file__).with_name("reaction_times_plot.png")
    plot_t_distribution(t_stat, df, p_value, t_crit, outfile)


if __name__ == "__main__":
    main()
