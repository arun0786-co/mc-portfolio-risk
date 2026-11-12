"""Figures for the report. White background, dark text, one accent colour."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#b3261e"
DARK = "#222222"
GREY = "#8a8a8a"
LIGHT = "#d0d0d0"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def pnl_distribution(sims: dict, alpha: float, horizon: int, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    lo = min(np.quantile(s.pnl, 0.0005) for s in sims.values())
    hi = max(np.quantile(s.pnl, 0.9995) for s in sims.values())
    bins = np.linspace(lo, hi, 140)
    styles = {"normal": (GREY, "-"), "bootstrap": (DARK, "-"), "t": (ACCENT, "-")}
    for name, s in sims.items():
        c, ls = styles[name]
        ax.hist(s.pnl * 100, bins=bins * 100, histtype="step", color=c, ls=ls, lw=1.2,
                density=True, label=name)
        ax.axvline(-np.quantile(s.losses, alpha) * 100, color=c, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(f"{horizon}-day portfolio return (%)")
    ax.set_ylabel("density (log scale)")
    ax.set_title(f"simulated {horizon}-day return, dashed lines mark the {alpha:.0%} VaR", loc="left")
    ax.legend(frameon=False)
    _save(fig, path)


def fan(values: np.ndarray, path, label: str):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    t = np.arange(values.shape[1])
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], axis=0) * 100
    ax.fill_between(t, q[0], q[6], color=LIGHT, lw=0, label="1% to 99%")
    ax.fill_between(t, q[1], q[5], color="#b0b0b0", lw=0, label="5% to 95%")
    ax.fill_between(t, q[2], q[4], color=GREY, lw=0, label="25% to 75%")
    ax.plot(t, q[3], color=ACCENT, lw=1.4, label="median")
    ax.set_xlabel("trading days ahead")
    ax.set_ylabel("portfolio value (start = 100)")
    ax.set_title(f"range of simulated portfolio paths ({label})", loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    _save(fig, path)


def components(names, weights, comp, path, alpha):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(names))
    share = comp / comp.sum() * 100
    ax.bar(x - 0.2, np.asarray(weights) * 100, 0.4, color=LIGHT, label="weight")
    ax.bar(x + 0.2, share, 0.4, color=ACCENT, label=f"share of {alpha:.0%} ES")
    ax.set_xticks(x, [n.replace(".NS", "") for n in names], rotation=30, ha="right")
    ax.set_ylabel("% of portfolio")
    ax.set_title("where the tail loss comes from", loc="left")
    ax.set_ylim(0, max(share.max(), np.max(weights) * 100) * 1.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _save(fig, path)


def convergence(losses, alpha, path, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(losses)
    ns = np.unique(np.geomspace(500, len(perm), 25).astype(int))
    reps = 200
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for which, c, lab in ((0, DARK, "VaR"), (1, ACCENT, "ES")):
        mid, lo, hi = [], [], []
        for n in ns:
            est = np.empty(reps)
            for r in range(reps):
                sub = perm[rng.integers(0, len(perm), n)]
                v = np.quantile(sub, alpha)
                est[r] = v if which == 0 else sub[sub >= v].mean()
            mid.append(np.median(est))
            lo.append(np.quantile(est, 0.025))
            hi.append(np.quantile(est, 0.975))
        ax.fill_between(ns, np.array(lo) * 100, np.array(hi) * 100, color=c, alpha=0.15, lw=0)
        ax.plot(ns, np.array(mid) * 100, color=c, lw=1.2, label=f"{alpha:.0%} {lab}")
    ax.axvline(20_000, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("number of simulated paths")
    ax.set_ylabel("loss (%)")
    ax.set_title("Monte Carlo error against path count (95% band)", loc="left")
    ax.legend(frameon=False)
    _save(fig, path)


def backtest(bt, models, alpha, path):
    fig, axes = plt.subplots(len(models), 1, figsize=(7, 2.3 * len(models)), sharex=True)
    x = bt.index if bt.index is not None else np.arange(len(bt.losses))
    for ax, m in zip(np.atleast_1d(axes), models):
        v = bt.var[m][alpha]
        hit = bt.losses > v
        ax.plot(x, bt.losses * 100, color=LIGHT, lw=0.6)
        ax.plot(x, v * 100, color=DARK, lw=0.9)
        ax.scatter(np.asarray(x)[hit], bt.losses[hit] * 100, color=ACCENT, s=9, zorder=3)
        ax.set_ylabel("daily loss (%)")
        ax.set_title(f"{m}: {alpha:.0%} one-day VaR, {int(hit.sum())} breaches "
                     f"({len(hit) * (1 - alpha):.0f} expected)", loc="left", fontsize=10)
    _save(fig, path)
