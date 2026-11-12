"""Risk measures computed from simulated losses. Losses are positive numbers."""

import numpy as np


def var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Value at risk and expected shortfall at confidence alpha (for example 0.99)."""
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def var_es_ci(losses, alpha, n_boot=200, level=0.95, seed=0):
    """Bootstrap confidence intervals for the Monte Carlo error in VaR and ES."""
    rng = np.random.default_rng(seed)
    n = len(losses)
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        out[b] = var_es(losses[rng.integers(0, n, n)], alpha)
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return np.quantile(out[:, 0], q), np.quantile(out[:, 1], q)


def component_es(asset_pnl: np.ndarray, alpha: float) -> np.ndarray:
    """Euler allocation of ES: mean loss of each holding over the portfolio's tail paths.

    The components add up to the portfolio ES exactly.
    """
    losses = -asset_pnl.sum(axis=1)
    var = np.quantile(losses, alpha)
    return -asset_pnl[losses >= var].mean(axis=0)


def standalone_es(asset_pnl: np.ndarray, alpha: float) -> np.ndarray:
    """ES of each holding on its own, ignoring diversification."""
    return np.array([var_es(-asset_pnl[:, i], alpha)[1] for i in range(asset_pnl.shape[1])])


def max_drawdown(values: np.ndarray) -> np.ndarray:
    """Largest peak-to-trough fall along each path, as a positive fraction."""
    peak = np.maximum.accumulate(values, axis=1)
    return (1.0 - values / peak).max(axis=1)


def horizon_table(values: np.ndarray, days, alphas) -> dict:
    """VaR and ES at several horizons read off the same simulated paths."""
    out = {}
    for d in days:
        losses = 1.0 - values[:, d]
        out[d] = {a: var_es(losses, a) for a in alphas}
    return out
