"""Walk-forward backtest of one-day VaR and ES forecasts against realised losses.

Every forecast for day t uses returns up to day t - 1 only.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2

from .estimate import fit_t_dof, ledoit_wolf
from .risk import var_es

BT_MODELS = ("hist", "normal", "t", "ewma", "ewma_t")


@dataclass
class Backtest:
    losses: np.ndarray  # (days,) realised portfolio loss
    var: dict  # model -> alpha -> (days,)
    es: dict
    index: object = None


def _forecast(shocks, scale, chol, w, alphas):
    r = (scale * shocks) @ chol.T
    losses = -((np.exp(r) - 1.0) @ w)
    return [var_es(losses, a) for a in alphas]


def run_backtest(returns, weights, alphas=(0.95, 0.99), window=500, lam=0.94,
                 n_paths=20_000, refit=21, seed=0, index=None) -> Backtest:
    x = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    n, p = x.shape
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, p))  # common random numbers across days
    u = rng.random(n_paths)
    port = -((np.exp(x) - 1.0) @ w)  # realised daily loss, weights reset each day

    days = n - window
    var = {m: {a: np.empty(days) for a in alphas} for m in BT_MODELS}
    es = {m: {a: np.empty(days) for a in alphas} for m in BT_MODELS}

    s_ewma = np.cov(x[:window], rowvar=False)
    white = []  # returns whitened by the EWMA forecast that was live on their day
    nu = nu_e = 8.0
    scale_t = scale_e = None

    for k, t in enumerate(range(window, n)):
        win = x[t - window : t]
        cov, _ = ledoit_wolf(win)
        chol = np.linalg.cholesky(cov)
        chol_e = np.linalg.cholesky(s_ewma)

        if k % refit == 0:
            nu = fit_t_dof(win, cov)
            scale_t = np.sqrt((nu - 2) / chi2.ppf(u, nu))[:, None]
            if len(white) >= 250:
                nu_e = fit_t_dof(np.array(white[-window:]), np.eye(p))
            else:
                nu_e = nu
            scale_e = np.sqrt((nu_e - 2) / chi2.ppf(u, nu_e))[:, None]

        hist_losses = port[t - window : t]
        forecasts = {
            "hist": [var_es(hist_losses, a) for a in alphas],
            "normal": _forecast(z, 1.0, chol, w, alphas),
            "t": _forecast(z, scale_t, chol, w, alphas),
            "ewma": _forecast(z, 1.0, chol_e, w, alphas),
            "ewma_t": _forecast(z, scale_e, chol_e, w, alphas),
        }
        for m, f in forecasts.items():
            for a, (v, e) in zip(alphas, f):
                var[m][a][k] = v
                es[m][a][k] = e

        white.append(np.linalg.solve(chol_e, x[t]))
        s_ewma = lam * s_ewma + (1 - lam) * np.outer(x[t], x[t])

    idx = None if index is None else index[window:]
    return Backtest(port[window:], var, es, idx)


def kupiec(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage test. Returns (LR statistic, p-value)."""
    n, x = len(breaches), int(breaches.sum())
    p = 1 - alpha
    phat = x / n
    lr = -2 * (xlogy(n - x, 1 - p) + xlogy(x, p) - xlogy(n - x, 1 - phat) - xlogy(x, phat))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen(breaches: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test: do breaches cluster?"""
    b = breaches.astype(int)
    prev, cur = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (cur == 0))
    n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0))
    n11 = np.sum((prev == 1) & (cur == 1))
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2 * (null - alt), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def es_z2(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Acerbi and Szekely (2014) Z2 statistic. Zero when ES is right, negative when too low."""
    hit = losses > var
    return float(1.0 - np.sum(losses[hit] / es[hit]) / (len(losses) * (1 - alpha)))


def score(bt: Backtest, alphas) -> list[dict]:
    rows = []
    for m in BT_MODELS:
        for a in alphas:
            hit = bt.losses > bt.var[m][a]
            rows.append(
                {
                    "model": m,
                    "alpha": a,
                    "days": len(hit),
                    "breaches": int(hit.sum()),
                    "expected": len(hit) * (1 - a),
                    "kupiec_p": kupiec(hit, a)[1],
                    "indep_p": christoffersen(hit)[1],
                    "z2": es_z2(bt.losses, bt.var[m][a], bt.es[m][a], a),
                }
            )
    return rows
