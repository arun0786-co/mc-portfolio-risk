"""Simulate joint paths of daily log returns for a buy-and-hold portfolio."""

from dataclasses import dataclass

import numpy as np

MODELS = ("normal", "t", "bootstrap")


@dataclass
class SimResult:
    values: np.ndarray  # (paths, horizon + 1) portfolio value, starts at 1
    asset_pnl: np.ndarray  # (paths, assets) P&L of each holding at the horizon

    @property
    def pnl(self) -> np.ndarray:
        return self.values[:, -1] - 1.0

    @property
    def losses(self) -> np.ndarray:
        return -self.pnl


def _bootstrap_index(rng, n_obs: int, m: int, horizon: int, block: float) -> np.ndarray:
    """Stationary bootstrap (Politis and Romano 1994): blocks of geometric length."""
    starts = rng.integers(0, n_obs, size=(m, horizon))
    new = rng.random((m, horizon)) < 1.0 / block
    idx = np.empty((m, horizon), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, horizon):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def draw_returns(rng, m, horizon, mu, cov, model, nu=None, hist=None, block=10.0):
    """Daily log returns of shape (m, horizon, assets)."""
    p = len(mu)
    if model == "bootstrap":
        if hist is None:
            raise ValueError("bootstrap needs historical returns")
        centred = hist - hist.mean(axis=0) + mu
        return centred[_bootstrap_index(rng, len(hist), m, horizon, block)]
    chol = np.linalg.cholesky(cov)
    shocks = rng.standard_normal((m, horizon, p)) @ chol.T
    if model == "normal":
        return mu + shocks
    if model == "t":
        if nu is None or nu <= 2:
            raise ValueError("t model needs nu > 2")
        # one chi-square per day shared by all assets: joint fat tails and tail dependence,
        # scaled so the covariance still equals cov
        g = rng.chisquare(nu, size=(m, horizon, 1))
        return mu + np.sqrt((nu - 2) / g) * shocks
    raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")


def simulate(
    mu,
    cov,
    weights,
    horizon: int,
    n_paths: int,
    model: str = "normal",
    nu: float | None = None,
    hist: np.ndarray | None = None,
    block: float = 10.0,
    seed: int = 0,
    chunk: int = 5000,
) -> SimResult:
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.ones((n_paths, horizon + 1))
    asset_pnl = np.empty((n_paths, len(w)))
    for lo in range(0, n_paths, chunk):
        m = min(chunk, n_paths - lo)
        r = draw_returns(rng, m, horizon, mu, cov, model, nu=nu, hist=hist, block=block)
        growth = np.exp(np.cumsum(r, axis=1))  # (m, horizon, assets)
        values[lo : lo + m, 1:] = growth @ w
        asset_pnl[lo : lo + m] = w * (growth[:, -1, :] - 1.0)
    return SimResult(values, asset_pnl)
