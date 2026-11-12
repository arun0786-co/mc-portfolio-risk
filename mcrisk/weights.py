"""Simple long-only weighting schemes to compare against the configured portfolio."""

import numpy as np


def equal(cov: np.ndarray) -> np.ndarray:
    p = len(cov)
    return np.full(p, 1.0 / p)


def inverse_vol(cov: np.ndarray) -> np.ndarray:
    w = 1.0 / np.sqrt(np.diag(cov))
    return w / w.sum()


def risk_parity(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """Equal risk contribution weights: every w_i * (cov w)_i is the same."""
    w = inverse_vol(cov)
    for _ in range(max_iter):
        new = w / np.sqrt(w * (cov @ w))
        new /= new.sum()
        # damped update, the plain fixed point can oscillate
        new = 0.5 * (w + new)
        if np.abs(new - w).max() < tol:
            return new
        w = new
    return w


SCHEMES = {"equal": equal, "inverse_vol": inverse_vol, "risk_parity": risk_parity}
