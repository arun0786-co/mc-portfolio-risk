"""Estimate return, volatility and correlation inputs from daily prices."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="any")


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity.

    Returns the shrunk covariance and the shrinkage weight in [0, 1].
    """
    x = x - x.mean(axis=0)
    n, p = x.shape
    s = x.T @ x / n
    m = np.trace(s) / p
    d2 = np.sum((s - m * np.eye(p)) ** 2) / p
    # sum_k ||x_k x_k' - S||_F^2, expanded so no (n, p, p) array is needed
    sq = np.sum(x**2, axis=1)
    b2 = (np.sum(sq**2) - n * np.sum(s**2)) / (n**2 * p)
    b2 = min(max(b2, 0.0), d2)
    w = b2 / d2 if d2 > 0 else 0.0
    return w * m * np.eye(p) + (1 - w) * s, w


def ewma_cov(x: np.ndarray, lam: float = 0.94, init: np.ndarray | None = None) -> np.ndarray:
    """RiskMetrics exponentially weighted covariance after the last row of x."""
    s = np.cov(x, rowvar=False) if init is None else init.copy()
    for r in x:
        s = lam * s + (1 - lam) * np.outer(r, r)
    return s


def t_loglik(nu: float, d: np.ndarray, logdet: float, p: int) -> float:
    """Log likelihood of a multivariate t whose covariance is fixed.

    d holds squared Mahalanobis distances under the covariance, logdet its log determinant.
    The t scatter matrix is cov * (nu - 2) / nu, so the covariance does not move with nu.
    """
    n = len(d)
    k = (nu - 2) / nu
    return (
        n * (gammaln((nu + p) / 2) - gammaln(nu / 2) - 0.5 * p * np.log(nu * np.pi))
        - 0.5 * n * (logdet + p * np.log(k))
        - 0.5 * (nu + p) * np.sum(np.log1p(d / (k * nu)))
    )


def fit_t_dof(x: np.ndarray, cov: np.ndarray, lo: float = 2.1, hi: float = 100.0) -> float:
    """Maximum likelihood degrees of freedom of a multivariate t, covariance held fixed."""
    x = x - x.mean(axis=0)
    p = x.shape[1]
    d = np.einsum("ij,ij->i", x, np.linalg.solve(cov, x.T).T)
    logdet = np.linalg.slogdet(cov)[1]
    res = minimize_scalar(lambda nu: -t_loglik(nu, d, logdet, p), bounds=(lo, hi), method="bounded")
    return float(res.x)


def summarise(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised return and volatility per asset, with the standard error of the return."""
    mu = returns.mean() * TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    years = len(returns) / TRADING_DAYS
    return pd.DataFrame(
        {
            "ann_return": mu,
            "return_se": vol / np.sqrt(years),
            "ann_vol": vol,
            "skew": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "worst_day": returns.min(),
        }
    )


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)
