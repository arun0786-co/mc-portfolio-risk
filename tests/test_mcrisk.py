import numpy as np
import pytest
from scipy.stats import norm

from mcrisk.backtest import christoffersen, es_z2, kupiec, run_backtest, score
from mcrisk.estimate import ewma_cov, fit_t_dof, ledoit_wolf
from mcrisk.risk import component_es, max_drawdown, var_es
from mcrisk.simulate import draw_returns, simulate
from mcrisk.weights import risk_parity

COV = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, 0.03], [0.0, 0.03, 0.0625]]) / 252
W = np.array([0.5, 0.3, 0.2])


def test_var_es_match_normal_closed_form():
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(2_000_000)
    var, es = var_es(losses, 0.99)
    assert var == pytest.approx(norm.ppf(0.99), abs=0.01)
    assert es == pytest.approx(norm.pdf(norm.ppf(0.99)) / 0.01, abs=0.015)


def test_single_asset_var_matches_lognormal():
    sigma, h = 0.02, 10
    s = simulate([0.0], [[sigma**2]], [1.0], h, 400_000, seed=1)
    exact = 1 - np.exp(norm.ppf(0.01) * sigma * np.sqrt(h))
    assert var_es(s.losses, 0.99)[0] == pytest.approx(exact, rel=0.01)


@pytest.mark.parametrize("model", ["normal", "t"])
def test_simulated_covariance_matches_input(model):
    rng = np.random.default_rng(2)
    r = draw_returns(rng, 400_000, 1, np.zeros(3), COV, model, nu=6.0)[:, 0, :]
    assert np.allclose(np.cov(r, rowvar=False), COV, rtol=0.06, atol=2e-6)


def test_t_has_fatter_tails_than_normal_at_same_covariance():
    n = simulate(np.zeros(3), COV, W, 1, 300_000, "normal", seed=3)
    t = simulate(np.zeros(3), COV, W, 1, 300_000, "t", nu=4.0, seed=3)
    assert var_es(t.losses, 0.999)[1] > 1.3 * var_es(n.losses, 0.999)[1]


def test_bootstrap_keeps_rows_and_blocks_together():
    hist = np.arange(300, dtype=float).reshape(100, 3)
    rng = np.random.default_rng(4)
    r = draw_returns(rng, 2000, 30, hist.mean(axis=0), None, "bootstrap", hist=hist, block=10)
    row = r[..., 0] / 3
    assert np.allclose(r[..., 1] - r[..., 0], 1) and np.allclose(row, np.round(row))
    step = np.diff(row, axis=1) % 100
    assert np.mean(step == 1) == pytest.approx(0.9, abs=0.02)


def test_same_seed_same_paths():
    a = simulate(np.zeros(3), COV, W, 5, 1000, "t", nu=5.0, seed=9, chunk=300)
    b = simulate(np.zeros(3), COV, W, 5, 1000, "t", nu=5.0, seed=9, chunk=300)
    assert np.array_equal(a.values, b.values)


def test_component_es_adds_up_to_portfolio_es():
    s = simulate(np.zeros(3), COV, W, 21, 50_000, "t", nu=5.0, seed=5)
    comp = component_es(s.asset_pnl, 0.99)
    assert comp.sum() == pytest.approx(var_es(s.losses, 0.99)[1], rel=1e-12)
    assert np.allclose(s.asset_pnl.sum(axis=1), s.pnl)


def test_max_drawdown():
    v = np.array([[1.0, 1.2, 0.9, 1.5, 1.2], [1.0, 1.1, 1.2, 1.3, 1.4]])
    assert np.allclose(max_drawdown(v), [0.25, 0.0])


def test_ledoit_wolf_shrinks_small_samples_more():
    rng = np.random.default_rng(6)
    chol = np.linalg.cholesky(COV)
    small = rng.standard_normal((20, 3)) @ chol.T
    large = rng.standard_normal((20_000, 3)) @ chol.T
    s_small, w_small = ledoit_wolf(small)
    s_large, w_large = ledoit_wolf(large)
    assert 0 <= w_large < w_small <= 1
    assert np.allclose(s_large, COV, rtol=0.1, atol=2e-6)
    assert np.all(np.linalg.eigvalsh(s_small) > 0)


def test_ewma_recursion():
    x = np.array([[1.0, 0.0], [0.0, 2.0]])
    s = ewma_cov(x, lam=0.5, init=np.zeros((2, 2)))
    assert np.allclose(s, [[0.25, 0.0], [0.0, 2.0]])


def test_fit_t_dof_recovers_truth():
    rng = np.random.default_rng(7)
    nu = 5.0
    r = draw_returns(rng, 60_000, 1, np.zeros(3), COV, "t", nu=nu)[:, 0, :]
    assert fit_t_dof(r, COV) == pytest.approx(nu, abs=0.3)


def test_risk_parity_equalises_contributions():
    w = risk_parity(COV)
    rc = w * (COV @ w)
    assert w.sum() == pytest.approx(1) and np.allclose(rc, rc.mean(), rtol=1e-6)


def test_kupiec_known_value():
    # 250 days, 99% VaR, 7 breaches: LR = 5.497 (textbook traffic-light example)
    b = np.zeros(250, bool)
    b[:7] = True
    lr, p = kupiec(b, 0.99)
    assert lr == pytest.approx(5.497, abs=0.005) and p < 0.05
    b = np.zeros(1000, bool)
    b[::100] = True
    assert kupiec(b, 0.99)[0] == pytest.approx(0, abs=1e-9)
    assert kupiec(np.zeros(250, bool), 0.99)[0] == pytest.approx(-2 * 250 * np.log(0.99))


def test_christoffersen_flags_clustered_breaches():
    spread = np.zeros(1000, bool)
    spread[::50] = True
    clustered = np.zeros(1000, bool)
    clustered[100:120] = True
    assert christoffersen(spread)[1] > 0.05
    assert christoffersen(clustered)[1] < 1e-6


def test_z2_sign():
    rng = np.random.default_rng(8)
    losses = rng.standard_normal(200_000)
    var = np.full_like(losses, norm.ppf(0.975))
    es = np.full_like(losses, norm.pdf(norm.ppf(0.975)) / 0.025)
    assert es_z2(losses, var, es, 0.975) == pytest.approx(0, abs=0.03)
    assert es_z2(losses, var, es / 2, 0.975) < -0.9


def test_backtest_is_calibrated_on_normal_data_and_has_no_lookahead():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((1500, 3)) @ np.linalg.cholesky(COV).T
    bt = run_backtest(x, W, window=250, n_paths=5000, seed=0)
    rows = {(r["model"], r["alpha"]): r for r in score(bt, (0.95, 0.99))}
    assert rows[("normal", 0.95)]["kupiec_p"] > 0.01
    assert abs(rows[("normal", 0.95)]["z2"]) < 0.25
    # changing the future must not change earlier forecasts
    y = x.copy()
    y[1000:] *= 5
    bt2 = run_backtest(y, W, window=250, n_paths=5000, seed=0)
    for m in bt.var:
        assert np.array_equal(bt.var[m][0.99][:751], bt2.var[m][0.99][:751])
    assert not np.array_equal(bt.var["ewma"][0.99][751:], bt2.var["ewma"][0.99][751:])
