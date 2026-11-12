"""Run the full analysis for a portfolio config and write a markdown report with figures."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import BT_MODELS, run_backtest, score
from .data import load_prices
from .estimate import TRADING_DAYS, cov_to_corr, fit_t_dof, ledoit_wolf, log_returns, summarise
from .risk import component_es, horizon_table, max_drawdown, standalone_es, var_es, var_es_ci
from .simulate import MODELS, simulate
from .weights import SCHEMES


def load_config(path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    n = len(cfg["tickers"])
    w = np.asarray(cfg.get("weights", [1.0 / n] * n), dtype=float)
    if len(w) != n or abs(w.sum() - 1) > 1e-8:
        raise ValueError("weights must match tickers and sum to 1")
    cfg["weights"] = w
    cfg.setdefault("horizon", 21)
    cfg.setdefault("paths", 50_000)
    cfg.setdefault("alphas", [0.95, 0.99])
    cfg.setdefault("drift", "historical")
    cfg.setdefault("block", 10)
    cfg.setdefault("seed", 0)
    cfg.setdefault("backtest", {})
    return cfg


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def crisis_window(x: np.ndarray, w: np.ndarray, length: int = 63) -> tuple[int, int]:
    """The stretch of `length` days where the portfolio was most volatile."""
    port = pd.Series(x @ w)
    vol = port.rolling(length).std()
    end = int(vol.idxmax()) + 1
    return end - length, end


def run(config_path, out_dir="results", refresh=False, log=print) -> dict:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tickers, w = cfg["tickers"], cfg["weights"]
    H, N, alphas, seed = cfg["horizon"], cfg["paths"], cfg["alphas"], cfg["seed"]
    a_hi = max(alphas)

    prices = load_prices(tickers, cfg["start"], cfg["end"], refresh=refresh)
    rets = log_returns(prices)
    x = rets.to_numpy()
    log(f"{len(rets)} daily returns, {rets.index[0].date()} to {rets.index[-1].date()}")

    stats = summarise(rets)
    cov, shrink = ledoit_wolf(x)
    corr = cov_to_corr(cov)
    mu = x.mean(axis=0) if cfg["drift"] == "historical" else np.zeros(len(w))
    nu = fit_t_dof(x, cov)
    port_vol = float(np.sqrt(w @ cov @ w * TRADING_DAYS))
    log(f"portfolio vol {pct(port_vol)} a year, t degrees of freedom {nu:.2f}")

    def sim(model, mu_=mu, cov_=cov, w_=w, hist_=x, s=seed):
        return simulate(mu_, cov_, w_, H, N, model=model, nu=nu, hist=hist_,
                        block=cfg["block"], seed=s)

    sims = {m: sim(m) for m in MODELS}
    main = sims["t"]

    md = [f"# Portfolio risk report", ""]
    md += [
        f"{len(tickers)} stocks, {rets.index[0].date()} to {rets.index[-1].date()} "
        f"({len(rets)} trading days). {N:,} simulated paths per model, {H}-day horizon, "
        f"buy and hold, drift = {cfg['drift']}, seed {seed}. Losses are fractions of the "
        f"starting portfolio value.",
        "",
        "## Inputs estimated from history",
        "",
    ]
    md.append(
        table(
            ["stock", "weight", "return / yr", "std error", "vol / yr", "skew", "excess kurtosis", "worst day"],
            [
                [t, pct(wi, 1), pct(r.ann_return, 1), pct(r.return_se, 1), pct(r.ann_vol, 1),
                 f"{r.skew:.2f}", f"{r.excess_kurtosis:.1f}", pct(np.expm1(r.worst_day), 1)]
                for t, wi, r in zip(tickers, w, stats.itertuples())
            ],
        )
    )
    off = corr[np.triu_indices(len(w), 1)]
    md += [
        "",
        f"Portfolio volatility {pct(port_vol, 1)} a year. Pairwise correlations run from "
        f"{off.min():.2f} to {off.max():.2f}, mean {off.mean():.2f}. Ledoit-Wolf shrinkage "
        f"weight {shrink:.4f}. Fitted multivariate t degrees of freedom: {nu:.2f}.",
        "",
        "Correlation matrix:",
        "",
        table(
            [""] + [t.replace(".NS", "") for t in tickers],
            [[t.replace(".NS", "")] + [f"{c:.2f}" for c in row] for t, row in zip(tickers, corr)],
        ),
        "",
        f"## {H}-day VaR and expected shortfall",
        "",
    ]

    rows, summary = [], {}
    for m, s in sims.items():
        for a in alphas:
            v, e = var_es(s.losses, a)
            (vl, vh), (el, eh) = var_es_ci(s.losses, a, seed=seed)
            summary[(m, a)] = (v, e)
            rows.append([m, f"{a:.0%}", pct(v), f"{pct(vl)} to {pct(vh)}", pct(e), f"{pct(el)} to {pct(eh)}"])
    md.append(table(["model", "level", "VaR", "VaR 95% CI", "ES", "ES 95% CI"], rows))
    md += ["", "The intervals are bootstrap intervals for Monte Carlo error only. "
           "They say nothing about error in the estimated inputs.", ""]

    plots.pnl_distribution(sims, a_hi, H, out / "pnl_distribution.png")
    plots.fan(main.values, out / "fan.png", f"t model, {H} days")
    plots.convergence(main.losses, a_hi, out / "convergence.png", seed=seed)
    md += ["![return distribution](pnl_distribution.png)", "", "![fan chart](fan.png)", "",
           "![convergence](convergence.png)", ""]

    # other facts about the distribution, t model
    dd = max_drawdown(main.values)
    md += ["## Shape of the loss distribution (t model)", ""]
    md.append(
        table(
            ["measure", "value"],
            [
                ["probability of any loss at the horizon", pct(np.mean(main.pnl < 0), 1)],
                ["probability of losing more than 5%", pct(np.mean(main.pnl < -0.05), 1)],
                ["probability of losing more than 10%", pct(np.mean(main.pnl < -0.10), 2)],
                ["median return", pct(np.median(main.pnl))],
                ["worst simulated path", pct(main.pnl.min(), 1)],
                ["median max drawdown inside the horizon", pct(np.median(dd))],
                ["95th percentile max drawdown", pct(np.quantile(dd, 0.95))],
                ["99th percentile max drawdown", pct(np.quantile(dd, 0.99))],
            ],
        )
    )

    days = sorted({d for d in (1, 5, 10, H) if d <= H})
    ht = horizon_table(main.values, days, alphas)
    v1 = ht[days[0]][a_hi][0]
    md += ["", "## Horizon scaling (t model)", ""]
    md.append(
        table(
            ["days", f"{a_hi:.0%} VaR", f"{a_hi:.0%} ES", "1-day VaR x sqrt(days)"],
            [[d, pct(ht[d][a_hi][0]), pct(ht[d][a_hi][1]), pct(v1 * np.sqrt(d))] for d in days],
        )
    )

    # risk contributions
    comp = component_es(main.asset_pnl, a_hi)
    alone = standalone_es(main.asset_pnl, a_hi)
    plots.components(tickers, w, comp, out / "components.png", a_hi)
    md += ["", f"## Where the {a_hi:.0%} ES comes from (t model)", ""]
    md.append(
        table(
            ["stock", "weight", "ES contribution", "share of ES", "stand-alone ES"],
            [[t, pct(wi, 1), pct(c), pct(c / comp.sum(), 1), pct(s)]
             for t, wi, c, s in zip(tickers, w, comp, alone)],
        )
    )
    md += [
        "",
        f"Contributions add up to the portfolio ES of {pct(comp.sum())}. The stand-alone "
        f"figures add up to {pct(alone.sum())}, so diversification removes "
        f"{pct(1 - comp.sum() / alone.sum(), 1)} of the tail loss.",
        "",
        "![components](components.png)",
        "",
    ]

    # stress scenarios
    sd = np.sqrt(np.diag(cov))
    c_hi = 0.5 * corr + 0.5
    np.fill_diagonal(c_hi, 1.0)
    lo_i, hi_i = crisis_window(x, w)
    cov_crisis, _ = ledoit_wolf(x[lo_i:hi_i])
    scen = {
        "baseline": sim("t"),
        "volatility x 1.5": sim("t", cov_=cov * 2.25),
        "correlations halfway to 1": sim("t", cov_=c_hi * np.outer(sd, sd)),
        f"covariance of {rets.index[lo_i].date()} to {rets.index[hi_i - 1].date()}": sim("t", cov_=cov_crisis),
        "zero drift": sim("t", mu_=np.zeros(len(w))),
    }
    md += ["## Stress scenarios (t model)", ""]
    base_es = var_es(scen["baseline"].losses, a_hi)[1]
    rows = []
    for name, s in scen.items():
        v, e = var_es(s.losses, a_hi)
        rows.append([name, pct(v), pct(e), f"{e / base_es:.2f}x"])
    md.append(table(["scenario", f"{a_hi:.0%} VaR", f"{a_hi:.0%} ES", "ES vs baseline"], rows))
    md += ["", "The covariance scenario uses the 63 days in the sample where the portfolio was most volatile.", ""]

    # weighting schemes
    md += ["## Other ways to weight the same stocks (t model)", ""]
    rows = []
    for name, fn in {"configured": lambda c: w, **SCHEMES}.items():
        ww = fn(cov)
        if name != "configured" and np.allclose(ww, w):
            continue
        s = sim("t", w_=ww)
        v, e = var_es(s.losses, a_hi)
        rows.append([name, pct(np.sqrt(ww @ cov @ ww * TRADING_DAYS), 1), pct(v), pct(e),
                     pct(ww.max(), 1), pct(ww.min(), 1)])
    md.append(table(["scheme", "vol / yr", f"{a_hi:.0%} VaR", f"{a_hi:.0%} ES", "largest weight", "smallest weight"], rows))

    # backtest
    bcfg = cfg["backtest"]
    bt = run_backtest(x, w, alphas=tuple(alphas), window=bcfg.get("window", 500),
                      lam=bcfg.get("lam", 0.94), n_paths=bcfg.get("paths", 20_000),
                      seed=seed, index=rets.index)
    scores = score(bt, alphas)
    plots.backtest(bt, ["hist", "ewma_t"], a_hi, out / "backtest.png")
    md += ["", "## Backtest of one-day forecasts", "",
           f"Walk forward over {len(bt.losses)} days ({bt.index[0].date()} to {bt.index[-1].date()}). "
           f"Each forecast uses only earlier data. A breach is a day whose loss exceeded the VaR forecast. "
           f"Kupiec tests the breach count, Christoffersen tests whether breaches cluster; "
           f"p below 0.05 rejects the model. Z2 is the Acerbi-Szekely ES statistic: about 0 when "
           f"ES is right, negative when ES is too small.", ""]
    md.append(
        table(
            ["model", "level", "breaches", "expected", "Kupiec p", "Christoffersen p", "ES Z2"],
            [[r["model"], f"{r['alpha']:.0%}", r["breaches"], f"{r['expected']:.1f}",
              f"{r['kupiec_p']:.3f}", f"{r['indep_p']:.3f}", f"{r['z2']:+.2f}"] for r in scores],
        )
    )
    md += ["", "![backtest](backtest.png)", ""]

    (out / "report.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out / 'report.md'}")
    return {"summary": summary, "scores": scores, "nu": nu, "port_vol": port_vol}
