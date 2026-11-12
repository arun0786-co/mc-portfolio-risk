"""Download adjusted daily closes and cache them on disk."""

from pathlib import Path

import pandas as pd


def load_prices(tickers, start, end, cache_dir="data", refresh=False) -> pd.DataFrame:
    cache = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(df.columns):
            return df[list(tickers)].dropna(how="any")

    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    df = raw["Close"][list(tickers)]
    missing = [t for t in tickers if df[t].notna().sum() == 0]
    if missing:
        raise ValueError(f"no prices returned for {missing}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df.dropna(how="any")
