"""Data loading for the Nifty forecasting project.

Prices come from Yahoo Finance via `yfinance`. The default ticker is the
Nifty 50 index (^NSEI). Any Yahoo ticker works, e.g. "RELIANCE.NS".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKER = "^NSEI"  # Nifty 50 index


def load_prices(
    ticker: str = DEFAULT_TICKER,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = "data",
) -> pd.Series:
    """Download daily closing prices and return them as a clean Series.

    A CSV copy is cached in `cache_dir` so the app still works offline
    after the first successful download.
    """
    cache_path = None
    if cache_dir is not None:
        safe = ticker.replace("^", "").replace(".", "_")
        cache_path = Path(cache_dir) / f"{safe}.csv"

    try:
        import warnings

        import yfinance as yf

        with warnings.catch_warnings():  # silence pandas deprecation noise from inside yfinance
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker!r}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # newer yfinance returns MultiIndex columns
            close = close.iloc[:, 0]
        series = clean_series(close)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            series.to_frame("close").to_csv(cache_path)
        return series
    except Exception as err:  # network down, ticker wrong, etc.
        if cache_path is not None and cache_path.exists():
            return load_csv(cache_path)
        raise RuntimeError(f"Could not load prices for {ticker!r}: {err}") from err


def load_csv(path: str | Path) -> pd.Series:
    """Load a CSV with a date column and a `close` (or `Close`) column."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = "close" if "close" in df.columns else "Close"
    return clean_series(df[col])


def clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated()].sort_index()
    s.name = "close"
    return s.astype(float)


def synthetic_prices(n: int = 1500, seed: int = 0, start_price: float = 10000.0) -> pd.Series:
    """Random-walk prices for tests and offline demos. NOT real market data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.01, n)
    prices = start_price * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(prices, index=idx, name="close")


def train_test_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    """Chronological split — never shuffle time series."""
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be between 1 and len(series) - 1")
    return series.iloc[:-test_size], series.iloc[-test_size:]
