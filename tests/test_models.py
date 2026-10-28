import numpy as np
import pandas as pd
import pytest

from src import models
from src.data import clean_series, synthetic_prices, train_test_split
from src.evaluate import metrics, run_backtest


@pytest.fixture(scope="module")
def series():
    return synthetic_prices(800, seed=1)


def test_split_is_chronological(series):
    train, test = train_test_split(series, 100)
    assert len(test) == 100
    assert train.index.max() < test.index.min()


def test_naive_uses_previous_close(series):
    train, test = train_test_split(series, 50)
    pred = models.naive_forecast(train, test)
    assert pred.iloc[0] == train.iloc[-1]
    assert np.allclose(pred.iloc[1:].values, test.iloc[:-1].values)


def test_arima_has_no_lookahead(series):
    """Changing a future price must not change earlier predictions."""
    train, test = train_test_split(series, 60)
    p1, _ = models.arima_forecast(train, test, order=(1, 1, 0))
    test2 = test.copy()
    test2.iloc[30:] *= 1.5
    p2, _ = models.arima_forecast(train, test2, order=(1, 1, 0))
    assert np.allclose(p1.iloc[:31].values, p2.iloc[:31].values)


def test_arima_close_to_naive_on_random_walk(series):
    train, test = train_test_split(series, 100)
    pred, _ = models.arima_forecast(train, test)
    naive = models.naive_forecast(train, test)
    assert np.mean(np.abs(pred - naive) / naive) < 0.01


def test_future_band_widens(series):
    fut = models.arima_future(series, 10, (0, 1, 0))
    width = fut["upper"] - fut["lower"]
    assert width.is_monotonic_increasing
    assert (fut["lower"] < fut["forecast"]).all() and (fut["forecast"] < fut["upper"]).all()


def test_metrics_perfect_prediction():
    a = pd.Series([1.0, 2.0, 3.0])
    m = metrics(a, a, pd.Series([0.5, 1.5, 2.5]))
    assert m["RMSE"] == 0 and m["Direction acc %"] == 100


def test_clean_series_drops_bad_rows():
    s = pd.Series([1, None, -3, 4], index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-04"]))
    out = clean_series(s)
    assert list(out.values) == [1.0, 4.0] and out.index.is_monotonic_increasing


def test_backtest_runs(series):
    res = run_backtest(series, test_size=80, use_lstm=False)
    assert res["metrics"].iloc[0]["RMSE vs naive"] == 1.0
    assert len(res["predictions"]) == 80


@pytest.mark.skipif(not models.torch_available(), reason="torch not installed")
def test_lstm_runs_without_lookahead(series):
    train, test = train_test_split(series, 40)
    p1 = models.lstm_forecast(train, test, epochs=3)
    test2 = test.copy()
    test2.iloc[20:] *= 1.3
    p2 = models.lstm_forecast(train, test2, epochs=3)
    assert len(p1) == 40 and np.isfinite(p1).all()
    assert np.allclose(p1.iloc[:21].values, p2.iloc[:21].values, rtol=1e-5)
