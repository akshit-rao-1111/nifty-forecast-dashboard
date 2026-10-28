"""Metrics and the full backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models


def metrics(actual: pd.Series, pred: pd.Series, prev: pd.Series) -> dict[str, float]:
    """Error metrics plus directional accuracy (did it call up/down right?)."""
    a, p, b = actual.values, pred.values, prev.values
    err = p - a
    pred_move, true_move = np.sign(p - b), np.sign(a - b)
    called = pred_move != 0  # naive never predicts a move, so direction is N/A for it
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE %": float(np.mean(np.abs(err / a)) * 100),
        "Direction acc %": float((pred_move[called] == true_move[called]).mean() * 100)
        if called.any()
        else float("nan"),
    }


def run_backtest(
    series: pd.Series, test_size: int = 250, use_lstm: bool = True, lstm_kwargs: dict | None = None
) -> dict:
    from .data import train_test_split

    train, test = train_test_split(series, test_size)
    prev = pd.concat([train.iloc[-1:], test]).shift(1).iloc[1:]

    preds = {
        "Naive (tomorrow = today)": models.naive_forecast(train, test),
        "Drift": models.drift_forecast(train, test),
    }
    arima_pred, arima_info = models.arima_forecast(train, test)
    preds[f"ARIMA{arima_info.order}"] = arima_pred
    if use_lstm and models.torch_available():
        preds["LSTM"] = models.lstm_forecast(train, test, **(lstm_kwargs or {}))

    table = pd.DataFrame({name: metrics(test, p, prev) for name, p in preds.items()}).T
    naive_rmse = table.iloc[0]["RMSE"]
    table["RMSE vs naive"] = table["RMSE"] / naive_rmse
    return {
        "train": train,
        "test": test,
        "predictions": pd.DataFrame(preds),
        "metrics": table.round(3),
        "arima_order": arima_info.order,
    }
