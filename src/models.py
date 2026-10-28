"""Forecasting models.

Every model answers the same question: given all closing prices up to day t,
what will the close be on day t+1? Predictions are made walk-forward over the
test period. Each model sees only real past data, never the future.

Models
------
- naive  : tomorrow = today. Hard to beat; every other model must be compared to it.
- drift  : today + the average daily change seen in training.
- arima  : ARIMA(p,1,q) on log prices, order picked by AIC.
- lstm   : small PyTorch LSTM predicting the next daily log return.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- baselines
def naive_forecast(train: pd.Series, test: pd.Series) -> pd.Series:
    history = pd.concat([train.iloc[-1:], test])
    return pd.Series(history.shift(1).iloc[1:].values, index=test.index, name="naive")


def drift_forecast(train: pd.Series, test: pd.Series) -> pd.Series:
    mean_log_ret = np.diff(np.log(train.values)).mean()
    prev = pd.concat([train.iloc[-1:], test]).shift(1).iloc[1:].values
    return pd.Series(prev * np.exp(mean_log_ret), index=test.index, name="drift")


# ------------------------------------------------------------------- ARIMA
@dataclass
class ArimaInfo:
    order: tuple[int, int, int]
    aic: float


def select_arima_order(log_train: np.ndarray, max_p: int = 2, max_q: int = 2) -> ArimaInfo:
    from statsmodels.tsa.arima.model import ARIMA

    best = ArimaInfo(order=(0, 1, 0), aic=np.inf)
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = ARIMA(log_train, order=(p, 1, q), trend="t").fit()
            except Exception:
                continue
        if res.aic < best.aic:
            best = ArimaInfo(order=(p, 1, q), aic=float(res.aic))
    return best


def arima_forecast(
    train: pd.Series, test: pd.Series, order: tuple[int, int, int] | None = None
) -> tuple[pd.Series, ArimaInfo]:
    """One-step-ahead ARIMA forecasts over the test period.

    Parameters are estimated on train only, then held fixed while the model
    is fed each new real observation (no look-ahead, no refitting).
    """
    from statsmodels.tsa.arima.model import ARIMA

    log_train = np.log(train.values)
    log_all = np.log(np.concatenate([train.values, test.values]))
    info = select_arima_order(log_train) if order is None else ArimaInfo(order, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(log_train, order=info.order, trend="t").fit()
        if np.isnan(info.aic):
            info = ArimaInfo(info.order, float(res.aic))
        full = res.apply(log_all)
        pred_log = full.get_prediction(start=len(train), end=len(log_all) - 1).predicted_mean

    return pd.Series(np.exp(pred_log), index=test.index, name="arima"), info


def arima_future(series: pd.Series, steps: int, order: tuple[int, int, int]) -> pd.DataFrame:
    """Forecast `steps` business days past the end of the data, with a 95% band."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(np.log(series.values), order=order, trend="t").fit()
        fc = res.get_forecast(steps)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    idx = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame(
        {
            "forecast": np.exp(fc.predicted_mean),
            "lower": np.exp(ci[:, 0]),
            "upper": np.exp(ci[:, 1]),
        },
        index=idx,
    )


# -------------------------------------------------------------------- LSTM
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _windows(x: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([x[i : i + lookback] for i in range(len(x) - lookback)])
    y = x[lookback:]
    return X, y


def lstm_forecast(
    train: pd.Series,
    test: pd.Series,
    lookback: int = 30,
    hidden: int = 32,
    epochs: int = 40,
    lr: float = 1e-3,
    seed: int = 42,
) -> pd.Series:
    """Train an LSTM on daily log returns, then predict one step ahead over test.

    Working in returns (not raw prices) keeps the data roughly stationary and
    stops the network from just memorising the price level.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    prices = np.concatenate([train.values, test.values])
    rets = np.diff(np.log(prices))  # rets[i] = return from day i to day i+1
    n_train_rets = len(train) - 1

    # scale with training statistics only
    mu, sd = rets[:n_train_rets].mean(), rets[:n_train_rets].std()
    z = (rets - mu) / sd

    X_tr, y_tr = _windows(z[:n_train_rets], lookback)
    n_val = max(1, int(0.1 * len(X_tr)))
    X_fit, y_fit, X_val, y_val = X_tr[:-n_val], y_tr[:-n_val], X_tr[-n_val:], y_tr[-n_val:]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.head(out[:, -1]).squeeze(-1)

    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    model, loss_fn = Net(), nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xf, yf, Xv, yv = t(X_fit), t(y_fit), t(X_val), t(y_val)

    best_state, best_val, patience = None, np.inf, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xf))
        for i in range(0, len(Xf), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        if val < best_val - 1e-5:
            best_val, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:  # early stopping
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # walk-forward: to predict return k (into test day), use the real returns before it
    preds = []
    model.eval()
    with torch.no_grad():
        for k in range(n_train_rets, len(rets)):
            window = t(z[k - lookback : k]).unsqueeze(0)
            preds.append(model(window).item() * sd + mu)
    prev_prices = prices[len(train) - 1 : -1]
    return pd.Series(prev_prices * np.exp(preds), index=test.index, name="lstm")
