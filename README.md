# 📈 Nifty Forecast Dashboard

**ARIMA vs. LSTM vs. simple baselines for forecasting the Nifty 50, with an honest evaluation.**

An interactive Streamlit dashboard that downloads Indian market data, trains classical and
deep-learning forecasting models, and tests them **walk-forward** on data they have never
seen. It also tells you plainly when they fail to beat "tomorrow = today."

> ⚠️ Educational project. Not investment advice. Do not trade on these forecasts.

---

## What it does

| | |
|---|---|
| **Data** | Daily closes from Yahoo Finance: Nifty 50, Nifty Bank, Sensex, or any NSE stock (`TCS.NS`, `RELIANCE.NS`…) |
| **Models** | Naive · Drift · ARIMA(p,1,q) with AIC order selection · PyTorch LSTM on log returns |
| **Evaluation** | One-step-ahead walk-forward backtest; RMSE, MAE, MAPE, directional accuracy, error relative to naive |
| **Dashboard** | Interactive backtest chart, metrics table, ARIMA future forecast with a 95% interval |

## Quick start

```bash
git clone https://github.com/akshit-rao-1111/nifty-forecast-dashboard.git
cd nifty-forecast-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Command-line backtest (saves `results/metrics.csv` and `results/backtest.png`):

```bash
python scripts/run_backtest.py                        # Nifty 50
python scripts/run_backtest.py --ticker TCS.NS        # any Yahoo ticker
python scripts/run_backtest.py --demo                 # offline, synthetic data
```

Run tests: `pytest -q`

## How it works

```
prices ──► chronological split ──► train  ──► fit model (train only)
                                │
                                └► test   ──► for each day t: predict close(t) using data up to t-1
                                                 ──► compare with actual ──► metrics
```

- **Naive:** tomorrow's close = today's close. This is the baseline every model must beat.
- **Drift:** today's close × the average daily return seen in training.
- **ARIMA:** fitted on log prices; `(p, q)` chosen from 0–2 by lowest AIC. Parameters are
  estimated on the training set and **kept fixed** while the model is fed each real new observation.
- **LSTM:** 1 layer, 32 units, 30-day lookback. It predicts the next daily **log return** rather
  than the price, so it can't score well just by staying near the current price level.
  Scaling uses training statistics only, with early stopping on a validation slice.

## Results: being honest

**Nifty 50**: data from 2015-01-01, the last 250 trading days up to Oct 2026 used as the test set,
one-day-ahead forecasts (`python scripts/run_backtest.py`):

| Model | RMSE (pts) | MAE (pts) | MAPE % | Direction acc % | RMSE vs naive |
|---|---|---|---|---|---|
| **Naive (tomorrow = today)** | **205.97** | **154.51** | **0.640** | n/a | **1.000** |
| Drift | 206.89 | 155.24 | 0.643 | 47.6 | 1.004 |
| ARIMA(0, 1, 0) | 206.87 | 155.23 | 0.643 | 47.6 | 1.004 |
| LSTM | 206.51 | 154.94 | 0.642 | 46.8 | 1.003 |

**No model beat the naive forecast.** Every model came within 0.4% of it, none called the
direction better than a coin flip, and ARIMA's AIC search chose `(0,1,0)`, which *is* a random walk.

**This is the expected result, not a bug.** Research on liquid equity indices consistently
finds that price-only models struggle to beat a random walk at a one-day horizon. The point
of the project is to show *why*:

1. **Liquid indices are close to a random walk day to day.** Public information is already
   priced in, so past prices alone carry very little signal about tomorrow.
2. **Price-level RMSE is misleading.** Any model that stays near today's price looks
   "accurate". Always compare against naive (see the `RMSE vs naive` column).
3. **LSTMs on noisy data tend to predict roughly zero change,** which collapses them into the
   naive model.
4. **Many online tutorials leak future data.** Common mistakes are scaling on the full
   dataset, shuffling time series, or plotting multi-step forecasts that are really one-step
   predictions using actual prices. These leaks produce beautiful charts that are useless in
   practice. This repo has tests that check for look-ahead (`tests/test_models.py`).

## Limitations

- Uses only past prices. No news, earnings, FII/DII flows, global markets, VIX, or macro data.
- No transaction costs, slippage or taxes. Even a small edge would likely disappear after these.
- A single test window. Results change with market regime, so they need checking across
  several periods (rolling-origin cross-validation) before drawing conclusions.
- Yahoo Finance data is unofficial and can have gaps.

## Ideas to extend

- [ ] Add exogenous features (VIX, USD/INR, S&P 500 overnight return) → ARIMAX / multivariate LSTM
- [ ] Rolling-origin evaluation over several test windows
- [ ] Diebold–Mariano test for whether differences in error are statistically significant
- [ ] Forecast volatility (GARCH) instead of price, which is more predictable
- [ ] Deploy to Streamlit Community Cloud

## Project structure

```
├── app.py                 # Streamlit dashboard
├── src/
│   ├── data.py            # download, cache, clean, split
│   ├── models.py          # naive, drift, ARIMA, LSTM
│   └── evaluate.py        # metrics + backtest runner
├── scripts/run_backtest.py
├── tests/test_models.py   # includes no-look-ahead tests
└── requirements.txt
```

## Tech

Python · pandas · statsmodels · PyTorch · Streamlit · Plotly · yfinance · pytest

## License

MIT
