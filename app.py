"""Nifty Forecast Dashboard — run with:  streamlit run app.py"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import models
from src.data import load_prices, synthetic_prices
from src.evaluate import run_backtest

st.set_page_config(page_title="Nifty Forecast Dashboard", page_icon="📈", layout="wide")

TICKERS = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Index / stock", list(TICKERS) + ["Custom ticker"])
    ticker = st.text_input("Yahoo ticker", "INFY.NS") if choice == "Custom ticker" else TICKERS[choice]
    start = st.date_input("History from", value=None, help="Default: 2015-01-01")
    test_days = st.slider("Test period (trading days)", 60, 500, 250, step=10)
    use_lstm = st.checkbox("Train LSTM (slower)", value=models.torch_available(),
                           disabled=not models.torch_available(),
                           help=None if models.torch_available() else "Install torch to enable")
    future_days = st.slider("Future forecast (business days)", 5, 60, 20)
    demo = st.checkbox("Use synthetic demo data (offline)", value=False)
    run = st.button("Run backtest", type="primary", width="stretch")

st.title("📈 Nifty Forecast Dashboard")
st.caption(
    "ARIMA vs. LSTM vs. simple baselines, evaluated walk-forward on unseen data. "
    "**Educational project, not investment advice.**"
)


@st.cache_data(show_spinner="Downloading prices…", ttl=3600)
def get_data(ticker: str, start: str):
    return load_prices(ticker, start=start)


@st.cache_data(show_spinner="Training models…")
def get_backtest(series, test_days: int, use_lstm: bool):
    return run_backtest(series, test_size=test_days, use_lstm=use_lstm)


if not run and "ran" not in st.session_state:
    st.info("Pick an index in the sidebar and press **Run backtest**.")
    st.stop()
st.session_state["ran"] = True

try:
    series = synthetic_prices() if demo else get_data(ticker, str(start or "2015-01-01"))
except RuntimeError as e:
    st.error(f"{e}\n\nTry again, check the ticker, or tick *Use synthetic demo data*.")
    st.stop()

if len(series) < test_days + 200:
    st.error("Not enough history for this test period. Pick an earlier start date.")
    st.stop()

res = get_backtest(series, test_days, use_lstm)
table, preds, test = res["metrics"], res["predictions"], res["test"]

# ------------------------------------------------------------------ headline
label = "Synthetic demo" if demo else choice if choice != "Custom ticker" else ticker
c1, c2, c3, c4 = st.columns(4)
c1.metric(label, f"{series.iloc[-1]:,.0f}", f"{(series.iloc[-1] / series.iloc[-2] - 1) * 100:+.2f}% last day")
best = table["RMSE"].idxmin()
c2.metric("Lowest error", best.split(" (")[0])
c3.metric("Days tested", len(test))
c4.metric("ARIMA order (p,d,q)", str(res["arima_order"]))

beats = table[table["RMSE vs naive"] < 0.995].index.drop(table.index[0], errors="ignore")
if len(beats) == 0:
    st.warning(
        "**No model clearly beat the naive forecast (tomorrow = today).** "
        "That is the expected result for a liquid index: daily price moves are close to "
        "random. See *Why this happens* below."
    )
else:
    st.success(f"Beat the naive baseline on RMSE by over 0.5%: {', '.join(beats)}. "
               "Check that this holds across other test periods before trusting it.")

# ------------------------------------------------------------------ charts
tab1, tab2, tab3 = st.tabs(["Backtest", "Future forecast", "Why this happens"])

with tab1:
    show = st.multiselect("Models to plot", list(preds.columns), default=list(preds.columns))
    fig = go.Figure()
    ctx = res["train"].iloc[-120:]
    fig.add_scatter(x=ctx.index, y=ctx.values, name="Training data", line=dict(color="#9aa0a6"))
    fig.add_scatter(x=test.index, y=test.values, name="Actual", line=dict(color="#111", width=2))
    for name in show:
        fig.add_scatter(x=preds.index, y=preds[name], name=name, line=dict(width=1.3))
    fig.update_layout(height=480, hovermode="x unified", margin=dict(t=20, b=10),
                      legend=dict(orientation="h", y=-0.15), yaxis_title="Close")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Error metrics (lower is better, except direction)")
    st.dataframe(table.style.format("{:.3f}").highlight_min(subset=["RMSE", "MAE", "MAPE %"], color="#c8e6c9"),
                 width="stretch")
    st.caption("**RMSE vs naive** < 1 means the model beat 'tomorrow = today'. "
               "**Direction acc %** = how often it called up/down correctly; 50% is a coin flip.")

with tab2:
    fut = models.arima_future(series, future_days, res["arima_order"])
    hist = series.iloc[-250:]
    fig2 = go.Figure()
    fig2.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#111"))
    fig2.add_scatter(x=fut.index, y=fut["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig2.add_scatter(x=fut.index, y=fut["lower"], fill="tonexty", line=dict(width=0),
                     fillcolor="rgba(66,133,244,0.2)", name="95% range")
    fig2.add_scatter(x=fut.index, y=fut["forecast"], name="ARIMA forecast", line=dict(color="#4285f4"))
    fig2.update_layout(height=450, hovermode="x unified", margin=dict(t=20, b=10), yaxis_title="Close")
    st.plotly_chart(fig2, width="stretch")
    st.caption("Look at how fast the shaded band widens. The model is saying it does not know "
               "where the index will be in a few weeks, and that is the honest answer.")

with tab3:
    st.markdown("""
- **Markets are close to a random walk day to day.** Today's price already reflects public
  information, so yesterday's chart says very little about tomorrow's move.
- **The naive forecast is a strong baseline.** Any model that predicts price levels gets
  most of its accuracy just by staying near today's price. A low RMSE means nothing unless
  it beats the naive one.
- **LSTMs often learn to copy the last value.** On noisy data the safest thing for a
  network to do is predict roughly zero change, which turns it into the naive model.
- **No look-ahead here.** Models are trained on the training period only and predict each
  test day using real data up to the day before. Many online tutorials leak future data
  and show misleadingly perfect charts.
- **What's missing:** news, earnings, global markets, FII flows, volatility regimes and
  transaction costs. A real trading system would need all of these and more.
""")
