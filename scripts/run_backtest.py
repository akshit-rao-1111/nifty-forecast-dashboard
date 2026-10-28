"""Command-line backtest. Saves the metrics table and a chart to results/.

Usage:
    python scripts/run_backtest.py                 # Nifty 50
    python scripts/run_backtest.py --ticker TCS.NS --test-days 200
    python scripts/run_backtest.py --demo          # offline, synthetic data
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import DEFAULT_TICKER, load_prices, synthetic_prices  # noqa: E402
from src.evaluate import run_backtest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=DEFAULT_TICKER)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--test-days", type=int, default=250)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--demo", action="store_true", help="use synthetic random-walk data")
    args = ap.parse_args()

    series = synthetic_prices() if args.demo else load_prices(args.ticker, start=args.start)
    res = run_backtest(series, args.test_days, use_lstm=not args.no_lstm)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res["metrics"].to_csv(out / "metrics.csv")
    print(res["metrics"].to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res["test"], color="black", lw=2, label="Actual")
    for name in res["predictions"]:
        ax.plot(res["predictions"][name], lw=1, label=name)
    ax.set_title(f"{'Synthetic demo' if args.demo else args.ticker}: one-day-ahead forecasts on the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "backtest.png", dpi=130)
    print(f"\nSaved {out / 'metrics.csv'} and {out / 'backtest.png'}")


if __name__ == "__main__":
    main()
