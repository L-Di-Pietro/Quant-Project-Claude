"""
download_data.py — Data sourcing and cleaning script.

Downloads historical OHLCV data from Yahoo Finance, cleans it
(forward-fill, adjusted close), and saves to Parquet format.

Usage
-----
    python data/download_data.py --symbols AAPL MSFT GOOG --start 2015-01-01 --end 2024-12-31

The resulting files are saved in the ``data/`` directory as
``<SYMBOL>.parquet``.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yfinance as yf


def download_and_clean(
    symbol: str,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    output_dir: str = ".",
) -> str:
    """
    Download OHLCV data for *symbol*, clean it, and save as Parquet.

    Cleaning steps:
        1. Forward-fill missing values.
        2. Drop any remaining NaN rows (at the very start).
        3. Ensure Adj Close column is present.
        4. Sort by date ascending.

    Returns the path to the saved file.
    """
    print(f"[download] Fetching {symbol} from {start} to {end} ...")
    ticker = yf.Ticker(symbol)
    df: pd.DataFrame = ticker.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol}. "
            "Check the ticker symbol and date range."
        )

    # Standardise column names
    rename_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if lower == "adj_close":
            rename_map[col] = "Adj Close"
        elif lower in ("open", "high", "low", "close", "volume"):
            rename_map[col] = col.title()

    # If there's no Adj Close but there is a Close, copy it
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    df.sort_index(inplace=True)

    # Forward-fill then drop leading NaN
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Keep only OHLCV + Adj Close
    keep_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    # Ensure index name is "Date"
    df.index.name = "Date"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol}.parquet")
    df.to_parquet(out_path, engine="pyarrow")
    print(f"[download] Saved {len(df)} rows → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical market data and save as Parquet."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT"],
        help="Ticker symbols to download (default: AAPL MSFT).",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date in YYYY-MM-DD format (default: 2015-01-01).",
    )
    parser.add_argument(
        "--end",
        default="2024-12-31",
        help="End date in YYYY-MM-DD format (default: 2024-12-31).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save parquet files (default: same as this script).",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    for symbol in args.symbols:
        try:
            download_and_clean(
                symbol=symbol,
                start=args.start,
                end=args.end,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to download {symbol}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
