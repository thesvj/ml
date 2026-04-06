import argparse
import os

import numpy as np
import pandas as pd

from features import compute_features


FEATURE_COLUMNS = [
    "ret_5",
    "ret_30",
    "rel_ret_5",
    "rank_ret_5",
    "vol_30",
    "risk_adj_ret",
    "vol_z",
    "pv_signal",
    "dist_high",
    "z_price",
    "residual",
    "market_ret_5",
    "corr_market_30",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _pivot_feature_matrix(df: pd.DataFrame, value_col: str, timestamps: list[str], tickers: list[str]) -> np.ndarray:
    pivot = (
        df.pivot_table(index="timestamp", columns="ticker", values=value_col, aggfunc="mean")
        .reindex(index=timestamps, columns=tickers)
        .fillna(0.0)
    )
    return pivot.to_numpy(dtype=np.float64)


def build_dataset_csv(input_csv: str, output_csv: str) -> None:
    eps = 1e-6
    df = pd.read_csv(input_csv)

    required_cols = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {input_csv}: {missing_cols}")

    # Ensure deterministic ordering for matrix construction and final output.
    timestamps = sorted(df["timestamp"].astype(str).unique().tolist())
    tickers = sorted(df["ticker"].astype(str).unique().tolist())

    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df["ticker"] = df["ticker"].astype(str)

    open_raw = _pivot_feature_matrix(df, "open", timestamps, tickers)
    high_raw = _pivot_feature_matrix(df, "high", timestamps, tickers)
    low_raw = _pivot_feature_matrix(df, "low", timestamps, tickers)
    close_raw = _pivot_feature_matrix(df, "close", timestamps, tickers)
    volume_raw = _pivot_feature_matrix(df, "volume", timestamps, tickers)

    # Keep raw OHLCV for dataset output; use clipped arrays for stable log/ratio features.
    open_ = np.maximum(open_raw, eps)
    high = np.maximum(high_raw, eps)
    low = np.maximum(low_raw, eps)
    close = np.maximum(close_raw, eps)
    volume = np.maximum(volume_raw, eps)

    features = compute_features(open_, high, low, close, volume)

    kept_timestamps = timestamps[30:]
    rows = []
    for t_idx, ts in enumerate(kept_timestamps):
        src_t_idx = t_idx + 30
        for n_idx, ticker in enumerate(tickers):
            feature_values = features[t_idx, n_idx].tolist()
            row = {
                "timestamp": ts,
                "ticker": ticker,
                "open": open_raw[src_t_idx, n_idx],
                "high": high_raw[src_t_idx, n_idx],
                "low": low_raw[src_t_idx, n_idx],
                "close": close_raw[src_t_idx, n_idx],
                "volume": volume_raw[src_t_idx, n_idx],
            }
            row.update(dict(zip(FEATURE_COLUMNS, feature_values)))
            rows.append(row)

    out_df = pd.DataFrame(rows, columns=["timestamp", "ticker", *OHLCV_COLUMNS, *FEATURE_COLUMNS])
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    out_df.fillna(0.0, inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"Saved dataset CSV: {output_csv}")
    print(f"Rows: {len(out_df)}, Columns: {len(out_df.columns)}")
    print(f"Tickers: {len(tickers)}, Timestamps kept: {len(kept_timestamps)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature-engineered dataset.csv from processed_data.csv")
    parser.add_argument("--input", default="data/processed_data.csv", help="Input processed OHLCV CSV")
    parser.add_argument("--output", default="data/dataset.csv", help="Output feature dataset CSV")
    args = parser.parse_args()

    build_dataset_csv(args.input, args.output)


if __name__ == "__main__":
    main()
