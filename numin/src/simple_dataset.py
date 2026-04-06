import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


def create_stock_feature_dataset(
    input_csv: str,
    output_csv: Optional[str] = None,
    return_horizon: int = 1,
) -> pd.DataFrame:
    """
    Create a simple per-day stock feature dataset and next-day return target.

    Each row in the returned dataframe represents one (timestamp, ticker) sample.
    Target is the forward return for `return_horizon` day(s):
        target_return = close[t + horizon] / close[t] - 1
    """
    if return_horizon < 1:
        raise ValueError("return_horizon must be >= 1")

    df = pd.read_csv(input_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ticker"] = df["ticker"].astype(str)

    # Keep deterministic order and robust numeric conversion.
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    eps = 1e-8
    grouped = df.groupby("ticker", group_keys=False)

    # Basic single-day features.
    df["ret_1d"] = grouped["close"].pct_change()
    df["log_ret_1d"] = np.log((df["close"] + eps) / (grouped["close"].shift(1) + eps))
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["oc_change"] = (df["close"] - df["open"]) / (df["open"] + eps)
    df["volume_change"] = grouped["volume"].pct_change()

    # Short rolling context features.
    df["ret_5d_mean"] = grouped["ret_1d"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["ret_10d_std"] = grouped["ret_1d"].transform(lambda s: s.rolling(10, min_periods=2).std())
    df["vol_5d_mean"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean())

    # Forward return target for day n (or horizon n).
    df["target_return"] = grouped["close"].shift(-return_horizon) / (df["close"] + eps) - 1.0

    # Remove rows that cannot form full features/targets.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


@dataclass
class SequenceSample:
    x: torch.Tensor
    y: torch.Tensor
    timestamp: str
    ticker: str


class StockReturnSequenceDataset(Dataset):
    """
    Uses previous `lookback` days to predict target return of current day.

    If user thinks in terms of "n-1 previous days -> predict day n", then:
        lookback = n - 1
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int,
        feature_columns: Sequence[str],
        target_column: str = "target_return",
        ticker: Optional[str] = None,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in dataframe")

        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        df = data.copy()
        if "timestamp" not in df.columns or "ticker" not in df.columns:
            raise ValueError("Dataframe must contain 'timestamp' and 'ticker' columns")

        if ticker is not None:
            df = df[df["ticker"] == ticker].copy()

        if df.empty:
            raise ValueError("No rows available after filtering")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        self.lookback = lookback
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.samples: list[SequenceSample] = []

        for tick, g in df.groupby("ticker", sort=True):
            g = g.reset_index(drop=True)
            features = g[self.feature_columns].to_numpy(dtype=np.float32)
            targets = g[self.target_column].to_numpy(dtype=np.float32)
            timestamps = g["timestamp"].dt.strftime("%Y-%m-%d").to_numpy()

            # Sample i predicts target at i using [i-lookback, ..., i-1].
            for i in range(self.lookback, len(g)):
                x_window = np.ascontiguousarray(features[i - self.lookback : i])
                x = torch.from_numpy(x_window)
                y = torch.tensor(targets[i], dtype=torch.float32)
                self.samples.append(SequenceSample(x=x, y=y, timestamp=timestamps[i], ticker=tick))

        if not self.samples:
            raise ValueError("No training samples could be created. Increase data size or reduce lookback.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return {
            "x": sample.x,
            "y": sample.y,
            "timestamp": sample.timestamp,
            "ticker": sample.ticker,
        }


DEFAULT_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1d",
    "log_ret_1d",
    "hl_spread",
    "oc_change",
    "volume_change",
    "ret_5d_mean",
    "ret_10d_std",
    "vol_5d_mean",
]


def create_stock_dataloader(
    data: pd.DataFrame,
    lookback: int,
    batch_size: int = 32,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = "target_return",
    ticker: Optional[str] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = StockReturnSequenceDataset(
        data=data,
        lookback=lookback,
        feature_columns=feature_columns or DEFAULT_FEATURE_COLUMNS,
        target_column=target_column,
        ticker=ticker,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_stock_features_for_day(
    data: pd.DataFrame,
    ticker: str,
    day: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Return feature values for one stock on one day."""
    feat_cols = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    missing_features = [c for c in feat_cols if c not in data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    day_ts = pd.to_datetime(day)
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    row = df[(df["ticker"] == ticker) & (df["timestamp"] == day_ts)]
    if row.empty:
        raise ValueError(f"No row found for ticker='{ticker}' and day='{day}'")

    return row.iloc[0][feat_cols]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple stock feature dataset creator and dataloader utility")
    parser.add_argument("--input", required=True, help="Input OHLCV CSV")
    parser.add_argument("--output", default="", help="Optional output CSV for feature dataset")
    parser.add_argument("--lookback", type=int, default=9, help="Number of previous days (n-1)")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_df = create_stock_feature_dataset(
        input_csv=args.input,
        output_csv=args.output if args.output else None,
        return_horizon=1,
    )

    loader = create_stock_dataloader(
        data=dataset_df,
        lookback=args.lookback,
        batch_size=args.batch_size,
    )

    first_batch = next(iter(loader))
    print("Created feature dataset and DataLoader")
    print(f"Rows: {len(dataset_df)}")
    print(f"Samples: {len(loader.dataset)}")
    print(f"Batch x shape: {tuple(first_batch['x'].shape)}")
    print(f"Batch y shape: {tuple(first_batch['y'].shape)}")


if __name__ == "__main__":
    main()
