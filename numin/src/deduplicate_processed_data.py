import argparse
import os

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


def deduplicate_processed_data(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    # Average numeric OHLCV values for duplicate (timestamp, ticker) rows.
    dedup = (
        df[REQUIRED_COLUMNS]
        .groupby(["timestamp", "ticker"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["timestamp", "ticker"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dedup.to_csv(output_csv, index=False)

    before_rows = len(df)
    after_rows = len(dedup)
    print(f"Saved deduplicated file: {output_csv}")
    print(f"Rows before: {before_rows}")
    print(f"Rows after : {after_rows}")
    print(f"Removed    : {before_rows - after_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate processed OHLCV CSV by timestamp and ticker")
    parser.add_argument("--input", default="data/processed_data.csv", help="Input processed data CSV")
    parser.add_argument("--output", default="data/processed_data_dedup.csv", help="Output deduplicated CSV")
    args = parser.parse_args()

    deduplicate_processed_data(args.input, args.output)


if __name__ == "__main__":
    main()
