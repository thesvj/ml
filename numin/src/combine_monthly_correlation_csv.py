import argparse
from pathlib import Path

import pandas as pd


BASE_COLUMNS = ["ticker_1", "ticker_2", "correlation", "overlap_obs"]
PERIOD_COLUMNS = ["year_month", "year"]


def _detect_period_column(df: pd.DataFrame, file_path: Path) -> str:
    for col in PERIOD_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"Missing period column in {file_path}. Expected one of: {PERIOD_COLUMNS}")


def combine_monthly_csvs(input_dir: str, output_csv: str) -> None:
    input_path = Path(input_dir)
    files = sorted(input_path.glob("stock_corr_half_*.csv"))

    if not files:
        raise FileNotFoundError(f"No monthly correlation CSV files found in: {input_dir}")

    frames: list[pd.DataFrame] = []
    common_period_col: str | None = None
    for file in files:
        df = pd.read_csv(file)
        period_col = _detect_period_column(df, file)

        if common_period_col is None:
            common_period_col = period_col
        elif period_col != common_period_col:
            raise ValueError(
                f"Mixed period columns in input files. Found both '{common_period_col}' and '{period_col}'."
            )

        required_columns = [*BASE_COLUMNS, period_col]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {file}: {missing}")

        frames.append(df[required_columns])

    combined = pd.concat(frames, ignore_index=True)
    period_col = common_period_col or "year_month"

    # Remove exact duplicate rows if any and keep stable order with sorting.
    combined = combined.drop_duplicates(subset=[*BASE_COLUMNS, period_col])
    combined = combined.sort_values([period_col, "ticker_1", "ticker_2"]).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Saved combined monthly correlation CSV: {output_path}")
    print(f"Monthly files combined: {len(files)}")
    print(f"Total rows: {len(combined)}")
    print(f"Period column: {period_col}")
    print(f"Unique periods: {combined[period_col].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine all monthly correlation CSV files into one CSV")
    parser.add_argument(
        "--input-dir",
        default="data/correlation_exports/monthly_last5y",
        help="Directory containing stock_corr_half_YYYY-MM.csv files",
    )
    parser.add_argument(
        "--output",
        default="data/correlation_exports/monthly_last5y/stock_corr_half_all_months.csv",
        help="Output path for combined monthly CSV",
    )
    args = parser.parse_args()

    combine_monthly_csvs(args.input_dir, args.output)


if __name__ == "__main__":
    main()
