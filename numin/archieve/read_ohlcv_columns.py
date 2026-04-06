#!/usr/bin/env python3
"""Read an OHLCV CSV file and print all column names."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an OHLCV CSV file and print all columns."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the OHLCV CSV file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        print("No columns found (empty file or missing header).")
        return

    print("Columns:")
    for idx, column in enumerate(header, start=1):
        print(f"{idx}. {column}")


if __name__ == "__main__":
    main()
