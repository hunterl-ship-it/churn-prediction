#!/usr/bin/env python3
"""Prepare ecommerce demo datasets for fraud and return workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = ("is_fraud", "risk_label", "is_returned")
DROP_COLUMNS = ("order_id", "order_date", "customer_support_contacts")


def load_source(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in DROP_COLUMNS:
        if column in data.columns:
            data = data.drop(columns=[column])
    return data


def prepare(data: pd.DataFrame, train_fraction: float = 0.8):
    split_index = int(train_fraction * len(data))

    returns_train = data.iloc[:split_index].copy()
    returns_test = data.iloc[split_index:].copy()
    if "is_returned" in returns_test.columns:
        returns_test = returns_test.drop(columns=["is_returned"])
    for column in ("is_fraud", "risk_label"):
        if column in returns_test.columns:
            returns_test = returns_test.drop(columns=[column])

    fraud_train = data.iloc[:split_index].copy()
    if "is_fraud" in fraud_train.columns:
        fraud_train = fraud_train[fraud_train["is_fraud"] == 0]
    fraud_test = data.iloc[split_index:].copy()
    for frame in (fraud_train, fraud_test):
        for column in LABEL_COLUMNS:
            if column in frame.columns:
                frame.drop(columns=[column], inplace=True)

    return returns_train, returns_test, fraud_train, fraud_test


def main():
    parser = argparse.ArgumentParser(description="Prepare ecommerce demo CSVs")
    parser.add_argument(
        "--source",
        default="../fraud-detection/synthetic_ecommerce_order_risk_dataset.csv",
        help="Path to synthetic ecommerce source CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/ecommerce",
        help="Directory for prepared demo files",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        source = Path("datasets/ecommerce/synthetic_ecommerce_order_risk_dataset.csv")
    if not source.exists():
        raise SystemExit(f"Source dataset not found: {args.source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_source(source)
    returns_train, returns_test, fraud_train, fraud_test = prepare(data)

    returns_train.to_csv(output_dir / "train.csv", index=False)
    returns_test.to_csv(output_dir / "test.csv", index=False)
    fraud_train.to_csv(output_dir / "fraud_train.csv", index=False)
    fraud_test.to_csv(output_dir / "fraud_test.csv", index=False)

    print(f"Wrote demo files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
