#!/usr/bin/env python3
"""Create stratified eval.csv holdouts for hero demo datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def stratified_sample(dataframe: pd.DataFrame, label_column: str, n_rows: int, seed: int = 42) -> pd.DataFrame:
    fraction = min(1.0, n_rows / len(dataframe))
    parts = []
    for _, group in dataframe.groupby(label_column, sort=False):
        parts.append(group.sample(frac=fraction, random_state=seed))
    return pd.concat(parts, ignore_index=True).head(n_rows)


def prepare_churn_eval(train_path: Path, output_path: Path, n_rows: int, seed: int):
    data = pd.read_csv(train_path)
    if "Churn" not in data.columns:
        raise ValueError(f"{train_path} must include a Churn column")
    eval_data = stratified_sample(data, "Churn", n_rows, seed)
    eval_data.to_csv(output_path, index=False)
    print(f"Wrote {len(eval_data):,} rows -> {output_path} (positive rate {eval_data['Churn'].mean():.1%})")


def prepare_healthcare_eval(train_path: Path, output_path: Path, n_rows: int, seed: int):
    data = pd.read_csv(train_path)
    if "no_show" not in data.columns:
        raise ValueError(f"{train_path} must include a no_show column")
    eval_data = stratified_sample(data, "no_show", n_rows, seed)
    eval_data.to_csv(output_path, index=False)
    print(f"Wrote {len(eval_data):,} rows -> {output_path} (positive rate {eval_data['no_show'].mean():.1%})")


def main():
    parser = argparse.ArgumentParser(description="Prepare labeled eval splits for hero demos")
    parser.add_argument("--eval-rows", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    prepare_churn_eval(
        root / "datasets/churn/train.csv",
        root / "datasets/churn/eval.csv",
        args.eval_rows,
        args.seed,
    )
    prepare_healthcare_eval(
        root / "datasets/healthcare/train.csv",
        root / "datasets/healthcare/eval.csv",
        args.eval_rows,
        args.seed,
    )


if __name__ == "__main__":
    main()
