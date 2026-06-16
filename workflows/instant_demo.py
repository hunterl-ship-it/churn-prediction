"""Load pre-baked instant demo artifacts for hero workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from woodwide.evaluation import load_metrics_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def artifacts_dir(page_id: str) -> Path:
    return PROJECT_ROOT / "demo_artifacts" / page_id


def artifacts_available(page_id: str) -> bool:
    directory = artifacts_dir(page_id)
    required = ("metrics.json", "at_risk.csv", "intervention_plan.csv")
    return directory.is_dir() and all((directory / name).exists() for name in required)


def load_metrics(page_id: str) -> dict | None:
    return load_metrics_json(str(artifacts_dir(page_id) / "metrics.json"))


def load_at_risk(page_id: str) -> pd.DataFrame:
    path = artifacts_dir(page_id) / "at_risk.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_intervention_plan(page_id: str) -> pd.DataFrame:
    path = artifacts_dir(page_id) / "intervention_plan.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_driver_chart(page_id: str) -> pd.DataFrame:
    path = artifacts_dir(page_id) / "driver_chart.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_metadata(page_id: str) -> dict | None:
    path = artifacts_dir(page_id) / "metadata.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_explanations(page_id: str) -> pd.DataFrame:
    path = artifacts_dir(page_id) / "explanations.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame()


def merge_exemplar_explanations(dataframe: pd.DataFrame, explanations: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or explanations.empty or "id" not in explanations.columns:
        return dataframe
    merged = dataframe.copy()
    for column in explanations.columns:
        if column == "id":
            continue
        if column not in merged.columns:
            merged[column] = None
    explanation_by_id = explanations.set_index("id", drop=False)
    for index, row in merged.iterrows():
        row_id = row.get("id")
        if row_id in explanation_by_id.index:
            for column in explanations.columns:
                if column != "id":
                    merged.at[index, column] = explanation_by_id.at[row_id, column]
    return merged
