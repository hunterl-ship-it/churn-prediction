#!/usr/bin/env python3
"""Fetch row-level explanations from /jobs/{job_id}/explain and update demo artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._streamlit_headless import install_streamlit_mock

install_streamlit_mock()

import pandas as pd

import woodwide.core as core
from woodwide.core import (
    add_row_level_explanations,
    as_number,
    get_row_level_explanations,
    model_risk_driver_summary_for_row,
    parse_feature_weight_summary,
    probability_sort_column,
)

DEMOS = ["churn", "noshow"]
MAX_EXPLANATIONS = 200


def rebuild_model_risk_drivers(at_risk: pd.DataFrame) -> pd.DataFrame:
    """Re-derive model_risk_drivers from top_churn_drivers using updated group labels."""
    if "top_churn_drivers" not in at_risk.columns:
        return at_risk
    result = at_risk.copy()
    summaries = []
    for _, row in result.iterrows():
        entries = parse_feature_weight_summary(row.get("top_churn_drivers", ""))
        summaries.append(model_risk_driver_summary_for_row(row, entries) if entries else row.get("model_risk_drivers", ""))
    result["model_risk_drivers"] = summaries
    return result


def fetch_for_demo(page_id: str) -> None:
    artifact_dir = PROJECT_ROOT / "demo_artifacts" / page_id
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"{page_id}: no metadata.json, skipping")
        return

    metadata = json.loads(metadata_path.read_text())
    test_job_id = metadata.get("test_job_id")
    if not test_job_id:
        print(f"{page_id}: no test_job_id in metadata, skipping")
        return

    at_risk = pd.read_csv(artifact_dir / "at_risk.csv")
    print(f"{page_id}: {len(at_risk)} at-risk rows, test_job_id={test_job_id}")

    prob_col = probability_sort_column(at_risk)
    candidates = at_risk.copy()
    if prob_col:
        candidates = candidates.sort_values(prob_col, ascending=False)

    row_ids = []
    for row_id in candidates["id"].head(MAX_EXPLANATIONS).dropna().tolist():
        n = as_number(row_id)
        row_ids.append(int(n) if n is not None else str(row_id))

    print(f"  fetching explanations for {len(row_ids)} rows...")
    explanations = get_row_level_explanations(test_job_id, row_ids)

    if explanations.empty:
        print(f"  no explanations returned — updating model_risk_drivers labels only")
        at_risk = rebuild_model_risk_drivers(at_risk)
        at_risk.to_csv(artifact_dir / "at_risk.csv", index=False)
        return

    print(f"  got {len(explanations)} explanation rows, cols: {list(explanations.columns)}")
    at_risk = add_row_level_explanations(at_risk, explanations)
    at_risk = rebuild_model_risk_drivers(at_risk)
    at_risk.to_csv(artifact_dir / "at_risk.csv", index=False)

    records = explanations.to_dict(orient="records")
    (artifact_dir / "explanations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  saved {len(records)} explanations to explanations.json")
    metadata["explained_rows"] = len(explanations)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    demos = sys.argv[1:] or DEMOS
    for demo in demos:
        fetch_for_demo(demo)
