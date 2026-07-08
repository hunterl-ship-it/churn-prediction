#!/usr/bin/env python3
"""Enrich demo artifacts: realistic per-row explanations + real API factors/clusters."""

from __future__ import annotations

import hashlib
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
    analysis_dataframe_for_modeling,
    build_interventions,
    clustering_dataframe_for_modeling,
    dataframe_to_csv_bytes,
    dataset_name_for_dataframe,
    get_or_create_dataset_from_bytes,
    inference_cache_key,
    model_risk_driver_summary_for_row,
    parse_cluster_inference_payload,
    parse_feature_weight_summary,
    probability_sort_column,
    run_cached_model_inference,
    train_model,
    wait_for_training_complete,
)

DEMOS = ["churn", "noshow"]


# ── Per-row explanation generation ──────────────────────────────────────────

CHURN_TEMPLATES = [
    "Predicted churn ({prob:.0%} risk): top signals are {d1} and {d2}.",
    "High churn likelihood ({prob:.0%}): {d1} and {d2} are the strongest indicators.",
    "{prob:.0%} churn probability — model flagged {d1}, {d2}, and {d3}.",
    "Model scored this customer {prob:.0%} for churn, driven by {d1} and {d2}.",
]

NOSHOW_TEMPLATES = [
    "Predicted no-show ({prob:.0%} risk): top signals are {d1} and {d2}.",
    "High no-show likelihood ({prob:.0%}): {d1} and {d2} are the strongest indicators.",
    "{prob:.0%} no-show probability — model flagged {d1}, {d2}, and {d3}.",
    "Model scored this appointment {prob:.0%} for no-show, driven by {d1} and {d2}.",
]


def _driver_label(feature: str, value) -> str:
    label = feature.replace("_", " ")
    if value != "" and value is not None and not (isinstance(value, float) and pd.isna(value)):
        return f"{label} ({value})"
    return label


def generate_explanation(row, templates: list[str], idx: int) -> str:
    prob = float(row.get("prediction_prob", row.get("risk_probability", 0.5)))
    entries = parse_feature_weight_summary(row.get("top_churn_drivers", ""))
    positive = sorted([e for e in entries if e[1] > 0], key=lambda e: -e[1])
    if not positive:
        return templates[idx % len(templates)].format(
            prob=prob, d1="risk profile", d2="historical patterns", d3="engagement signals"
        )
    drivers = [_driver_label(f, row.get(f, "")) for f, _ in positive]
    d1 = drivers[0] if len(drivers) > 0 else "risk profile"
    d2 = drivers[1] if len(drivers) > 1 else "historical patterns"
    d3 = drivers[2] if len(drivers) > 2 else "engagement signals"
    return templates[idx % len(templates)].format(prob=prob, d1=d1, d2=d2, d3=d3)


def enrich_explanations(at_risk: pd.DataFrame, page_id: str) -> pd.DataFrame:
    templates = NOSHOW_TEMPLATES if page_id == "noshow" else CHURN_TEMPLATES
    result = at_risk.copy()
    result["prediction_class_explanation"] = [
        generate_explanation(row, templates, i)
        for i, (_, row) in enumerate(result.iterrows())
    ]
    return result


# ── Real Wood Wide factor + cluster training ─────────────────────────────────

def run_api_factors_and_clusters(at_risk: pd.DataFrame, page_id: str):
    modeling_df, input_columns = analysis_dataframe_for_modeling(at_risk)
    cluster_df, cluster_columns = clustering_dataframe_for_modeling(modeling_df)

    # Fall back to all at_risk columns when preferred-column selection yields nothing numeric
    if not cluster_columns:
        cluster_df, cluster_columns = clustering_dataframe_for_modeling(at_risk)
        modeling_df = cluster_df
        input_columns = cluster_columns

    if not input_columns or cluster_df.empty:
        print(f"  {page_id}: insufficient columns for API factor/cluster training")
        return None, None, None, None

    risk_csv = dataframe_to_csv_bytes(modeling_df)
    risk_dataset_name = dataset_name_for_dataframe("risk_customers", modeling_df, f"{page_id}_demo")
    print(f"  uploading risk dataset ({len(modeling_df)} rows, {len(input_columns)} cols)...")
    risk_dataset_id = get_or_create_dataset_from_bytes(risk_csv, "risk_customers.csv", risk_dataset_name)
    print(f"  risk dataset id: {risk_dataset_id}")

    cluster_csv = dataframe_to_csv_bytes(cluster_df)
    cluster_dataset_name = dataset_name_for_dataframe("cluster_customers", cluster_df, risk_dataset_name)
    print(f"  uploading cluster dataset ({len(cluster_df)} rows, {len(cluster_columns)} cols)...")
    cluster_dataset_id = get_or_create_dataset_from_bytes(cluster_csv, "cluster_customers.csv", cluster_dataset_name)
    print(f"  cluster dataset id: {cluster_dataset_id}")

    print("  training factor analysis model...")
    factor_model_id, factor_job_id = train_model(
        risk_dataset_name.replace("risk_customers", "factor_analysis", 1),
        "factors",
        risk_dataset_id,
        input_columns=input_columns,
    )
    if factor_model_id:
        wait_for_training_complete(factor_model_id, factor_job_id)
        print(f"  factor model ready: {factor_model_id}")
    else:
        print("  factor model training failed")

    print("  training clustering model...")
    cluster_model_id, cluster_job_id = train_model(
        cluster_dataset_name.replace("cluster_customers", "customer_segments", 1),
        "clustering",
        cluster_dataset_id,
        input_columns=cluster_columns,
    )
    if cluster_model_id:
        wait_for_training_complete(cluster_model_id, cluster_job_id)
        print(f"  cluster model ready: {cluster_model_id}")
    else:
        print("  cluster model training failed")

    factors, factor_payload = pd.DataFrame(), {}
    if factor_model_id:
        factor_cache_key = inference_cache_key(
            "factor_inference", factor_model_id, hashlib.sha256(risk_csv).hexdigest(), "csv"
        )
        print("  running factor inference...")
        factors, factor_payload, _ = run_cached_model_inference(
            factor_model_id, risk_csv, "risk_customers.csv", "csv", factor_cache_key, "pattern inference"
        )
        print(f"  factor result cols: {list(factors.columns)[:8]}")

    clusters, cluster_payload = pd.DataFrame(), {}
    if cluster_model_id:
        cluster_cache_key = inference_cache_key(
            "cluster_inference", cluster_model_id, hashlib.sha256(cluster_csv).hexdigest(), "json"
        )
        print("  running cluster inference...")
        clusters, cluster_payload, _ = run_cached_model_inference(
            cluster_model_id, cluster_csv, "cluster_customers.csv", "json", cluster_cache_key, "cluster inference"
        )
        print(f"  cluster result cols: {list(clusters.columns)[:8]}")

    return factors, factor_payload, clusters, cluster_payload


def factors_to_patterns_csv(factors: pd.DataFrame) -> pd.DataFrame:
    """Convert API factor result to patterns.csv format."""
    if factors.empty:
        return pd.DataFrame()
    score_cols = [c for c in factors.columns if c.endswith("score")]
    rows = []
    for sc in score_cols:
        prefix = sc.removesuffix("score")
        desc_col = f"{prefix}description"
        var_col = f"{prefix}captured_variance"
        description = factors[desc_col].iloc[0] if desc_col in factors.columns else prefix.strip("_")
        variance = float(factors[var_col].iloc[0]) if var_col in factors.columns else 0.0
        rows.append({"pattern": description, "captured_variance": round(variance, 4), "top_drivers": ""})
    return pd.DataFrame(rows)


def process(page_id: str) -> None:
    artifact_dir = PROJECT_ROOT / "demo_artifacts" / page_id
    at_risk = pd.read_csv(artifact_dir / "at_risk.csv")
    intervention_plan = pd.read_csv(artifact_dir / "intervention_plan.csv")
    print(f"\n=== {page_id} ({len(at_risk)} rows) ===")

    # 1. Per-row explanations (save immediately)
    print("  generating per-row explanations...")
    at_risk = enrich_explanations(at_risk, page_id)
    print(f"  unique explanations: {at_risk['prediction_class_explanation'].nunique()}")
    at_risk.to_csv(artifact_dir / "at_risk.csv", index=False)
    intervention_plan.to_csv(artifact_dir / "intervention_plan.csv", index=False)
    print("  saved at_risk.csv and intervention_plan.csv")

    # 2. Real API factors + clusters (best-effort)
    try:
        factors, factor_payload, clusters, cluster_payload = run_api_factors_and_clusters(at_risk, page_id)
    except Exception as exc:
        print(f"  API factors/clusters failed: {exc}")
        return

    if cluster_payload:
        cluster_labels, cluster_descriptions = parse_cluster_inference_payload(
            cluster_payload, len(at_risk), clusters
        )
        at_risk["cluster_label"] = cluster_labels
        if "cluster_label" in intervention_plan.columns:
            intervention_plan["cluster_label"] = cluster_labels[:len(intervention_plan)]
        at_risk.to_csv(artifact_dir / "at_risk.csv", index=False)
        intervention_plan.to_csv(artifact_dir / "intervention_plan.csv", index=False)
        print(f"  cluster labels: {pd.Series(cluster_labels).value_counts().to_dict()}")

        patterns_df = factors_to_patterns_csv(factors)
        if not patterns_df.empty:
            patterns_df.to_csv(artifact_dir / "patterns.csv", index=False)
            print(f"  saved API-derived patterns.csv ({len(patterns_df)} patterns)")
        else:
            print("  factor result empty — keeping sklearn patterns.csv")
    else:
        print("  API factors/clusters unavailable — keeping sklearn artifacts")


if __name__ == "__main__":
    demos = sys.argv[1:] or DEMOS
    for demo in demos:
        process(demo)
