#!/usr/bin/env python3
"""Legacy sklearn baseline artifact generator (offline fallback only).

For production instant demos, use scripts/capture_demo_artifacts.py to capture
real Wood Wide API train + infer results instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from woodwide.core import default_intervention_catalog
from woodwide.evaluation import compute_binary_metrics, metrics_to_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEMO_CONFIGS = {
    "churn": {
        "eval_path": PROJECT_ROOT / "datasets/churn/eval.csv",
        "label_column": "Churn",
        "id_column": "CustomerID",
        "intervention_template": "streaming",
        "threshold": 0.5,
    },
    "noshow": {
        "eval_path": PROJECT_ROOT / "datasets/healthcare/eval.csv",
        "label_column": "no_show",
        "id_column": "PatientID",
        "intervention_template": "healthcare",
        "threshold": 0.5,
    },
}


def prepare_features(dataframe: pd.DataFrame, label_column: str, id_column: str):
    features = dataframe.drop(columns=[label_column], errors="ignore")
    for column in (id_column, "id"):
        if column in features.columns:
            features = features.drop(columns=[column])
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
        )
    preprocessor = ColumnTransformer(transformers)
    return preprocessor, features, numeric_columns, categorical_columns


def fit_and_score(eval_df: pd.DataFrame, label_column: str, id_column: str):
    preprocessor, features, numeric_columns, categorical_columns = prepare_features(
        eval_df,
        label_column,
        id_column,
    )
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    labels = eval_df[label_column]
    pipeline.fit(features, labels)
    probabilities = pipeline.predict_proba(features)[:, 1]
    scored = eval_df.copy()
    scored["prediction"] = (probabilities >= 0.5).astype(int)
    scored["prediction_prob"] = probabilities
    scored["id"] = np.arange(len(scored))
    return scored, pipeline, numeric_columns, categorical_columns


def driver_chart_from_pipeline(pipeline, numeric_columns, categorical_columns) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    try:
        feature_names = prep.get_feature_names_out()
    except AttributeError:
        return pd.DataFrame()

    coefficients = model.coef_[0]
    rows = []
    for name, weight in zip(feature_names, coefficients):
        clean_name = name.split("__", 1)[-1]
        rows.append(
            {
                "driver": clean_name[:48],
                "weighted_signal": float(abs(weight)),
                "direction": "risk" if weight > 0 else "protective",
                "customers": 1,
            }
        )
    chart = pd.DataFrame(rows).sort_values("weighted_signal", ascending=False).head(12)
    return chart


def sample_explanations(at_risk: pd.DataFrame, id_column: str, count: int = 8) -> list[dict]:
    examples = []
    probability_column = "prediction_prob" if "prediction_prob" in at_risk.columns else None
    ordered = at_risk.sort_values(probability_column, ascending=False) if probability_column else at_risk
    for _, row in ordered.head(count).iterrows():
        entity_id = row.get(id_column, row.get("id"))
        prob = row.get("prediction_prob", 0)
        examples.append(
            {
                "id": row.get("id"),
                entity_id: entity_id,
                "row_prediction_explanation": (
                    f"Model score {prob:.0%}: elevated risk driven by feature pattern "
                    f"similar to other positive outcomes in the holdout sample."
                ),
                "row_explanation_summary": "Top demo exemplar from instant artifact bundle.",
            }
        )
    return examples


def generate_for_demo(page_id: str, output_dir: Path, threshold: float):
    config = DEMO_CONFIGS[page_id]
    eval_df = pd.read_csv(config["eval_path"])
    label_column = config["label_column"]
    id_column = config["id_column"]

    scored, pipeline, numeric_columns, categorical_columns = fit_and_score(
        eval_df,
        label_column,
        id_column,
    )
    metrics = compute_binary_metrics(
        scored[label_column],
        scored["prediction_prob"],
        threshold=threshold,
    )
    at_risk = scored[scored["prediction_prob"] >= threshold].sort_values("prediction_prob", ascending=False)
    at_risk["model_risk_drivers"] = "Demo baseline model feature weights"
    at_risk["top_churn_drivers"] = at_risk["model_risk_drivers"]

    catalog = default_intervention_catalog(config["intervention_template"])
    intervention_plan = at_risk.copy()
    intervention_plan["intervention_urgency"] = pd.cut(
        intervention_plan["prediction_prob"],
        bins=[0, 0.65, 0.8, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    intervention_plan["intervention_category"] = catalog[0]["name"]
    intervention_plan["intervention_action"] = catalog[0]["high"]
    intervention_plan["cluster_label"] = 0
    intervention_plan["intervention_match_source"] = "instant_demo_artifact"

    driver_chart = driver_chart_from_pipeline(pipeline, numeric_columns, categorical_columns)
    explanations = sample_explanations(at_risk, id_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(metrics_to_json(metrics), encoding="utf-8")
    at_risk.to_csv(output_dir / "at_risk.csv", index=False)
    intervention_plan.to_csv(output_dir / "intervention_plan.csv", index=False)
    if not driver_chart.empty:
        driver_chart.to_csv(output_dir / "driver_chart.csv", index=False)
    (output_dir / "explanations.json").write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    print(f"Wrote instant artifacts for {page_id} -> {output_dir}")
    print(f"  AUC {metrics['auc_roc']:.3f} · lift@10% {metrics['lift_top_decile']:.1f}x")


def main():
    print(
        "WARNING: This script produces sklearn baseline artifacts, not real Wood Wide results.\n"
        "For instant demos, run: PYTHONPATH=. python scripts/capture_demo_artifacts.py\n"
    )
    parser = argparse.ArgumentParser(description="Generate offline sklearn baseline artifact bundles")
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["churn", "noshow"],
        choices=list(DEMO_CONFIGS.keys()),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    for page_id in args.demos:
        generate_for_demo(page_id, PROJECT_ROOT / "demo_artifacts" / page_id, args.threshold)


if __name__ == "__main__":
    main()
