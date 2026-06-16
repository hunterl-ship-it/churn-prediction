#!/usr/bin/env python3
"""Capture instant demo artifacts from a real Wood Wide API train + infer run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._streamlit_headless import HeadlessStop, install_streamlit_mock

install_streamlit_mock()

import pandas as pd

import woodwide.core as core
from woodwide.core import (
    add_original_customer_features,
    add_prediction_descriptions,
    add_row_level_explanations,
    as_number,
    build_interventions,
    dataset_name_for_dataframe,
    dataset_name_for_upload,
    filter_at_risk_customers,
    get_or_create_dataset_from_bytes,
    get_row_level_explanations,
    prediction_input_columns_for_training,
    probability_sort_column,
    read_uploaded_csv,
    risk_driver_chart_data,
    risk_driver_chart_data_from_feature_contrast,
    risk_driver_chart_data_from_feature_entries,
    run_prediction_inference,
    train_model,
    wait_for_training_complete,
)
from woodwide.evaluation import compute_binary_metrics, metrics_to_json, oriented_probability_series
from workflows.supervised_outcome import CHURN_DEMO_CONFIG, NOSHOW_DEMO_CONFIG, OutcomeDemoConfig

DEMO_CONFIGS: dict[str, OutcomeDemoConfig] = {
    "churn": CHURN_DEMO_CONFIG,
    "noshow": NOSHOW_DEMO_CONFIG,
}


class CsvUpload:
    def __init__(self, path: str | Path):
        path = Path(path)
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._bytes

    def seek(self, position: int) -> None:
        return None


def configure_capture_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / ".capture_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    core.LOCAL_CACHE_DIR = str(cache_dir)
    core.LOCAL_CACHE_DB_PATH = os.path.join(core.LOCAL_CACHE_DIR, "woodwide_jobs.sqlite3")
    core.init_local_cache()


def train_prediction_model(
    config: OutcomeDemoConfig,
    train_path: Path,
    sample_rows: int | None,
    force_retrain: bool = False,
) -> tuple[str, str]:
    train_df = pd.read_csv(train_path)
    if sample_rows and len(train_df) > sample_rows:
        train_df = train_df.sample(n=sample_rows, random_state=42)
        print(f"  Training on {sample_rows:,} sampled rows (of {len(pd.read_csv(train_path)):,})")
    else:
        print(f"  Training on {len(train_df):,} rows")

    label_column = config.get_label_column(train_df, f"capture:{config.page_id}")
    input_columns = prediction_input_columns_for_training(train_df, label_column)
    dataset_name = dataset_name_for_dataframe(config.dataset_prefix, train_df, train_path.stem)
    file_bytes = train_df.to_csv(index=False).encode("utf-8")

    print(f"  Uploading dataset {dataset_name}...")
    dataset_id = get_or_create_dataset_from_bytes(file_bytes, train_path.name, dataset_name)

    model_name = dataset_name.replace(config.model_train_name_from, config.model_train_name_to, 1)
    if force_retrain:
        model_name = f"{model_name}_retrain_{uuid.uuid4().hex[:8]}"
        print(f"  Force retrain: new model name {model_name}")
    print(f"  Training model {model_name}...")
    model_id, job_id = train_model(
        model_name,
        "prediction",
        dataset_id,
        label_column=label_column,
        input_columns=input_columns,
    )

    training_status = wait_for_training_complete(model_id, job_id)
    if training_status is not True:
        raise RuntimeError(f"Model training did not complete successfully (status={training_status})")

    return model_id, label_column


def infer_csv(model_id: str, csv_path: Path) -> tuple[pd.DataFrame, dict, str | None]:
    upload = CsvUpload(csv_path)
    scored, payload, job_id = run_prediction_inference(model_id, upload, filename=csv_path.name)
    return scored, payload, job_id


def build_driver_chart(
    at_risk: pd.DataFrame,
    scored: pd.DataFrame,
    feature_entries_key: str,
) -> pd.DataFrame:
    feature_entries = core.st.session_state.get(feature_entries_key, {})
    driver_chart = risk_driver_chart_data_from_feature_entries(feature_entries, at_risk)
    if driver_chart.empty:
        driver_chart = risk_driver_chart_data(at_risk)
    if driver_chart.empty:
        driver_chart = risk_driver_chart_data_from_feature_contrast(at_risk, scored)
    return driver_chart


def build_intervention_plan(at_risk: pd.DataFrame) -> pd.DataFrame:
    cluster_labels = [0] * len(at_risk)
    return build_interventions(at_risk, pd.DataFrame(), cluster_labels, [""])


def explanations_to_json(explanations: pd.DataFrame) -> list[dict]:
    if explanations.empty:
        return []
    return json.loads(explanations.to_json(orient="records"))


def capture_demo(
    page_id: str,
    output_dir: Path,
    threshold: float,
    max_explanations: int,
    reuse_model_id: str | None,
    sample_train_rows: int | None,
    force_retrain: bool = False,
) -> None:
    if not core.api_key:
        raise SystemExit("WOODWIDE_API_KEY is required. Set it in .env or the environment.")

    config = DEMO_CONFIGS[page_id]
    train_path = PROJECT_ROOT / config.default_train_path
    test_path = PROJECT_ROOT / config.default_test_path
    eval_path = PROJECT_ROOT / config.default_eval_path

    print(f"\n=== Capturing {page_id} demo artifacts ===")
    print(f"  API: {core.base_url}")

    if reuse_model_id:
        model_id = reuse_model_id
        eval_df = pd.read_csv(eval_path)
        label_column = config.get_label_column(eval_df, f"capture:{config.page_id}")
        print(f"  Reusing model {model_id}")
    else:
        model_id, label_column = train_prediction_model(
            config,
            train_path,
            sample_train_rows,
            force_retrain=force_retrain,
        )

    print("  Running holdout eval inference...")
    eval_source = pd.read_csv(eval_path)
    scored_eval, eval_payload, eval_job_id = infer_csv(model_id, eval_path)
    probability_column = probability_sort_column(scored_eval)
    if not probability_column:
        raise RuntimeError("Eval inference did not return a probability column.")

    oriented_probs, scores_flipped = oriented_probability_series(
        eval_source[label_column],
        scored_eval[probability_column],
    )
    scored_eval = scored_eval.copy()
    scored_eval["risk_probability"] = oriented_probs

    metrics = compute_binary_metrics(
        eval_source[label_column],
        scored_eval[probability_column],
        threshold=threshold,
    )
    print(
        f"  Eval metrics: AUC {metrics['auc_roc']:.3f} · "
        f"lift@10% {metrics['lift_top_decile']:.1f}x · "
        f"{metrics['eval_rows']:,} rows"
    )

    print("  Running test-set inference...")
    test_df = pd.read_csv(test_path)
    scored_test, test_payload, test_job_id = infer_csv(model_id, test_path)
    scored_test = add_original_customer_features(scored_test, test_df)
    scored_test = add_prediction_descriptions(
        scored_test,
        test_payload,
        test_job_id,
        model_id,
    )
    core.st.session_state[config.feature_entries_key] = core.st.session_state.get(
        config.feature_entries_key,
        core.st.session_state.get("churn_feature_entries_by_label", {}),
    )

    test_probability_column = probability_sort_column(scored_test)
    if test_probability_column:
        test_probs = pd.to_numeric(scored_test[test_probability_column], errors="coerce")
        scored_test["risk_probability"] = 1.0 - test_probs if scores_flipped else test_probs

    at_risk = filter_at_risk_customers(scored_test, threshold)
    print(f"  At-risk rows: {len(at_risk):,} (threshold {threshold:.0%})")

    explanations = pd.DataFrame()
    if max_explanations and test_job_id and "id" in at_risk.columns:
        probability_column = probability_sort_column(at_risk)
        candidates = at_risk.copy()
        if probability_column:
            candidates = candidates.assign(
                _explain_sort=pd.to_numeric(candidates[probability_column], errors="coerce")
            ).sort_values("_explain_sort", ascending=False)
        row_ids = []
        for row_id in candidates["id"].head(max_explanations).dropna().tolist():
            row_id_number = as_number(row_id)
            row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))
        if row_ids:
            print(f"  Fetching row explanations for {len(row_ids)} records...")
            explanations = get_row_level_explanations(test_job_id, row_ids)
            at_risk = add_row_level_explanations(at_risk, explanations)

    driver_chart = build_driver_chart(at_risk, scored_test, config.feature_entries_key)
    intervention_plan = build_intervention_plan(at_risk)

    metadata = {
        "source": "woodwide_api",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page_id,
        "model_id": model_id,
        "eval_job_id": eval_job_id,
        "test_job_id": test_job_id,
        "label_column": label_column,
        "threshold": threshold,
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "test_path": str(test_path.relative_to(PROJECT_ROOT)),
        "eval_path": str(eval_path.relative_to(PROJECT_ROOT)),
        "api_base_url": core.base_url,
        "scores_flipped": scores_flipped,
        "metrics_summary": {
            "auc_roc": metrics["auc_roc"],
            "lift_top_decile": metrics["lift_top_decile"],
            "eval_rows": metrics["eval_rows"],
        },
        "at_risk_rows": len(at_risk),
        "explained_rows": len(explanations),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(metrics_to_json(metrics), encoding="utf-8")
    at_risk.to_csv(output_dir / "at_risk.csv", index=False)
    intervention_plan.to_csv(output_dir / "intervention_plan.csv", index=False)
    if not driver_chart.empty:
        driver_chart.to_csv(output_dir / "driver_chart.csv", index=False)
    (output_dir / "explanations.json").write_text(
        json.dumps(explanations_to_json(explanations), indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"  Wrote artifacts -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture instant demo artifacts from a real Wood Wide API run",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        default=["churn", "noshow"],
        choices=list(DEMO_CONFIGS.keys()),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-explanations", type=int, default=25)
    parser.add_argument(
        "--reuse-model-id",
        action="append",
        default=[],
        metavar="PAGE:MODEL_ID",
        help="Skip training for a demo, e.g. churn:019e88ae-901b-79b5-b4d0-f4294aef7e92",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=None,
        help="Train on a random sample of N rows (faster dev capture)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Isolated Wood Wide job cache directory (default: .capture_cache/)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train a fresh model (new model name) instead of reusing cached/existing models",
    )
    args = parser.parse_args()

    reuse_by_page = {}
    for entry in args.reuse_model_id:
        page_id, model_id = entry.split(":", 1)
        reuse_by_page[page_id] = model_id

    configure_capture_cache(args.cache_dir)

    for page_id in args.demos:
        try:
            capture_demo(
                page_id,
                PROJECT_ROOT / "demo_artifacts" / page_id,
                args.threshold,
                args.max_explanations,
                reuse_by_page.get(page_id),
                args.sample_train,
                force_retrain=args.retrain,
            )
        except HeadlessStop as exc:
            raise SystemExit(f"Capture aborted for {page_id}: {exc}") from exc


if __name__ == "__main__":
    main()
