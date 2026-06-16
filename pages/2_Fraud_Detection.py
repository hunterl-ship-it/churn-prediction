"""Fraud detection workflow page."""

import pandas as pd
import streamlit as st

from actions.fraud_review import build_fraud_review_plan
from shared.ui import (
    apply_page_style,
    configure_demo_app,
    dataset_source_label,
    download_button_for_path,
    init_shared_session,
    render_api_sidebar,
    render_brand_header,
    render_demo_run_button,
    resolve_csv_sources,
    workflow_not_ready_message,
)

from workflows.fraud import (
    FRAUD_LABEL_COLUMN,
    prepare_fraud_anomaly_datasets,
    prepare_fraud_prediction_datasets,
)
from woodwide.core import (
    DEFAULT_FRAUD_ANOMALY_TRAIN_DATASET_PATH,
    DEFAULT_FRAUD_PREDICTION_TRAIN_DATASET_PATH,
    DEFAULT_FRAUD_SCORING_DATASET_PATH,
    PREVIEW_ROW_COUNT,
    add_original_customer_features,
    add_prediction_descriptions,
    add_row_level_explanations,
    anomaly_score_column,
    api_key,
    as_number,
    dataset_name_for_upload,
    filter_flagged_transactions,
    filter_high_fraud_risk,
    get_or_create_dataset,
    get_or_create_dataset_from_bytes,
    get_or_start_model_training,
    get_row_level_explanations,
    prediction_input_columns_for_training,
    probability_sort_column,
    read_uploaded_csv,
    run_anomaly_inference,
    run_prediction_inference,
    train_model,
)

configure_demo_app("Fraud Detection")
init_shared_session()
apply_page_style()

render_brand_header(
    title="Fraud Detection",
    subtitle=(
        "Score transactions for fraud risk and build a prioritized review queue. "
        "Uses supervised prediction on is_fraud by default; switch to anomaly mode when labels are unavailable."
    ),
)

with st.sidebar:
    render_api_sidebar()
    st.divider()
    st.header("Detection mode")
    fraud_mode = st.radio(
        "Model approach",
        options=("prediction", "anomaly"),
        format_func=lambda value: "Prediction (is_fraud label)" if value == "prediction" else "Anomaly (normal-only, no labels)",
        index=0,
        help="Prediction optimizes for fraud specifically. Anomaly flags unusual transactions vs a normal baseline.",
    )

    st.divider()
    st.header("Inputs")
    if fraud_mode == "prediction":
        train_label = f"Fraud-labeled history (must include `{FRAUD_LABEL_COLUMN}`)"
    else:
        train_label = "Normal transactions only (no fraud labels)"
    uploaded_train = st.file_uploader(
        train_label,
        type=["csv"],
        help="Upload your own training CSV, or use the demo datasets button below.",
    )
    uploaded_test = st.file_uploader(
        "Transactions to score",
        type=["csv"],
        help="Upload your own scoring CSV, or use the demo datasets button below.",
    )
    default_train_path = (
        DEFAULT_FRAUD_PREDICTION_TRAIN_DATASET_PATH
        if fraud_mode == "prediction"
        else DEFAULT_FRAUD_ANOMALY_TRAIN_DATASET_PATH
    )

    st.divider()
    st.header("Demo Datasets")
    render_demo_run_button(
        f"fraud_use_demo:{fraud_mode}",
        default_train_path,
        DEFAULT_FRAUD_SCORING_DATASET_PATH,
    )
    train_file, test_file = resolve_csv_sources(
        f"fraud_use_demo:{fraud_mode}",
        uploaded_train,
        uploaded_test,
        default_train_path,
        DEFAULT_FRAUD_SCORING_DATASET_PATH,
    )

    with st.expander("Download demo CSVs"):
        if fraud_mode == "prediction":
            download_button_for_path(
                "Download train.csv (with is_fraud)",
                DEFAULT_FRAUD_PREDICTION_TRAIN_DATASET_PATH,
                "fraud_train_labeled.csv",
            )
        else:
            download_button_for_path(
                "Download fraud_train.csv (normal only)",
                DEFAULT_FRAUD_ANOMALY_TRAIN_DATASET_PATH,
                "fraud_train.csv",
            )
        download_button_for_path(
            "Download fraud_test.csv (scoring)",
            DEFAULT_FRAUD_SCORING_DATASET_PATH,
            "fraud_test.csv",
        )

    raw_file = st.file_uploader(
        "Or upload raw ecommerce CSV",
        type=["csv"],
        help="Optional: full dataset with is_fraud for one-click prep.",
    )

    st.divider()
    st.header("Demo Settings")
    preview_row_count = st.slider("Preview rows", 100, 5000, PREVIEW_ROW_COUNT, 100)
    score_label = "Fraud probability" if fraud_mode == "prediction" else "Anomaly score"
    fraud_threshold = st.slider(
        f"{score_label} threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50 if fraud_mode == "prediction" else 0.62,
        step=0.01,
        help="Transactions at or above this score are flagged for review.",
    )
    max_row_explanations = st.slider("Row explanations", 0, 50, 10, 5) if fraud_mode == "prediction" else 0
    show_raw_model_outputs = st.toggle("Show raw model outputs", value=False)

    if raw_file:
        if st.button("Prepare datasets from raw CSV", use_container_width=True):
            raw_df = read_uploaded_csv(raw_file)
            try:
                if fraud_mode == "prediction":
                    prepared_train, prepared_test = prepare_fraud_prediction_datasets(raw_df)
                else:
                    prepared_train, prepared_test = prepare_fraud_anomaly_datasets(raw_df)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["prepared_fraud_train"] = prepared_train
                st.session_state["prepared_fraud_test"] = prepared_test
                st.session_state["prepared_fraud_mode"] = fraud_mode
                st.success("Prepared training and scoring files from raw upload.")

prepared_train = st.session_state.get("prepared_fraud_train")
prepared_test = st.session_state.get("prepared_fraud_test")
prepared_mode = st.session_state.get("prepared_fraud_mode")

if prepared_train is not None and prepared_mode != fraud_mode:
    st.warning("Prepared datasets were built for a different mode. Re-run prep or upload mode-specific files.")
    prepared_train = None
    prepared_test = None

training_source = train_file if train_file is not None else prepared_train
scoring_source = test_file if test_file is not None else prepared_test

if not training_source and not scoring_source:
    workflow_not_ready_message()
    st.stop()

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Mode", "Prediction" if fraud_mode == "prediction" else "Anomaly")
metric_col_2.metric("Training file", dataset_source_label(training_source))
metric_col_3.metric("Scoring file", dataset_source_label(scoring_source))
metric_col_4.metric("API connection", "Ready" if api_key else "Missing key")

model_id = None


def training_bytes_and_name(source):
    if isinstance(source, pd.DataFrame):
        training_df = source
        train_bytes = training_df.to_csv(index=False).encode()
        dataset_name = f"fraud_{fraud_mode}_prepared_{hash(train_bytes) & 0xFFFFFFFF:08x}"
    else:
        training_df = read_uploaded_csv(source)
        train_bytes = source.getvalue()
        prefix = "fraud_labeled" if fraud_mode == "prediction" else "fraud_normal"
        dataset_name = dataset_name_for_upload(prefix, source)
    return training_df, train_bytes, dataset_name


def scoring_bytes_and_frame(source):
    if isinstance(source, pd.DataFrame):
        scoring_df = source
        file_bytes = scoring_df.to_csv(index=False).encode()
    else:
        scoring_df = read_uploaded_csv(source)
        source.seek(0)
        file_bytes = source.getvalue()
    return scoring_df, file_bytes


if training_source is not None:
    training_df, train_bytes, dataset_name = training_bytes_and_name(training_source)

    if fraud_mode == "prediction":
        st.subheader("1. Fraud Prediction Model")
        section_note(f"Supervised classification on `{FRAUD_LABEL_COLUMN}` (model_type: prediction).")
        if FRAUD_LABEL_COLUMN not in training_df.columns:
            st.error(f"Training data must include `{FRAUD_LABEL_COLUMN}` for prediction mode.")
            st.stop()
        input_columns = prediction_input_columns_for_training(training_df, FRAUD_LABEL_COLUMN)
        pending_prefix = f"fraud-prediction:{dataset_name}"
        model_type = "prediction"
        training_label = "fraud prediction"
        ready_message = "Fraud prediction model is ready."
        train_kwargs = {
            "label_column": FRAUD_LABEL_COLUMN,
            "input_columns": input_columns,
        }
    else:
        st.subheader("1. Anomaly Model")
        section_note("Unsupervised baseline on normal transactions only (model_type: anomaly, no label).")
        pending_prefix = f"fraud-anomaly:{dataset_name}"
        model_type = "anomaly"
        training_label = "fraud anomaly"
        ready_message = "Fraud anomaly model is ready."
        train_kwargs = {}

    with st.spinner("Uploading training dataset...", show_time=True):
        if isinstance(training_source, pd.DataFrame):
            dataset_id = get_or_create_dataset_from_bytes(train_bytes, "fraud_train.csv", dataset_name)
        else:
            training_source.seek(0)
            dataset_id = get_or_create_dataset(training_source, dataset_name)

    model_ready_key = f"{fraud_mode}:{dataset_name}"

    with st.spinner(f"Training {training_label} model...", show_time=True):
        model_id = get_or_start_model_training(
            st.session_state.model_ids,
            model_ready_key,
            pending_prefix,
            training_label,
            ready_message,
            lambda: train_model(
                dataset_name.replace("fraud_labeled", "fraud_prediction", 1).replace(
                    "fraud_normal", "fraud_detection", 1
                ),
                model_type,
                dataset_id,
                **train_kwargs,
            ),
        )

review_plan = None

if scoring_source is not None:
    if model_id is None:
        st.warning("Upload training data and wait for the model to finish training.")
    else:
        st.subheader("2. Flagged Transactions")
        section_note(
            f"Transactions at or above {fraud_threshold:.0%} "
            f"{'fraud probability' if fraud_mode == 'prediction' else 'anomaly score'} are flagged for review."
        )

        scoring_df, file_bytes = scoring_bytes_and_frame(scoring_source)

        with st.spinner("Scoring transactions...", show_time=True):
            if fraud_mode == "prediction":
                if isinstance(scoring_source, pd.DataFrame):
                    scoring_upload = type(
                        "Upload",
                        (),
                        {"getvalue": lambda self: file_bytes, "name": "fraud_test.csv"},
                    )()
                    scored, inference_payload, inference_job_id = run_prediction_inference(
                        model_id,
                        scoring_upload,
                        filename="fraud_test.csv",
                    )
                else:
                    scoring_source.seek(0)
                    scored, inference_payload, inference_job_id = run_prediction_inference(
                        model_id,
                        scoring_source,
                        filename="fraud_test.csv",
                    )
                scored = add_original_customer_features(scored, scoring_df)
                scored = add_prediction_descriptions(scored, inference_payload, inference_job_id, model_id)
                flagged = filter_high_fraud_risk(scored, fraud_threshold)

                if max_row_explanations and inference_job_id and "id" in flagged.columns:
                    probability_column = probability_sort_column(flagged)
                    candidates = flagged.copy()
                    if probability_column:
                        candidates = candidates.assign(
                            _sort=pd.to_numeric(candidates[probability_column], errors="coerce")
                        ).sort_values("_sort", ascending=False)
                    row_ids = []
                    for row_id in candidates["id"].head(max_row_explanations).dropna().tolist():
                        row_id_number = as_number(row_id)
                        row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))
                    with st.spinner("Explaining highest-risk transactions...", show_time=True):
                        explanations = get_row_level_explanations(inference_job_id, row_ids)
                        flagged = add_row_level_explanations(flagged, explanations)
            else:
                scored, _, _ = run_anomaly_inference(model_id, file_bytes, "fraud_test.csv")
                scored = add_original_customer_features(scored, scoring_df)
                flagged = filter_flagged_transactions(scored, fraud_threshold)

            review_plan = build_fraud_review_plan(flagged)

        if show_raw_model_outputs:
            st.dataframe(scored, use_container_width=True, height=360)

if review_plan is not None:
    score_column = probability_sort_column(review_plan) if fraud_mode == "prediction" else anomaly_score_column(review_plan)
    st.subheader("3. Fraud Review Plan")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Flagged transactions", f"{len(review_plan):,}")
    metric_b.metric("High severity", f"{(review_plan['review_severity'] == 'high').sum():,}")
    metric_c.metric("Threshold", f"{fraud_threshold:.0%}")

    preferred = [
        score_column,
        "prediction",
        "fraud_score",
        "review_severity",
        "review_category",
        "likely_signal",
        "recommended_action",
        "signal_explanation",
        "row_prediction_explanation",
        "order_value_eur",
        "payment_method",
        "address_mismatch",
        "high_risk_ip",
    ]
    visible = [column for column in preferred if column and column in review_plan.columns]
    if not visible:
        visible = review_plan.columns.tolist()

    plan_tab, download_tab = st.tabs(["Preview", "Download"])
    with plan_tab:
        st.dataframe(review_plan[visible].head(preview_row_count), use_container_width=True, height=520)
    with download_tab:
        st.download_button(
            "Download fraud review plan CSV",
            review_plan.to_csv(index=False).encode("utf-8"),
            "fraud_review_plan.csv",
            "text/csv",
            use_container_width=True,
        )
