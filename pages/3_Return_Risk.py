"""Return risk workflow page."""

import streamlit as st

from actions.return_actions import build_return_action_plan
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
from woodwide.core import (
    DEFAULT_RETURNS_TEST_DATASET_PATH,
    DEFAULT_RETURNS_TRAIN_DATASET_PATH,
    PREVIEW_ROW_COUNT,
    add_original_customer_features,
    add_prediction_descriptions,
    add_row_level_explanations,
    api_key,
    as_number,
    dataset_name_for_upload,
    filter_high_return_risk,
    get_or_create_dataset,
    get_or_start_model_training,
    get_row_level_explanations,
    prediction_input_columns_for_training,
    probability_sort_column,
    read_uploaded_csv,
    run_prediction_inference,
    train_model,
)

configure_demo_app("Return Risk")
init_shared_session()
apply_page_style()

LABEL_COLUMN = "is_returned"

render_brand_header(
    title="Return Risk",
    subtitle="Train a return prediction model, score orders, and generate proactive return-prevention actions.",
)

with st.sidebar:
    render_api_sidebar()
    st.divider()
    st.header("Inputs")
    uploaded_train = st.file_uploader(
        "Order history (training)",
        type=["csv"],
        help="Upload your own training CSV, or use the demo datasets button below.",
    )
    uploaded_test = st.file_uploader(
        "Orders to score",
        type=["csv"],
        help="Upload your own scoring CSV, or use the demo datasets button below.",
    )

    st.divider()
    st.header("Demo Datasets")
    render_demo_run_button(
        "returns_use_demo",
        DEFAULT_RETURNS_TRAIN_DATASET_PATH,
        DEFAULT_RETURNS_TEST_DATASET_PATH,
    )
    train_file, test_file = resolve_csv_sources(
        "returns_use_demo",
        uploaded_train,
        uploaded_test,
        DEFAULT_RETURNS_TRAIN_DATASET_PATH,
        DEFAULT_RETURNS_TEST_DATASET_PATH,
    )

    with st.expander("Download demo CSVs"):
        download_button_for_path("Download train.csv", DEFAULT_RETURNS_TRAIN_DATASET_PATH, "train.csv")
        download_button_for_path("Download test.csv", DEFAULT_RETURNS_TEST_DATASET_PATH, "test.csv")

    st.divider()
    st.header("Demo Settings")
    preview_row_count = st.slider("Preview rows", 100, 5000, PREVIEW_ROW_COUNT, 100)
    return_threshold = st.slider(
        "High return-risk threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
    )
    max_row_explanations = st.slider("Row explanations", 0, 50, 10, 5)
    show_raw_model_outputs = st.toggle("Show raw model outputs", value=False)

if not train_file and not test_file:
    workflow_not_ready_message()
    st.stop()

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Training file", dataset_source_label(train_file))
metric_col_2.metric("Scoring file", dataset_source_label(test_file))
metric_col_3.metric("API connection", "Ready" if api_key else "Missing key")

model_id = None

if train_file:
    st.subheader("1. Return Prediction Model")
    section_note(f"Supervised prediction on `{LABEL_COLUMN}` (model_type: prediction).")
    training_df = read_uploaded_csv(train_file)
    if LABEL_COLUMN not in training_df.columns:
        st.error(f"Training data must include a `{LABEL_COLUMN}` column.")
        st.stop()

    dataset_name = dataset_name_for_upload("return_history", train_file)
    input_columns = prediction_input_columns_for_training(training_df, LABEL_COLUMN)

    with st.spinner("Uploading training dataset...", show_time=True):
        dataset_id = get_or_create_dataset(train_file, dataset_name)

    with st.spinner("Training return model...", show_time=True):
        model_id = get_or_start_model_training(
            st.session_state.model_ids,
            dataset_name,
            f"returns:{dataset_name}",
            "return prediction",
            "Return prediction model is ready.",
            lambda: train_model(
                dataset_name.replace("return_history", "return_prediction", 1),
                "prediction",
                dataset_id,
                label_column=LABEL_COLUMN,
                input_columns=input_columns,
            ),
        )

action_plan = None

if test_file:
    if not train_file:
        st.warning("Upload training data first.")
    elif not model_id:
        st.warning("Wait for model training to finish before running inference.")
    else:
        st.subheader("2. High Return-Risk Orders")
        section_note(f"Orders at or above {return_threshold:.0%} return probability are prioritized.")

        with st.spinner("Scoring orders...", show_time=True):
            scoring_df = read_uploaded_csv(test_file)
            scored, inference_payload, inference_job_id = run_prediction_inference(
                model_id,
                test_file,
                filename="returns_test.csv",
            )
            scored = add_original_customer_features(scored, scoring_df)
            scored = add_prediction_descriptions(scored, inference_payload, inference_job_id, model_id)
            flagged = filter_high_return_risk(scored, return_threshold)

            if max_row_explanations and inference_job_id and "id" in flagged.columns:
                probability_column = probability_sort_column(flagged)
                candidates = flagged.copy()
                if probability_column:
                    candidates = candidates.assign(
                        _sort=probability_column
                    ).sort_values("_sort", ascending=False)
                row_ids = []
                for row_id in candidates["id"].head(max_row_explanations).dropna().tolist():
                    row_id_number = as_number(row_id)
                    row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))
                with st.spinner("Explaining highest-risk orders...", show_time=True):
                    explanations = get_row_level_explanations(inference_job_id, row_ids)
                    flagged = add_row_level_explanations(flagged, explanations)

            action_plan = build_return_action_plan(flagged)

        if show_raw_model_outputs:
            st.dataframe(scored, use_container_width=True, height=360)

if action_plan is not None:
    st.subheader("3. Return Action Plan")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("High-risk orders", f"{len(action_plan):,}")
    metric_b.metric("High severity", f"{(action_plan['action_severity'] == 'high').sum():,}")
    metric_c.metric("Threshold", f"{return_threshold:.0%}")

    preferred = [
        probability_sort_column(action_plan),
        "return_score",
        "action_severity",
        "action_category",
        "likely_signal",
        "recommended_action",
        "signal_explanation",
        "order_value_eur",
        "product_category",
        "late_delivery_risk",
    ]
    visible = [column for column in preferred if column and column in action_plan.columns]
    if not visible:
        visible = action_plan.columns.tolist()

    plan_tab, download_tab = st.tabs(["Preview", "Download"])
    with plan_tab:
        st.dataframe(action_plan[visible].head(preview_row_count), use_container_width=True, height=520)
    with download_tab:
        st.download_button(
            "Download return action plan CSV",
            action_plan.to_csv(index=False).encode("utf-8"),
            "return_action_plan.csv",
            "text/csv",
            use_container_width=True,
        )
