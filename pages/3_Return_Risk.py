"""Return risk workflow page."""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from actions.return_actions import build_return_action_plan
from shared.dashboard import init_dashboard_state, render_widget_controls, render_dashboard_header
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
    section_note,
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

PAGE_ID = "returns"

ALL_RETURN_WIDGETS = {
    "raw_output": "Raw Scored Orders",
    "action_metrics": "Action Plan Metrics",
    "action_plan": "Return Action Plan Table",
    "severity_pie": "Action Severity Pie Chart",
    "category_pie": "Action Category Pie Chart",
}

configure_demo_app("Return Risk")
init_shared_session()
apply_page_style()
init_dashboard_state(PAGE_ID, list(ALL_RETURN_WIDGETS.keys()))

LABEL_COLUMN = "is_returned"

render_brand_header(
    title="Return Risk",
    subtitle="Train a return prediction model, score orders, and generate proactive return-prevention actions.",
)

with st.sidebar:
    render_api_sidebar()
    st.divider()
    st.header("Demo Datasets")
    render_demo_run_button(
        "returns_use_demo",
        DEFAULT_RETURNS_TRAIN_DATASET_PATH,
        DEFAULT_RETURNS_TEST_DATASET_PATH,
    )
    train_file, test_file = resolve_csv_sources(
        "returns_use_demo",
        DEFAULT_RETURNS_TRAIN_DATASET_PATH,
        DEFAULT_RETURNS_TEST_DATASET_PATH,
    )

    with st.expander("Download demo CSVs"):
        download_button_for_path("Download train.csv", DEFAULT_RETURNS_TRAIN_DATASET_PATH, "train.csv")
        download_button_for_path("Download test.csv", DEFAULT_RETURNS_TEST_DATASET_PATH, "test.csv")

    st.divider()
    st.header("Demo Settings")
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

# ── Compute scored + action plan ────────────────────────────────────────────────
action_plan = None
scored = None

if test_file:
    if not train_file:
        st.warning("Select training data first.")
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
                        _sort=pd.to_numeric(candidates[probability_column], errors="coerce")
                    ).sort_values("_sort", ascending=False)
                row_ids = []
                for row_id in candidates["id"].head(max_row_explanations).dropna().tolist():
                    row_id_number = as_number(row_id)
                    row_ids.append(int(row_id_number) if row_id_number is not None else str(row_id))
                with st.spinner("Explaining highest-risk orders...", show_time=True):
                    explanations = get_row_level_explanations(inference_job_id, row_ids)
                    flagged = add_row_level_explanations(flagged, explanations)

            action_plan = build_return_action_plan(flagged)

# ── Dashboard widget rendering ──────────────────────────────────────────────────
if action_plan is not None:
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
    visible = [c for c in preferred if c and c in action_plan.columns] or action_plan.columns.tolist()

    severity_pie_data = {}
    if "action_severity" in action_plan.columns:
        severity_pie_data = {
            str(k).title(): int(v)
            for k, v in action_plan.groupby("action_severity").size().to_dict().items()
        }

    category_pie_data = {}
    if "action_category" in action_plan.columns:
        category_pie_data = {
            str(k)[:28]: int(v)
            for k, v in action_plan.groupby("action_category").size().to_dict().items()
        }

    data_dict = {
        "metrics": {
            "High-risk orders": f"{len(action_plan):,}",
            "High severity": f"{(action_plan['action_severity'] == 'high').sum():,}" if "action_severity" in action_plan.columns else "N/A",
            "Threshold": f"{return_threshold:.0%}",
        },
        "severity_pie_data": severity_pie_data,
        "category_pie_data": category_pie_data,
        "action_plan_df": action_plan,
        "raw_output_df": scored if scored is not None else pd.DataFrame(),
    }

    render_dashboard_header(PAGE_ID, ALL_RETURN_WIDGETS, data_dict=data_dict)

    active_widgets = st.session_state[f"{PAGE_ID}_active_widgets"]

    def _pie(data: dict, title: str):
        if not data:
            st.info("No data available.")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#0B3D2E", "#175cd3", "#f04438", "#f79009", "#668575", "#d8e2dc"]
        ax.pie(
            list(data.values()),
            labels=list(data.keys()),
            autopct="%1.1f%%",
            colors=colors[: len(data)],
            startangle=140,
        )
        ax.axis("equal")
        ax.set_title(title, fontsize=12, weight="bold", color="#0B3D2E")
        fig.patch.set_facecolor("#ffffff")
        st.pyplot(fig)
        plt.close()

    for idx, widget_id in enumerate(active_widgets):
        if widget_id not in ALL_RETURN_WIDGETS:
            continue
        widget_name = ALL_RETURN_WIDGETS[widget_id]
        collapsed = render_widget_controls(PAGE_ID, idx, widget_id, widget_name)
        if collapsed:
            continue

        if widget_id == "raw_output":
            if not show_raw_model_outputs:
                st.info("Enable **Show raw model outputs** in the sidebar to display this widget.")
            elif scored is not None:
                row_sel, _ = st.columns([1, 3])
                with row_sel:
                    n = st.selectbox(
                        "Rows to display",
                        [10, 25, 50, 100, 250, 500, 1000],
                        index=3,
                        key=f"{PAGE_ID}_{widget_id}_rows",
                    )
                st.dataframe(scored.head(n), use_container_width=True, height=360)
            else:
                st.info("No raw output available yet.")

        elif widget_id == "action_metrics":
            m1, m2, m3 = st.columns(3)
            m1.metric("High-risk orders", f"{len(action_plan):,}")
            if "action_severity" in action_plan.columns:
                m2.metric("High severity", f"{(action_plan['action_severity'] == 'high').sum():,}")
            m3.metric("Threshold", f"{return_threshold:.0%}")

        elif widget_id == "action_plan":
            row_sel, _ = st.columns([1, 3])
            with row_sel:
                n = st.selectbox(
                    "Rows to display",
                    [10, 25, 50, 100, 250, 500, 1000],
                    index=3,
                    key=f"{PAGE_ID}_{widget_id}_rows",
                )
            plan_tab, dl_tab = st.tabs(["Preview", "Download"])
            with plan_tab:
                st.dataframe(action_plan[visible].head(n), use_container_width=True, height=520)
            with dl_tab:
                st.download_button(
                    "Download return action plan CSV",
                    action_plan.to_csv(index=False).encode("utf-8"),
                    "return_action_plan.csv",
                    "text/csv",
                    use_container_width=True,
                    key=f"{PAGE_ID}_{widget_id}_dl",
                )

        elif widget_id == "severity_pie":
            _pie(severity_pie_data, "Action Severity Distribution")

        elif widget_id == "category_pie":
            _pie(category_pie_data, "Action Category Distribution")

        st.divider()
